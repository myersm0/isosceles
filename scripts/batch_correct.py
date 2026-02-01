#!/usr/bin/env python3
"""
Batch processing for LLM-based annotation corrections.

Reads a single .conllu file, sends sentences to Anthropic's batch API
for surface corrections (lemma, upos, feats), and writes corrected output.

Workflow:
    # Prepare
    python batch_correct.py prepare tombe.conllu -p prompt_surface.txt -o work/
    # Creates: work/tombe.jsonl, work/tombe.mapping.json

    # Submit (seeds 1h cache, then submits batch)
    python batch_correct.py submit work/tombe.jsonl
    # Creates: work/tombe.state.json

    # Check / apply
    python batch_correct.py resume work/tombe.state.json
    # When done, writes work/tombe.conllu

    # Bulk
    for f in spacy/*.conllu; do python batch_correct.py prepare "$f" -p prompt.txt -o work/; done
    for f in work/*.jsonl; do python batch_correct.py submit "$f"; done
    python batch_correct.py resume work/*.state.json

    # Utilities
    python batch_correct.py list
    python batch_correct.py status <batch_id>
"""

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request


def parse_conllu(filepath):
	sentences = []
	current_tokens = []
	current_comments = []
	sent_id = None
	sent_text = None

	with open(filepath, encoding="utf-8") as f:
		for line in f:
			line = line.rstrip("\n")
			if not line:
				if current_tokens:
					sentences.append((sent_id, sent_text, current_comments, current_tokens))
					current_tokens = []
					current_comments = []
					sent_id = None
					sent_text = None
			elif line.startswith("#"):
				current_comments.append(line)
				if line.startswith("# sent_id"):
					sent_id = line.split("=", 1)[1].strip()
				elif line.startswith("# text"):
					sent_text = line.split("=", 1)[1].strip()
			else:
				fields = line.split("\t")
				if len(fields) >= 10 and "-" not in fields[0] and "." not in fields[0]:
					current_tokens.append({
						"id": int(fields[0]),
						"form": fields[1],
						"lemma": fields[2],
						"upos": fields[3],
						"xpos": fields[4],
						"feats": fields[5] if fields[5] != "_" else None,
						"head": int(fields[6]) if fields[6] != "_" else 0,
						"deprel": fields[7],
						"deps": fields[8],
						"misc": fields[9],
					})

		if current_tokens:
			sentences.append((sent_id, sent_text, current_comments, current_tokens))

	return sentences


def write_conllu(sentences, filepath):
	with open(filepath, "w", encoding="utf-8") as f:
		for sent_id, sent_text, comments, tokens in sentences:
			for comment in comments:
				f.write(comment + "\n")
			for t in tokens:
				feats = t.get("feats") or "_"
				f.write(
					f"{t['id']}\t{t['form']}\t{t['lemma']}\t{t['upos']}\t{t['xpos']}"
					f"\t{feats}\t{t['head']}\t{t['deprel']}\t{t['deps']}\t{t['misc']}\n"
				)
			f.write("\n")


def format_tokens_for_llm(tokens):
	lines = []
	for t in tokens:
		feats = t.get("feats") or "_"
		lines.append(f"{t['id']}\t{t['form']}\t{t['lemma']}\t{t['upos']}\t{feats}")
	return "\n".join(lines)


def parse_corrections_json(response_text):
	match = re.search(r"\{[\s\S]*\}", response_text)
	if match:
		try:
			return json.loads(match.group()).get("corrections", [])
		except json.JSONDecodeError:
			pass
	return []


def apply_corrections(tokens, corrections):
	tokens = copy.deepcopy(tokens)
	token_map = {t["id"]: t for t in tokens}
	safe_fields = {"lemma", "upos", "xpos", "feats"}
	n_applied = 0

	for c in corrections:
		field = c.get("field")
		value = c.get("value")
		if field not in safe_fields:
			continue
		try:
			tok_id = int(c.get("id"))
		except (ValueError, TypeError):
			continue
		if tok_id not in token_map:
			continue
		token_map[tok_id][field] = value
		n_applied += 1

	return tokens, n_applied


# --- Commands ---


def cmd_prepare(args):
	input_path = Path(args.conllu_file)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	prompt = Path(args.prompt).read_text(encoding="utf-8")

	sentences = parse_conllu(input_path)
	if not sentences:
		sys.exit(f"No sentences found in {input_path}")

	stem = input_path.stem
	requests = []
	mapping = []

	for i, (sent_id, sent_text, comments, tokens) in enumerate(sentences):
		requests.append({
			"custom_id": f"req_{i}",
			"params": {
				"model": args.model,
				"max_tokens": 1024,
				"system": [
					{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
				],
				"messages": [
					{"role": "user", "content": format_tokens_for_llm(tokens)}
				]
			}
		})
		mapping.append({"idx": i, "sent_id": sent_id})

	jsonl_path = output_dir / f"{stem}.jsonl"
	with open(jsonl_path, "w", encoding="utf-8") as f:
		for req in requests:
			f.write(json.dumps(req, ensure_ascii=False) + "\n")

	mapping_path = output_dir / f"{stem}.mapping.json"
	with open(mapping_path, "w", encoding="utf-8") as f:
		json.dump({
			"input_conllu": str(input_path.resolve()),
			"sentences": mapping,
		}, f, ensure_ascii=False, indent=2)

	size_mb = jsonl_path.stat().st_size / 1_000_000
	print(f"{stem}: {len(requests)} sentences, {size_mb:.2f} MB → {jsonl_path}", file=sys.stderr)


def cmd_submit(args):
	jsonl_path = Path(args.jsonl_file)
	stem = jsonl_path.stem

	mapping_path = jsonl_path.with_suffix(".mapping.json")
	input_conllu = None
	if mapping_path.exists():
		with open(mapping_path, encoding="utf-8") as f:
			meta = json.load(f)
		input_conllu = meta.get("input_conllu")

	with open(jsonl_path, encoding="utf-8") as f:
		raw_requests = [json.loads(line) for line in f]

	def make_request(r):
		return Request(
			custom_id=r["custom_id"],
			params=MessageCreateParamsNonStreaming(**r["params"])
		)

	client = anthropic.Anthropic()

	# Warm 1h cache with a single-request seed batch
	print(f"Warming cache ({stem})...", file=sys.stderr)
	seed_batch = client.messages.batches.create(requests=[make_request(raw_requests[0])])
	while True:
		status = client.messages.batches.retrieve(seed_batch.id)
		if status.processing_status == "ended":
			break
		time.sleep(5)
	print(f"Cache warm. Submitting {len(raw_requests)} requests...", file=sys.stderr)

	# Submit full batch — requests should hit the warm cache
	requests = [make_request(r) for r in raw_requests]
	batch = client.messages.batches.create(requests=requests)

	state_path = jsonl_path.with_suffix(".state.json")
	state = {
		"batch_id": batch.id,
		"seed_batch_id": seed_batch.id,
		"stem": stem,
		"input_conllu": input_conllu,
		"output_dir": str(jsonl_path.parent.resolve()),
		"created_at": str(batch.created_at),
		"expires_at": str(batch.expires_at),
	}
	with open(state_path, "w", encoding="utf-8") as f:
		json.dump(state, f, indent=2)

	print(f"Batch ID: {batch.id}", file=sys.stderr)
	print(f"State: {state_path}", file=sys.stderr)


def cmd_resume(args):
	client = anthropic.Anthropic()

	for state_file in args.state_files:
		state_path = Path(state_file)
		if not state_path.name.endswith(".state.json"):
			print(f"  ✗   {state_path.name}: not a .state.json file, skipping", file=sys.stderr)
			continue
		with open(state_path, encoding="utf-8") as f:
			state = json.load(f)

		batch_id = state["batch_id"]
		batch = client.messages.batches.retrieve(batch_id)
		counts = batch.request_counts
		stem = state["stem"]

		if batch.processing_status != "ended":
			print(
				f"  ... {stem}: {batch.processing_status} "
				f"({counts.succeeded} done, {counts.processing} pending)",
				file=sys.stderr
			)
			continue

		if counts.errored > 0 or counts.expired > 0:
			print(
				f"  !   {stem}: {counts.errored} errored, {counts.expired} expired",
				file=sys.stderr
			)

		output_dir = Path(state["output_dir"])
		mapping_path = output_dir / f"{stem}.mapping.json"
		with open(mapping_path, encoding="utf-8") as f:
			meta = json.load(f)
		mapping = {m["idx"]: m for m in meta["sentences"]}

		input_conllu = state.get("input_conllu")
		if not input_conllu:
			print(f"  ✗   {stem}: no input_conllu in state, skipping", file=sys.stderr)
			continue

		sentences = parse_conllu(input_conllu)
		sent_map = {}
		sent_order = []
		for sent_id, sent_text, comments, tokens in sentences:
			sent_map[sent_id] = (sent_id, sent_text, comments, tokens)
			sent_order.append(sent_id)

		total_corrections = 0
		for result in client.messages.batches.results(batch_id):
			if result.result.type != "succeeded":
				continue
			text = result.result.message.content[0].text
			corrections = parse_corrections_json(text)
			if not corrections:
				continue

			idx = int(result.custom_id.split("_")[1])
			sent_id = mapping[idx]["sent_id"]

			if sent_id in sent_map:
				old = sent_map[sent_id]
				new_tokens, n = apply_corrections(old[3], corrections)
				sent_map[sent_id] = (old[0], old[1], old[2], new_tokens)
				total_corrections += n

		corrected = [sent_map[sid] for sid in sent_order]
		output_path = output_dir / f"{stem}.conllu"
		write_conllu(corrected, output_path)

		print(f"  ✓   {stem}: {total_corrections} corrections → {output_path}", file=sys.stderr)


def cmd_status(args):
	client = anthropic.Anthropic()
	batch = client.messages.batches.retrieve(args.batch_id)
	counts = batch.request_counts

	print(f"Batch ID:   {batch.id}")
	print(f"Status:     {batch.processing_status}")
	print(f"Created:    {batch.created_at}")
	print(f"Expires:    {batch.expires_at}")
	print(f"Succeeded:  {counts.succeeded}")
	print(f"Processing: {counts.processing}")
	print(f"Errored:    {counts.errored}")
	print(f"Expired:    {counts.expired}")


def cmd_list(args):
	client = anthropic.Anthropic()

	print(f"{'ID':<40} {'Status':<12} {'Done':>6} {'Pend':>6} {'Created'}")
	print("-" * 85)

	for batch in client.messages.batches.list(limit=args.limit):
		counts = batch.request_counts
		created = str(batch.created_at)[:19] if batch.created_at else "?"
		print(
			f"{batch.id:<40} {batch.processing_status:<12} "
			f"{counts.succeeded:>6} {counts.processing:>6} {created}"
		)


def main():
	parser = argparse.ArgumentParser(description="Batch LLM annotation corrections")
	sub = parser.add_subparsers(dest="command", required=True)

	p = sub.add_parser("prepare", help="Prepare batch requests from a .conllu file")
	p.add_argument("conllu_file", help="Input .conllu file")
	p.add_argument("-p", "--prompt", required=True, help="Prompt file")
	p.add_argument("-o", "--output-dir", required=True, help="Output directory")
	p.add_argument("-m", "--model", default="claude-sonnet-4-5")
	p.set_defaults(func=cmd_prepare)

	p = sub.add_parser("submit", help="Submit a prepared .jsonl batch")
	p.add_argument("jsonl_file", help=".jsonl file from prepare")
	p.set_defaults(func=cmd_submit)

	p = sub.add_parser("resume", help="Check status and apply completed batches")
	p.add_argument("state_files", nargs="+", help="One or more .state.json files")
	p.set_defaults(func=cmd_resume)

	p = sub.add_parser("status", help="Check a single batch by ID")
	p.add_argument("batch_id")
	p.set_defaults(func=cmd_status)

	p = sub.add_parser("list", help="List recent batches")
	p.add_argument("-n", "--limit", type=int, default=10)
	p.set_defaults(func=cmd_list)

	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
