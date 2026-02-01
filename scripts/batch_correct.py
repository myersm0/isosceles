#!/usr/bin/env python3
"""
Batch processing for LLM-based annotation corrections.

Reads .conllu files (produced by annotate.py -b spacy), sends them to 
Anthropic's batch API for surface corrections (lemma, upos, feats), 
and writes corrected .conllu files.

Persistent workflow (can close laptop between steps):

    # Step 1: Prepare batch requests
    python batch_correct.py prepare input_dir/ -p prompt_surface.txt -o mybatch.jsonl
    # Creates: mybatch.jsonl, mybatch.mapping.json
    
    # Step 2: Submit batch
    python batch_correct.py submit mybatch.jsonl
    # Creates: mybatch.state.json (saves batch_id)
    
    # Step 3: Check status later (non-blocking)
    python batch_correct.py resume mybatch.state.json
    # Shows status, tells you when ready
    
    # Step 4: Apply when ready
    python batch_correct.py resume mybatch.state.json -i input_dir/ -o output_dir/

Alternative: blocking workflow (keeps terminal open):

    python batch_correct.py run input_dir/ output_dir/ -p prompt_surface.txt
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request


def parse_conllu(filepath):
	"""Parse a CoNLL-U file into list of (sent_id, sent_text, tokens)."""
	sentences = []
	current_tokens = []
	sent_id = None
	sent_text = None
	
	with open(filepath, encoding="utf-8") as f:
		for line in f:
			line = line.rstrip("\n")
			if not line:
				if current_tokens:
					sentences.append((sent_id, sent_text, current_tokens))
					current_tokens = []
					sent_id = None
					sent_text = None
			elif line.startswith("#"):
				if line.startswith("# sent_id"):
					sent_id = line.split("=", 1)[1].strip()
				elif line.startswith("# text"):
					sent_text = line.split("=", 1)[1].strip()
			else:
				fields = line.split("\t")
				if len(fields) >= 10 and not "-" in fields[0] and not "." in fields[0]:
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
			sentences.append((sent_id, sent_text, current_tokens))
	
	return sentences


def write_conllu(sentences, filepath):
	"""Write sentences to CoNLL-U format."""
	with open(filepath, "w", encoding="utf-8") as f:
		for sent_id, sent_text, tokens in sentences:
			if sent_id:
				f.write(f"# sent_id = {sent_id}\n")
			if sent_text:
				f.write(f"# text = {sent_text}\n")
			for t in tokens:
				feats = t.get("feats") or "_"
				f.write(f"{t['id']}\t{t['form']}\t{t['lemma']}\t{t['upos']}\t{t['xpos']}\t{feats}\t{t['head']}\t{t['deprel']}\t{t['deps']}\t{t['misc']}\n")
			f.write("\n")


def format_tokens_for_llm(tokens):
	"""Format tokens for LLM input (surface mode: ID, FORM, LEMMA, UPOS, FEATS)."""
	lines = []
	for t in tokens:
		feats = t.get("feats") or "_"
		lines.append(f"{t['id']}\t{t['form']}\t{t['lemma']}\t{t['upos']}\t{feats}")
	return "\n".join(lines)


def parse_corrections_json(response_text):
	"""Extract corrections from LLM response."""
	match = re.search(r"\{[\s\S]*\}", response_text)
	if match:
		try:
			return json.loads(match.group()).get("corrections", [])
		except json.JSONDecodeError:
			pass
	return []


def apply_corrections(tokens, corrections):
	"""Apply corrections to tokens (surface fields only)."""
	import copy
	tokens = copy.deepcopy(tokens)
	token_map = {t["id"]: t for t in tokens}
	safe_fields = {"lemma", "upos", "xpos", "feats"}
	n_applied = 0
	
	for c in corrections:
		tok_id = c.get("id")
		field = c.get("field")
		value = c.get("value")
		if field not in safe_fields:
			continue
		try:
			tok_id = int(tok_id)
		except (ValueError, TypeError):
			continue
		if tok_id not in token_map:
			continue
		token_map[tok_id][field] = value
		n_applied += 1
	
	return tokens, n_applied


def cmd_prepare(args):
	"""Prepare batch requests from input files."""
	input_dir = Path(args.input_dir)
	prompt = Path(args.prompt).read_text(encoding="utf-8")
	
	if input_dir.is_file() and input_dir.suffix == ".conllu":
		conllu_files = [input_dir]
	else:
		conllu_files = sorted(input_dir.glob("*.conllu"))
	
	if not conllu_files:
		sys.exit(f"No .conllu files found in {input_dir}")
	
	requests = []
	mapping = []
	
	for conllu_file in conllu_files:
		sentences = parse_conllu(conllu_file)
		for sent_id, sent_text, tokens in sentences:
			parse_str = format_tokens_for_llm(tokens)
			idx = len(requests)
			custom_id = f"req_{idx}"
			
			mapping.append({"idx": idx, "file": conllu_file.stem, "sent_id": sent_id})
			
			request = {
				"custom_id": custom_id,
				"params": {
					"model": args.model,
					"max_tokens": 1024,
					"system": [
						{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}
					],
					"messages": [
						{"role": "user", "content": parse_str}
					]
				}
			}
			requests.append(request)
	
	output_path = Path(args.output)
	with open(output_path, "w", encoding="utf-8") as f:
		for req in requests:
			f.write(json.dumps(req, ensure_ascii=False) + "\n")
	
	mapping_path = output_path.with_suffix(".mapping.json")
	with open(mapping_path, "w", encoding="utf-8") as f:
		json.dump(mapping, f, ensure_ascii=False, indent=2)
	
	print(f"Prepared {len(requests)} requests from {len(conllu_files)} files", file=sys.stderr)
	print(f"Output: {output_path}", file=sys.stderr)
	print(f"Mapping: {mapping_path}", file=sys.stderr)
	
	size_mb = output_path.stat().st_size / 1_000_000
	print(f"Size: {size_mb:.2f} MB (limit: 256 MB)", file=sys.stderr)
	
	if len(requests) > 100_000:
		print(f"WARNING: {len(requests)} requests exceeds 100k limit", file=sys.stderr)
	if size_mb > 256:
		print(f"WARNING: {size_mb:.2f} MB exceeds 256 MB limit", file=sys.stderr)


def cmd_submit(args):
	"""Submit batch requests."""
	client = anthropic.Anthropic()
	jsonl_path = Path(args.jsonl_file)
	
	with open(jsonl_path, encoding="utf-8") as f:
		raw_requests = [json.loads(line) for line in f]
	
	requests = [
		Request(
			custom_id=r["custom_id"],
			params=MessageCreateParamsNonStreaming(**r["params"])
		)
		for r in raw_requests
	]
	
	print(f"Submitting batch with {len(requests)} requests...", file=sys.stderr)
	batch = client.messages.batches.create(requests=requests)
	
	state_path = jsonl_path.with_suffix(".state.json")
	state = {
		"batch_id": batch.id,
		"jsonl_file": str(jsonl_path),
		"mapping_file": str(jsonl_path.with_suffix(".mapping.json")),
		"created_at": batch.created_at,
		"expires_at": batch.expires_at,
	}
	with open(state_path, "w", encoding="utf-8") as f:
		json.dump(state, f, indent=2, default=str)
	
	print(f"Batch ID: {batch.id}", file=sys.stderr)
	print(f"Status: {batch.processing_status}", file=sys.stderr)
	print(f"Expires: {batch.expires_at}", file=sys.stderr)
	print(f"State saved to: {state_path}", file=sys.stderr)
	
	print(batch.id)


def cmd_poll(args):
	"""Poll batch until complete."""
	client = anthropic.Anthropic()
	batch_id = args.batch_id
	
	while True:
		batch = client.messages.batches.retrieve(batch_id)
		counts = batch.request_counts
		
		print(
			f"Status: {batch.processing_status} | "
			f"processing: {counts.processing}, succeeded: {counts.succeeded}, "
			f"errored: {counts.errored}, canceled: {counts.canceled}, expired: {counts.expired}",
			file=sys.stderr
		)
		
		if batch.processing_status == "ended":
			print(f"\nBatch complete. Results URL: {batch.results_url}", file=sys.stderr)
			return batch
		
		time.sleep(args.interval)


def cmd_status(args):
	"""Check batch status once (no polling)."""
	client = anthropic.Anthropic()
	batch = client.messages.batches.retrieve(args.batch_id)
	counts = batch.request_counts
	
	print(f"Batch ID: {batch.id}")
	print(f"Status: {batch.processing_status}")
	print(f"Created: {batch.created_at}")
	print(f"Expires: {batch.expires_at}")
	print(f"Requests:")
	print(f"  Processing: {counts.processing}")
	print(f"  Succeeded: {counts.succeeded}")
	print(f"  Errored: {counts.errored}")
	print(f"  Canceled: {counts.canceled}")
	print(f"  Expired: {counts.expired}")
	
	if batch.processing_status == "ended":
		print(f"\nReady to apply. Run:")
		print(f"  python batch_correct.py apply {batch.id} <input_dir> <output_dir> -m <mapping.json>")


def cmd_resume(args):
	"""Resume from state file: check status, apply if ready."""
	state_path = Path(args.state_file)
	with open(state_path, encoding="utf-8") as f:
		state = json.load(f)
	
	batch_id = state["batch_id"]
	mapping_file = state["mapping_file"]
	
	client = anthropic.Anthropic()
	batch = client.messages.batches.retrieve(batch_id)
	counts = batch.request_counts
	
	print(f"Batch ID: {batch_id}")
	print(f"Status: {batch.processing_status}")
	print(f"  Processing: {counts.processing}")
	print(f"  Succeeded: {counts.succeeded}")
	print(f"  Errored: {counts.errored}")
	
	if batch.processing_status != "ended":
		print(f"\nStill processing. Check again later:")
		print(f"  python batch_correct.py resume {state_path}")
		return
	
	if not args.output_dir or not args.input_dir:
		print(f"\nReady to apply. Run:")
		print(f"  python batch_correct.py resume {state_path} -i <input_dir> -o <output_dir>")
		return
	
	print(f"\nApplying results...")
	args.batch_id = batch_id
	args.mapping = mapping_file
	cmd_apply(args)


def cmd_list(args):
	"""List recent batches."""
	client = anthropic.Anthropic()
	
	print(f"{'ID':<35} {'Status':<12} {'Succeeded':>10} {'Processing':>10} {'Created'}")
	print("-" * 90)
	
	for batch in client.messages.batches.list(limit=args.limit):
		counts = batch.request_counts
		created = str(batch.created_at)[:19] if batch.created_at else "?"
		print(f"{batch.id:<35} {batch.processing_status:<12} {counts.succeeded:>10} {counts.processing:>10} {created}")


def cmd_apply(args):
	"""Apply batch results to .conllu files."""
	client = anthropic.Anthropic()
	input_dir = Path(args.input_dir)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	
	mapping_path = Path(args.mapping)
	with open(mapping_path, encoding="utf-8") as f:
		mapping = {m["idx"]: m for m in json.load(f)}
	
	print(f"Retrieving results for batch {args.batch_id}...", file=sys.stderr)
	
	corrections_by_file = {}
	total_corrections = 0
	succeeded = 0
	errored = 0
	
	for result in client.messages.batches.results(args.batch_id):
		if result.result.type == "succeeded":
			succeeded += 1
			text = result.result.message.content[0].text
			corr = parse_corrections_json(text)
			total_corrections += len(corr)
			
			idx = int(result.custom_id.split("_")[1])
			m = mapping[idx]
			file_stem, sent_id = m["file"], m["sent_id"]
			
			if file_stem not in corrections_by_file:
				corrections_by_file[file_stem] = {}
			corrections_by_file[file_stem][sent_id] = corr
		else:
			errored += 1
			print(f"  Error: {result.custom_id} - {result.result.type}", file=sys.stderr)
	
	print(f"Results: {succeeded} succeeded, {errored} errored", file=sys.stderr)
	print(f"Total corrections: {total_corrections}", file=sys.stderr)
	
	if input_dir.is_file():
		conllu_files = [input_dir]
	else:
		conllu_files = sorted(input_dir.glob("*.conllu"))
	
	for conllu_file in conllu_files:
		file_corrections = corrections_by_file.get(conllu_file.stem, {})
		sentences = parse_conllu(conllu_file)
		
		corrected = []
		n_applied = 0
		for sent_id, sent_text, tokens in sentences:
			if sent_id in file_corrections:
				tokens, n = apply_corrections(tokens, file_corrections[sent_id])
				n_applied += n
			corrected.append((sent_id, sent_text, tokens))
		
		output_path = output_dir / conllu_file.name
		write_conllu(corrected, output_path)
		print(f"  {conllu_file.name}: {n_applied} corrections applied", file=sys.stderr)


def cmd_run(args):
	"""Run full pipeline: prepare → submit → poll → apply."""
	import tempfile
	
	with tempfile.TemporaryDirectory() as tmpdir:
		jsonl_path = Path(tmpdir) / "batch_requests.jsonl"
		mapping_path = jsonl_path.with_suffix(".mapping.json")
		
		args.output = str(jsonl_path)
		cmd_prepare(args)
		
		client = anthropic.Anthropic()
		with open(jsonl_path, encoding="utf-8") as f:
			raw_requests = [json.loads(line) for line in f]
		
		requests = [
			Request(
				custom_id=r["custom_id"],
				params=MessageCreateParamsNonStreaming(**r["params"])
			)
			for r in raw_requests
		]
		
		print(f"\nSubmitting batch with {len(requests)} requests...", file=sys.stderr)
		batch = client.messages.batches.create(requests=requests)
		batch_id = batch.id
		print(f"Batch ID: {batch_id}", file=sys.stderr)
		
		args.batch_id = batch_id
		args.interval = 60
		cmd_poll(args)
		
		args.mapping = str(mapping_path)
		cmd_apply(args)


def main():
	parser = argparse.ArgumentParser(description="Batch processing for LLM annotation corrections")
	subparsers = parser.add_subparsers(dest="command", required=True)
	
	p_prepare = subparsers.add_parser("prepare", help="Prepare batch requests from .conllu files")
	p_prepare.add_argument("input_dir", help="Directory with .conllu files (or single file)")
	p_prepare.add_argument("--prompt", "-p", required=True, help="Path to prompt file")
	p_prepare.add_argument("--output", "-o", default="batch_requests.jsonl", help="Output JSONL file")
	p_prepare.add_argument("--model", "-m", default="claude-sonnet-4-5", help="Model to use")
	p_prepare.set_defaults(func=cmd_prepare)
	
	p_submit = subparsers.add_parser("submit", help="Submit batch requests")
	p_submit.add_argument("jsonl_file", help="JSONL file with requests")
	p_submit.set_defaults(func=cmd_submit)
	
	p_poll = subparsers.add_parser("poll", help="Poll batch until complete")
	p_poll.add_argument("batch_id", help="Batch ID")
	p_poll.add_argument("--interval", "-i", type=int, default=60, help="Poll interval in seconds")
	p_poll.set_defaults(func=cmd_poll)
	
	p_status = subparsers.add_parser("status", help="Check batch status once")
	p_status.add_argument("batch_id", help="Batch ID")
	p_status.set_defaults(func=cmd_status)
	
	p_resume = subparsers.add_parser("resume", help="Resume from state file")
	p_resume.add_argument("state_file", help="Path to .state.json file")
	p_resume.add_argument("--input-dir", "-i", help="Directory with original .conllu files")
	p_resume.add_argument("--output-dir", "-o", help="Output directory (if ready to apply)")
	p_resume.set_defaults(func=cmd_resume)
	
	p_list = subparsers.add_parser("list", help="List recent batches")
	p_list.add_argument("--limit", "-n", type=int, default=10, help="Number of batches to show")
	p_list.set_defaults(func=cmd_list)
	
	p_apply = subparsers.add_parser("apply", help="Apply batch results")
	p_apply.add_argument("batch_id", help="Batch ID")
	p_apply.add_argument("input_dir", help="Directory with original .conllu files")
	p_apply.add_argument("output_dir", help="Output directory for corrected files")
	p_apply.add_argument("--mapping", "-m", required=True, help="Path to mapping JSON file")
	p_apply.set_defaults(func=cmd_apply)
	
	p_run = subparsers.add_parser("run", help="Run full pipeline")
	p_run.add_argument("input_dir", help="Directory with .conllu files")
	p_run.add_argument("output_dir", help="Output directory")
	p_run.add_argument("--prompt", "-p", required=True, help="Path to prompt file")
	p_run.add_argument("--model", "-m", default="claude-sonnet-4-5", help="Model to use")
	p_run.set_defaults(func=cmd_run)
	
	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
