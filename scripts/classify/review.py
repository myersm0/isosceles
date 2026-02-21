"""
Sonnet review: takes classifier flags, sends each to Claude Sonnet
with a focused prompt, collects confirmed corrections.

Usage (integrated):
  python -m classify input.conllu --db littre.db --review

Usage (standalone, from saved flags):
  python -m classify.review input.conllu --flags flags.json

Usage (batch mode, generates JSONL for Anthropic batch API):
  python -m classify.review input.conllu --flags flags.json --batch
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

from .conllu import read_conllu, parse_block


ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-4-5"

prompts_dir = Path(__file__).parent / "prompts"

template_files = {
	("littre", "lemma_not_found"): "review_littre_notfound.txt",
	("littre", "upos_mismatch"): "review_littre_upos.txt",
	("lemma", ""): "review_lemma.txt",
	("tense", ""): "review_tense.txt",
	("aux", ""): "review_aux.txt",
	("que", ""): "review_que.txt",
}


def load_review_system():
	return (prompts_dir / "review_system.txt").read_text(encoding="utf-8").strip()


def load_review_templates():
	"""Load all review templates from individual files."""
	templates = {}
	for (task, issue), filename in template_files.items():
		path = prompts_dir / filename
		if path.exists():
			key = f"{task}:{issue}" if issue else task
			templates[key] = path.read_text(encoding="utf-8").strip()
	return templates


def build_token_lookup(conllu_path):
	"""Build a lookup: sent_id → {token_id → full_token_dict, 'text' → sent_text}."""
	blocks = read_conllu(conllu_path)
	lookup = {}
	for block in blocks:
		sent_id, sent_text, tokens = parse_block(block)
		token_map = {}
		for t in tokens:
			token_map[t["id"]] = t
		lookup[sent_id] = {"text": sent_text, "tokens": token_map}
	return lookup


def build_review_prompt(flag, token_lookup, templates):
	"""Build user prompt for a single flag."""
	sent_id = flag["sent_id"]
	sent_data = token_lookup.get(sent_id)
	if not sent_data:
		return None

	token_id = flag.get("token_id") or flag.get("id")
	if isinstance(token_id, str):
		token_id = int(token_id)

	token = sent_data["tokens"].get(token_id)
	if not token:
		return None

	task = flag["task"]
	issue = flag.get("issue", "")
	template_key = f"{task}:{issue}" if issue else task

	if template_key not in templates and task in templates:
		template_key = task

	if template_key not in templates:
		return None

	template = templates[template_key]

	# Build substitution dict from flag + token
	subs = {
		"token_id": token_id,
		"form": token["form"],
		"lemma": token["lemma"],
		"upos": token["upos"],
		"feats": token["feats"],
	}

	# Add task-specific fields from the flag
	for key in ("reason", "issue", "current", "expected",
				"current_upos", "expected_upos", "detail",
				"littre_pos", "next_form", "next_upos"):
		if key in flag:
			subs[key] = flag[key]

	# Fill in next token info for aux if not in flag
	if task == "aux" and "next_form" not in subs:
		for other_id in sorted(sent_data["tokens"].keys()):
			if other_id > token_id:
				other = sent_data["tokens"][other_id]
				if other["upos"] not in ("PUNCT", "ADV"):
					subs["next_form"] = other["form"]
					subs["next_upos"] = other["upos"]
					break
		subs.setdefault("next_form", "—")
		subs.setdefault("next_upos", "—")

	try:
		context = template.format(**subs)
	except KeyError as e:
		context = template
		for k, v in subs.items():
			context = context.replace("{" + k + "}", str(v))

	sent_text = sent_data["text"] or ""
	user_prompt = f"SENTENCE: {sent_text}\n\n{context}"

	return user_prompt


def extract_json(text):
	"""Extract JSON object from text that may contain reasoning before/after."""
	text = text.strip()
	# Strip markdown fences
	if text.startswith("```"):
		text = text.split("\n", 1)[1] if "\n" in text else text[3:]
		if text.endswith("```"):
			text = text[:-3]
		text = text.strip()
	# Try direct parse first
	try:
		return json.loads(text)
	except json.JSONDecodeError:
		pass
	# Find {"action" in the text
	idx = text.find('{"action"')
	if idx < 0:
		return None
	# Find matching closing brace
	depth = 0
	for i in range(idx, len(text)):
		if text[i] == "{":
			depth += 1
		elif text[i] == "}":
			depth -= 1
			if depth == 0:
				try:
					return json.loads(text[idx:i + 1])
				except json.JSONDecodeError:
					return None
	return None


def call_sonnet(system_prompt, user_prompt, model=DEFAULT_MODEL):
	"""Make a direct Anthropic API call."""
	api_key = os.environ.get("ANTHROPIC_API_KEY")
	if not api_key:
		print("Error: ANTHROPIC_API_KEY not set", file=sys.stderr)
		sys.exit(1)

	payload = {
		"model": model,
		"max_tokens": 256,
		"system": system_prompt,
		"messages": [
			{"role": "user", "content": user_prompt},
		],
	}

	headers = {
		"x-api-key": api_key,
		"content-type": "application/json",
		"anthropic-version": "2023-06-01",
	}

	try:
		response = requests.post(ANTHROPIC_URL, json=payload, headers=headers, timeout=30)
		response.raise_for_status()
		data = response.json()
		content = data["content"][0]["text"]
		return extract_json(content)
	except (requests.RequestException, KeyError, IndexError) as e:
		print(f"  API error: {e}", file=sys.stderr)
		return None


def generate_batch_jsonl(flags, token_lookup, templates, system_prompt, output_path, model=DEFAULT_MODEL):
	"""Generate Anthropic batch API JSONL file."""
	with open(output_path, "w") as f:
		for i, flag in enumerate(flags):
			user_prompt = build_review_prompt(flag, token_lookup, templates)
			if not user_prompt:
				continue

			request = {
				"custom_id": f"flag-{i:04d}-{flag['sent_id']}-tok{flag.get('token_id', flag.get('id', '?'))}",
				"params": {
					"model": model,
					"max_tokens": 256,
					"system": system_prompt,
					"messages": [
						{"role": "user", "content": user_prompt},
					],
				},
			}
			f.write(json.dumps(request) + "\n")

	print(f"Wrote {len(flags)} requests to {output_path}")


def run_review(conllu_path, flags, model=DEFAULT_MODEL, batch_output=None):
	"""Run Sonnet review on classifier flags.

	Returns list of confirmed patches.
	"""
	token_lookup = build_token_lookup(conllu_path)
	system_prompt = load_review_system()
	templates = load_review_templates()

	if batch_output:
		generate_batch_jsonl(flags, token_lookup, templates, system_prompt, batch_output, model)
		return []

	patches = []
	for i, flag in enumerate(flags):
		user_prompt = build_review_prompt(flag, token_lookup, templates)
		if not user_prompt:
			print(f"  [{i}] Could not build prompt for {flag}")
			continue

		sent_id = flag["sent_id"]
		token_id = flag.get("token_id") or flag.get("id")
		form = flag.get("form", "?")
		task = flag["task"]

		print(f"[{sent_id} tok {token_id}] \"{form}\" ({task}) ... ", end="", flush=True)

		start = time.time()
		response = call_sonnet(system_prompt, user_prompt, model)
		elapsed = time.time() - start

		if not response:
			print(f"error ({elapsed:.1f}s)")
			continue

		action = response.get("action")

		if action == "correct":
			fields = response.get("fields", {})
			print(f"CORRECT → {fields} ({elapsed:.1f}s)")
			patches.append({
				"sent_id": sent_id,
				"token_id": int(token_id),
				"fields": fields,
				"flag": flag,
			})
		elif action == "no_change":
			reason = response.get("reason", "")
			print(f"no_change: {reason} ({elapsed:.1f}s)")
		else:
			print(f"unexpected response: {response} ({elapsed:.1f}s)")

	print(f"\n--- Review Summary ---")
	print(f"Flags reviewed: {len(flags)}")
	print(f"Confirmed corrections: {len(patches)}")

	return patches
