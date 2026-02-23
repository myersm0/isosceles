"""
Submit batch JSONL to Anthropic Message Batches API, poll for completion,
parse results into patches.

Usage:
  # Submit and wait
  python -m classify.batch submit batch.jsonl

  # Submit, save batch ID, come back later
  python -m classify.batch submit batch.jsonl --no-wait --id-file batch_id.txt

  # Check status of a running batch
  python -m classify.batch status msgbatch_abc123

  # Poll until done and download results
  python -m classify.batch poll msgbatch_abc123 --results-out results.jsonl

  # Parse results into patches and apply
  python -m classify.batch apply input.conllu results.jsonl --output corrected.conllu
"""

import argparse
import json
import os
import sys
import time

import requests


API_BASE = "https://api.anthropic.com/v1/messages/batches"


def get_headers():
	api_key = os.environ.get("ANTHROPIC_API_KEY")
	if not api_key:
		print("Error: ANTHROPIC_API_KEY not set", file=sys.stderr)
		sys.exit(1)
	return {
		"x-api-key": api_key,
		"anthropic-version": "2023-06-01",
		"content-type": "application/json",
	}


def submit_batch(jsonl_path):
	"""Read JSONL file and submit as a batch."""
	batch_requests = []
	with open(jsonl_path) as f:
		for line in f:
			line = line.strip()
			if line:
				batch_requests.append(json.loads(line))

	if not batch_requests:
		print("No requests found in JSONL file", file=sys.stderr)
		sys.exit(1)

	print(f"Submitting batch with {len(batch_requests)} requests...")

	response = requests.post(
		API_BASE,
		headers=get_headers(),
		json={"requests": batch_requests},
		timeout=60,
	)
	response.raise_for_status()
	batch = response.json()

	batch_id = batch["id"]
	status = batch["processing_status"]
	print(f"Batch created: {batch_id}")
	print(f"Status: {status}")

	return batch


def check_status(batch_id):
	"""Check status of a batch."""
	response = requests.get(
		f"{API_BASE}/{batch_id}",
		headers=get_headers(),
		timeout=30,
	)
	response.raise_for_status()
	return response.json()


def poll_until_done(batch_id, interval=30):
	"""Poll batch until processing ends."""
	print(f"Polling batch {batch_id} every {interval}s...")

	while True:
		batch = check_status(batch_id)
		status = batch["processing_status"]
		counts = batch.get("request_counts", {})
		succeeded = counts.get("succeeded", 0)
		processing = counts.get("processing", 0)
		errored = counts.get("errored", 0)
		total = succeeded + processing + errored + counts.get("canceled", 0) + counts.get("expired", 0)

		print(f"  [{status}] {succeeded}/{total} succeeded, "
			  f"{processing} processing, {errored} errored")

		if status == "ended":
			return batch

		time.sleep(interval)


def download_results(batch):
	"""Download results from a completed batch."""
	results_url = batch.get("results_url")
	if not results_url:
		print("No results_url in batch response", file=sys.stderr)
		return []

	headers = get_headers()
	del headers["content-type"]

	response = requests.get(results_url, headers=headers, timeout=120)
	response.raise_for_status()

	results = []
	for line in response.text.strip().split("\n"):
		if line.strip():
			results.append(json.loads(line))

	return results


def parse_results(results):
	"""Parse batch results into patches compatible with apply.py."""
	patches = []
	errors = []

	for result in results:
		custom_id = result.get("custom_id", "?")
		result_data = result.get("result", {})
		result_type = result_data.get("type")

		if result_type != "succeeded":
			errors.append((custom_id, result_type, result_data.get("error", {}).get("message", "")))
			continue

		message = result_data.get("message", {})
		stop_reason = message.get("stop_reason", "?")
		content_blocks = message.get("content", [])
		text = ""
		for block in content_blocks:
			if block.get("type") == "text":
				text += block.get("text", "")

		from .review import extract_json, ASSISTANT_PREFILL

		# First try: parse response as-is (model may have returned complete JSON)
		response = extract_json(text.strip()) if text.strip() else None

		# If the raw response is a bare fields dict, wrap it
		if response is not None and "action" not in response:
			if any(k in response for k in ("lemma", "upos", "feats")):
				response = {"action": "correct", "fields": response}
			elif "reason" in response:
				response = {"action": "no_change", "reason": response["reason"]}
			else:
				response = None

		# Second try: prepend prefill and parse
		if response is None:
			full_text = ASSISTANT_PREFILL + text
			response = extract_json(full_text)

		if response is None:
			full_text = ASSISTANT_PREFILL + text
			errors.append((
				custom_id, "parse_error",
				f"stop={stop_reason} text={repr(full_text[:300])}",
			))
			continue

		action = response.get("action")

		# Parse sent_id and token_id from custom_id
		# Format: flag-NNNN-SENT_ID-tokN
		parts = custom_id.split("-", 2)
		remaining = parts[2] if len(parts) > 2 else custom_id
		tok_idx = remaining.rfind("-tok")
		if tok_idx >= 0:
			sent_id = remaining[:tok_idx]
			try:
				token_id = int(remaining[tok_idx + 4:])
			except ValueError:
				token_id = None
		else:
			sent_id = remaining
			token_id = None

		if action == "correct":
			fields = response.get("fields", {})
			print(f"  [{custom_id}] CORRECT → {fields}")
			patches.append({
				"sent_id": sent_id,
				"token_id": token_id,
				"fields": fields,
				"custom_id": custom_id,
			})
		elif action == "no_change":
			reason = response.get("reason", "")
			print(f"  [{custom_id}] no_change: {reason}")
		else:
			errors.append((custom_id, "unexpected_action", str(response)[:100]))

	if errors:
		print(f"\n{len(errors)} errors:")
		for custom_id, error_type, detail in errors:
			print(f"  [{custom_id}] {error_type}: {detail}")

	print(f"\n{len(patches)} corrections, "
		  f"{len(results) - len(patches) - len(errors)} no_change, "
		  f"{len(errors)} errors")

	return patches


def main():
	parser = argparse.ArgumentParser(description="Anthropic batch API operations")
	subparsers = parser.add_subparsers(dest="command", required=True)

	# submit
	sub = subparsers.add_parser("submit", help="Submit batch JSONL")
	sub.add_argument("jsonl", help="Path to batch JSONL file")
	sub.add_argument("--no-wait", action="store_true", help="Don't wait for completion")
	sub.add_argument("--id-file", help="Save batch ID to this file")
	sub.add_argument("--results-out", help="Save raw results JSONL to this file")
	sub.add_argument("--patches-out", help="Save parsed patches JSON to this file")
	sub.add_argument("--interval", type=int, default=30, help="Poll interval in seconds")

	# status
	sub = subparsers.add_parser("status", help="Check batch status")
	sub.add_argument("batch_id", help="Batch ID")

	# poll
	sub = subparsers.add_parser("poll", help="Poll until done and download results")
	sub.add_argument("batch_id", help="Batch ID")
	sub.add_argument("--results-out", help="Save raw results JSONL")
	sub.add_argument("--patches-out", help="Save parsed patches JSON")
	sub.add_argument("--interval", type=int, default=30, help="Poll interval in seconds")

	# apply
	sub = subparsers.add_parser("apply", help="Parse results and apply corrections")
	sub.add_argument("input", help="Original CoNLL-U file")
	sub.add_argument("results", help="Results JSONL or patches JSON file")
	sub.add_argument("--output", "-o", required=True, help="Corrected CoNLL-U output path")

	args = parser.parse_args()

	if args.command == "submit":
		batch = submit_batch(args.jsonl)
		batch_id = batch["id"]

		if args.id_file:
			with open(args.id_file, "w") as f:
				f.write(batch_id + "\n")
			print(f"Batch ID saved to {args.id_file}")

		if args.no_wait:
			print(f"Use: python -m classify.batch poll {batch_id}")
			return

		batch = poll_until_done(batch_id, args.interval)
		results = download_results(batch)

		if args.results_out:
			with open(args.results_out, "w") as f:
				for r in results:
					f.write(json.dumps(r, ensure_ascii=False) + "\n")
			print(f"Raw results saved to {args.results_out}")

		patches = parse_results(results)

		if args.patches_out:
			with open(args.patches_out, "w") as f:
				json.dump(patches, f, indent=2, ensure_ascii=False)
			print(f"Patches saved to {args.patches_out}")

	elif args.command == "status":
		batch = check_status(args.batch_id)
		print(json.dumps(batch, indent=2))

	elif args.command == "poll":
		batch = poll_until_done(args.batch_id, args.interval)
		results = download_results(batch)

		if args.results_out:
			with open(args.results_out, "w") as f:
				for r in results:
					f.write(json.dumps(r, ensure_ascii=False) + "\n")
			print(f"Raw results saved to {args.results_out}")

		patches = parse_results(results)

		if args.patches_out:
			with open(args.patches_out, "w") as f:
				json.dump(patches, f, indent=2, ensure_ascii=False)
			print(f"Patches saved to {args.patches_out}")

	elif args.command == "apply":
		results_path = args.results
		if results_path.endswith(".json"):
			with open(results_path) as f:
				patches = json.load(f)
			print(f"Loaded {len(patches)} patches from {results_path}")
		else:
			results = []
			with open(results_path) as f:
				for line in f:
					if line.strip():
						results.append(json.loads(line))
			print(f"Loaded {len(results)} results from {results_path}")
			patches = parse_results(results)

		from .apply import apply_patches
		apply_patches(args.input, patches, args.output)


if __name__ == "__main__":
	main()
