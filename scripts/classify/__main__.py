"""Unified CoNLL-U classifier pipeline.

Usage:
  # Classify only (print flags)
  python -m classify input.conllu --db littre.db

  # Classify + review with Sonnet + apply corrections
  python -m classify input.conllu --db littre.db --review --output corrected.conllu

  # Classify + generate batch JSONL for Anthropic batch API
  python -m classify input.conllu --db littre.db --review --batch review_batch.jsonl

  # Classify specific tasks
  python -m classify input.conllu --db littre.db --tasks littre,tense,aux

  # Save flags to JSON for later review
  python -m classify input.conllu --db littre.db --flags-out flags.json

  # Review previously saved flags
  python -m classify input.conllu --flags-in flags.json --review --output corrected.conllu
"""

import argparse
import json
import sys

from .conllu import read_conllu, parse_block
from .config import task_defaults, all_tasks
from . import littre as littre_mod
from . import llm_classifier


def build_parser():
	parser = argparse.ArgumentParser(
		description="Classify and correct CoNLL-U annotation errors",
	)
	parser.add_argument("input", help="CoNLL-U file to classify")
	parser.add_argument("--db", help="Path to littre.db (required for littre task)")
	parser.add_argument(
		"--tasks", default="all",
		help=f"Comma-separated task list, or 'all' (default: all). "
			 f"Available: {', '.join(all_tasks)}",
	)
	parser.add_argument("--strict", action="store_true",
		help="Include low-confidence flags (Littré UPOS mismatch)")
	parser.add_argument("--no-think", action="store_true",
		help="Disable thinking mode for LLM classifiers")

	# Review options
	parser.add_argument("--review", action="store_true",
		help="Send flags to Sonnet for review")
	parser.add_argument("--batch", metavar="PATH",
		help="Generate batch JSONL instead of direct API calls")
	parser.add_argument("--review-model", default=None,
		help="Model for review (default: claude-sonnet-4-5)")

	# Output options
	parser.add_argument("--output", "-o", metavar="PATH",
		help="Write corrected CoNLL-U to this path")
	parser.add_argument("--flags-out", metavar="PATH",
		help="Save flags to JSON file")
	parser.add_argument("--flags-in", metavar="PATH",
		help="Load flags from JSON file (skip classification)")

	for task_name, defaults in task_defaults.items():
		if defaults["type"] == "llm":
			parser.add_argument(
				f"--model-{task_name}",
				default=defaults["model"],
				help=f"Model for {task_name} task (default: {defaults['model']})",
			)

	return parser


def run_classifiers(args, blocks, parsed_blocks, tasks):
	"""Run all selected classifiers. Returns list of flags."""
	all_flags = []

	for task_name in tasks:
		defaults = task_defaults[task_name]

		if defaults["type"] == "dictionary":
			print(f"=== {task_name} (Littré lookup) ===")
			flags = littre_mod.run(
				blocks, parsed_blocks,
				db_path=args.db,
				strict=args.strict,
			)
			for flag in flags:
				print(f"  [{flag['sent_id']}] tok {flag['token_id']} "
					  f"\"{flag['form']}\" lemma=\"{flag['lemma']}\" "
					  f"({flag['upos']}): {flag['issue']} — {flag['detail']}")
			print(f"  [{task_name}] {len(flags)} flags\n")
			all_flags.extend(flags)

		elif defaults["type"] == "llm":
			model = getattr(args, f"model_{task_name}", defaults["model"])
			think = defaults.get("think", True) and not args.no_think

			print(f"=== {task_name} ({model}) ===")
			flags = llm_classifier.run(
				blocks, parsed_blocks,
				task_name=task_name,
				model=model,
				think=think,
			)
			all_flags.extend(flags)

			ensemble = defaults.get("ensemble", [])
			for ensemble_config in ensemble:
				ensemble_model = ensemble_config["model"]
				ensemble_think = ensemble_config.get("think", True) and not args.no_think
				print(f"=== {task_name} ({ensemble_model}, ensemble) ===")
				ensemble_flags = llm_classifier.run(
					blocks, parsed_blocks,
					task_name=task_name,
					model=ensemble_model,
					think=ensemble_think,
				)
				all_flags.extend(ensemble_flags)

	return all_flags


def main():
	parser = build_parser()
	args = parser.parse_args()

	blocks = read_conllu(args.input)
	parsed_blocks = [parse_block(lines) for lines in blocks]
	print(f"Loaded {len(blocks)} sentences from {args.input}\n")

	# --- Classification phase ---
	if args.flags_in:
		with open(args.flags_in) as f:
			all_flags = json.load(f)
		print(f"Loaded {len(all_flags)} flags from {args.flags_in}\n")
	else:
		if args.tasks == "all":
			tasks = all_tasks
		else:
			tasks = [t.strip() for t in args.tasks.split(",")]
			for t in tasks:
				if t not in all_tasks:
					print(f"Unknown task: {t}. Available: {', '.join(all_tasks)}",
						  file=sys.stderr)
					sys.exit(1)

		if "littre" in tasks and not args.db:
			print("--db required for littre task", file=sys.stderr)
			sys.exit(1)

		all_flags = run_classifiers(args, blocks, parsed_blocks, tasks)

	# --- Summary ---
	print("=" * 60)
	print(f"TOTAL: {len(all_flags)} flags\n")
	by_task = {}
	for flag in all_flags:
		by_task[flag["task"]] = by_task.get(flag["task"], 0) + 1
	for task, count in sorted(by_task.items()):
		print(f"  {task}: {count}")
	print()

	# --- Save flags ---
	if args.flags_out:
		with open(args.flags_out, "w") as f:
			json.dump(all_flags, f, indent=2, ensure_ascii=False)
		print(f"Saved {len(all_flags)} flags to {args.flags_out}\n")

	# --- Review phase ---
	if not args.review and not args.batch:
		if not args.flags_out:
			print("Use --review to send flags to Sonnet, "
				  "--batch to generate batch JSONL, "
				  "or --flags-out to save flags.")
		return

	if not all_flags:
		print("No flags to review.")
		return

	from .review import run_review, DEFAULT_MODEL

	model = args.review_model or DEFAULT_MODEL
	patches = run_review(
		args.input, all_flags,
		model=model,
		batch_output=args.batch,
	)

	if args.batch:
		return

	# --- Apply phase ---
	if patches and args.output:
		from .apply import apply_patches
		apply_patches(args.input, patches, args.output)
	elif patches:
		print(f"\n{len(patches)} corrections confirmed. "
			  f"Use --output to write corrected CoNLL-U.")
		print("\nPatches:")
		for p in patches:
			print(f"  [{p['sent_id']} tok {p['token_id']}] {p['fields']}")


if __name__ == "__main__":
	main()
