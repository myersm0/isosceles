"""Unified CoNLL-U classifier pipeline.

Usage:
  python -m classify input.conllu --db littre.db
  python -m classify input.conllu --db littre.db --tasks littre,tense,aux
  python -m classify input.conllu --db littre.db --tasks tense --model-tense gemma3:12b
"""

import argparse
import sys

from .conllu import read_conllu, parse_block
from .config import task_defaults, all_tasks
from . import littre as littre_mod
from . import llm_classifier


def build_parser():
	parser = argparse.ArgumentParser(
		description="Classify CoNLL-U annotation errors",
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

	for task_name, defaults in task_defaults.items():
		if defaults["type"] == "llm":
			parser.add_argument(
				f"--model-{task_name}",
				default=defaults["model"],
				help=f"Model for {task_name} task (default: {defaults['model']})",
			)

	return parser


def main():
	parser = build_parser()
	args = parser.parse_args()

	if args.tasks == "all":
		tasks = all_tasks
	else:
		tasks = [t.strip() for t in args.tasks.split(",")]
		for t in tasks:
			if t not in all_tasks:
				print(f"Unknown task: {t}. Available: {', '.join(all_tasks)}", file=sys.stderr)
				sys.exit(1)

	if "littre" in tasks and not args.db:
		print("--db required for littre task", file=sys.stderr)
		sys.exit(1)

	blocks = read_conllu(args.input)
	parsed_blocks = [parse_block(lines) for lines in blocks]
	print(f"Loaded {len(blocks)} sentences from {args.input}\n")

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
				sent_id = flag["sent_id"]
				detail = flag["detail"]
				form = flag["form"]
				lemma = flag["lemma"]
				upos = flag["upos"]
				issue = flag["issue"]
				print(f"  [{sent_id}] tok {flag['token_id']} "
					  f'"{form}" lemma="{lemma}" ({upos}): {issue} — {detail}')
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

	print("=" * 60)
	print(f"TOTAL: {len(all_flags)} flags across {len(tasks)} tasks\n")

	by_task = {}
	for flag in all_flags:
		task = flag["task"]
		by_task[task] = by_task.get(task, 0) + 1
	for task, count in sorted(by_task.items()):
		model = flag.get("model", "")
		print(f"  {task}: {count}")


if __name__ == "__main__":
	main()
