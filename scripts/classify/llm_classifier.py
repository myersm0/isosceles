"""Generic LLM classifier: runs an Ollama model with a task-specific prompt."""

import time
from pathlib import Path

from .ollama_client import call_ollama
from .filters import task_filters


prompts_dir = Path(__file__).parent / "prompts"


def load_prompt(task_name):
	path = prompts_dir / f"classify_{task_name}.txt"
	return path.read_text(encoding="utf-8").strip()


def run(blocks, parsed_blocks, task_name, model, think=True, timeout=120):
	"""Run a single LLM classifier pass. Returns list of flag dicts."""
	system_prompt = load_prompt(task_name)
	filter_fn, format_fn = task_filters[task_name]

	all_flags = []
	total_time = 0
	skipped = 0

	for sent_id, sent_text, tokens in parsed_blocks:
		candidates = filter_fn(tokens)

		if not candidates:
			skipped += 1
			continue

		user_prompt = (
			f"Sentence: {sent_text}\n\n"
			f"Tokens to check:\n{format_fn(candidates, tokens)}"
		)

		start = time.time()
		response = call_ollama(
			model, system_prompt, user_prompt,
			think=think, timeout=timeout,
		)
		elapsed = time.time() - start
		total_time += elapsed

		if not response or "flags" not in response:
			print(f"[{sent_id}] {task_name}/{model} — no response ({elapsed:.1f}s)")
			continue

		flags = response["flags"]
		count = len(candidates)

		if flags:
			print(f"[{sent_id}] {count} {task_name} — {len(flags)} flagged ({elapsed:.1f}s)")
			for flag in flags:
				detail = _format_flag_detail(task_name, flag)
				print(f"  {detail}")
				all_flags.append({
					"task": task_name,
					"model": model,
					"sent_id": sent_id,
					**flag,
				})
		else:
			print(f"[{sent_id}] {count} {task_name} — clean ({elapsed:.1f}s)")

	queried = len(parsed_blocks) - skipped
	avg = total_time / max(1, queried)
	print(f"  [{task_name}/{model}] {total_time:.1f}s total, {avg:.1f}s/sent, "
		  f"{len(all_flags)} flags, {skipped} skipped\n")

	return all_flags


def _format_flag_detail(task_name, flag):
	token_id = flag.get("id", "?")
	form = flag.get("form", "?")

	if task_name == "lemma":
		lemma = flag.get("lemma", "?")
		reason = flag.get("reason", "")
		return f'tok {token_id} "{form}" lemma="{lemma}": {reason}'

	elif task_name == "tense":
		issue = flag.get("issue", "?")
		current = flag.get("current", "?")
		expected = flag.get("expected", "?")
		reason = flag.get("reason", "")
		return f'tok {token_id} "{form}" [{issue}]: {current} → {expected} — {reason}'

	elif task_name == "aux":
		current = flag.get("current_upos", "?")
		expected = flag.get("expected_upos", "?")
		reason = flag.get("reason", "")
		return f'tok {token_id} "{form}": {current} → {expected} — {reason}'

	elif task_name == "que":
		current = flag.get("current_upos", "?")
		expected = flag.get("expected_upos", "?")
		reason = flag.get("reason", "")
		return f'tok {token_id} "{form}": {current} → {expected} — {reason}'

	else:
		return str(flag)
