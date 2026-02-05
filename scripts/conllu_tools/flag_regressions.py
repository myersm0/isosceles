#!/usr/bin/env python3
"""
Flag potential regressions between original and LLM-corrected CoNLL-U files.

Usage:
  python flag_regressions.py original.conllu corrected.conllu
  python flag_regressions.py original.conllu corrected.conllu --fix
  python flag_regressions.py original.conllu corrected.conllu --fix --output fixed.conllu --log fixes.log
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

try:
	from .validator import parse_conllu_block
except ImportError:
	from validator import parse_conllu_block


@dataclass
class Flag:
	severity: str
	category: str
	sent_id: str
	sent_text: str
	token_id: int
	form: str
	message: str
	sentence_index: int = -1
	orig_token: Optional[dict] = None
	corr_token: Optional[dict] = None


CLITIC_FORMS = frozenset(
	["le", "la", "les", "l'", "l\u2019", "me", "m'", "m\u2019",
	 "te", "t'", "t\u2019", "se", "s'", "s\u2019", "nous", "vous"]
)
INTENSITY_MARKERS = frozenset(
	["si", "tel", "telle", "tels", "telles", "tant", "tellement", "assez", "trop"]
)
SUBJECT_PRONOUNS_AFTER_SI = frozenset(["il", "ils", "elle", "elles", "on"])


def parse_feats(feats_string: str) -> dict[str, str]:
	if feats_string == "_":
		return {}
	result = {}
	for pair in feats_string.split("|"):
		parts = pair.split("=", 1)
		if len(parts) == 2:
			result[parts[0]] = parts[1]
	return result


def is_valid_infinitive(lemma: str) -> bool:
	lower = lemma.lower()
	return lower.endswith("er") or lower.endswith("ir") or lower.endswith("re")


def format_token_row(tok: dict) -> str:
	return "\t".join([
		str(tok["id"]), tok["form"], tok["lemma"], tok["upos"],
		tok.get("xpos", "_"), tok.get("feats", "_"),
		str(tok["head"]), tok["deprel"],
		tok.get("deps", "_"), tok.get("misc", "_"),
	])


def read_conllu_blocks(path: str) -> list[list[str]]:
	blocks: list[list[str]] = []
	current: list[str] = []
	with open(path, encoding="utf-8") as f:
		for line in f:
			line = line.rstrip("\n")
			if line == "" and current:
				blocks.append(current)
				current = []
			else:
				current.append(line)
	if current:
		blocks.append(current)
	return blocks


def parse_block(lines: list[str]) -> tuple[str, str, list[dict]]:
	return parse_conllu_block("\n".join(lines))


def preceding_forms(tokens: list[dict], index: int, window: int) -> list[str]:
	start = max(0, index - window)
	return [tokens[j]["form"].lower() for j in range(start, index)]


def flag_regressions(
	original: list[tuple[str, str, list[dict]]],
	corrected: list[tuple[str, str, list[dict]]],
) -> list[Flag]:
	flags: list[Flag] = []

	if len(original) != len(corrected):
		flags.append(Flag(
			"hard", "sentence_count_mismatch", "", "",
			0, "", f"original has {len(original)}, corrected has {len(corrected)} sentences",
		))
		return flags

	for sent_idx, ((orig_sid, orig_text, orig_tokens), (corr_sid, corr_text, corr_tokens)) in enumerate(zip(original, corrected)):
		sid = corr_sid or orig_sid
		text = corr_text or orig_text
		orig_by_id = {t["id"]: t for t in orig_tokens}

		for idx, ct in enumerate(corr_tokens):
			ot = orig_by_id.get(ct["id"])
			if ot is None:
				flags.append(Flag(
					"hard", "token_id_mismatch", sid, text,
					ct["id"], ct["form"],
					f"token ID {ct['id']} not found in original",
					sentence_index=sent_idx,
				))
				continue

			cf = parse_feats(ct.get("feats", "_"))
			of = parse_feats(ot.get("feats", "_"))
			changed_lemma = ot["lemma"] != ct["lemma"]
			changed_upos = ot["upos"] != ct["upos"]
			changed_feats = ot.get("feats", "_") != ct.get("feats", "_")
			anything_changed = changed_lemma or changed_upos or changed_feats

			def add_flag(severity, category, message):
				flags.append(Flag(
					severity, category, sid, text,
					ct["id"], ct["form"], message,
					sentence_index=sent_idx,
					orig_token=ot, corr_token=ct,
				))

			# ── all-token checks ──

			if ct["upos"] in ("VERB", "AUX") and ct["lemma"] != "_":
				if not is_valid_infinitive(ct["lemma"]):
					add_flag("hard", "non_infinitive_lemma",
						f"VERB/AUX lemma '{ct['lemma']}' not a valid infinitive")

			if cf.get("VerbForm") == "Part":
				if ct["lemma"].lower() == ct["form"].lower():
					add_flag("hard", "participle_self_lemma",
						"participle with lemma == surface form")

			form_lower = ct["form"].lower()
			if form_lower in ("fût", "eût") and ct["upos"] in ("AUX", "VERB"):
				if cf.get("Mood") != "Sub" or cf.get("Tense") != "Imp":
					add_flag("hard", "circumflex_mood",
						"circumflexed form must be Mood=Sub|Tense=Imp")

			if form_lower in ("fut", "eut") and ct["upos"] in ("AUX", "VERB"):
				if cf.get("Mood") == "Sub":
					add_flag("hard", "non_circumflex_subjunctive",
						"non-circumflexed form should not be Mood=Sub")

			if not anything_changed:
				continue

			# ── changed-token checks ──

			if form_lower in ("s'", "s\u2019"):
				if ot["upos"] == "SCONJ" and ct["upos"] == "PRON":
					if idx + 1 < len(corr_tokens):
						next_form = corr_tokens[idx + 1]["form"].lower()
						if next_form in SUBJECT_PRONOUNS_AFTER_SI:
							add_flag("hard", "si_to_se_regression",
								f"s' before '{next_form}': likely conditional si, not reflexive se")

			if form_lower in CLITIC_FORMS:
				if ot["upos"] == "PRON" and ct["upos"] == "DET":
					add_flag("hard", "clitic_pron_to_det",
						"clitic pronoun changed to DET")

			if ot["upos"] == "AUX" and ct["upos"] in ("AUX", "VERB"):
				orig_tense = of.get("Tense", "")
				corr_tense = cf.get("Tense", "")
				if orig_tense and corr_tense and orig_tense != corr_tense:
					for offset in range(1, 4):
						j = idx + offset
						if j >= len(corr_tokens):
							break
						fj = parse_feats(corr_tokens[j].get("feats", "_"))
						if fj.get("VerbForm") == "Part":
							add_flag("soft", "aux_tense_in_compound",
								f"AUX tense {orig_tense}→{corr_tense} before participle")
							break

			if form_lower in ("partie", "part"):
				if ot["upos"] == "NOUN" and ct["upos"] == "VERB":
					head_tok = orig_by_id.get(ot["head"])
					if head_tok and head_tok["lemma"].lower() in ("faire", "fait"):
						add_flag("hard", "faire_partie_lvc",
							"partie in 'faire partie de' should stay NOUN")

			if form_lower in ("que", "qu'", "qu\u2019"):
				if ot["upos"] != "SCONJ" and ct["upos"] == "SCONJ":
					prec = preceding_forms(corr_tokens, idx, 3)
					if "est" in prec and any(f in prec for f in ("-ce", "ce")):
						add_flag("hard", "interrogative_que_to_sconj",
							"que in qu'est-ce que should not be SCONJ")
					if ot["upos"] == "PRON":
						wider = set(preceding_forms(corr_tokens, idx, 5))
						if not wider & INTENSITY_MARKERS:
							add_flag("soft", "que_to_sconj_no_trigger",
								"que PRON→SCONJ without nearby intensity marker")

	return deduplicate_flags(flags)


def deduplicate_flags(flags: list[Flag]) -> list[Flag]:
	seen: dict[tuple[int, int], Flag] = {}
	for flag in flags:
		key = (flag.sentence_index, flag.token_id)
		if key not in seen:
			seen[key] = flag
		else:
			existing = seen[key]
			if flag.severity == "hard" and existing.severity == "soft":
				flag.message = f"{flag.message}; {existing.message}"
				seen[key] = flag
			else:
				existing.message = f"{existing.message}; {flag.message}"
	return list(seen.values())


def format_flag(flag: Flag) -> str:
	label = "HARD" if flag.severity == "hard" else "SOFT"
	lines = [f"[{label}] {flag.category} — {flag.message}"]
	lines.append(f"  Sentence {flag.sent_id}: {flag.sent_text}")
	lines.append(f"  Token {flag.token_id} \"{flag.form}\"")
	if flag.orig_token:
		lines.append(f"  orig: {format_token_row(flag.orig_token)}")
	if flag.corr_token:
		lines.append(f"  corr: {format_token_row(flag.corr_token)}")
	return "\n".join(lines)


def revert_token_in_block(block_lines: list[str], token_id: int, orig_token: dict):
	target = str(token_id)
	for i, line in enumerate(block_lines):
		if line.startswith("#") or not line.strip():
			continue
		parts = line.split("\t")
		if len(parts) >= 6 and parts[0] == target:
			parts[2] = orig_token["lemma"]
			parts[3] = orig_token["upos"]
			parts[5] = orig_token.get("feats", "_")
			block_lines[i] = "\t".join(parts)
			return


def write_conllu_file(path: str, blocks: list[list[str]]):
	with open(path, "w", encoding="utf-8") as f:
		for block in blocks:
			for line in block:
				f.write(line + "\n")
			f.write("\n")


def interactive_review(
	flags: list[Flag],
	corr_blocks: list[list[str]],
	output_path: str,
	log_path: str,
) -> int:
	reverted = 0
	log_entries: list[str] = []
	timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

	for i, flag in enumerate(flags):
		print(f"\n{'='*72}")
		print(f"  Flag {i + 1}/{len(flags)}")
		print(f"{'='*72}")
		print(format_flag(flag))

		can_revert = (
			flag.orig_token is not None
			and flag.corr_token is not None
			and flag.sentence_index >= 0
			and (
				flag.orig_token["lemma"] != flag.corr_token["lemma"]
				or flag.orig_token["upos"] != flag.corr_token["upos"]
				or flag.orig_token.get("feats", "_") != flag.corr_token.get("feats", "_")
			)
		)

		if can_revert:
			try:
				response = input("\n  [a]ccept revert / [Enter] skip > ").strip().lower()
			except (EOFError, KeyboardInterrupt):
				print("\nAborted.")
				break
			if response == "a":
				revert_token_in_block(
					corr_blocks[flag.sentence_index],
					flag.token_id, flag.orig_token,
				)
				reverted += 1
				entry = (
					f"{timestamp} REVERTED [{flag.category}] "
					f"Sentence {flag.sent_id} Token {flag.token_id} \"{flag.form}\": "
					f"{flag.corr_token['upos']} {flag.corr_token['lemma']} "
					f"→ {flag.orig_token['upos']} {flag.orig_token['lemma']}"
				)
				log_entries.append(entry)
				print("  ✓ reverted")
		else:
			try:
				input("\n  (no revert available) [Enter] to continue > ")
			except (EOFError, KeyboardInterrupt):
				print("\nAborted.")
				break

	if reverted > 0:
		write_conllu_file(output_path, corr_blocks)
		print(f"\nWrote {output_path} with {reverted} reversion(s)")

	if log_entries:
		with open(log_path, "a", encoding="utf-8") as f:
			for entry in log_entries:
				f.write(entry + "\n")
		print(f"Appended {len(log_entries)} entries to {log_path}")

	return reverted


def main():
	parser = argparse.ArgumentParser(
		description="Flag potential regressions in LLM-corrected CoNLL-U files."
	)
	parser.add_argument("original", help="Original (pre-correction) CoNLL-U file")
	parser.add_argument("corrected", help="LLM-corrected CoNLL-U file")
	parser.add_argument("--fix", action="store_true",
		help="Interactive mode: review flags and optionally revert regressions")
	parser.add_argument("--output",
		help="Output path for fixed file (default: overwrite corrected)")
	parser.add_argument("--log", default="regression_fixes.log",
		help="Log file for accepted reverts (default: regression_fixes.log)")
	args = parser.parse_args()

	orig_blocks = read_conllu_blocks(args.original)
	corr_blocks = read_conllu_blocks(args.corrected)

	orig_sentences = [parse_block(b) for b in orig_blocks]
	corr_sentences = [parse_block(b) for b in corr_blocks]

	flags = flag_regressions(orig_sentences, corr_sentences)

	hard_flags = [f for f in flags if f.severity == "hard"]
	soft_flags = [f for f in flags if f.severity == "soft"]

	if not args.fix:
		if hard_flags:
			print("=== HARD FLAGS (likely regressions) ===\n")
			for f in hard_flags:
				print(format_flag(f))
				print()
		if soft_flags:
			print("=== SOFT FLAGS (review recommended) ===\n")
			for f in soft_flags:
				print(format_flag(f))
				print()
		total_tokens = sum(len(toks) for _, _, toks in corr_sentences)
		print("---")
		print(f"{len(orig_sentences)} sentences, {total_tokens} tokens")
		print(f"{len(hard_flags)} hard flags, {len(soft_flags)} soft flags")
	else:
		if not flags:
			print("No flags to review.")
			return
		output_path = args.output or args.corrected
		n = interactive_review(flags, corr_blocks, output_path, args.log)
		total = len(flags)
		print(f"\n{n} reverted out of {total} flags reviewed")


if __name__ == "__main__":
	main()
