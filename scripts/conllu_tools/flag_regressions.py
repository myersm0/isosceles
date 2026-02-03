#!/usr/bin/env python3
"""
Flag potential regressions between original and LLM-corrected CoNLL-U files.
"""

import sys
from dataclasses import dataclass, field
from typing import Optional

try:
	from .validator import parse_conllu_block
except ImportError:
	from validator import parse_conllu_block


@dataclass
class Flag:
	severity: str  # "hard" or "soft"
	category: str
	sent_id: str
	sent_text: str
	token_id: int
	form: str
	original: str
	corrected: str
	message: str


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


def read_conllu_sentences(path: str) -> list[tuple[str, str, list[dict]]]:
	sentences = []
	current_block = []
	with open(path, encoding="utf-8") as f:
		for line in f:
			line = line.rstrip("\n")
			if line == "" and current_block:
				sent_id, sent_text, tokens = parse_conllu_block("\n".join(current_block))
				if tokens:
					sentences.append((sent_id or "", sent_text or "", tokens))
				current_block = []
			else:
				current_block.append(line)
	if current_block:
		sent_id, sent_text, tokens = parse_conllu_block("\n".join(current_block))
		if tokens:
			sentences.append((sent_id or "", sent_text or "", tokens))
	return sentences


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
			"hard", "sentence_count_mismatch", "", "", 0, "",
			str(len(original)), str(len(corrected)),
			"sentence count mismatch between files",
		))
		return flags

	for (orig_sid, orig_text, orig_tokens), (corr_sid, corr_text, corr_tokens) in zip(original, corrected):
		sid = corr_sid or orig_sid
		text = corr_text or orig_text
		orig_by_id = {t["id"]: t for t in orig_tokens}

		for idx, ct in enumerate(corr_tokens):
			ot = orig_by_id.get(ct["id"])
			if ot is None:
				flags.append(Flag(
					"hard", "token_id_mismatch", sid, text,
					ct["id"], ct["form"], "_", "_",
					f"token ID {ct['id']} not found in original",
				))
				continue

			cf = parse_feats(ct.get("feats", "_"))
			of = parse_feats(ot.get("feats", "_"))
			changed_lemma = ot["lemma"] != ct["lemma"]
			changed_upos = ot["upos"] != ct["upos"]
			changed_feats = ot.get("feats", "_") != ct.get("feats", "_")
			anything_changed = changed_lemma or changed_upos or changed_feats

			# ── all-token checks (regardless of whether changed) ──

			if ct["upos"] in ("VERB", "AUX") and ct["lemma"] != "_":
				if not is_valid_infinitive(ct["lemma"]):
					flags.append(Flag(
						"hard", "non_infinitive_lemma", sid, text,
						ct["id"], ct["form"],
						f"{ot['upos']} {ot['lemma']}", f"{ct['upos']} {ct['lemma']}",
						f"VERB/AUX lemma '{ct['lemma']}' not a valid infinitive",
					))

			if cf.get("VerbForm") == "Part":
				if ct["lemma"].lower() == ct["form"].lower():
					flags.append(Flag(
						"hard", "participle_self_lemma", sid, text,
						ct["id"], ct["form"],
						ot["lemma"], ct["lemma"],
						"participle with lemma == surface form",
					))

			# circumflex checks apply to final output regardless of change
			form_lower = ct["form"].lower()
			if form_lower in ("fût", "eût") and ct["upos"] in ("AUX", "VERB"):
				if cf.get("Mood") != "Sub" or cf.get("Tense") != "Imp":
					flags.append(Flag(
						"hard", "circumflex_mood", sid, text,
						ct["id"], ct["form"],
						ot.get("feats", "_"), ct.get("feats", "_"),
						"circumflexed form must be Mood=Sub|Tense=Imp",
					))

			if form_lower in ("fut", "eut") and ct["upos"] in ("AUX", "VERB"):
				if cf.get("Mood") == "Sub":
					flags.append(Flag(
						"hard", "non_circumflex_subjunctive", sid, text,
						ct["id"], ct["form"],
						ot.get("feats", "_"), ct.get("feats", "_"),
						"non-circumflexed form should not be Mood=Sub",
					))

			if not anything_changed:
				continue

			# ── changed-token checks ──

			# s' SCONJ→PRON before subject pronoun
			if form_lower in ("s'", "s\u2019"):
				if ot["upos"] == "SCONJ" and ct["upos"] == "PRON":
					if idx + 1 < len(corr_tokens):
						next_form = corr_tokens[idx + 1]["form"].lower()
						if next_form in SUBJECT_PRONOUNS_AFTER_SI:
							flags.append(Flag(
								"hard", "si_to_se_regression", sid, text,
								ct["id"], ct["form"],
								"SCONJ si", "PRON se",
								f"s' before '{next_form}': likely conditional si",
							))

			# clitic PRON→DET
			if form_lower in CLITIC_FORMS:
				if ot["upos"] == "PRON" and ct["upos"] == "DET":
					flags.append(Flag(
						"hard", "clitic_pron_to_det", sid, text,
						ct["id"], ct["form"],
						"PRON", "DET",
						"clitic pronoun changed to DET",
					))

			# AUX tense changed when followed by participle
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
							flags.append(Flag(
								"soft", "aux_tense_in_compound", sid, text,
								ct["id"], ct["form"],
								f"Tense={orig_tense}", f"Tense={corr_tense}",
								"auxiliary tense changed in compound tense",
							))
							break

			# partie NOUN→VERB under faire (faire partie de)
			if form_lower in ("partie", "part"):
				if ot["upos"] == "NOUN" and ct["upos"] == "VERB":
					head_tok = orig_by_id.get(ot["head"])
					if head_tok and head_tok["lemma"].lower() in ("faire", "fait"):
						flags.append(Flag(
							"hard", "faire_partie_lvc", sid, text,
							ct["id"], ct["form"],
							f"NOUN {ot['lemma']}", f"VERB {ct['lemma']}",
							"partie in 'faire partie de' should stay NOUN",
						))

			# que/qu' changed to SCONJ
			if form_lower in ("que", "qu'", "qu\u2019"):
				if ot["upos"] != "SCONJ" and ct["upos"] == "SCONJ":
					prec = preceding_forms(corr_tokens, idx, 3)
					if "est" in prec and any(f in prec for f in ("-ce", "ce")):
						flags.append(Flag(
							"hard", "interrogative_que_to_sconj", sid, text,
							ct["id"], ct["form"],
							ot["upos"], "SCONJ",
							"que in qu'est-ce que should not be SCONJ",
						))

					if ot["upos"] == "PRON":
						wider = set(preceding_forms(corr_tokens, idx, 5))
						if not wider & INTENSITY_MARKERS:
							flags.append(Flag(
								"soft", "que_to_sconj_no_trigger", sid, text,
								ct["id"], ct["form"],
								"PRON", "SCONJ",
								"que PRON→SCONJ without nearby intensity marker",
							))

	return flags


def format_flag(flag: Flag) -> str:
	severity_label = "HARD" if flag.severity == "hard" else "SOFT"
	lines = [f"[{severity_label}] {flag.category}"]
	if flag.sent_id:
		preview = flag.sent_text[:80] + "…" if len(flag.sent_text) > 80 else flag.sent_text
		lines.append(f"  Sentence {flag.sent_id}: {preview}")
	lines.append(f"  Token {flag.token_id} \"{flag.form}\": {flag.message}")
	if flag.original and flag.corrected:
		lines.append(f"  {flag.original} → {flag.corrected}")
	return "\n".join(lines)


def main():
	if len(sys.argv) < 3:
		print("usage: python flag_regressions.py <original.conllu> <corrected.conllu>", file=sys.stderr)
		sys.exit(1)

	original = read_conllu_sentences(sys.argv[1])
	corrected = read_conllu_sentences(sys.argv[2])
	flags = flag_regressions(original, corrected)

	hard_flags = [f for f in flags if f.severity == "hard"]
	soft_flags = [f for f in flags if f.severity == "soft"]

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

	total_tokens = sum(len(toks) for _, _, toks in corrected)
	print("---")
	print(f"{len(original)} sentences, {total_tokens} tokens")
	print(f"{len(hard_flags)} hard flags, {len(soft_flags)} soft flags")


if __name__ == "__main__":
	main()
