"""Cross-pipeline comparison task.

Compares Stanza and spaCy annotations to find disagreements.
Convention differences (ligature normalization + edit distance ≤ 2)
are filtered. Remaining disagreements are classified into tiers
and flagged for review.
"""

from pathlib import Path

try:
	from conllu_tools.compare import compare_files
except ImportError:
	import sys
	for parent in [
		Path(__file__).resolve().parent.parent,
		Path(__file__).resolve().parent.parent / "annotate",
		Path(__file__).resolve().parent.parent.parent / "scripts" / "annotate",
	]:
		if (parent / "conllu_tools" / "compare.py").exists():
			sys.path.insert(0, str(parent))
			break
	from conllu_tools.compare import compare_files


def _normalize_ligatures(text):
	return (
		text
		.replace("\u0153", "oe").replace("\u0152", "Oe")
		.replace("\u00e6", "ae").replace("\u00c6", "Ae")
	)


def _edit_distance(a, b):
	if len(a) < len(b):
		return _edit_distance(b, a)
	if len(b) == 0:
		return len(a)
	prev = list(range(len(b) + 1))
	for i, ca in enumerate(a):
		curr = [i + 1]
		for j, cb in enumerate(b):
			curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + (ca != cb)))
		prev = curr
	return prev[len(b)]


def _is_near_match(lemma_a, lemma_b):
	a = _normalize_ligatures(lemma_a.lower())
	b = _normalize_ligatures(lemma_b.lower())
	if a == b:
		return True
	return _edit_distance(a, b) <= 2


# GSD pronoun/possessive lemma conventions that differ from spaCy
_convention_pairs = {
	("lui", "il"), ("moi", "je"), ("eux", "il"), ("eux", "le"),
	("eux", "son"), ("son", "leur"), ("son", "notre"), ("son", "votre"),
	("madame", "monsieur"), ("puis", "pouvoir"),
}


def _is_convention_diff(stanza_lemma, spacy_lemma):
	return (stanza_lemma.lower(), spacy_lemma.lower()) in _convention_pairs


def _parse_feats(feats_str):
	if not feats_str or feats_str == "_":
		return {}
	result = {}
	for pair in feats_str.split("|"):
		parts = pair.split("=", 1)
		if len(parts) == 2:
			result[parts[0]] = parts[1]
	return result


def find_spacy_file(stanza_path, spacy_dir):
	stanza_name = Path(stanza_path).name
	spacy_path = Path(spacy_dir) / stanza_name
	if spacy_path.exists():
		return str(spacy_path)
	for candidate in Path(spacy_dir).rglob(stanza_name):
		return str(candidate)
	return None


def _compare_lemmas(result, littre_checker):
	flags = []
	skipped_near = 0
	skipped_same = 0
	skipped_convention = 0

	for tc in result.all_comparisons():
		vals = tc.values("lemma")
		stanza_lemma = vals.get("stanza", "")
		spacy_lemma = vals.get("spacy", "")

		if stanza_lemma == spacy_lemma:
			skipped_same += 1
			continue

		if _is_near_match(stanza_lemma, spacy_lemma):
			skipped_near += 1
			continue

		if _is_convention_diff(stanza_lemma, spacy_lemma):
			skipped_convention += 1
			continue

		stanza_tok = tc.pipelines["stanza"]

		flag = {
			"task": "compare",
			"sent_id": tc.sent_id,
			"id": tc.token_id,
			"token_id": tc.token_id,
			"form": tc.form,
			"lemma": stanza_lemma,
			"upos": stanza_tok.get("upos", "_"),
			"feats": stanza_tok.get("feats", "_"),
			"suggested": spacy_lemma,
			"issue": "lemma_disagreement",
		}

		if littre_checker:
			stanza_found = bool(littre_checker.lookup(stanza_lemma))
			spacy_found = bool(littre_checker.lookup(spacy_lemma))

			if not stanza_found and spacy_found:
				flag["tier"] = 1
				flag["reason"] = (
					f"stanza lemma '{stanza_lemma}' not in Littré; "
					f"spaCy suggests '{spacy_lemma}'"
				)
			elif not stanza_found and not spacy_found:
				flag["tier"] = 3
				flag["reason"] = (
					f"neither lemma in Littré: "
					f"stanza='{stanza_lemma}', spaCy='{spacy_lemma}'"
				)
			else:
				flag["tier"] = 3
				flag["reason"] = (
					f"both valid lemmas: "
					f"stanza='{stanza_lemma}', spaCy='{spacy_lemma}'"
				)
		else:
			flag["tier"] = 3
			flag["reason"] = (
				f"stanza='{stanza_lemma}', spaCy='{spacy_lemma}'"
			)

		flags.append(flag)

	tier1 = sum(1 for f in flags if f.get("tier") == 1)
	tier3 = sum(1 for f in flags if f.get("tier") == 3)

	for f in flags:
		tier = f.get("tier", "?")
		print(f"  [{f['sent_id']}] tok {f['id']} \"{f['form']}\" "
			  f"T{tier}: {f['lemma']} \u2192 {f['suggested']} \u2014 {f['reason']}")

	total_compared = skipped_same + skipped_near + skipped_convention + len(flags)
	print(f"  [compare] lemma: {total_compared} compared, "
		  f"{skipped_same} agree, {skipped_near} near-match, "
		  f"{skipped_convention} convention, "
		  f"{len(flags)} flagged (T1: {tier1}, T3: {tier3})")

	return flags


def _compare_tense(result):
	flags = []
	skipped_same = 0
	skipped_non_finite = 0
	skipped_convention = 0

	for tc in result.all_comparisons():
		stanza_tok = tc.pipelines["stanza"]
		spacy_tok = tc.pipelines["spacy"]

		if stanza_tok.get("upos", "_") not in ("VERB", "AUX"):
			continue

		stanza_feats = _parse_feats(stanza_tok.get("feats", "_"))
		spacy_feats = _parse_feats(spacy_tok.get("feats", "_"))

		if stanza_feats.get("VerbForm") != "Fin":
			skipped_non_finite += 1
			continue

		stanza_mood = stanza_feats.get("Mood", "")
		stanza_tense = stanza_feats.get("Tense", "")
		spacy_mood = spacy_feats.get("Mood", "")
		spacy_tense = spacy_feats.get("Tense", "")

		if stanza_mood == spacy_mood and stanza_tense == spacy_tense:
			skipped_same += 1
			continue

		# Imperative/indicative is a convention difference, not an error
		if (
			{stanza_mood, spacy_mood} == {"Imp", "Ind"}
			and stanza_tense == spacy_tense
		):
			skipped_convention += 1
			continue

		stanza_label = f"{stanza_mood} {stanza_tense}".strip()
		spacy_label = f"{spacy_mood} {spacy_tense}".strip()

		flag = {
			"task": "compare_tense",
			"sent_id": tc.sent_id,
			"id": tc.token_id,
			"token_id": tc.token_id,
			"form": tc.form,
			"lemma": stanza_tok.get("lemma", "_"),
			"upos": stanza_tok.get("upos", "_"),
			"feats": stanza_tok.get("feats", "_"),
			"issue": "tense_disagreement",
			"current": stanza_label,
			"expected": spacy_label,
			"spacy_feats": spacy_tok.get("feats", "_"),
			"reason": f"stanza={stanza_label}, spaCy={spacy_label}",
		}

		flags.append(flag)

	for f in flags:
		print(f"  [{f['sent_id']}] tok {f['id']} \"{f['form']}\" "
			  f"tense: {f['current']} \u2192 {f['expected']}")

	print(f"  [compare] tense: {skipped_same} agree, "
		  f"{skipped_non_finite} non-finite, "
		  f"{skipped_convention} convention, "
		  f"{len(flags)} flagged")

	return flags


def run(stanza_path, spacy_dir, parsed_blocks=None, littre_checker=None):
	"""Run cross-pipeline comparison and emit flags.

	Returns: list of flag dicts
	"""
	spacy_path = find_spacy_file(stanza_path, spacy_dir)
	if not spacy_path:
		print(f"  [compare] No matching spaCy file for {stanza_path}")
		return []

	print(f"=== compare (stanza vs spacy) ===")
	print(f"  spaCy file: {spacy_path}")

	result = compare_files(stanza=stanza_path, spacy=spacy_path)

	if result.warnings:
		mwt_warns = sum(1 for w in result.warnings if "MWT" in w or "unmatched" in w)
		other_warns = len(result.warnings) - mwt_warns
		if mwt_warns:
			print(f"  [warn] {mwt_warns} MWT/alignment skips")
		if other_warns:
			for w in result.warnings:
				if "MWT" not in w and "unmatched" not in w:
					print(f"  [warn] {w}")

	lemma_flags = _compare_lemmas(result, littre_checker)
	tense_flags = _compare_tense(result)

	print()
	return lemma_flags + tense_flags
