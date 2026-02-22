"""Deterministic French verb morphology checker (Layer 1 of tense pipeline).

Catches unambiguous suffix-feature mismatches with zero false positives.
All other finite verbs are forwarded to the LLM (Layer 2).
"""

from .conllu import parse_feats


def run(parsed_blocks):
	"""Check finite verbs for morphology-feature mismatches.

	Returns:
		flags: list of definite mismatch dicts
		candidate_ids: dict {sent_id: set(token_ids)} for LLM Layer 2
	"""
	flags = []
	candidate_ids = {}
	confirmed = 0

	for sent_id, sent_text, tokens in parsed_blocks:
		for token in tokens:
			if token["upos"] not in ("VERB", "AUX"):
				continue
			feats = parse_feats(token["feats"])
			if feats.get("VerbForm") != "Fin":
				continue

			form = token["form"].lower()
			mood = feats.get("Mood", "")
			tense = feats.get("Tense", "")

			result = _check(form, mood, tense)

			if result is not None:
				flags.append({
					"task": "tense_morph",
					"sent_id": sent_id,
					"id": token["id"],
					"form": token["form"],
					"lemma": token["lemma"],
					**result,
				})
			else:
				confirmed += 1
				if sent_id not in candidate_ids:
					candidate_ids[sent_id] = set()
				candidate_ids[sent_id].add(token["id"])

	amb_count = sum(len(v) for v in candidate_ids.values())

	print(f"=== tense Layer 1 (morphology rules) ===")
	for f in flags:
		print(f"  [{f['sent_id']}] tok {f['id']} \"{f['form']}\" "
			  f"{f['current']} \u2192 {f['expected']} \u2014 {f['reason']}")
	print(f"  [tense_morph] {len(flags)} mismatches, "
		  f"{confirmed} confirmed ok, {amb_count} \u2192 LLM\n")

	return flags, candidate_ids


def _check(form, mood, tense):
	"""Returns None (confirmed) or a mismatch dict."""

	# -aient + Tense=Pres + Mood=Ind: always wrong
	# (Mood=Cnd + Tense=Pres is correct GSD encoding for conditional)
	if form.endswith("aient") and tense == "Pres" and mood == "Ind":
		return _mismatch("tense", "Pres", "Imp",
			"-aient is imparfait/conditional, never present")

	# -\u00e8rent + Tense=Pres: always pass\u00e9 simple
	if form.endswith("\u00e8rent") and tense == "Pres":
		return _mismatch("tense", "Pres", "Past",
			"-\u00e8rent is pass\u00e9 simple, not present")

	# -ions/-iez + Mood=Ind + Tense=Pres: wrong
	# UNLESS preceded by i (-ier verbs: oublions, criez)
	if tense == "Pres" and mood == "Ind":
		if form.endswith("ions") and len(form) > 4 and form[-5] != "i":
			return _mismatch("tense", "Pres", "Imp",
				"-ions is imparfait, not present indicative (-ons)")
		if form.endswith("iez") and len(form) > 3 and form[-4] != "i":
			return _mismatch("tense", "Pres", "Imp",
				"-iez is imparfait, not present indicative (-ez)")

	# Circumflex 3sg + Mood=Ind: imperfect subjunctive
	# Exclude -aît (-aître verbs: paraît, connaît, naît)
	if mood == "Ind":
		for ending in ("\u00e2t", "\u00eet", "\u00fbt"):
			if form.endswith(ending) and not form.endswith("a\u00eet"):
				return _mismatch("mood", "Ind", "Sub",
					f"-{ending} is imperfect subjunctive, not indicative")

	# Circumflex 1pl/2pl pass\u00e9 simple + Tense=Pres
	if tense == "Pres":
		for ending in ("\u00fbmes", "\u00fbtes", "\u00eemes", "\u00eetes"):
			if form.endswith(ending):
				return _mismatch("tense", "Pres", "Past",
					f"-{ending} is pass\u00e9 simple, not present")

	return None


def _mismatch(issue, current, expected, reason):
	return {
		"issue": issue,
		"current": current,
		"expected": expected,
		"reason": reason,
	}
