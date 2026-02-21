"""Littré dictionary-based lemma and UPOS validator."""

import sqlite3
import unicodedata


target_upos = {"NOUN", "VERB", "ADJ", "ADV", "AUX"}

skip_lemmas = {
	"être", "avoir", "faire", "aller", "pouvoir", "vouloir", "devoir",
	"savoir", "falloir", "il", "lui", "moi", "nous", "vous", "eux",
	"elle", "on", "se", "soi", "ce", "le", "la", "les", "un", "une",
	"de", "en", "y", "ne", "pas", "plus", "que", "qui", "où",
	"son", "mon", "ton", "tout", "même", "autre", "quel",
	"bien", "très", "beaucoup", "peu", "trop", "assez", "tant",
}

littre_pos_to_upos = {
	"v.": "VERB",
	"s.": "NOUN",
	"adj.": "ADJ",
	"adv.": "ADV",
	"prép.": "ADP",
	"conj.": "CCONJ",
	"pron.": "PRON",
	"art.": "DET",
	"interj.": "INTJ",
	"part.": "VERB",
}


def strip_accents(text):
	nfkd = unicodedata.normalize("NFKD", text)
	return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def map_pos_to_upos(pos_string):
	if not pos_string:
		return set()
	upos_set = set()
	pos_lower = pos_string.lower().strip()
	for prefix, upos in littre_pos_to_upos.items():
		if prefix in pos_lower:
			upos_set.add(upos)
	if "part. passé" in pos_lower or "part." in pos_lower:
		upos_set.add("ADJ")
		upos_set.add("VERB")
	if "s." in pos_lower:
		upos_set.add("PROPN")
	if "v." in pos_lower:
		upos_set.add("AUX")
	return upos_set


class LittreChecker:
	def __init__(self, db_path):
		self.conn = sqlite3.connect(db_path)
		self.conn.row_factory = sqlite3.Row
		self._cache = {}

	def lookup(self, lemma):
		if lemma in self._cache:
			return self._cache[lemma]

		upper = lemma.upper()
		ascii_lower = strip_accents(lemma).lower().replace(" ", "_").replace("-", "_")
		seen = set()
		results = []

		def add_rows(rows):
			for row in rows:
				key = (row["headword"], row["pos"])
				if key not in seen:
					seen.add(key)
					results.append(key)

		add_rows(self.conn.execute(
			"SELECT headword, pos FROM entries WHERE headword = ?",
			(upper,)
		).fetchall())

		add_rows(self.conn.execute(
			"SELECT headword, pos FROM entries "
			"WHERE (headword LIKE ? || ', %' OR headword LIKE ? || ' %' OR headword LIKE ? || ' (%')"
			" AND headword != ?",
			(upper, upper, upper, upper)
		).fetchall())

		add_rows(self.conn.execute(
			"SELECT headword, pos FROM entries "
			"WHERE entry_id = ? OR entry_id LIKE ? || '.%' OR entry_id LIKE ? || '\\_%' ESCAPE '\\'",
			(ascii_lower, ascii_lower, ascii_lower)
		).fetchall())

		self._cache[lemma] = results
		return results

	def close(self):
		self.conn.close()


def run(blocks, parsed_blocks, db_path, strict=False):
	"""Run Littré classifier. Returns list of flag dicts."""
	checker = LittreChecker(db_path)
	all_flags = []

	for sent_id, sent_text, tokens in parsed_blocks:
		for token in tokens:
			if token["upos"] not in target_upos:
				continue
			if token["lemma"].lower() in skip_lemmas:
				continue
			if token["lemma"][0].isupper() and token["upos"] == "PROPN":
				continue

			results = checker.lookup(token["lemma"])

			if not results:
				all_flags.append({
					"task": "littre",
					"sent_id": sent_id,
					"token_id": token["id"],
					"form": token["form"],
					"lemma": token["lemma"],
					"upos": token["upos"],
					"issue": "lemma_not_found",
					"detail": f"'{token['lemma']}' not in Littré",
				})
				continue

			if strict:
				found_upos = set()
				for headword, pos in results:
					found_upos.update(map_pos_to_upos(pos))
				if token["upos"] not in found_upos and found_upos:
					all_flags.append({
						"task": "littre",
						"sent_id": sent_id,
						"token_id": token["id"],
						"form": token["form"],
						"lemma": token["lemma"],
						"upos": token["upos"],
						"issue": "upos_mismatch",
						"detail": (
							f"'{token['lemma']}' in Littré as "
							f"{', '.join(pos for _, pos in results)} "
							f"→ expected {found_upos}, got {token['upos']}"
						),
					})

	checker.close()
	return all_flags
