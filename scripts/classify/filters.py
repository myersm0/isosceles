"""Pre-filter and formatting functions for each classifier task.

Each task needs:
  filter_fn(tokens) -> list of candidate tokens
  format_fn(candidates, all_tokens) -> string for the user prompt
"""

from .conllu import parse_feats


# --- lemma ---

def filter_lemma(tokens):
	return [t for t in tokens if t["upos"] in ("VERB", "AUX")]


def format_lemma(candidates, all_tokens):
	lines = []
	for t in candidates:
		lines.append(
			f"ID={t['id']} FORM=\"{t['form']}\" LEMMA=\"{t['lemma']}\" UPOS={t['upos']}"
		)
	return "\n".join(lines)


# --- tense ---

def filter_tense(tokens):
	finite = []
	for t in tokens:
		if t["upos"] not in ("VERB", "AUX"):
			continue
		feats = parse_feats(t["feats"])
		if feats.get("VerbForm") == "Fin":
			finite.append(t)
	return finite


def format_tense(candidates, all_tokens):
	lines = []
	for t in candidates:
		feats = parse_feats(t["feats"])
		mood = feats.get("Mood", "?")
		tense = feats.get("Tense", "?")
		person = feats.get("Person", "?")
		number = feats.get("Number", "?")
		lines.append(
			f"ID={t['id']} FORM=\"{t['form']}\" LEMMA=\"{t['lemma']}\" "
			f"Mood={mood} Tense={tense} Person={person} Number={number}"
		)
	return "\n".join(lines)


# --- aux ---

def filter_aux(tokens):
	return [t for t in tokens if t["lemma"] in ("avoir", "être")]


def format_aux(candidates, all_tokens):
	lines = []
	for t in candidates:
		next_token = None
		for other in all_tokens:
			if other["id"] > t["id"] and other["upos"] not in ("PUNCT", "ADV"):
				next_token = other
				break
		context = ""
		if next_token:
			context = f" NEXT=\"{next_token['form']}\" NEXT_UPOS={next_token['upos']}"
		lines.append(
			f"ID={t['id']} FORM=\"{t['form']}\" LEMMA=\"{t['lemma']}\" "
			f"UPOS={t['upos']} DEPREL={t['deprel']}{context}"
		)
	return "\n".join(lines)


# --- que ---

def filter_que(tokens):
	result = []
	for t in tokens:
		lower = t["form"].lower().rstrip("'\u2019")
		if t["lemma"] == "que" or lower in ("que", "qu"):
			result.append(t)
	return result


def format_que(candidates, all_tokens):
	lines = []
	for t in candidates:
		lines.append(
			f"ID={t['id']} FORM=\"{t['form']}\" UPOS={t['upos']} "
			f"FEATS={t['feats']} DEPREL={t['deprel']}"
		)
	return "\n".join(lines)


task_filters = {
	"lemma": (filter_lemma, format_lemma),
	"tense": (filter_tense, format_tense),
	"aux": (filter_aux, format_aux),
	"que": (filter_que, format_que),
}
