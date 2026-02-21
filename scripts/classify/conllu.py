"""CoNLL-U reading and parsing utilities."""


def read_conllu(path):
	"""Read a CoNLL-U file into a list of sentence blocks (lists of lines)."""
	blocks = []
	current = []
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


def parse_block(lines):
	"""Parse a CoNLL-U block into sent_id, sent_text, and token list."""
	sent_id = None
	sent_text = None
	tokens = []
	for line in lines:
		if line.startswith("# sent_id"):
			sent_id = line.split("=", 1)[1].strip()
		elif line.startswith("# text"):
			sent_text = line.split("=", 1)[1].strip()
		elif not line.startswith("#"):
			fields = line.split("\t")
			if len(fields) >= 10 and "-" not in fields[0] and "." not in fields[0]:
				tokens.append({
					"id": int(fields[0]),
					"form": fields[1],
					"lemma": fields[2],
					"upos": fields[3],
					"feats": fields[5] if fields[5] != "_" else "_",
					"deprel": fields[7],
				})
	return sent_id, sent_text, tokens


def parse_feats(feats_string):
	"""Parse a UD feature string into a dict."""
	if feats_string == "_":
		return {}
	result = {}
	for pair in feats_string.split("|"):
		parts = pair.split("=", 1)
		if len(parts) == 2:
			result[parts[0]] = parts[1]
	return result
