#!/usr/bin/env python3
"""
UD parse validator and deterministic fixer.
"""

from typing import Optional


# Genuine lemma errors: (form_lower, wrong_lemma, upos) → correct_lemma
lemma_error_rules = {
	# "sous" (preposition) mislemmatized as "sou" (old coin)
	("sous", "sou", "ADP"): "sous",
	# "puis" (adverb "then") mislemmatized as "pouvoir"
	("puis", "pouvoir", "ADV"): "puis",
}


def parse_conllu_block(text: str) -> tuple[Optional[str], Optional[str], list[dict]]:
	"""Parse a CoNLL-U sentence block into (sent_id, sent_text, tokens)."""
	sent_id = None
	sent_text = None
	tokens = []
	
	for line in text.strip().split("\n"):
		if line.startswith("# sent_id = "):
			sent_id = line[12:]
		elif line.startswith("# text = "):
			sent_text = line[9:]
		elif line.startswith("#") or not line.strip():
			continue
		else:
			parts = line.split("\t")
			if len(parts) != 10:
				continue
			if "-" in parts[0]:
				continue
			try:
				tok_id = int(parts[0])
			except ValueError:
				continue
			tokens.append({
				"id": tok_id,
				"form": parts[1],
				"lemma": parts[2],
				"upos": parts[3],
				"xpos": parts[4],
				"feats": parts[5],
				"head": int(parts[6]) if parts[6] != "_" else 0,
				"deprel": parts[7],
				"deps": parts[8],
				"misc": parts[9],
			})
	
	return sent_id, sent_text, tokens


def tokens_to_conllu(tokens: list[dict], sent_id: str = None, sent_text: str = None) -> str:
	"""Convert token list to CoNLL-U format."""
	lines = []
	if sent_id:
		lines.append(f"# sent_id = {sent_id}")
	if sent_text:
		lines.append(f"# text = {sent_text}")
	for tok in tokens:
		row = "\t".join([
			str(tok["id"]),
			tok["form"],
			tok["lemma"],
			tok["upos"],
			tok.get("xpos", "_"),
			tok.get("feats", "_"),
			str(tok["head"]),
			tok["deprel"],
			tok.get("deps", "_"),
			tok.get("misc", "_"),
		])
		lines.append(row)
	return "\n".join(lines)


def detect_cycles(tokens: list[dict]) -> list[list[int]]:
	"""Detect cycles in dependency tree. Returns list of cycles (each a list of token IDs)."""
	cycles = []
	tok_by_id = {t["id"]: t for t in tokens}
	seen_cycles = set()
	
	for start_tok in tokens:
		visited = set()
		current = start_tok["id"]
		path = []
		
		while current != 0:
			if current in visited:
				cycle_start = path.index(current)
				cycle = path[cycle_start:]
				min_idx = cycle.index(min(cycle))
				cycle = tuple(cycle[min_idx:] + cycle[:min_idx])
				if cycle not in seen_cycles:
					seen_cycles.add(cycle)
					cycles.append(list(cycle))
				break
			
			if current not in tok_by_id:
				break
			
			visited.add(current)
			path.append(current)
			current = tok_by_id[current]["head"]
	
	return cycles


def detect_multiple_roots(tokens: list[dict]) -> list[int]:
	"""Detect multiple root nodes (HEAD=0)."""
	roots = [t["id"] for t in tokens if t["head"] == 0]
	return roots if len(roots) > 1 else []


def detect_self_loops(tokens: list[dict]) -> list[int]:
	"""Detect tokens pointing to themselves."""
	return [t["id"] for t in tokens if t["head"] == t["id"]]


def detect_invalid_heads(tokens: list[dict]) -> list[dict]:
	"""Detect tokens pointing to non-existent HEADs."""
	valid_ids = {t["id"] for t in tokens}
	valid_ids.add(0)
	return [{"id": t["id"], "head": t["head"]} for t in tokens if t["head"] not in valid_ids]


def validate_tree(tokens: list[dict]) -> dict:
	"""
	Validate dependency tree structure.
	
	Returns: {
		"valid": bool,
		"cycles": [...],
		"multiple_roots": [...],
		"self_loops": [...],
		"invalid_heads": [...]
	}
	"""
	issues = {
		"cycles": detect_cycles(tokens),
		"multiple_roots": detect_multiple_roots(tokens),
		"self_loops": detect_self_loops(tokens),
		"invalid_heads": detect_invalid_heads(tokens),
	}
	
	valid = not any([
		issues["cycles"],
		issues["multiple_roots"],
		issues["self_loops"],
		issues["invalid_heads"],
	])
	
	return {"valid": valid, **issues}


def apply_deterministic_fixes(tokens: list[dict]) -> tuple[list[dict], list[str]]:
	"""
	Apply deterministic lemma fixes for genuine errors only.
	Returns: (tokens, list of changes made)
	"""
	changes = []
	
	for tok in tokens:
		form_lower = tok["form"].lower()
		lemma_lower = tok["lemma"].lower()
		upos = tok["upos"]
		key = (form_lower, lemma_lower, upos)
		
		if key in lemma_error_rules:
			old_lemma = tok["lemma"]
			new_lemma = lemma_error_rules[key]
			tok["lemma"] = new_lemma
			changes.append(f"Token {tok['id']}: lemma {old_lemma} → {new_lemma}")
	
	return tokens, changes
