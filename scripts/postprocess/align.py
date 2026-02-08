#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer


def load_sentences_conllu(path):
	sentences = []
	current_text = None
	current_forms = []
	with open(path, encoding="utf-8") as f:
		for line in f:
			line = line.rstrip("\n")
			if line.startswith("# text = "):
				current_text = line[9:]
			elif line == "":
				if current_text is not None:
					sentences.append(current_text)
				elif current_forms:
					sentences.append(" ".join(current_forms))
				current_text = None
				current_forms = []
			elif not line.startswith("#"):
				fields = line.split("\t")
				if len(fields) >= 2 and "-" not in fields[0] and "." not in fields[0]:
					current_forms.append(fields[1])
	if current_text is not None:
		sentences.append(current_text)
	elif current_forms:
		sentences.append(" ".join(current_forms))
	return sentences


def load_sentences_json(path):
	doc = json.loads(Path(path).read_text(encoding="utf-8"))
	return [
		sent.get("text", " ".join(t["form"] for t in sent["tokens"]))
		for sent in doc["sentences"]
	]


def load_sentences(path):
	path = Path(path)
	if path.suffix == ".conllu":
		return load_sentences_conllu(path)
	elif path.suffix == ".json":
		return load_sentences_json(path)
	else:
		raise ValueError(f"Unknown file format: {path.suffix} (expected .conllu or .json)")


def dp_align(scores, gap_penalty=0.5, merge_penalty=0.1):
	n, m = scores.shape
	dp = np.full((n + 1, m + 1), np.inf)
	back = np.zeros((n + 1, m + 1, 2), dtype=int)
	dp[0, 0] = 0

	for i in range(n + 1):
		for j in range(m + 1):
			if i == 0 and j == 0:
				continue
			candidates = []
			if i >= 1 and j >= 1:
				candidates.append((dp[i-1, j-1] + scores[i-1, j-1], i-1, j-1))
			if i >= 2 and j >= 1:
				merged = (scores[i-2, j-1] + scores[i-1, j-1]) / 2
				candidates.append((dp[i-2, j-1] + merged + merge_penalty, i-2, j-1))
			if i >= 1 and j >= 2:
				merged = (scores[i-1, j-2] + scores[i-1, j-1]) / 2
				candidates.append((dp[i-1, j-2] + merged + merge_penalty, i-1, j-2))
			if i >= 3 and j >= 1:
				merged = (scores[i-3, j-1] + scores[i-2, j-1] + scores[i-1, j-1]) / 3
				candidates.append((dp[i-3, j-1] + merged + merge_penalty * 2, i-3, j-1))
			if i >= 1 and j >= 3:
				merged = (scores[i-1, j-3] + scores[i-1, j-2] + scores[i-1, j-1]) / 3
				candidates.append((dp[i-1, j-3] + merged + merge_penalty * 2, i-1, j-3))
			if i >= 1:
				candidates.append((dp[i-1, j] + gap_penalty, i-1, j))
			if j >= 1:
				candidates.append((dp[i, j-1] + gap_penalty, i, j-1))

			if candidates:
				best = min(candidates, key=lambda x: x[0])
				dp[i, j] = best[0]
				back[i, j] = [best[1], best[2]]

	alignment = []
	i, j = n, m
	while i > 0 or j > 0:
		pi, pj = back[i, j]
		fr_idx = list(range(pi, i)) if i > pi else []
		en_idx = list(range(pj, j)) if j > pj else []
		if fr_idx or en_idx:
			if fr_idx and en_idx:
				score = float(np.mean([scores[fi, ej] for fi in fr_idx for ej in en_idx]))
			else:
				score = 0.0
			alignment.append((fr_idx, en_idx, score))
		i, j = int(pi), int(pj)

	return list(reversed(alignment))


def alignment_type(fr_idx, en_idx):
	nf, ne = len(fr_idx), len(en_idx)
	if nf == 0:
		return "en_gap"
	if ne == 0:
		return "fr_gap"
	if nf == 1 and ne == 1:
		return "1:1"
	if nf == 2 and ne == 1:
		return "2:1"
	if nf == 1 and ne == 2:
		return "1:2"
	if nf == 3 and ne == 1:
		return "3:1"
	if nf == 1 and ne == 3:
		return "1:3"
	return f"{nf}:{ne}"


def align_pair(fr_path, en_path, model):
	fr_sents = load_sentences(fr_path)
	en_sents = load_sentences(en_path)

	fr_emb = model.encode(fr_sents, convert_to_numpy=True, show_progress_bar=False)
	en_emb = model.encode(en_sents, convert_to_numpy=True, show_progress_bar=False)
	scores = 1 - (fr_emb @ en_emb.T)

	alignment = dp_align(scores)

	return alignment, len(fr_sents), len(en_sents)


def make_document(fr_stem, en_stem, corpus, alignment):
	return {
		"_id": f"{corpus}_{fr_stem}",
		"corpus": corpus,
		"fr_doc": f"{corpus}_fr_{fr_stem}",
		"en_doc": f"{corpus}_en_{en_stem}",
		"pairs": [
			{
				"fr": fr_idx,
				"en": en_idx,
				"score": round(1 - score, 3),
				"type": alignment_type(fr_idx, en_idx),
			}
			for fr_idx, en_idx, score in alignment
		],
	}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--fr-dir", required=True)
	ap.add_argument("--en-dir", required=True)
	ap.add_argument("--index", required=True, help="TSV with fr_file, en_file columns")
	ap.add_argument("--output-dir", required=True)
	ap.add_argument("--corpus", default="maupassant")
	args = ap.parse_args()

	fr_dir = Path(args.fr_dir)
	en_dir = Path(args.en_dir)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	print("Loading LaBSE...")
	model = SentenceTransformer("sentence-transformers/LaBSE")

	with open(args.index, newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f, delimiter="\t")
		rows = list(reader)

	for row in rows:
		fr_file = row.get("fr_file", "").strip()
		en_file = row.get("en_file", "").strip()
		if not fr_file or not en_file:
			continue

		fr_path = fr_dir / f"{fr_file}.conllu"
		en_path = en_dir / f"{en_file}.conllu"

		if not fr_path.exists():
			print(f"Missing: {fr_path}")
			continue
		if not en_path.exists():
			print(f"Missing: {en_path}")
			continue

		print(f"Aligning: {fr_file} <-> {en_file}")
		alignment, n_fr, n_en = align_pair(fr_path, en_path, model)
		print(f"  {n_fr} FR, {n_en} EN -> {len(alignment)} pairs")

		doc = make_document(fr_file, en_file, args.corpus, alignment)
		out_path = output_dir / f"{fr_file}_alignment.json"
		out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
	main()
