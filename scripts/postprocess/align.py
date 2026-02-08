#!/usr/bin/env python3
import argparse
import csv
import json
import sys
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


def compute_score_matrices(fr_sents, en_sents, model):
	fr_emb = model.encode(fr_sents, convert_to_numpy=True, show_progress_bar=False)
	en_emb = model.encode(en_sents, convert_to_numpy=True, show_progress_bar=False)
	scores_1_1 = 1 - (fr_emb @ en_emb.T)

	fr_bigrams = [f"{fr_sents[i]} {fr_sents[i+1]}" for i in range(len(fr_sents) - 1)]
	en_bigrams = [f"{en_sents[j]} {en_sents[j+1]}" for j in range(len(en_sents) - 1)]
	fr_trigrams = [
		f"{fr_sents[i]} {fr_sents[i+1]} {fr_sents[i+2]}"
		for i in range(len(fr_sents) - 2)
	]
	en_trigrams = [
		f"{en_sents[j]} {en_sents[j+1]} {en_sents[j+2]}"
		for j in range(len(en_sents) - 2)
	]

	scores_2_1 = None
	if fr_bigrams:
		fr_bi_emb = model.encode(fr_bigrams, convert_to_numpy=True, show_progress_bar=False)
		scores_2_1 = 1 - (fr_bi_emb @ en_emb.T)

	scores_1_2 = None
	if en_bigrams:
		en_bi_emb = model.encode(en_bigrams, convert_to_numpy=True, show_progress_bar=False)
		scores_1_2 = 1 - (fr_emb @ en_bi_emb.T)

	scores_3_1 = None
	if fr_trigrams:
		fr_tri_emb = model.encode(fr_trigrams, convert_to_numpy=True, show_progress_bar=False)
		scores_3_1 = 1 - (fr_tri_emb @ en_emb.T)

	scores_1_3 = None
	if en_trigrams:
		en_tri_emb = model.encode(en_trigrams, convert_to_numpy=True, show_progress_bar=False)
		scores_1_3 = 1 - (fr_emb @ en_tri_emb.T)

	return scores_1_1, scores_2_1, scores_1_2, scores_3_1, scores_1_3


def dp_align(score_matrices, gap_penalty=0.3, merge_penalty=0.2, band_fraction=0.1):
	scores_1_1, scores_2_1, scores_1_2, scores_3_1, scores_1_3 = score_matrices
	n, m = scores_1_1.shape
	dp = np.full((n + 1, m + 1), np.inf)
	back = np.zeros((n + 1, m + 1, 2), dtype=int)
	dp[0, 0] = 0

	for i in range(n + 1):
		for j in range(m + 1):
			if i == 0 and j == 0:
				continue
			if band_fraction is not None and n > 0 and m > 0:
				if abs(i / n - j / m) > band_fraction:
					continue
			candidates = []
			if i >= 1 and j >= 1:
				candidates.append((dp[i-1, j-1] + scores_1_1[i-1, j-1], i-1, j-1))
			if i >= 2 and j >= 1 and scores_2_1 is not None:
				candidates.append((dp[i-2, j-1] + scores_2_1[i-2, j-1] + merge_penalty, i-2, j-1))
			if i >= 1 and j >= 2 and scores_1_2 is not None:
				candidates.append((dp[i-1, j-2] + scores_1_2[i-1, j-2] + merge_penalty, i-1, j-2))
			if i >= 3 and j >= 1 and scores_3_1 is not None:
				candidates.append((dp[i-3, j-1] + scores_3_1[i-3, j-1] + merge_penalty * 3, i-3, j-1))
			if i >= 1 and j >= 3 and scores_1_3 is not None:
				candidates.append((dp[i-1, j-3] + scores_1_3[i-1, j-3] + merge_penalty * 3, i-1, j-3))
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
			nf, ne = len(fr_idx), len(en_idx)
			if nf >= 1 and ne >= 1:
				if nf == 1 and ne == 1:
					score = float(scores_1_1[fr_idx[0], en_idx[0]])
				elif nf == 2 and ne == 1:
					score = float(scores_2_1[fr_idx[0], en_idx[0]])
				elif nf == 1 and ne == 2:
					score = float(scores_1_2[fr_idx[0], en_idx[0]])
				elif nf == 3 and ne == 1:
					score = float(scores_3_1[fr_idx[0], en_idx[0]])
				elif nf == 1 and ne == 3:
					score = float(scores_1_3[fr_idx[0], en_idx[0]])
				else:
					score = float(scores_1_1[fr_idx[0], en_idx[0]])
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


def write_alignment_tsv(out_path, fr_filename, en_filename, alignment):
	with open(out_path, "w", encoding="utf-8") as f:
		for fr_idx, en_idx, score in alignment:
			if not fr_idx or not en_idx:
				continue
			for fi in fr_idx:
				for ej in en_idx:
					f.write(f"{fr_filename}\t{fi}\t{en_filename}\t{ej}\n")


def print_alignment(fr_sents, en_sents, alignment):
	for fr_idx, en_idx, score in alignment:
		atype = alignment_type(fr_idx, en_idx)
		sim = round(1 - score, 3) if fr_idx and en_idx else None
		fr_text = " ||| ".join(fr_sents[i] for i in fr_idx) if fr_idx else ""
		en_text = " ||| ".join(en_sents[j] for j in en_idx) if en_idx else ""
		header = f"[{atype}]"
		if sim is not None:
			header += f" sim={sim}"
		print(header)
		if fr_text:
			print(f"  FR: {fr_text}")
		if en_text:
			print(f"  EN: {en_text}")
		print()


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--fr-dir", required=True)
	ap.add_argument("--en-dir", required=True)
	ap.add_argument("--index", required=True, help="TSV with fr_file, en_file columns")
	ap.add_argument("--output-dir", required=True)
	ap.add_argument("--gap-penalty", type=float, default=0.3)
	ap.add_argument("--merge-penalty", type=float, default=0.2)
	ap.add_argument("--band", type=float, default=0.1,
		help="Band fraction around diagonal (0 to disable)")
	args = ap.parse_args()

	fr_dir = Path(args.fr_dir)
	en_dir = Path(args.en_dir)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	print("Loading LaBSE...", file=sys.stderr)
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
			print(f"Missing: {fr_path}", file=sys.stderr)
			continue
		if not en_path.exists():
			print(f"Missing: {en_path}", file=sys.stderr)
			continue

		fr_sents = load_sentences(fr_path)
		en_sents = load_sentences(en_path)
		print(
			f"Aligning: {fr_file} <-> {en_file} "
			f"({len(fr_sents)} FR, {len(en_sents)} EN)",
			file=sys.stderr,
		)

		score_matrices = compute_score_matrices(fr_sents, en_sents, model)
		band = args.band if args.band > 0 else None
		alignment = dp_align(score_matrices, args.gap_penalty, args.merge_penalty, band)

		print_alignment(fr_sents, en_sents, alignment)

		fr_filename = f"{fr_file}.conllu"
		en_filename = f"{en_file}.conllu"
		tsv_path = output_dir / f"{fr_file}_alignment.tsv"
		write_alignment_tsv(tsv_path, fr_filename, en_filename, alignment)
		print(f"  -> {tsv_path}", file=sys.stderr)


if __name__ == "__main__":
	main()
