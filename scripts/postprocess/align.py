#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

def load_sentences(json_path):
	doc = json.loads(Path(json_path).read_text(encoding="utf-8"))
	return [sent.get("text", " ".join(t["form"] for t in sent["tokens"])) 
			for sent in doc["sentences"]]

def gale_church_cost(len_s, len_t, mean_ratio=1.1, var=6.8):
	if len_t == 0:
		return float("inf") if len_s > 0 else 0
	delta = (len_s - len_t * mean_ratio) / math.sqrt(len_t * var)
	return abs(delta)

def dp_align(scores, gap_penalty=0.5):
	n, m = scores.shape
	dp = np.full((n + 1, m + 1), float("inf"))
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
				candidates.append((dp[i-2, j-1] + merged + 0.1, i-2, j-1))
			if i >= 1 and j >= 2:
				merged = (scores[i-1, j-2] + scores[i-1, j-1]) / 2
				candidates.append((dp[i-1, j-2] + merged + 0.1, i-1, j-2))
			if i >= 1 and j == 0:
				candidates.append((dp[i-1, j] + gap_penalty, i-1, j))
			elif i >= 1:
				candidates.append((dp[i-1, j] + gap_penalty, i-1, j))
			if j >= 1 and i == 0:
				candidates.append((dp[i, j-1] + gap_penalty, i, j-1))
			elif j >= 1:
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
			alignment.append((fr_idx, en_idx))
		i, j = pi, pj
	
	return list(reversed(alignment))

def gale_church_matrix(fr_sents, en_sents):
	n, m = len(fr_sents), len(en_sents)
	fr_lens = [len(s) for s in fr_sents]
	en_lens = [len(s) for s in en_sents]
	scores = np.zeros((n, m))
	for i in range(n):
		for j in range(m):
			scores[i, j] = gale_church_cost(fr_lens[i], en_lens[j])
	return scores

def labse_matrix(fr_sents, en_sents, model):
	fr_emb = model.encode(fr_sents, convert_to_numpy=True)
	en_emb = model.encode(en_sents, convert_to_numpy=True)
	similarity = fr_emb @ en_emb.T
	return 1 - similarity

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("fr_json")
	ap.add_argument("en_json")
	ap.add_argument("--method", choices=["gc", "labse", "both"], default="both")
	ap.add_argument("--output", "-o")
	args = ap.parse_args()

	fr_sents = load_sentences(args.fr_json)
	en_sents = load_sentences(args.en_json)
	
	print(f"FR: {len(fr_sents)} sentences, EN: {len(en_sents)} sentences", file=__import__("sys").stderr)

	if args.method in ("gc", "both"):
		gc_scores = gale_church_matrix(fr_sents, en_sents)
		gc_align = dp_align(gc_scores)
	
	if args.method in ("labse", "both"):
		print("Loading LaBSE...", file=__import__("sys").stderr)
		model = SentenceTransformer("sentence-transformers/LaBSE")
		labse_scores = labse_matrix(fr_sents, en_sents, model)
		labse_align = dp_align(labse_scores)

	out = open(args.output, "w", encoding="utf-8") if args.output else __import__("sys").stdout
	
	if args.method == "both":
		print("method\tfr_idx\ten_idx\tfr_text\ten_text", file=out)
		for fr_idx, en_idx in gc_align:
			fr_text = " ||| ".join(fr_sents[i] for i in fr_idx) if fr_idx else ""
			en_text = " ||| ".join(en_sents[j] for j in en_idx) if en_idx else ""
			print(f"gc\t{fr_idx}\t{en_idx}\t{fr_text[:100]}\t{en_text[:100]}", file=out)
		for fr_idx, en_idx in labse_align:
			fr_text = " ||| ".join(fr_sents[i] for i in fr_idx) if fr_idx else ""
			en_text = " ||| ".join(en_sents[j] for j in en_idx) if en_idx else ""
			print(f"labse\t{fr_idx}\t{en_idx}\t{fr_text[:100]}\t{en_text[:100]}", file=out)
	else:
		align = gc_align if args.method == "gc" else labse_align
		print("fr_idx\ten_idx\tfr_text\ten_text", file=out)
		for fr_idx, en_idx in align:
			fr_text = " ||| ".join(fr_sents[i] for i in fr_idx) if fr_idx else ""
			en_text = " ||| ".join(en_sents[j] for j in en_idx) if en_idx else ""
			print(f"{fr_idx}\t{en_idx}\t{fr_text[:100]}\t{en_text[:100]}", file=out)

if __name__ == "__main__":
	main()