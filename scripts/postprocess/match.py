#!/usr/bin/env python3
"""
Match parallel texts by fuzzy proper noun overlap.

Used to generate initial title mappings before manual review.

Usage:
    python match.py data/maupassant/fr/conllu data/maupassant/en/conllu
    python match.py data/maupassant/fr/conllu data/maupassant/en/conllu --debug
"""
import argparse
from pathlib import Path

from rapidfuzz.distance import Levenshtein
from unidecode import unidecode


def parse_conllu(path):
	tokens = []
	with open(path, encoding="utf-8") as f:
		for line in f:
			line = line.rstrip("\n")
			if not line or line.startswith("#"):
				continue
			fields = line.split("\t")
			if len(fields) >= 4 and "-" not in fields[0] and "." not in fields[0]:
				tokens.append({"form": fields[1], "upos": fields[3]})
	return tokens


def extract_propns(tokens):
	return [unidecode(tok["form"].lower()) for tok in tokens if tok["upos"] == "PROPN"]


def fuzzy_multiset_overlap(names_a, names_b):
	if not names_a and not names_b:
		return 1.0
	if not names_a or not names_b:
		return 0.0
	b_available = list(names_b)
	matches = 0
	for a in names_a:
		best_idx = None
		best_dist = 2
		for i, b in enumerate(b_available):
			d = Levenshtein.distance(a, b)
			if d < best_dist:
				best_dist = d
				best_idx = i
		if best_idx is not None:
			matches += 1
			b_available.pop(best_idx)
	return matches / max(len(names_a), len(names_b))


def within_length_ratio(count_a, count_b, threshold=0.35):
	if count_a == 0 or count_b == 0:
		return False
	ratio = count_a / count_b
	return (1 - threshold) <= ratio <= (1 + threshold)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("source_dir")
	ap.add_argument("target_dir")
	ap.add_argument("--high-threshold", type=float, default=0.7)
	ap.add_argument("--low-threshold", type=float, default=0.25)
	ap.add_argument("--margin", type=float, default=2.0)
	ap.add_argument("--length-tolerance", type=float, default=0.35)
	ap.add_argument("--debug", action="store_true")
	args = ap.parse_args()

	source_docs = {}
	for p in Path(args.source_dir).glob("*.conllu"):
		tokens = parse_conllu(p)
		source_docs[p.stem] = {
			"propns": extract_propns(tokens),
			"tokens": len(tokens),
		}

	target_docs = {}
	for p in Path(args.target_dir).glob("*.conllu"):
		tokens = parse_conllu(p)
		target_docs[p.stem] = {
			"propns": extract_propns(tokens),
			"tokens": len(tokens),
		}

	if not args.debug:
		print("source_file\ttarget_file\tscore\trunner_up\tsource_tokens\ttarget_tokens")

	matched = 0
	unmatched = 0

	for source_name, source_data in sorted(source_docs.items()):
		if args.debug:
			print(f"\n=== {source_name} ===")
			print(f"  tokens: {source_data['tokens']}")
			propn_preview = source_data["propns"][:20]
			ellipsis = "..." if len(source_data["propns"]) > 20 else ""
			print(f"  propns ({len(source_data['propns'])}): {propn_preview}{ellipsis}")

		candidates = []
		for target_name, target_data in target_docs.items():
			length_ok = within_length_ratio(
				source_data["tokens"], target_data["tokens"], args.length_tolerance
			)
			score = fuzzy_multiset_overlap(source_data["propns"], target_data["propns"])
			candidates.append({
				"name": target_name,
				"score": score,
				"tokens": target_data["tokens"],
				"propns": target_data["propns"],
				"length_ok": length_ok,
			})

		candidates.sort(key=lambda x: x["score"], reverse=True)
		valid = [c for c in candidates if c["length_ok"]]

		if args.debug:
			print("  top 5 candidates:")
			for c in candidates[:5]:
				length_flag = "" if c["length_ok"] else " [LENGTH REJECT]"
				ratio = source_data["tokens"] / c["tokens"] if c["tokens"] else 0
				print(f"    {c['score']:.3f} {c['name']} (tokens={c['tokens']}, ratio={ratio:.2f}){length_flag}")

		best = None
		runner_up_score = 0

		if len(valid) >= 2:
			runner_up_score = valid[1]["score"]
			if valid[0]["score"] >= args.high_threshold:
				best = valid[0]
			elif valid[0]["score"] >= args.low_threshold and valid[0]["score"] > args.margin * runner_up_score:
				best = valid[0]
		elif len(valid) == 1 and valid[0]["score"] >= args.low_threshold:
			best = valid[0]

		if not args.debug:
			if best:
				print(f"{source_name}\t{best['name']}\t{best['score']:.3f}\t{runner_up_score:.3f}\t{source_data['tokens']}\t{best['tokens']}")
				matched += 1
			else:
				print(f"{source_name}\t\t\t\t{source_data['tokens']}\t")
				unmatched += 1

	if args.debug:
		print(f"\n=== SUMMARY ===")
		print(f"Matched: {matched}")
		print(f"Unmatched: {unmatched}")


if __name__ == "__main__":
	main()
