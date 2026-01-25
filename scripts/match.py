#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from unidecode import unidecode
from rapidfuzz.distance import Levenshtein

def extract_propns(doc):
	propns = []
	for sent in doc["sentences"]:
		for tok in sent["tokens"]:
			if tok["upos"] == "PROPN":
				propns.append(unidecode(tok["form"].lower()))
	return propns

def token_count(doc):
	return sum(len(sent["tokens"]) for sent in doc["sentences"])

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
	ap.add_argument("fr_dir")
	ap.add_argument("en_dir")
	ap.add_argument("--high-threshold", type=float, default=0.7)
	ap.add_argument("--low-threshold", type=float, default=0.25)
	ap.add_argument("--margin", type=float, default=2.0)
	ap.add_argument("--length-tolerance", type=float, default=0.35)
	ap.add_argument("--debug", action="store_true")
	args = ap.parse_args()

	fr_docs = {}
	for p in Path(args.fr_dir).glob("*.json"):
		doc = json.loads(p.read_text(encoding="utf-8"))
		fr_docs[p.stem] = {
			"propns": extract_propns(doc),
			"tokens": token_count(doc),
		}

	en_docs = {}
	for p in Path(args.en_dir).glob("*.json"):
		doc = json.loads(p.read_text(encoding="utf-8"))
		en_docs[p.stem] = {
			"propns": extract_propns(doc),
			"tokens": token_count(doc),
		}

	if not args.debug:
		print("fr_file\ten_file\tscore\trunner_up\tfr_tokens\ten_tokens")

	matched = 0
	unmatched = 0

	for fr_name, fr_data in sorted(fr_docs.items()):
		if args.debug:
			print(f"\n=== {fr_name} ===")
			print(f"  tokens: {fr_data['tokens']}")
			print(f"  propns ({len(fr_data['propns'])}): {fr_data['propns'][:20]}{'...' if len(fr_data['propns']) > 20 else ''}")

		candidates = []
		for en_name, en_data in en_docs.items():
			length_ok = within_length_ratio(fr_data["tokens"], en_data["tokens"], args.length_tolerance)
			score = fuzzy_multiset_overlap(fr_data["propns"], en_data["propns"])
			candidates.append({
				"name": en_name,
				"score": score,
				"tokens": en_data["tokens"],
				"propns": en_data["propns"],
				"length_ok": length_ok,
			})

		candidates.sort(key=lambda x: x["score"], reverse=True)
		valid = [c for c in candidates if c["length_ok"]]

		if args.debug:
			print(f"  top 5 candidates:")
			for c in candidates[:5]:
				length_flag = "" if c["length_ok"] else " [LENGTH REJECT]"
				ratio = fr_data["tokens"] / c["tokens"] if c["tokens"] else 0
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
				print(f"{fr_name}\t{best['name']}\t{best['score']:.3f}\t{runner_up_score:.3f}\t{fr_data['tokens']}\t{best['tokens']}")
				matched += 1
			else:
				print(f"{fr_name}\t\t\t\t{fr_data['tokens']}\t")
				unmatched += 1

	if args.debug:
		print(f"\n=== SUMMARY ===")
		print(f"Matched: {matched}")
		print(f"Unmatched: {unmatched}")

if __name__ == "__main__":
	main()
