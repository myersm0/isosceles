#!/usr/bin/env python3
import argparse
import sys


def parse_conllu(path):
	sentences = []
	current = []
	with open(path, encoding="utf-8") as f:
		for line in f:
			line = line.rstrip("\n")
			if line == "":
				if current:
					sentences.append(current)
					current = []
			else:
				current.append(line)
	if current:
		sentences.append(current)
	return sentences


def sentence_text(lines):
	for line in lines:
		if line.startswith("# text = "):
			return line[9:]
	tokens = [line.split("\t")[1] for line in lines if not line.startswith("#")]
	return " ".join(tokens)


def token_lines(lines):
	return [line for line in lines if not line.startswith("#")]


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("file_a")
	parser.add_argument("file_b")
	args = parser.parse_args()

	sentences_a = parse_conllu(args.file_a)
	sentences_b = parse_conllu(args.file_b)

	if len(sentences_a) != len(sentences_b):
		print(
			f"Warning: different number of sentences ({len(sentences_a)} vs {len(sentences_b)})",
			file=sys.stderr,
		)

	for i in range(min(len(sentences_a), len(sentences_b))):
		tokens_a = token_lines(sentences_a[i])
		tokens_b = token_lines(sentences_b[i])
		diffs = []

		for j in range(min(len(tokens_a), len(tokens_b))):
			if tokens_a[j] != tokens_b[j]:
				diffs.append(f"< {tokens_a[j]}\n> {tokens_b[j]}")
				fields_a = tokens_a[j].split("\t")
				fields_b = tokens_b[j].split("\t")
				if len(fields_a) >= 4 and len(fields_b) >= 4:
					upos_a, upos_b = fields_a[3], fields_b[3]
					lemma_a, lemma_b = fields_a[2], fields_b[2]
					if lemma_a != lemma_b and (upos_a == "PRON" or upos_b == "PRON"):
						print(
							f"PRON lemma change in sentence {i + 1} token {fields_a[0]}: "
							f"{fields_a[1]} lemma {lemma_a}→{lemma_b} (upos: {upos_a}/{upos_b})",
							file=sys.stderr,
						)

		for j in range(len(tokens_a), len(tokens_b)):
			diffs.append(f"> {tokens_b[j]}")
		for j in range(len(tokens_b), len(tokens_a)):
			diffs.append(f"< {tokens_a[j]}")

		if diffs:
			print(f"=== Sentence {i + 1}: {sentence_text(sentences_a[i])}")
			for diff in diffs:
				print(diff)
			print()


if __name__ == "__main__":
	main()
