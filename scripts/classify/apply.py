"""
Apply confirmed patches to a CoNLL-U file, producing corrected output.
"""

import sys
from pathlib import Path

from .conllu import read_conllu, parse_block


def apply_patches(conllu_path, patches, output_path):
	"""Apply patches to CoNLL-U and write corrected file.

	patches: list of {sent_id, token_id, fields: {lemma?, upos?, feats?}}
	"""
	patch_lookup = {}
	for patch in patches:
		key = (patch["sent_id"], patch["token_id"])
		patch_lookup[key] = patch["fields"]

	blocks = read_conllu(conllu_path)
	applied = 0

	with open(output_path, "w", encoding="utf-8") as out:
		for block in blocks:
			sent_id = None
			for line in block:
				if line.startswith("# sent_id"):
					sent_id = line.split("=", 1)[1].strip()
					break

			for line in block:
				if line.startswith("#") or line.strip() == "":
					out.write(line + "\n")
					continue

				fields = line.split("\t")
				if len(fields) < 10 or "-" in fields[0] or "." in fields[0]:
					out.write(line + "\n")
					continue

				token_id = int(fields[0])
				key = (sent_id, token_id)

				if key in patch_lookup:
					patch = patch_lookup[key]
					if "lemma" in patch:
						fields[2] = patch["lemma"]
					if "upos" in patch:
						fields[3] = patch["upos"]
					if "feats" in patch:
						fields[5] = patch["feats"]
					applied += 1

				out.write("\t".join(fields) + "\n")

			out.write("\n")

	print(f"Applied {applied} patches to {output_path}")
	print(f"  ({len(patches)} patches provided, {len(patches) - applied} not matched)")
