#!/usr/bin/env python3
"""Extract words that Littré lists as both adj. and adv."""

import sqlite3
import sys

db_path = sys.argv[1] if len(sys.argv) > 1 else "../b4e000/data/littre.db"

conn = sqlite3.connect(db_path)

rows = conn.execute("""
	SELECT LOWER(headword) as hw, pos
	FROM entries
	WHERE LOWER(pos) LIKE '%adj.%'
	   OR LOWER(pos) LIKE '%adv.%'
""").fetchall()

by_word = {}
for hw, pos in rows:
	# Normalize: take first word of multi-word headwords
	# e.g. "SOUDAIN, AINE" → "soudain"
	word = hw.split(",")[0].split(" ")[0].strip().lower()
	if not word:
		continue
	pos_lower = pos.lower() if pos else ""
	if word not in by_word:
		by_word[word] = {"adj": False, "adv": False, "raw_pos": []}
	if "adj." in pos_lower:
		by_word[word]["adj"] = True
	if "adv." in pos_lower:
		by_word[word]["adv"] = True
	by_word[word]["raw_pos"].append(pos)

both = {w: info for w, info in by_word.items() if info["adj"] and info["adv"]}

print(f"# Words with both adj. and adv. in Littré: {len(both)}")
print(f"# (from {len(by_word)} total words with either tag)")
print()
for word in sorted(both):
	pos_list = ", ".join(set(both[word]["raw_pos"]))
	print(f"{word:20s} {pos_list}")

conn.close()
