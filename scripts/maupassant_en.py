#!/usr/bin/env python3
"""
Process McMaster et al. English translations of Maupassant from Project Gutenberg.

Usage:
    python maupassant_en.py download
    python maupassant_en.py toc
    python maupassant_en.py list
    python maupassant_en.py segment
"""

import re
import sys
import time
import urllib.request
from pathlib import Path


base_dir = Path(__file__).parent.parent
data_dir = base_dir / "data" / "maupassant_en"
raw_dir = data_dir / "raw"
txt_dir = data_dir / "txt"
headers = {"User-Agent": "CorpusIsosceles/1.0 (academic research)"}

gutenberg_volumes = [
	(3077, "Vol 01"), (3078, "Vol 02"), (3079, "Vol 03"), (3080, "Vol 04"),
	(3081, "Vol 05"), (3082, "Vol 06"), (3083, "Vol 07"), (3084, "Vol 08"),
	(3085, "Vol 09"), (3086, "Vol 10"), (3087, "Vol 11"), (3088, "Vol 12"),
	(3089, "Vol 13"),
]

title_variants = {
	"THE PARRICIDE": "A PARRICIDE",
	"THE TRIP OF THE HORLA": "THE TRIP OF LE HORLA",
	"FAREWELL": "FAREWELL!",
	"JULIE ROMAINE": "JULIE ROMAIN",
	"MADAME HUSSON'S ROSIER": "MADAME HUSSON'S “ROSIER”",
	"A COWARD": "COWARD",
	"WALTER SCHNAFF'S ADVENTURE": "WALTER SCHNAFFS' ADVENTURE",
	"THE DIARY OF A MAD MAN": "THE DIARY OF A MADMAN",
	"THE PENGUINS ROCK": "THE PENGUINS' ROCK",
	"A DEAD WOMAN'S SECRET": "DEAD WOMAN'S SECRET",
	"A FATHERS CONFESSION": "A FATHER'S CONFESSION",
	"THE ENGLISHMEN OF ETRETAT": "THE ENGLISHMAN OF ETRETAT",
	"A PIECE OF STRING": "THE PIECE OF STRING",
}


def safe_print(text):
	try:
		print(text)
	except UnicodeEncodeError:
		print(text.encode("ascii", "replace").decode("ascii"))


def download_gutenberg(ebook_id, output_path):
	url = f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"
	try:
		req = urllib.request.Request(url, headers=headers)
		with urllib.request.urlopen(req, timeout=30) as response:
			text = response.read().decode("utf-8-sig", errors="replace")
			output_path.write_text(text, encoding="utf-8")
			return True
	except Exception as e:
		print(f"  Error: {e}")
		return False


def cmd_download():
	raw_dir.mkdir(parents=True, exist_ok=True)
	print(f"\n=== Downloading McMaster translations from Project Gutenberg ===\n")

	for ebook_id, vol_name in gutenberg_volumes:
		outfile = raw_dir / f"pg{ebook_id}.txt"
		if outfile.exists():
			print(f"  pg{ebook_id}.txt ({vol_name}) - exists, skipping")
			continue
		print(f"  pg{ebook_id}.txt ({vol_name}) - downloading...")
		if download_gutenberg(ebook_id, outfile):
			print(f"    -> {outfile}")
		time.sleep(1)

	print(f"\nRaw files in: {raw_dir}")


def strip_gutenberg_boilerplate(text):
	start_markers = [
		"*** START OF THE PROJECT GUTENBERG EBOOK",
		"*** START OF THIS PROJECT GUTENBERG EBOOK",
		"*END*THE SMALL PRINT",
	]
	end_markers = [
		"*** END OF THE PROJECT GUTENBERG EBOOK",
		"*** END OF THIS PROJECT GUTENBERG EBOOK",
		"End of the Project Gutenberg",
		"End of Project Gutenberg",
	]

	start_idx = 0
	for marker in start_markers:
		idx = text.find(marker)
		if idx != -1:
			newline_after = text.find("\n", idx)
			if newline_after != -1:
				start_idx = max(start_idx, newline_after + 1)

	end_idx = len(text)
	for marker in end_markers:
		idx = text.find(marker)
		if idx != -1:
			end_idx = min(end_idx, idx)

	return text[start_idx:end_idx].strip()


def normalize_text(text):
	text = text.replace("\r\n", "\n")
	paragraphs = re.split(r"\n\s*\n", text)
	normalized = []
	for para in paragraphs:
		para = para.strip()
		if not para:
			continue
		para = re.sub(r"\n", " ", para)
		para = re.sub(r"  +", " ", para)
		normalized.append(para)
	return "\n\n".join(normalized)


def normalize_filename(title):
	name = title.lower()
	name = re.sub(r"[^a-z0-9]+", "_", name)
	name = re.sub(r"_+", "_", name).strip("_")
	return name[:60]


def parse_toc(text):
	lines = text.split("\n")

	vol_idx = None
	volume_num = None
	for i, line in enumerate(lines):
		match = re.match(r"^VOLUME\s+([IVXLC]+)\.?\s*$", line.strip())
		if match:
			vol_idx = i
			volume_num = match.group(1)
			break

	if vol_idx is None:
		return None, []

	toc_titles = []
	for i in range(vol_idx + 1, min(vol_idx + 100, len(lines))):
		line = lines[i]
		stripped = line.strip()

		if not stripped:
			continue

		if line.startswith("     ") or line.startswith("\t"):
			if stripped.isupper() and 2 < len(stripped) < 80:
				if any(skip in stripped for skip in [
					"GUY DE MAUPASSANT", "A STUDY BY", "TRANSLATED BY",
					"CONTENTS", "INTRODUCTION", "PREFACE"
				]):
					continue
				toc_titles.append(stripped)
		else:
			if stripped.isupper() and len(stripped) > 2 and toc_titles:
				break

	return volume_num, toc_titles


def find_title_in_body(lines, toc_title, start_from=0):
	for i, line in enumerate(lines):
		if i < start_from:
			continue
		if line.startswith(" ") or line.startswith("\t"):
			continue
		stripped = line.strip()
		if stripped == toc_title:
			return i

	if toc_title in title_variants:
		variant = title_variants[toc_title]
		for i, line in enumerate(lines):
			if i < start_from:
				continue
			if line.startswith(" ") or line.startswith("\t"):
				continue
			stripped = line.strip()
			if stripped == variant:
				return i

	return None


def segment_volume(text, source_name):
	volume_num, toc_titles = parse_toc(text)

	if not toc_titles:
		safe_print(f"  Warning: No TOC found in {source_name}")
		return []

	stories = []
	lines = text.split("\n")

	first_title = toc_titles[0]
	first_story_line = find_title_in_body(lines, first_title, start_from=0)

	if first_story_line is None:
		safe_print(f"  Warning: Could not find first story '{first_title}'")
		return []

	title_positions = []
	for toc_title in toc_titles:
		line_num = find_title_in_body(lines, toc_title, start_from=first_story_line)
		if line_num is not None:
			title_positions.append((line_num, toc_title))
		else:
			safe_print(f"  Warning: TOC title not found in body: {toc_title}")

	title_positions.sort(key=lambda x: x[0])

	for idx, (start_line, title) in enumerate(title_positions):
		if idx + 1 < len(title_positions):
			end_line = title_positions[idx + 1][0]
		else:
			end_line = len(lines)

		story_lines = lines[start_line + 1:end_line]
		story_text = "\n".join(story_lines).strip()
		story_text = normalize_text(story_text)

		if len(story_text) > 200:
			stories.append({
				"title": title.title(),
				"title_raw": title,
				"volume": volume_num,
				"source": source_name,
				"text": story_text,
				"words": len(story_text.split()),
			})

	return stories


def cmd_list():
	if not raw_dir.exists():
		print(f"No raw directory found: {raw_dir}")
		print("Run: python maupassant_en.py download")
		return

	raw_files = sorted(raw_dir.glob("*.txt"))
	if not raw_files:
		print(f"No .txt files in {raw_dir}")
		return

	print(f"\n=== McMaster Volumes ({len(raw_files)} files) ===\n")

	total_stories = 0
	total_words = 0

	for filepath in raw_files:
		text = filepath.read_text(encoding="utf-8", errors="replace")
		stripped = strip_gutenberg_boilerplate(text)
		volume_num, toc_titles = parse_toc(stripped)
		stories = segment_volume(stripped, filepath.stem)
		words = sum(s["words"] for s in stories)
		total_stories += len(stories)
		total_words += words
		found = len(stories)
		expected = len(toc_titles)
		status = "" if found == expected else f" (TOC has {expected})"
		safe_print(f"  {filepath.name:<20} Vol {volume_num or '?':<4} {found:>3} stories, {words:>7} words{status}")

	print(f"\n  Total: {total_stories} stories, {total_words:,} words")


def cmd_toc():
	if not raw_dir.exists():
		print(f"No raw directory: {raw_dir}")
		return

	for filepath in sorted(raw_dir.glob("*.txt")):
		text = filepath.read_text(encoding="utf-8", errors="replace")
		text = strip_gutenberg_boilerplate(text)
		volume_num, toc_titles = parse_toc(text)

		safe_print(f"\n=== {filepath.name} (Volume {volume_num}) - {len(toc_titles)} entries ===")
		for title in toc_titles:
			safe_print(f"  {title}")


def cmd_segment(filepath=None):
	txt_dir.mkdir(parents=True, exist_ok=True)

	if filepath:
		files = [Path(filepath)]
	else:
		if not raw_dir.exists():
			print(f"No raw directory: {raw_dir}")
			return
		files = sorted(raw_dir.glob("*.txt"))

	if not files:
		print("No files to process")
		return

	all_metadata = []
	seen_titles = {}

	for filepath in files:
		safe_print(f"\n=== {filepath.name} ===")
		text = filepath.read_text(encoding="utf-8", errors="replace")
		text = strip_gutenberg_boilerplate(text)
		stories = segment_volume(text, filepath.stem)

		for story in stories:
			title = story["title"]
			filename = normalize_filename(story["title_raw"])

			if filename in seen_titles:
				seen_titles[filename] += 1
				filename = f"{filename}_{seen_titles[filename]}"
			else:
				seen_titles[filename] = 1

			out_path = txt_dir / f"{filename}.txt"
			out_path.write_text(story["text"], encoding="utf-8")

			safe_print(f"  {title[:45]:<45} {story['words']:>6} words")

			all_metadata.append({
				"file": filename,
				"title": title,
				"volume": story.get("volume", ""),
				"source": story["source"],
				"words": story["words"],
			})

	if all_metadata:
		meta_path = data_dir / "metadata.tsv"
		with open(meta_path, "w", encoding="utf-8") as f:
			f.write("file\ttitle\tvolume\tsource\twords\n")
			for m in all_metadata:
				f.write(f"{m['file']}\t{m['title']}\t{m['volume']}\t{m['source']}\t{m['words']}\n")

		print(f"\n=== Summary ===")
		print(f"Stories: {len(all_metadata)}")
		print(f"Words: {sum(m['words'] for m in all_metadata):,}")
		print(f"Metadata: {meta_path}")
		print(f"Texts: {txt_dir}")


def main():
	if len(sys.argv) < 2:
		print(__doc__)
		return

	cmd = sys.argv[1]
	if cmd == "download":
		cmd_download()
	elif cmd == "toc":
		cmd_toc()
	elif cmd == "list":
		cmd_list()
	elif cmd == "segment":
		filepath = sys.argv[2] if len(sys.argv) > 2 else None
		cmd_segment(filepath)
	else:
		print(f"Unknown command: {cmd}")
		print(__doc__)


if __name__ == "__main__":
	main()
