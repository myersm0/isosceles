#!/usr/bin/env python3
"""
Download ELTeC corpora and extract clean text.

Usage:
    python eltec.py download fra
    python eltec.py download eng
    python eltec.py extract fra
    python eltec.py extract eng
"""

import json
import re
import sys
import time
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


base_dir = Path(__file__).parent.parent
data_dir = base_dir / "data"
headers = {"User-Agent": "CorpusIsosceles/1.0 (academic research)"}


def normalize_unicode(text):
	text = unicodedata.normalize("NFC", text)
	replacements = {
		"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
		"\u2014": "--", "\u2013": "-", "\u2026": "...",
		"\u00a0": " ", "\u200b": "", "\u00ad": "", "\ufeff": "",
	}
	for orig, repl in replacements.items():
		text = text.replace(orig, repl)
	return text


def download_eltec(lang):
	print(f"=== Downloading ELTeC-{lang} ===\n")

	xml_dir = data_dir / f"eltec_{lang}" / "xml"
	xml_dir.mkdir(parents=True, exist_ok=True)

	api_url = f"https://api.github.com/repos/COST-ELTeC/ELTeC-{lang}/contents/level1"
	print("Fetching file list from GitHub API...")

	try:
		req = urllib.request.Request(api_url, headers={**headers, "Accept": "application/vnd.github.v3+json"})
		with urllib.request.urlopen(req, timeout=30) as response:
			items = json.loads(response.read().decode("utf-8"))
		files = [item["name"] for item in items if item["name"].endswith(".xml")]
		print(f"Found {len(files)} XML files\n")
	except Exception as e:
		print(f"API failed: {e}")
		print(f"Download manually from: https://github.com/COST-ELTeC/ELTeC-{lang}")
		return

	for i, filename in enumerate(files, 1):
		dest = xml_dir / filename
		if dest.exists():
			print(f"[{i:3}/{len(files)}] {filename} (exists)")
			continue

		url = f"https://raw.githubusercontent.com/COST-ELTeC/ELTeC-{lang}/master/level1/{filename}"
		print(f"[{i:3}/{len(files)}] {filename}")

		try:
			req = urllib.request.Request(url, headers=headers)
			with urllib.request.urlopen(req, timeout=30) as response:
				dest.write_bytes(response.read())
			time.sleep(0.3)
		except Exception as e:
			print(f"         FAILED: {e}")

	print(f"\nDone. XMLs in: {xml_dir}")


def extract_text_from_tei(xml_path):
	ns = {"tei": "http://www.tei-c.org/ns/1.0"}

	tree = ET.parse(xml_path)
	root = tree.getroot()

	author_elem = root.find(".//tei:author", ns)
	title_elem = root.find(".//tei:title[@type='main']", ns)
	if title_elem is None:
		title_elem = root.find(".//tei:title", ns)

	author = author_elem.text.strip() if author_elem is not None and author_elem.text else "Unknown"
	title = title_elem.text.strip() if title_elem is not None and title_elem.text else xml_path.stem

	body = root.find(".//tei:body", ns)
	if body is None:
		body = root.find(".//body")
	if body is None:
		return None, None, None

	paragraphs = []
	for elem in body.iter():
		tag_local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
		if tag_local in ("p", "head"):
			text_parts = []
			if elem.text:
				text_parts.append(elem.text)
			for child in elem:
				if child.text:
					text_parts.append(child.text)
				if child.tail:
					text_parts.append(child.tail)
			para_text = "".join(text_parts)
			para_text = re.sub(r"\s+", " ", para_text).strip()
			if para_text:
				paragraphs.append(para_text)

	full_text = "\n\n".join(paragraphs)
	full_text = normalize_unicode(full_text)
	return author, title, full_text


def extract_eltec(lang):
	print(f"=== Extracting ELTeC-{lang} ===\n")

	xml_dir = data_dir / f"eltec_{lang}" / "xml"
	txt_dir = data_dir / f"eltec_{lang}" / "txt"
	txt_dir.mkdir(parents=True, exist_ok=True)

	if not xml_dir.exists():
		print(f"No XML directory found. Run: python eltec.py download {lang}")
		return

	xml_files = sorted(xml_dir.glob("*.xml"))
	print(f"Found {len(xml_files)} XML files\n")

	manifest = []

	for i, xml_path in enumerate(xml_files, 1):
		try:
			author, title, text = extract_text_from_tei(xml_path)
			if text is None:
				print(f"[{i:3}/{len(xml_files)}] {xml_path.name} - NO TEXT FOUND")
				continue

			txt_path = txt_dir / f"{xml_path.stem}.txt"
			txt_path.write_text(text, encoding="utf-8")

			words = len(text.split())
			print(f"[{i:3}/{len(xml_files)}] {author[:20]:<20} {title[:35]:<35} {words:>7} words")

			manifest.append({"file": xml_path.stem, "author": author, "title": title, "words": words})
		except Exception as e:
			print(f"[{i:3}/{len(xml_files)}] {xml_path.name} - ERROR: {e}")

	manifest_path = txt_dir / "_manifest.tsv"
	with open(manifest_path, "w", encoding="utf-8") as f:
		f.write("file\tauthor\ttitle\twords\n")
		for item in manifest:
			f.write(f"{item['file']}\t{item['author']}\t{item['title']}\t{item['words']}\n")

	total_words = sum(item["words"] for item in manifest)
	print(f"\nExtracted {len(manifest)} texts, {total_words:,} words total")
	print(f"Text files in: {txt_dir}")
	print(f"Manifest: {manifest_path}")


def main():
	if len(sys.argv) < 3:
		print(__doc__)
		return

	cmd = sys.argv[1]
	lang = sys.argv[2]

	if lang not in ("fra", "eng"):
		print(f"Unknown language: {lang}. Use 'fra' or 'eng'.")
		return

	if cmd == "download":
		download_eltec(lang)
	elif cmd == "extract":
		extract_eltec(lang)
	else:
		print(f"Unknown command: {cmd}")


if __name__ == "__main__":
	main()
