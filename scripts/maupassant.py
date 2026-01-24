#!/usr/bin/env python3
"""
Download Maupassant texts from maupassant.free.fr (based on the Pléiade edition).

Usage:
    python maupassant.py list
    python maupassant.py download
"""

import html
import re
import sys
import time
import unicodedata
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


base_dir = Path(__file__).parent.parent
data_dir = base_dir / "data" / "maupassant_fr"
headers = {"User-Agent": "CorpusIsosceles/1.0 (academic research)"}
base_url = "http://maupassant.free.fr"


def safe_print(text):
	try:
		print(text)
	except UnicodeEncodeError:
		print(text.encode("ascii", "replace").decode("ascii"))


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


def fetch_url(url):
	for attempt in range(3):
		try:
			req = urllib.request.Request(url, headers=headers)
			with urllib.request.urlopen(req, timeout=30) as response:
				return response.read().decode("utf-8")
		except Exception as e:
			if attempt == 2:
				raise
			time.sleep(1)
	return None


class IndexParser(HTMLParser):
	def __init__(self):
		super().__init__()
		self.stories = []
		self.current_year = None
		self.in_link = False
		self.current_href = None
		self.current_title = ""

	def handle_starttag(self, tag, attrs):
		attrs_dict = dict(attrs)
		if tag == "a" and "name" in attrs_dict:
			name = attrs_dict["name"]
			if re.match(r"^\d{4}$", name):
				self.current_year = name
		if tag == "a" and "href" in attrs_dict:
			href = attrs_dict["href"]
			if href.startswith("textes/") and href.endswith(".html"):
				self.in_link = True
				self.current_href = href
				self.current_title = ""

	def handle_data(self, data):
		if self.in_link:
			self.current_title += data

	def handle_endtag(self, tag):
		if tag == "a" and self.in_link:
			self.in_link = False
			if self.current_href and self.current_title:
				self.stories.append({
					"href": self.current_href,
					"title": self.current_title.strip(),
					"year": self.current_year,
				})
			self.current_href = None
			self.current_title = ""


class TextExtractor(HTMLParser):
	def __init__(self):
		super().__init__()
		self.in_body = False
		self.text_parts = []
		self.skip_depth = 0

	def handle_starttag(self, tag, attrs):
		if tag == "body":
			self.in_body = True
		if tag in ("table", "script", "style", "form"):
			self.skip_depth += 1
		if tag in ("p", "br") and self.in_body and self.skip_depth == 0:
			self.text_parts.append("\n\n")

	def handle_endtag(self, tag):
		if tag in ("table", "script", "style", "form"):
			self.skip_depth = max(0, self.skip_depth - 1)
		if tag == "body":
			self.in_body = False

	def handle_data(self, data):
		if self.in_body and self.skip_depth == 0:
			self.text_parts.append(data)

	def get_text(self):
		text = "".join(self.text_parts)
		text = html.unescape(text)
		text = re.sub(r"\n{3,}", "\n\n", text)
		text = re.sub(r"[ \t]+", " ", text)
		text = re.sub(r"^ +", "", text, flags=re.MULTILINE)
		text = re.sub(r"Retour à la page d'accueil.*", "", text, flags=re.DOTALL)
		text = re.sub(r"maupassant\.free\.fr.*", "", text, flags=re.IGNORECASE)
		return text.strip()


def get_story_list():
	print("Fetching story index...")
	html_content = fetch_url(f"{base_url}/contes2.htm")
	parser = IndexParser()
	parser.feed(html_content)
	return parser.stories


def get_novel_list():
	return [
		{"href": "textes/vie.html", "title": "Une vie", "year": "1883"},
		{"href": "textes/belami.html", "title": "Bel-Ami", "year": "1885"},
		{"href": "textes/montoriol.html", "title": "Mont-Oriol", "year": "1887"},
		{"href": "textes/pierreetjean.html", "title": "Pierre et Jean", "year": "1888"},
		{"href": "textes/fortcommelamort.html", "title": "Fort comme la mort", "year": "1889"},
		{"href": "textes/notrecoeur.html", "title": "Notre cœur", "year": "1890"},
	]


def extract_metadata_and_body(text, story):
	lines = text.split("\n")

	metadata = {
		"file": Path(story["href"]).stem,
		"title": story["title"],
		"year": story.get("year", ""),
		"publication": "",
		"source_url": f"{base_url}/{story['href']}",
	}

	body_start = 0
	body_end = len(lines)

	for i, line in enumerate(lines):
		line_lower = line.lower().strip()
		if any(marker in line_lower for marker in [
			"guy de maupassant", "texte publié", "texte d'origine",
			"dialogues initiés", "tiret", "guillemet", "mis en ligne",
		]):
			body_start = i + 1
			if "texte publié" in line_lower:
				metadata["publication"] = line.strip()
			continue
		if line.strip() == story["title"]:
			body_start = i + 1
			continue
		if len(line.strip()) > 50:
			break

	for i in range(len(lines) - 1, body_start, -1):
		line = lines[i].strip()
		if not line:
			continue
		if any(marker in line.lower() for marker in ["phpmyvisites", "open source web analytics", "maupassant.free.fr"]):
			body_end = i
			continue
		if re.match(r"^\d{1,2}\s+\w+\s+\d{4}$", line) or re.match(r"^\w+\s+\d{4}$", line):
			body_end = i
			continue
		if len(line) > 20:
			break

	body_lines = lines[body_start:body_end]
	body = "\n".join(body_lines).strip()
	body = re.sub(r"\n{3,}", "\n\n", body)
	body = normalize_unicode(body)

	return metadata, body


def download_text(story, dest_dir, metadata_list):
	url = f"{base_url}/{story['href']}"
	filename = Path(story["href"]).stem
	dest_path = dest_dir / f"{filename}.txt"

	if dest_path.exists():
		return "exists"

	try:
		html_content = fetch_url(url)
		parser = TextExtractor()
		parser.feed(html_content)
		raw_text = parser.get_text()

		if len(raw_text) < 100:
			return "too_short"

		metadata, body = extract_metadata_and_body(raw_text, story)

		if len(body) < 50:
			return "no_body"

		dest_path.write_text(body, encoding="utf-8")
		metadata["words"] = len(body.split())
		metadata_list.append(metadata)

		time.sleep(0.5)
		return "ok"
	except Exception as e:
		return f"error: {e}"


def cmd_list():
	stories = get_story_list()
	novels = get_novel_list()

	safe_print(f"\n=== Novels ({len(novels)}) ===\n")
	for n in novels:
		safe_print(f"  {n['year']}  {n['title']}")

	safe_print(f"\n=== Contes by year ({len(stories)} total) ===\n")
	by_year = {}
	for s in stories:
		year = s.get("year", "unknown")
		if year not in by_year:
			by_year[year] = []
		by_year[year].append(s)

	for year in sorted(by_year.keys()):
		items = by_year[year]
		safe_print(f"{year}: {len(items)} stories")


def cmd_download():
	txt_dir = data_dir / "txt"
	txt_dir.mkdir(parents=True, exist_ok=True)

	all_metadata = []

	stories = get_story_list()
	print(f"\n=== Downloading {len(stories)} contes ===\n")
	for i, story in enumerate(stories, 1):
		result = download_text(story, txt_dir, all_metadata)
		status = "ok" if result == "ok" else ("skip" if result == "exists" else "FAIL")
		year = story.get("year", "????")
		title = story["title"][:40]
		print(f"[{i:3}/{len(stories)}] {status:4} {year} {title}")

	if all_metadata:
		metadata_path = data_dir / "metadata.tsv"
		with open(metadata_path, "w", encoding="utf-8") as f:
			f.write("file\ttitle\tyear\twords\tpublication\n")
			for m in sorted(all_metadata, key=lambda x: (x.get("year", ""), x.get("title", ""))):
				pub = m.get("publication", "").replace("\t", " ")
				f.write(f"{m['file']}\t{m['title']}\t{m['year']}\t{m.get('words', 0)}\t{pub}\n")
		print(f"\nMetadata: {metadata_path}")

	print(f"Texts: {txt_dir}")


def main():
	if len(sys.argv) < 2:
		print(__doc__)
		return
	cmd = sys.argv[1]
	if cmd == "list":
		cmd_list()
	elif cmd == "download":
		cmd_download()
	else:
		print(f"Unknown command: {cmd}")


if __name__ == "__main__":
	main()
