#!/usr/bin/env python3
"""
Download Poe texts from English WikiSource.

Usage:
    python poe.py list
    python poe.py download 1845
    python poe.py download 1850
    python poe.py download all
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
data_dir = base_dir / "data" / "poe_en"
headers = {"User-Agent": "CorpusIsosceles/1.0 (academic research)"}
base_url = "https://en.wikisource.org"

editions = {
	"1845": {
		"name": "Tales (1845)",
		"index_url": "/wiki/Tales_(Poe)",
		"link_prefix": "/wiki/Tales_(Poe)/",
	},
	"1850": {
		"name": "Works (Griswold 1850) Vol 1",
		"index_url": "/wiki/The_Works_of_the_Late_Edgar_Allan_Poe_(1850)/Volume_1",
		"link_prefix": "/wiki/The_Works_of_the_Late_Edgar_Allan_Poe_(1850)/Volume_1/",
	},
}


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


def normalize_filename(title):
	name = title.lower()
	name = re.sub(r"[^a-z0-9]+", "_", name)
	name = re.sub(r"_+", "_", name).strip("_")
	return name[:60]


class TOCParser(HTMLParser):
	def __init__(self, link_prefix):
		super().__init__()
		self.link_prefix = link_prefix
		self.stories = []
		self.in_link = False
		self.current_href = None
		self.current_title = ""

	def handle_starttag(self, tag, attrs):
		if tag == "a":
			attrs_dict = dict(attrs)
			href = attrs_dict.get("href", "")
			if href.startswith(self.link_prefix):
				self.in_link = True
				self.current_href = href
				self.current_title = ""

	def handle_data(self, data):
		if self.in_link:
			self.current_title += data

	def handle_endtag(self, tag):
		if tag == "a" and self.in_link:
			self.in_link = False
			title = self.current_title.strip()
			if title and self.current_href:
				self.stories.append({"href": self.current_href, "title": title})
			self.current_href = None
			self.current_title = ""


class TextExtractor(HTMLParser):
	skip_classes = [
		"thumb", "toc", "mw-heading", "sister", "noprint", "printfooter",
		"catlinks", "mw-footer", "mw-header", "ws-header", "ws-footer",
		"header", "footer", "navigation", "mw-editsection", "licensetpl",
		"mw-cite", "mw-references", "reference", "external", "ws-license",
		"ws-noexport", "ws-summary", "license-text", "prp-page",
	]
	skip_ids = [
		"catlinks", "footer", "mw-navigation", "mw-footer-container",
		"mw-head", "footer-info", "ws-header", "ws-footer",
		"header", "headerContainer", "jump-to-nav",
	]

	def __init__(self):
		super().__init__()
		self.in_content = False
		self.text_parts = []
		self.skip_depth = 0

	def handle_starttag(self, tag, attrs):
		attrs_dict = dict(attrs)
		cls = attrs_dict.get("class", "")
		elem_id = attrs_dict.get("id", "")

		if "mw-parser-output" in cls:
			self.in_content = True

		if tag in ("script", "style", "table", "sup", "footer", "header", "nav"):
			self.skip_depth += 1
		if tag == "div" and (any(x in cls for x in self.skip_classes) or any(x in elem_id for x in self.skip_ids)):
			self.skip_depth += 1
		if tag == "span" and any(x in cls for x in ["mw-editsection", "noprint"]):
			self.skip_depth += 1
		if tag in ("p", "br") and self.in_content and self.skip_depth == 0:
			self.text_parts.append("\n\n")

	def handle_endtag(self, tag):
		if tag in ("script", "style", "table", "sup", "div", "footer", "header", "nav", "span"):
			self.skip_depth = max(0, self.skip_depth - 1)

	def handle_data(self, data):
		if self.in_content and self.skip_depth == 0:
			self.text_parts.append(data)

	def get_text(self):
		text = "".join(self.text_parts)
		text = html.unescape(text)
		text = re.sub(r"\n{3,}", "\n\n", text)
		text = re.sub(r"[ \t]+", " ", text)
		text = re.sub(r"^ +", "", text, flags=re.MULTILINE)
		return text.strip()


def get_story_list(edition_key):
	edition = editions[edition_key]
	url = base_url + edition["index_url"]
	print(f"Fetching TOC: {edition['name']}")
	html_content = fetch_url(url)
	parser = TOCParser(edition["link_prefix"])
	parser.feed(html_content)
	return parser.stories


def extract_body(raw_text, title):
	lines = raw_text.split("\n")
	body_start = 0
	body_end = len(lines)

	header_markers = [
		"wikisource", "← ", "→", "category:", "this work", "public domain",
		"sister projects:", "wikipedia article", "commons category", "wikidata item",
		"published in", "volume", "by edgar allan poe",
		"the works of the late edgar allan poe",
	]

	for i, line in enumerate(lines):
		line_lower = line.strip().lower()
		if any(marker in line_lower for marker in header_markers):
			body_start = i + 1
			continue
		if re.match(r"^\d+$", line.strip()):
			body_start = i + 1
			continue
		title_normalized = re.sub(r"[^a-z]", "", title.lower())
		line_normalized = re.sub(r"[^a-z]", "", line_lower)
		if title_normalized and line_normalized == title_normalized:
			body_start = i + 1
			continue
		if len(line.strip()) > 50:
			break

	footer_markers = [
		"wikisource", "category", "public domain", "this work is",
		"this work was published", "retrieved from", "privacy policy",
		"about wikisource", "disclaimers", "code of conduct", "developers",
		"statistics", "cookie statement", "mobile view", "creative commons",
		"attribution-sharealike", "terms of use", "last edited",
		"add topic", "languages", "search", "hidden categories",
		"subpages", "headers applying",
	]

	for i in range(len(lines) - 1, body_start, -1):
		line = lines[i].strip()
		line_lower = line.lower()
		if not line:
			continue
		if any(marker in line_lower for marker in footer_markers):
			body_end = i
			continue
		if len(line) < 20:
			body_end = i
			continue
		if len(line) > 40:
			break

	body = "\n".join(lines[body_start:body_end]).strip()
	body = re.sub(r"\n{3,}", "\n\n", body)
	body = normalize_unicode(body)
	return body


def download_edition(edition_key):
	edition = editions[edition_key]
	stories = get_story_list(edition_key)

	if not stories:
		print("No stories found in TOC")
		return

	print(f"Found {len(stories)} stories\n")

	edition_dir = data_dir / edition_key
	txt_dir = edition_dir / "txt"
	txt_dir.mkdir(parents=True, exist_ok=True)

	metadata = []

	for i, story in enumerate(stories, 1):
		title = story["title"]
		filename = normalize_filename(title)
		dest_path = txt_dir / f"{filename}.txt"

		if dest_path.exists():
			print(f"[{i:2}/{len(stories)}] skip {title[:45]}")
			continue

		url = base_url + story["href"]
		print(f"[{i:2}/{len(stories)}] {title[:50]}")

		try:
			html_content = fetch_url(url)
			parser = TextExtractor()
			parser.feed(html_content)
			raw_text = parser.get_text()
			body = extract_body(raw_text, title)

			if len(body) < 100:
				print(f"           TOO SHORT ({len(body)} chars)")
				continue

			dest_path.write_text(body, encoding="utf-8")
			metadata.append({"file": filename, "title": title, "words": len(body.split()), "url": url})
			time.sleep(0.5)

		except Exception as e:
			print(f"           ERROR: {e}")

	if metadata:
		meta_path = edition_dir / "metadata.tsv"
		with open(meta_path, "w", encoding="utf-8") as f:
			f.write("file\ttitle\twords\n")
			for m in metadata:
				f.write(f"{m['file']}\t{m['title']}\t{m['words']}\n")
		print(f"\nMetadata: {meta_path}")

	print(f"Texts: {txt_dir}")


def cmd_list():
	for key, edition in editions.items():
		stories = get_story_list(key)
		safe_print(f"\n=== {edition['name']} ({len(stories)} stories) ===\n")
		for s in stories:
			safe_print(f"  {s['title']}")


def cmd_download(what):
	if what == "all":
		for key in editions:
			print(f"\n{'='*60}")
			download_edition(key)
	elif what in editions:
		download_edition(what)
	else:
		print(f"Unknown edition: {what}")
		print(f"Available: {', '.join(editions.keys())}, all")


def main():
	if len(sys.argv) < 2:
		print(__doc__)
		return

	cmd = sys.argv[1]
	if cmd == "list":
		cmd_list()
	elif cmd == "download":
		what = sys.argv[2] if len(sys.argv) > 2 else "all"
		cmd_download(what)
	else:
		print(f"Unknown command: {cmd}")


if __name__ == "__main__":
	main()
