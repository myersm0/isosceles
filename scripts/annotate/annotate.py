#!/usr/bin/env python3
"""
Annotation pipeline for the isosceles corpus.

Backends:
  stanza         - Stanza for tokenization and parsing (baseline)
  stanza+corenlp - CoreNLP segmentation + Stanza parsing
  corenlp        - CoreNLP for everything (requires CORENLP_HOME or --corenlp-url)
  spacy          - CoreNLP segmentation + spaCy parsing
  spacy+llm      - CoreNLP segmentation + spaCy parsing + LLM corrections

All CoreNLP-using backends support --corenlp-url to connect to a
standalone CoreNLP server, bypassing stanza's wrapper entirely.

Examples:
  python annotate.py data/maupassant/fr/txt data/maupassant/fr/conllu -l fr -b spacy
  python annotate.py data/eltec/fr/txt data/gold -l fr -b spacy+llm \\
      -m claude-opus-4-5 --chunks 1 --chunk-size 20 --seed 42
  python annotate.py data/txt data/conllu -l fr -b spacy --corenlp-url http://localhost:9000
"""
import argparse
import json
import os
import random
import re
import sys
import copy
import requests
from pathlib import Path
from conllu_tools import (
	apply_corrections,
	validate_tree,
	apply_deterministic_fixes,
)


# -----------------------------------------------------------------------------
# Native CoreNLP HTTP client (no stanza dependency)
# -----------------------------------------------------------------------------

class _Obj:
	"""Attribute-access wrapper for dicts."""
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)


def _wrap_corenlp_json(data):
	"""Convert CoreNLP JSON response into objects matching the protobuf interface."""
	sentences = []
	for sent_json in data.get("sentences", []):
		tokens = []
		for tok_json in sent_json.get("tokens", []):
			tokens.append(_Obj(
				word=tok_json.get("word", ""),
				after=tok_json.get("after", ""),
				before=tok_json.get("before", ""),
				lemma=tok_json.get("lemma", ""),
				pos=tok_json.get("pos", ""),
				tokenEndIndex=tok_json.get("index", 0),
			))

		edges = []
		for dep_json in sent_json.get("basicDependencies", []):
			edges.append(_Obj(
				source=dep_json.get("governor", 0),
				target=dep_json.get("dependent", 0),
				dep=dep_json.get("dep", ""),
			))

		sentences.append(_Obj(
			token=tokens,
			basicDependencies=_Obj(edge=edges),
		))

	return _Obj(sentence=sentences)


class NativeCoreNLPClient:
	"""
	Drop-in replacement for stanza.server.CoreNLPClient that talks
	to an already-running CoreNLP server over HTTP.
	"""

	def __init__(self, url, annotators=None, properties=None):
		self.url = url.rstrip("/")
		self.annotators = annotators or ["tokenize", "ssplit"]
		self.default_properties = properties or {}

	def __enter__(self):
		return self

	def __exit__(self, *args):
		pass

	def annotate(self, text, properties=None):
		props = {**self.default_properties}
		if properties:
			props.update(properties)
		props.setdefault("annotators", ",".join(self.annotators))
		props["outputFormat"] = "json"

		response = requests.post(
			self.url,
			params={"properties": json.dumps(props)},
			data=text.encode("utf-8"),
			headers={"Content-Type": "text/plain; charset=utf-8"},
		)
		response.raise_for_status()
		return _wrap_corenlp_json(response.json())


def make_corenlp_client(corenlp_url=None, annotators=None, properties=None, threads=None, be_quiet=True):
	"""Factory: returns NativeCoreNLPClient if url given, else stanza's."""
	if corenlp_url:
		return NativeCoreNLPClient(corenlp_url, annotators=annotators, properties=properties)
	from stanza.server import CoreNLPClient
	return CoreNLPClient(
		annotators=annotators,
		properties=properties,
		threads=threads or 5,
		be_quiet=be_quiet,
	)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def load_llm_prompt(path=None):
	if path is None:
		path = Path(__file__).parent / "prompt_fr.txt"
	return Path(path).read_text(encoding="utf-8")


def chunk_text(text, max_chars=9000):
	"""Split text into chunks for CoreNLP's character limit."""
	chunks = []
	start = 0
	n = len(text)
	
	while start < n:
		end = min(start + max_chars, n)
		if end == n:
			chunks.append(text[start:end])
			break
		
		window = text[start:end]
		cut = max(
			window.rfind("\n\n"),
			window.rfind("\n"),
			window.rfind(". "),
			window.rfind("? "),
			window.rfind("! "),
		)
		
		if cut == -1 or cut < max_chars * 0.5:
			cut = max_chars
		
		cut_end = start + cut + 1
		chunks.append(text[start:cut_end])
		start = cut_end
	
	return chunks


def build_corenlp_props(lang, ssplit="two", threads=None):
	"""Build CoreNLP properties dict."""
	props = {
		"pipelineLanguage": lang,
		"ssplit.newlineIsSentenceBreak": ssplit,
		"tokenize.whitespace": "false",  # might help?
	}
	if threads:
		props["threads"] = str(threads)
	return props


def select_files(input_dir, output_dir, fmt, limit=None, overwrite=False):
	"""Select input files, optionally limiting and checking for existing outputs."""
	input_dir = Path(input_dir)
	output_dir = Path(output_dir)
	ext = ".json" if fmt == "json" else ".conllu"
	
	all_files = sorted(input_dir.glob("*.txt"))
	
	if not overwrite:
		files = []
		skipped = 0
		for f in all_files:
			out_path = output_dir / (f.stem + ext)
			if out_path.exists():
				skipped += 1
			else:
				files.append(f)
		if skipped > 0:
			print(f"  Skipping {skipped} files with existing output", file=sys.stderr)
	else:
		files = list(all_files)
	
	if limit and limit < len(files):
		random.shuffle(files)
		files = files[:limit]
		files = sorted(files)
	
	return files


def sample_chunks(sentences, n_chunks, chunk_size, seed=None):
	"""Sample non-overlapping chunks of consecutive sentences."""
	if seed is not None:
		random.seed(seed)
	
	n_sentences = len(sentences)
	if n_sentences < chunk_size:
		return sentences
	
	max_chunks = n_sentences // chunk_size
	n_chunks = min(n_chunks, max_chunks)
	
	if n_chunks == 0:
		return sentences[:chunk_size] if sentences else []
	
	sampled = []
	used_ranges = []
	attempts = 0
	
	while len(sampled) < n_chunks and attempts < n_chunks * 20:
		attempts += 1
		start = random.randint(0, n_sentences - chunk_size)
		end = start + chunk_size
		
		overlap = any(not (end <= us or start >= ue) for us, ue in used_ranges)
		if not overlap:
			sampled.append((start, end))
			used_ranges.append((start, end))
	
	sampled.sort()
	
	result = []
	for chunk_idx, (start, end) in enumerate(sampled):
		for sent_id, sent_text, tokens in sentences[start:end]:
			result.append((f"{sent_id}:chunk{chunk_idx+1}", sent_text, tokens))
	
	return result


# -----------------------------------------------------------------------------
# Output formatters
# -----------------------------------------------------------------------------

def tokens_to_conllu(tokens, sent_id, sent_text):
	lines = [f"# sent_id = {sent_id}", f"# text = {sent_text}"]
	for t in tokens:
		row = [
			str(t["id"]), t["form"], t["lemma"], t["upos"], t.get("xpos", "_"),
			t.get("feats", "_"), str(t["head"]), t["deprel"],
			t.get("deps", "_"), t.get("misc", "_"),
		]
		lines.append("\t".join(row))
	lines.append("")
	return "\n".join(lines)


def write_output(out_path, doc_id, sentences, fmt):
	if fmt == "json":
		data = {
			"doc_id": doc_id,
			"sentences": [
				{"sent_id": sid, "text": txt, "tokens": toks}
				for sid, txt, toks in sentences
			]
		}
		out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
	else:
		lines = [tokens_to_conllu(toks, sid, txt) for sid, txt, toks in sentences]
		out_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Core parsing helpers
# -----------------------------------------------------------------------------

def load_spacy_model(lang, component_name="single_sentence"):
	"""Load spaCy transformer model with single-sentence processing."""
	import spacy
	from spacy.language import Language
	
	@Language.component(component_name)
	def force_single_sentence(doc):
		for token in doc:
			token.is_sent_start = False
		doc[0].is_sent_start = True
		return doc
	
	model_name = "fr_dep_news_trf" if lang == "fr" else "en_core_web_trf"
	nlp = spacy.load(model_name, disable=["ner"])
	nlp.add_pipe(component_name, before="parser")
	return nlp


def spacy_doc_to_tokens(doc):
	"""Convert spaCy Doc to list of token dicts."""
	non_space = [(i, tok) for i, tok in enumerate(doc) if tok.pos_ != "SPACE"]
	old_to_new = {old_i: new_i for new_i, (old_i, _) in enumerate(non_space, start=1)}
	
	tokens = []
	for new_i, (old_i, tok) in enumerate(non_space, start=1):
		head = 0 if tok.head == tok else old_to_new.get(tok.head.i, 0)
		tokens.append({
			"id": new_i,
			"form": tok.text,
			"lemma": tok.lemma_ or "_",
			"upos": tok.pos_ or "_",
			"xpos": tok.tag_ or "_",
			"feats": str(tok.morph) or "_",
			"head": head,
			"deprel": tok.dep_ or "_",
			"deps": "_",
			"misc": "_",
		})
	return tokens


def segment_and_parse(text, doc_id, corenlp_client, spacy_nlp, props):
	"""
	Segment text with CoreNLP and parse with spaCy.
	
	Returns: list of (sent_id, sent_text, tokens)
	"""
	sentences = []
	sent_idx = 0
	
	for text_chunk in chunk_text(text):
		doc = corenlp_client.annotate(text_chunk, properties=props)
		
		for sent in doc.sentence:
			sent_idx += 1
			sent_text = "".join(t.word + t.after for t in sent.token).strip()
			sent_text = " ".join(sent_text.split())
			
			if not sent_text:
				continue
			
			spacy_doc = spacy_nlp(sent_text)
			tokens = spacy_doc_to_tokens(spacy_doc)
			sentences.append((f"{doc_id}-{sent_idx}", sent_text, tokens))
	
	return sentences


# -----------------------------------------------------------------------------
# LLM correction helpers
# -----------------------------------------------------------------------------

def get_llm_client(model):
	"""Initialize appropriate LLM client based on model name."""
	if model.startswith("claude-"):
		import anthropic
		return anthropic.Anthropic()
	else:
		import openai
		return openai.OpenAI()


def get_llm_corrections(client, sent_text, tokens, lang, model, prompt, simplified=False):
	"""Get corrections from LLM."""
	if model.startswith("claude-"):
		return _anthropic_corrections(client, sent_text, tokens, lang, model, prompt, simplified)
	else:
		return _openai_corrections(client, sent_text, tokens, lang, model, prompt, simplified)


def _format_parse_for_llm(tokens, lang, simplified=False):
	lang_name = "French" if lang == "fr" else "English"
	if simplified:
		lines = [f"{t['id']}\t{t['form']}\t{t['lemma']}\t{t['upos']}\t{t.get('feats', '_') or '_'}" for t in tokens]
	else:
		lines = [f"{t['id']}\t{t['form']}\t{t['lemma']}\t{t['upos']}\t{t['head']}\t{t['deprel']}"
				 for t in tokens]
	return lang_name, "\n".join(lines)


def _parse_corrections_json(response_text):
	match = re.search(r"\{[\s\S]*\}", response_text)
	if match:
		try:
			return json.loads(match.group()).get("corrections", [])
		except json.JSONDecodeError:
			pass
	return []


def _anthropic_corrections(client, sent_text, tokens, lang, model, prompt, simplified=False):
	lang_name, parse_str = _format_parse_for_llm(tokens, lang, simplified)
	
	response = client.messages.create(
		model=model,
		max_tokens=1024,
		system=[{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}],
		messages=[{"role": "user", "content": f"{parse_str}"}] if simplified else
				 [{"role": "user", "content": f"{lang_name} sentence:\n{sent_text}\n\nParse:\n{parse_str}"}]
	)
	
	corrections = _parse_corrections_json(response.content[0].text)
	usage = response.usage
	cache_stats = {
		"cache_created": getattr(usage, "cache_creation_input_tokens", 0) or 0,
		"cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
	}
	return corrections, usage.input_tokens, usage.output_tokens, cache_stats


def _openai_corrections(client, sent_text, tokens, lang, model, prompt, simplified=False):
	lang_name, parse_str = _format_parse_for_llm(tokens, lang, simplified)
	
	user_content = parse_str if simplified else f"{lang_name} sentence:\n{sent_text}\n\nParse:\n{parse_str}"
	
	response = client.chat.completions.create(
		model=model,
		max_completion_tokens=1024,
		messages=[
			{"role": "system", "content": prompt},
			{"role": "user", "content": user_content}
		]
	)
	
	corrections = _parse_corrections_json(response.choices[0].message.content)
	usage = response.usage
	cache_stats = {"cache_created": 0, "cache_read": 0}
	return corrections, usage.prompt_tokens, usage.completion_tokens, cache_stats


def apply_non_structural_corrections(tokens, corrections):
	"""Apply only lemma/upos/xpos/feats corrections (cannot break tree validity)."""
	tokens = copy.deepcopy(tokens)
	token_map = {t["id"]: t for t in tokens}
	safe_fields = {"lemma", "upos", "xpos", "feats"}
	n_applied = 0
	
	for c in corrections:
		tok_id = c.get("id")
		field = c.get("field")
		value = c.get("value")
		if field not in safe_fields:
			continue
		try:
			tok_id = int(tok_id)
		except (ValueError, TypeError):
			continue
		if tok_id not in token_map:
			continue
		token_map[tok_id][field] = value
		n_applied += 1
	
	return tokens, n_applied


def log_rejected_corrections(sent_id, sent_text, tokens, corrections, validation):
	"""Log rejected corrections to stderr for later analysis."""
	structural = [c for c in corrections if c.get("field") in ("head", "deprel")]
	non_structural = [c for c in corrections if c.get("field") not in ("head", "deprel")]
	
	print(f"	[{sent_id}] Rejected: {validation}", file=sys.stderr)
	print(f"	  Text: {sent_text[:80]}{'...' if len(sent_text) > 80 else ''}", file=sys.stderr)
	
	if structural:
		print(f"	  Structural ({len(structural)}):", file=sys.stderr)
		token_map = {t["id"]: t["form"] for t in tokens}
		for c in structural[:5]:
			form = token_map.get(c.get("id"), "?")
			print(f"		{c.get('id')} '{form}': {c.get('field')} → {c.get('value')}", file=sys.stderr)
		if len(structural) > 5:
			print(f"		... +{len(structural) - 5} more", file=sys.stderr)
	
	if non_structural:
		print(f"	  Non-structural ({len(non_structural)}): will apply as partial fix", file=sys.stderr)


def apply_llm_corrections(sentences, llm_client, model, prompt, lang, mode="full"):
	"""
	Apply LLM corrections with validation and partial acceptance.
	
	mode="full": structural corrections with validation, falls back to non-structural
	mode="surface": lemma/upos/feats corrections only, no validation needed
	"""
	corrected = []
	simplified = (mode == "surface")
	stats = {
		"input_tokens": 0,
		"output_tokens": 0,
		"cache_created": 0,
		"cache_read": 0,
		"corrections": 0,
		"applied": 0,
		"rejected_structural": 0,
		"partial_applied": 0,
		"errors": 0,
	}
	
	for sent_id, sent_text, tokens in sentences:
		try:
			corrections, in_toks, out_toks, cache = get_llm_corrections(
				llm_client, sent_text, tokens, lang, model, prompt, simplified
			)
			stats["input_tokens"] += in_toks
			stats["output_tokens"] += out_toks
			stats["cache_created"] += cache["cache_created"]
			stats["cache_read"] += cache["cache_read"]
			stats["corrections"] += len(corrections)
			
			if corrections:
				if simplified:
					tokens, n_applied = apply_non_structural_corrections(tokens, corrections)
					stats["applied"] += n_applied
				else:
					candidate, n_applied = apply_corrections(tokens, corrections)
					validation = validate_tree(candidate)
					
					if validation["valid"]:
						tokens = candidate
						stats["applied"] += n_applied
					else:
						stats["rejected_structural"] += 1
						log_rejected_corrections(sent_id, sent_text, tokens, corrections, validation)
						tokens, n_partial = apply_non_structural_corrections(tokens, corrections)
						stats["partial_applied"] += n_partial
			
			if mode == "full":
				tokens, _ = apply_deterministic_fixes(tokens)
		
		except Exception as e:
			print(f"	[{sent_id}] ERROR: {e}", file=sys.stderr)
			stats["errors"] += 1
		
		corrected.append((sent_id, sent_text, tokens))
	
	return corrected, stats


# -----------------------------------------------------------------------------
# Stanza backend
# -----------------------------------------------------------------------------

def process_stanza(input_dir, output_dir, lang, fmt, limit=None, overwrite=False):
	import stanza
	
	input_dir = Path(input_dir)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	
	files = select_files(input_dir, output_dir, fmt, limit, overwrite)
	if not files:
		print("  No files to process", file=sys.stderr)
		return
	
	print("Loading Stanza model...", file=sys.stderr)
	if lang == "fr":
		nlp = stanza.Pipeline(lang=lang, package="gsd", processors="tokenize,mwt,pos,lemma,depparse")
	else:
		nlp = stanza.Pipeline(lang=lang, processors="tokenize,mwt,pos,lemma,depparse")
	ext = ".json" if fmt == "json" else ".conllu"
	
	for filepath in files:
		print(f"  {filepath.name}", file=sys.stderr)
		text = filepath.read_text(encoding="utf-8")
		doc = nlp(text)
		
		sentences = []
		for i, sent in enumerate(doc.sentences, 1):
			tokens = []
			for tok in sent.words:
				tokens.append({
					"id": tok.id,
					"form": tok.text,
					"lemma": tok.lemma or "_",
					"upos": tok.upos or "_",
					"xpos": tok.xpos or "_",
					"feats": tok.feats or "_",
					"head": tok.head,
					"deprel": tok.deprel or "_",
					"deps": "_",
					"misc": "_",
				})
			sentences.append((f"{filepath.stem}-{i}", sent.text, tokens))
		
		out_path = output_dir / (filepath.stem + ext)
		write_output(out_path, filepath.stem, sentences, fmt)


# -----------------------------------------------------------------------------
# Stanza + CoreNLP segmentation backend
# -----------------------------------------------------------------------------

def stanza_sent_to_tokens(sent):
	"""Convert a Stanza Sentence to list of token dicts."""
	tokens = []
	for tok in sent.words:
		tokens.append({
			"id": tok.id,
			"form": tok.text,
			"lemma": tok.lemma or "_",
			"upos": tok.upos or "_",
			"xpos": tok.xpos or "_",
			"feats": tok.feats or "_",
			"head": tok.head,
			"deprel": tok.deprel or "_",
			"deps": "_",
			"misc": "_",
		})
	return tokens


def segment_and_parse_stanza(text, doc_id, corenlp_client, stanza_nlp, props):
	"""Segment text with CoreNLP and parse with Stanza."""
	sentences = []
	sent_idx = 0

	for text_chunk in chunk_text(text):
		doc = corenlp_client.annotate(text_chunk, properties=props)

		for sent in doc.sentence:
			sent_idx += 1
			sent_text = "".join(t.word + t.after for t in sent.token).strip()
			sent_text = " ".join(sent_text.split())

			if not sent_text:
				continue

			stanza_doc = stanza_nlp(sent_text)
			if stanza_doc.sentences:
				tokens = stanza_sent_to_tokens(stanza_doc.sentences[0])
			else:
				continue
			sentences.append((f"{doc_id}-{sent_idx}", sent_text, tokens))

	return sentences


def process_stanza_corenlp(input_dir, output_dir, lang, fmt, limit=None, overwrite=False, ssplit="two", threads=None, corenlp_url=None):
	import stanza

	input_dir = Path(input_dir)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	files = select_files(input_dir, output_dir, fmt, limit, overwrite)
	if not files:
		print("  No files to process", file=sys.stderr)
		return

	props = build_corenlp_props(lang, ssplit)
	print(f"CoreNLP props: {props}", file=sys.stderr)

	print("Loading Stanza model...", file=sys.stderr)
	if lang == "fr":
		nlp = stanza.Pipeline(
			lang=lang,
			package="gsd",
			processors="tokenize,mwt,pos,lemma,depparse",
			tokenize_no_ssplit=True,
		)
	else:
		nlp = stanza.Pipeline(
			lang="en",
			processors={
				"tokenize": "gum",
				"mwt": "gum",
				"pos": "gum",
				"lemma": "gum",
				"depparse": "default",
			},
			package=None,
			tokenize_no_ssplit=True,
		)
	ext = ".json" if fmt == "json" else ".conllu"

	with make_corenlp_client(corenlp_url, annotators=["tokenize", "ssplit"], properties=props, threads=threads) as client:
		for filepath in files:
			print(f"  {filepath.name}", file=sys.stderr)
			text = filepath.read_text(encoding="utf-8")
			sentences = segment_and_parse_stanza(text, filepath.stem, client, nlp, props)

			out_path = output_dir / (filepath.stem + ext)
			write_output(out_path, filepath.stem, sentences, fmt)


# -----------------------------------------------------------------------------
# CoreNLP backend
# -----------------------------------------------------------------------------

def process_corenlp(input_dir, output_dir, lang, fmt, limit=None, overwrite=False, corenlp_url=None):
	from stanza.server import CoreNLPClient
	
	input_dir = Path(input_dir)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	
	files = select_files(input_dir, output_dir, fmt, limit, overwrite)
	if not files:
		print("  No files to process", file=sys.stderr)
		return
	
	ext = ".json" if fmt == "json" else ".conllu"
	props = {"pipelineLanguage": lang}
	annotators = ["tokenize", "ssplit", "pos", "lemma", "depparse"]
	
	with make_corenlp_client(corenlp_url, annotators=annotators, properties=props) as client:
		for filepath in files:
			print(f"  {filepath.name}", file=sys.stderr)
			text = filepath.read_text(encoding="utf-8")
			
			sentences = []
			sent_idx = 0
			
			for text_chunk in chunk_text(text):
				doc = client.annotate(text_chunk, properties=props)
				
				for sent in doc.sentence:
					sent_idx += 1
					tokens = []
					for tok in sent.token:
						dep = next((e for e in sent.basicDependencies.edge if e.target == tok.tokenEndIndex), None)
						tokens.append({
							"id": tok.tokenEndIndex,
							"form": tok.word,
							"lemma": tok.lemma or "_",
							"upos": tok.pos or "_",
							"xpos": "_",
							"feats": "_",
							"head": dep.source if dep else 0,
							"deprel": dep.dep if dep else "root",
							"deps": "_",
							"misc": "_",
						})
					
					sent_text = "".join(t.word + t.after for t in sent.token).strip()
					sentences.append((f"{filepath.stem}-{sent_idx}", sent_text, tokens))
			
			out_path = output_dir / (filepath.stem + ext)
			write_output(out_path, filepath.stem, sentences, fmt)


# -----------------------------------------------------------------------------
# spaCy backend (CoreNLP segmentation + spaCy parsing)
# -----------------------------------------------------------------------------

def process_spacy(input_dir, output_dir, lang, fmt, limit=None, overwrite=False, ssplit="two", threads=None, corenlp_url=None):
	input_dir = Path(input_dir)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	
	files = select_files(input_dir, output_dir, fmt, limit, overwrite)
	if not files:
		print("  No files to process", file=sys.stderr)
		return
	
	props = build_corenlp_props(lang, ssplit)
	print(f"CoreNLP props: {props}", file=sys.stderr)
	
	print("Loading spaCy model...", file=sys.stderr)
	nlp = load_spacy_model(lang, "single_sentence_spacy")
	ext = ".json" if fmt == "json" else ".conllu"
	
	with make_corenlp_client(corenlp_url, annotators=["tokenize", "ssplit"], properties=props, threads=threads) as client:
		for filepath in files:
			print(f"  {filepath.name}", file=sys.stderr)
			text = filepath.read_text(encoding="utf-8")
			sentences = segment_and_parse(text, filepath.stem, client, nlp, props)
			
			out_path = output_dir / (filepath.stem + ext)
			write_output(out_path, filepath.stem, sentences, fmt)


# -----------------------------------------------------------------------------
# spaCy + LLM backend
# -----------------------------------------------------------------------------

def process_spacy_llm(input_dir, output_dir, lang, fmt, model, limit=None, overwrite=False,
					  prompt_path=None, chunks=None, chunk_size=20, seed=None, ssplit="two", threads=None,
					  mode="full", corenlp_url=None):
	input_dir = Path(input_dir)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	
	files = select_files(input_dir, output_dir, fmt, limit, overwrite)
	if not files:
		print("  No files to process", file=sys.stderr)
		return
	
	# Setup
	if seed is not None:
		random.seed(seed)
		file_seeds = {f.stem: random.randint(0, 2**31) for f in files}
	else:
		file_seeds = {}
	
	prompt = load_llm_prompt(prompt_path)
	print(f"Loaded prompt: {len(prompt)} chars", file=sys.stderr)
	
	if chunks:
		print(f"Chunk sampling: {chunks} × {chunk_size} sentences per file", file=sys.stderr)
	
	props = build_corenlp_props(lang, ssplit)
	print(f"CoreNLP props: {props}", file=sys.stderr)
	
	print("Loading spaCy model...", file=sys.stderr)
	nlp = load_spacy_model(lang, "single_sentence_llm")
	llm_client = get_llm_client(model)
	ext = ".json" if fmt == "json" else ".conllu"
	
	total_stats = {
		"input_tokens": 0,
		"output_tokens": 0,
		"cache_created": 0,
		"cache_read": 0,
		"corrections": 0,
		"applied": 0,
		"rejected_structural": 0,
		"partial_applied": 0,
		"errors": 0,
		"sentences": 0,
	}
	
	with make_corenlp_client(corenlp_url, annotators=["tokenize", "ssplit"], properties=props, threads=threads) as client:
		for filepath in files:
			print(f"  {filepath.name}", file=sys.stderr)
			text = filepath.read_text(encoding="utf-8")
			
			# Segment and parse
			sentences = segment_and_parse(text, filepath.stem, client, nlp, props)
			
			# Sample if requested
			if chunks and chunks > 0:
				file_seed = file_seeds.get(filepath.stem)
				sentences = sample_chunks(sentences, chunks, chunk_size, seed=file_seed)
				print(f"	Sampled {len(sentences)} sentences", file=sys.stderr)
			
			sentences, stats = apply_llm_corrections(sentences, llm_client, model, prompt, lang, mode)
			
			total_stats["input_tokens"] += stats["input_tokens"]
			total_stats["output_tokens"] += stats["output_tokens"]
			total_stats["cache_created"] += stats["cache_created"]
			total_stats["cache_read"] += stats["cache_read"]
			total_stats["corrections"] += stats["corrections"]
			total_stats["applied"] += stats["applied"]
			total_stats["rejected_structural"] += stats["rejected_structural"]
			total_stats["partial_applied"] += stats["partial_applied"]
			total_stats["errors"] += stats["errors"]
			total_stats["sentences"] += len(sentences)
			
			# Write output
			out_path = output_dir / (filepath.stem + ext)
			write_output(out_path, filepath.stem, sentences, fmt)
	
	# Summary
	print(f"\n=== Summary ===", file=sys.stderr)
	print(f"Files: {len(files)}", file=sys.stderr)
	print(f"Sentences: {total_stats['sentences']}", file=sys.stderr)
	print(f"Tokens: {total_stats['input_tokens']:,} in / {total_stats['output_tokens']:,} out", file=sys.stderr)
	if model.startswith("claude-"):
		cache_total = total_stats["cache_created"] + total_stats["cache_read"]
		if cache_total > 0:
			hit_rate = total_stats["cache_read"] / cache_total * 100
			print(f"Cache: {total_stats['cache_read']:,} read / {total_stats['cache_created']:,} created ({hit_rate:.1f}% hit rate)", file=sys.stderr)
	print(f"Corrections proposed: {total_stats['corrections']}", file=sys.stderr)
	print(f"Corrections applied (full): {total_stats['applied']}", file=sys.stderr)
	print(f"Corrections applied (partial): {total_stats['partial_applied']}", file=sys.stderr)
	print(f"Rejected (invalid tree): {total_stats['rejected_structural']}", file=sys.stderr)
	print(f"Errors: {total_stats['errors']}", file=sys.stderr)
	
	if model.startswith("claude-"):
		# Pricing: cache writes +25%, cache reads -90%
		if "opus" in model:
			base_in, base_out = 15, 75
		else:
			base_in, base_out = 3, 15
		# input_tokens = non-cached content; cache tokens reported separately
		cost = (
			total_stats["input_tokens"] * base_in +
			total_stats["cache_created"] * base_in * 1.25 +
			total_stats["cache_read"] * base_in * 0.1 +
			total_stats["output_tokens"] * base_out
		) / 1_000_000
		print(f"Est. cost: ${cost:.4f}", file=sys.stderr)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
	ap = argparse.ArgumentParser(
		description="Annotation pipeline for the isosceles corpus",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Backends:
  stanza	  Stanza tokenization + parsing (baseline)
  stanza+corenlp  CoreNLP segmentation + Stanza parsing
  corenlp	 CoreNLP for everything (requires CORENLP_HOME)
  spacy	   CoreNLP segmentation + spaCy parsing
  spacy+llm   CoreNLP segmentation + spaCy parsing + LLM corrections

Examples:
  # Full document annotation (spaCy only)
  %(prog)s data/maupassant/fr/txt output/fr -l fr -b spacy
  
  # Surface corrections: lemma, UPOS, feats (recommended, most economical)
  %(prog)s data/eltec/fr/txt output/fr -l fr -b spacy+llm \\
	  -m claude-sonnet-4-5 -p prompt_surface.txt
  
  # Full structural corrections (experimental, may introduce regressions)
  %(prog)s data/maupassant/fr/txt output/fr -l fr -b spacy+llm \\
	  -m claude-sonnet-4-5 --mode full -p prompt_fr.txt
"""
	)
	ap.add_argument("input_dir", help="Directory containing .txt files")
	ap.add_argument("output_dir", help="Output directory for annotations")
	ap.add_argument("--lang", "-l", choices=["fr", "en"], required=True)
	ap.add_argument("--format", "-f", choices=["json", "conllu"], default="conllu")
	ap.add_argument("--backend", "-b", choices=["stanza", "stanza+corenlp", "corenlp", "spacy", "spacy+llm"], default="spacy")
	ap.add_argument("--model", "-m", default=None, help="LLM model (claude-* or gpt-*)")
	ap.add_argument("--limit", "-n", type=int, help="Process only N documents")
	ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
	ap.add_argument("--prompt", "-p", help="Path to LLM prompt file")
	ap.add_argument("--chunks", "-c", type=int, help="Sample N chunks per document")
	ap.add_argument("--chunk-size", type=int, default=20, help="Sentences per chunk (default: 20)")
	ap.add_argument("--seed", "-s", type=int, help="Random seed for reproducibility")
	ap.add_argument("--ssplit", choices=["always", "never", "two"], default="two",
					help="Newline sentence break mode (default: two)")
	ap.add_argument("--threads", "-t", type=int, default=5, help="CoreNLP threads (default: 5)")
	ap.add_argument("--corenlp-url", default=None,
					help="URL of a running CoreNLP server (e.g. http://localhost:9000). "
					     "Bypasses stanza's CoreNLPClient.")
	ap.add_argument("--mode", choices=["full", "surface"], default="surface",
					help="LLM correction mode: full (structural) or surface (lemma/upos/feats) (default: surface)")
	args = ap.parse_args()
	
	# Validation
	if args.backend in ("corenlp", "stanza+corenlp", "spacy", "spacy+llm"):
		if not args.corenlp_url and not os.environ.get("CORENLP_HOME"):
			sys.exit("Error: CORENLP_HOME not set (or use --corenlp-url)")
	
	if args.backend == "spacy+llm":
		if not args.model:
			sys.exit("Error: --model required for spacy+llm backend")
		if args.model.startswith("claude-") and not os.environ.get("ANTHROPIC_API_KEY"):
			sys.exit("Error: ANTHROPIC_API_KEY not set")
		if args.model.startswith(("gpt-", "o1", "o3")) and not os.environ.get("OPENAI_API_KEY"):
			sys.exit("Error: OPENAI_API_KEY not set")
	
	# Status
	print(f"Backend: {args.backend}", file=sys.stderr)
	print(f"Language: {args.lang}", file=sys.stderr)
	print(f"Input: {args.input_dir}", file=sys.stderr)
	print(f"Output: {args.output_dir}", file=sys.stderr)
	if args.backend in ("stanza+corenlp", "spacy", "spacy+llm", "corenlp"):
		print(f"Sentence split: newline={args.ssplit}, threads={args.threads}", file=sys.stderr)
		if args.corenlp_url:
			print(f"CoreNLP server: {args.corenlp_url}", file=sys.stderr)
	if args.backend == "spacy+llm":
		print(f"LLM: {args.model}, mode={args.mode}", file=sys.stderr)
	if args.chunks:
		print(f"Sampling: {args.chunks} × {args.chunk_size} sentences, seed={args.seed}", file=sys.stderr)
	print("", file=sys.stderr)
	
	# Dispatch
	if args.backend == "stanza":
		process_stanza(
			args.input_dir, args.output_dir, args.lang, args.format, args.limit, args.overwrite
		)
	elif args.backend == "stanza+corenlp":
		process_stanza_corenlp(
			args.input_dir, args.output_dir, args.lang, args.format, args.limit, args.overwrite, args.ssplit, args.threads, args.corenlp_url
		)
	elif args.backend == "corenlp":
		process_corenlp(
			args.input_dir, args.output_dir, args.lang, args.format, args.limit, args.overwrite, args.corenlp_url
		)
	elif args.backend == "spacy":
		process_spacy(
			args.input_dir, args.output_dir, args.lang, args.format, args.limit, args.overwrite, args.ssplit, args.threads, args.corenlp_url
		)
	elif args.backend == "spacy+llm":
		process_spacy_llm(
			args.input_dir, args.output_dir, args.lang, args.format, args.model,
			args.limit, args.overwrite, args.prompt, args.chunks, args.chunk_size, args.seed,
			args.ssplit, args.threads, args.mode, args.corenlp_url
		)
	
	print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
	main()
