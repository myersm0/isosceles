#!/usr/bin/env python3
"""
Sample and annotate chunks from ELTeC novels for training data generation.

Samples random chunks of consecutive sentences, preserving metadata for
stratification by decade, author, and other dimensions.

Usage:
  # Sample 10 chunks from a single novel (for testing)
  python sample_eltec.py novels/FRA001.txt output/ --chunks 10 --model claude-sonnet-4-20250514
  
  # Sample from all novels in a directory
  python sample_eltec.py novels/ output/ --chunks 100 --model claude-sonnet-4-20250514
  
  # Dry run to see what would be sampled
  python sample_eltec.py novels/ output/ --chunks 50 --dry-run
  
  # Use Opus for gold annotations
  python sample_eltec.py novels/ output/ --chunks 20 --model claude-opus-4-20250514 --tier gold
"""

import argparse
import json
import os
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime


# -----------------------------------------------------------------------------
# ELTeC metadata extraction
# -----------------------------------------------------------------------------

def extract_eltec_metadata(filepath):
	"""Extract metadata from ELTeC TEI XML file."""
	try:
		tree = ET.parse(filepath)
		root = tree.getroot()
		
		# Handle TEI namespace
		ns = {"tei": "http://www.tei-c.org/ns/1.0"}
		
		# Try to extract metadata
		metadata = {
			"file": filepath.name,
			"file_id": filepath.stem,
		}
		
		# Title
		title_el = root.find(".//tei:title", ns) or root.find(".//title")
		if title_el is not None and title_el.text:
			metadata["title"] = title_el.text.strip()
		
		# Author
		author_el = root.find(".//tei:author", ns) or root.find(".//author")
		if author_el is not None:
			author_text = "".join(author_el.itertext()).strip()
			metadata["author"] = author_text
		
		# Date/decade
		date_el = root.find(".//tei:date", ns) or root.find(".//date")
		if date_el is not None:
			date_text = date_el.get("when") or date_el.text or ""
			year_match = re.search(r"(\d{4})", date_text)
			if year_match:
				year = int(year_match.group(1))
				metadata["year"] = year
				metadata["decade"] = (year // 10) * 10
		
		return metadata
		
	except ET.ParseError:
		# Not XML, try to infer from filename
		return {
			"file": filepath.name,
			"file_id": filepath.stem,
		}


def extract_text_from_eltec(filepath):
	"""Extract plain text from ELTeC TEI XML or plain text file."""
	content = filepath.read_text(encoding="utf-8")
	
	# Check if it's XML
	if content.strip().startswith("<?xml") or content.strip().startswith("<TEI"):
		try:
			tree = ET.parse(filepath)
			root = tree.getroot()
			ns = {"tei": "http://www.tei-c.org/ns/1.0"}
			
			# Find body text
			body = root.find(".//tei:body", ns) or root.find(".//body")
			if body is not None:
				# Extract all text content
				text = "".join(body.itertext())
				# Normalize whitespace
				text = re.sub(r"\s+", " ", text).strip()
				return text
		except ET.ParseError:
			pass
	
	# Plain text
	return content


# -----------------------------------------------------------------------------
# Sentence segmentation
# -----------------------------------------------------------------------------

def segment_sentences(text, lang="fr"):
	"""Segment text into sentences using CoreNLP."""
	from stanza.server import CoreNLPClient
	
	sentences = []
	
	# Chunk text to avoid CoreNLP limits
	chunks = chunk_text(text, max_chars=9000)
	
	with CoreNLPClient(
		annotators=["tokenize", "ssplit"],
		properties={"pipelineLanguage": lang},
		be_quiet=True
	) as client:
		for chunk in chunks:
			try:
				doc = client.annotate(chunk, properties={"pipelineLanguage": lang})
				for sent in doc.sentence:
					sent_text = "".join(t.word + t.after for t in sent.token).strip()
					sent_text = " ".join(sent_text.split())
					if sent_text:
						sentences.append(sent_text)
			except Exception as e:
				print(f"  Warning: CoreNLP error on chunk: {e}", file=sys.stderr)
				continue
	
	return sentences


def chunk_text(text, max_chars=9000):
	"""Split text into chunks for CoreNLP processing."""
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


# -----------------------------------------------------------------------------
# Chunk sampling
# -----------------------------------------------------------------------------

def sample_chunks(sentences, n_chunks, chunk_size_min=15, chunk_size_max=30):
	"""Sample random chunks of consecutive sentences."""
	if len(sentences) < chunk_size_min:
		return []
	
	chunks = []
	max_attempts = n_chunks * 10
	attempts = 0
	
	while len(chunks) < n_chunks and attempts < max_attempts:
		attempts += 1
		
		# Random chunk size
		chunk_size = random.randint(chunk_size_min, min(chunk_size_max, len(sentences)))
		
		# Random start position
		max_start = len(sentences) - chunk_size
		if max_start < 0:
			continue
		start = random.randint(0, max_start)
		
		# Check for overlap with existing chunks
		overlap = False
		for existing_start, existing_end in [(c["start"], c["end"]) for c in chunks]:
			if not (start + chunk_size <= existing_start or start >= existing_end):
				overlap = True
				break
		
		if not overlap:
			chunks.append({
				"start": start,
				"end": start + chunk_size,
				"sentences": sentences[start:start + chunk_size]
			})
	
	return chunks


def compute_chunk_stats(sentences):
	"""Compute statistics for stratification."""
	if not sentences:
		return {}
	
	# Sentence length distribution
	lengths = [len(s.split()) for s in sentences]
	avg_length = sum(lengths) / len(lengths)
	
	# Dialogue ratio (simple heuristic: quotes)
	quote_chars = sum(s.count('"') + s.count('«') + s.count('»') + s.count("'") for s in sentences)
	total_chars = sum(len(s) for s in sentences)
	dialogue_ratio = quote_chars / total_chars if total_chars > 0 else 0
	
	return {
		"n_sentences": len(sentences),
		"avg_sent_length": round(avg_length, 1),
		"dialogue_ratio": round(dialogue_ratio, 3),
	}


# -----------------------------------------------------------------------------
# Annotation (spaCy + LLM)
# -----------------------------------------------------------------------------

def load_spacy_model(lang):
	"""Load appropriate spaCy model."""
	import spacy
	
	model_name = "fr_dep_news_trf" if lang == "fr" else "en_core_web_trf"
	return spacy.load(model_name)


def spacy_doc_to_tokens(doc):
	"""Convert spaCy doc to token list."""
	tokens = []
	for i, tok in enumerate(doc):
		tokens.append({
			"id": i + 1,
			"form": tok.text,
			"lemma": tok.lemma_,
			"upos": tok.pos_,
			"xpos": tok.tag_,
			"feats": str(tok.morph) if tok.morph else "_",
			"head": tok.head.i + 1 if tok.head != tok else 0,
			"deprel": tok.dep_.lower(),
			"deps": "_",
			"misc": "_",
		})
	return tokens


def annotate_chunk(nlp, llm_client, model, prompt, sentences, lang):
	"""Annotate a chunk of sentences with spaCy + LLM corrections."""
	from annotate import (
		get_llm_corrections,
		apply_corrections,
		tokens_to_conllu
	)
	
	results = []
	total_corrections = 0
	total_input = 0
	total_output = 0
	
	for i, sent_text in enumerate(sentences):
		# spaCy parse
		doc = nlp(sent_text)
		tokens = spacy_doc_to_tokens(doc)
		
		# LLM corrections
		try:
			corrections, input_toks, output_toks = get_llm_corrections(
				llm_client, sent_text, tokens, lang, model, prompt
			)
			total_input += input_toks
			total_output += output_toks
			
			if corrections:
				total_corrections += len(corrections)
				tokens = apply_corrections(tokens, corrections)
		except Exception as e:
			print(f"    Warning: LLM error on sentence {i}: {e}", file=sys.stderr)
		
		results.append({
			"sent_text": sent_text,
			"tokens": tokens,
			"n_tokens": len(tokens),
		})
	
	return results, {
		"total_corrections": total_corrections,
		"input_tokens": total_input,
		"output_tokens": total_output,
	}


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def write_chunk_output(output_dir, chunk_id, metadata, sentences_data):
	"""Write annotated chunk to CoNLL-U file with metadata."""
	lines = []
	
	# Document-level metadata as comments
	lines.append(f"# newdoc id = {chunk_id}")
	for key, value in metadata.items():
		lines.append(f"# {key} = {value}")
	lines.append("")
	
	# Sentences
	for i, sent_data in enumerate(sentences_data):
		sent_id = f"{chunk_id}-{i+1}"
		sent_text = sent_data["sent_text"]
		tokens = sent_data["tokens"]
		
		lines.append(f"# sent_id = {sent_id}")
		lines.append(f"# text = {sent_text}")
		
		for t in tokens:
			row = [
				str(t["id"]),
				t["form"],
				t["lemma"],
				t["upos"],
				t.get("xpos", "_"),
				t.get("feats", "_"),
				str(t["head"]),
				t["deprel"],
				t.get("deps", "_"),
				t.get("misc", "_"),
			]
			lines.append("\t".join(row))
		lines.append("")
	
	out_path = output_dir / f"{chunk_id}.conllu"
	out_path.write_text("\n".join(lines), encoding="utf-8")
	return out_path


def write_manifest(output_dir, manifest):
	"""Write manifest JSON with all chunk metadata."""
	manifest_path = output_dir / "manifest.json"
	manifest_path.write_text(
		json.dumps(manifest, ensure_ascii=False, indent=2),
		encoding="utf-8"
	)
	return manifest_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
	ap = argparse.ArgumentParser(
		description="Sample and annotate chunks from ELTeC novels",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Sample 10 chunks from one novel (testing)
  %(prog)s novel.txt output/ --chunks 10 --model claude-sonnet-4-20250514
  
  # Sample from all novels in directory  
  %(prog)s novels/ output/ --chunks 100 --model claude-sonnet-4-20250514
  
  # Dry run
  %(prog)s novels/ output/ --chunks 50 --dry-run
  
  # Gold tier with Opus
  %(prog)s novels/ output/ --chunks 20 --model claude-opus-4-20250514 --tier gold
"""
	)
	ap.add_argument("input", help="Input file or directory of ELTeC novels")
	ap.add_argument("output_dir", help="Output directory for annotated chunks")
	ap.add_argument("--chunks", "-c", type=int, default=10, help="Number of chunks to sample")
	ap.add_argument("--chunk-min", type=int, default=15, help="Minimum sentences per chunk")
	ap.add_argument("--chunk-max", type=int, default=30, help="Maximum sentences per chunk")
	ap.add_argument("--lang", "-l", default="fr", choices=["fr", "en"], help="Language")
	ap.add_argument("--model", "-m", default=None, help="LLM model (claude-* or gpt-*)")
	ap.add_argument("--tier", "-t", default="silver", choices=["gold", "silver"], help="Annotation tier")
	ap.add_argument("--prompt", "-p", default=None, help="Path to LLM prompt file")
	ap.add_argument("--dry-run", action="store_true", help="Show what would be sampled without processing")
	ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
	args = ap.parse_args()
	
	if args.seed:
		random.seed(args.seed)
	
	# Collect input files
	input_path = Path(args.input)
	if input_path.is_file():
		input_files = [input_path]
	else:
		input_files = sorted(input_path.glob("*.txt")) + sorted(input_path.glob("*.xml"))
	
	if not input_files:
		print(f"Error: No .txt or .xml files found in {args.input}", file=sys.stderr)
		sys.exit(1)
	
	print(f"Input files: {len(input_files)}", file=sys.stderr)
	print(f"Chunks to sample: {args.chunks}", file=sys.stderr)
	print(f"Chunk size: {args.chunk_min}-{args.chunk_max} sentences", file=sys.stderr)
	print(f"Tier: {args.tier}", file=sys.stderr)
	
	if args.dry_run:
		print(f"Mode: DRY RUN", file=sys.stderr)
	elif args.model:
		print(f"Model: {args.model}", file=sys.stderr)
	else:
		print("Error: --model required unless --dry-run", file=sys.stderr)
		sys.exit(1)
	
	# Check environment
	if not args.dry_run:
		if not os.environ.get("CORENLP_HOME"):
			print("Error: CORENLP_HOME not set", file=sys.stderr)
			sys.exit(1)
		
		if args.model.startswith("claude-") and not os.environ.get("ANTHROPIC_API_KEY"):
			print("Error: ANTHROPIC_API_KEY not set", file=sys.stderr)
			sys.exit(1)
		elif args.model.startswith(("gpt-", "o1", "o3")) and not os.environ.get("OPENAI_API_KEY"):
			print("Error: OPENAI_API_KEY not set", file=sys.stderr)
			sys.exit(1)
	
	# Setup output
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	
	# Load prompt
	if not args.dry_run:
		if args.prompt:
			prompt = Path(args.prompt).read_text(encoding="utf-8")
		else:
			default_prompt = Path(__file__).parent / "prompt_fr.txt"
			if default_prompt.exists():
				prompt = default_prompt.read_text(encoding="utf-8")
			else:
				print(f"Error: No prompt file found at {default_prompt}", file=sys.stderr)
				sys.exit(1)
	
	# Phase 1: Extract text and sentences from all files
	print(f"\n=== Phase 1: Extracting sentences ===", file=sys.stderr)
	
	all_sources = []
	for filepath in input_files:
		print(f"  {filepath.name}...", file=sys.stderr, end=" ", flush=True)
		
		metadata = extract_eltec_metadata(filepath)
		text = extract_text_from_eltec(filepath)
		
		if not args.dry_run:
			sentences = segment_sentences(text, args.lang)
		else:
			# Rough estimate for dry run
			sentences = text.split(". ")
		
		print(f"{len(sentences)} sentences", file=sys.stderr)
		
		all_sources.append({
			"filepath": filepath,
			"metadata": metadata,
			"sentences": sentences,
			"n_sentences": len(sentences),
		})
	
	total_sentences = sum(s["n_sentences"] for s in all_sources)
	print(f"\nTotal sentences available: {total_sentences}", file=sys.stderr)
	
	# Phase 2: Sample chunks
	print(f"\n=== Phase 2: Sampling chunks ===", file=sys.stderr)
	
	# Distribute chunks across sources proportionally
	chunks_per_source = {}
	remaining = args.chunks
	
	for source in all_sources:
		# Proportional allocation
		proportion = source["n_sentences"] / total_sentences
		n = max(1, int(args.chunks * proportion))
		n = min(n, remaining, source["n_sentences"] // args.chunk_min)
		chunks_per_source[source["filepath"].name] = n
		remaining -= n
	
	# Distribute any remaining
	while remaining > 0:
		for source in all_sources:
			if remaining <= 0:
				break
			max_chunks = source["n_sentences"] // args.chunk_min
			if chunks_per_source[source["filepath"].name] < max_chunks:
				chunks_per_source[source["filepath"].name] += 1
				remaining -= 1
	
	all_chunks = []
	for source in all_sources:
		n_chunks = chunks_per_source[source["filepath"].name]
		if n_chunks == 0:
			continue
		
		chunks = sample_chunks(
			source["sentences"],
			n_chunks,
			args.chunk_min,
			args.chunk_max
		)
		
		for i, chunk in enumerate(chunks):
			chunk_id = f"{source['metadata']['file_id']}-chunk{i+1:03d}"
			chunk["id"] = chunk_id
			chunk["source"] = source["filepath"].name
			chunk["metadata"] = source["metadata"].copy()
			chunk["stats"] = compute_chunk_stats(chunk["sentences"])
			all_chunks.append(chunk)
		
		print(f"  {source['filepath'].name}: {len(chunks)} chunks", file=sys.stderr)
	
	print(f"\nTotal chunks sampled: {len(all_chunks)}", file=sys.stderr)
	
	if args.dry_run:
		print(f"\n=== Dry Run Summary ===", file=sys.stderr)
		for chunk in all_chunks[:5]:
			print(f"  {chunk['id']}: {len(chunk['sentences'])} sentences", file=sys.stderr)
		if len(all_chunks) > 5:
			print(f"  ... and {len(all_chunks) - 5} more", file=sys.stderr)
		
		# Estimate cost
		est_sentences = sum(len(c["sentences"]) for c in all_chunks)
		est_cost_sonnet = est_sentences * 0.0025
		est_cost_opus = est_sentences * 0.05
		print(f"\nEstimated sentences: {est_sentences}", file=sys.stderr)
		print(f"Estimated cost (Sonnet): ${est_cost_sonnet:.2f}", file=sys.stderr)
		print(f"Estimated cost (Opus): ${est_cost_opus:.2f}", file=sys.stderr)
		return
	
	# Phase 3: Annotate chunks
	print(f"\n=== Phase 3: Annotating chunks ===", file=sys.stderr)
	
	# Load models
	print("Loading spaCy model...", file=sys.stderr)
	nlp = load_spacy_model(args.lang)
	
	print("Initializing LLM client...", file=sys.stderr)
	if args.model.startswith("claude-"):
		import anthropic
		llm_client = anthropic.Anthropic()
	else:
		import openai
		llm_client = openai.OpenAI()
	
	manifest = {
		"created": datetime.now().isoformat(),
		"tier": args.tier,
		"model": args.model,
		"lang": args.lang,
		"chunk_size": f"{args.chunk_min}-{args.chunk_max}",
		"chunks": [],
	}
	
	total_input_tokens = 0
	total_output_tokens = 0
	total_corrections = 0
	
	for i, chunk in enumerate(all_chunks):
		print(f"\n[{i+1}/{len(all_chunks)}] {chunk['id']}...", file=sys.stderr)
		
		sentences_data, stats = annotate_chunk(
			nlp, llm_client, args.model, prompt,
			chunk["sentences"], args.lang
		)
		
		total_input_tokens += stats["input_tokens"]
		total_output_tokens += stats["output_tokens"]
		total_corrections += stats["total_corrections"]
		
		# Write output
		chunk_metadata = {
			**chunk["metadata"],
			"chunk_start": chunk["start"],
			"chunk_end": chunk["end"],
			"tier": args.tier,
			"model": args.model,
			**chunk["stats"],
		}
		
		out_path = write_chunk_output(output_dir, chunk["id"], chunk_metadata, sentences_data)
		print(f"  → {out_path.name} ({len(sentences_data)} sentences, {stats['total_corrections']} corrections)", file=sys.stderr)
		
		# Update manifest
		manifest["chunks"].append({
			"id": chunk["id"],
			"file": out_path.name,
			"source": chunk["source"],
			"n_sentences": len(sentences_data),
			"n_corrections": stats["total_corrections"],
			**chunk_metadata,
		})
	
	# Write manifest
	write_manifest(output_dir, manifest)
	
	# Summary
	print(f"\n=== Summary ===", file=sys.stderr)
	print(f"Chunks processed: {len(all_chunks)}", file=sys.stderr)
	print(f"Total sentences: {sum(len(c['sentences']) for c in all_chunks)}", file=sys.stderr)
	print(f"Total corrections: {total_corrections}", file=sys.stderr)
	print(f"Input tokens: {total_input_tokens:,}", file=sys.stderr)
	print(f"Output tokens: {total_output_tokens:,}", file=sys.stderr)
	
	# Cost estimate
	if args.model.startswith("claude-"):
		if "opus" in args.model:
			input_cost = total_input_tokens * 15 / 1_000_000
			output_cost = total_output_tokens * 75 / 1_000_000
		else:  # sonnet
			input_cost = total_input_tokens * 3 / 1_000_000
			output_cost = total_output_tokens * 15 / 1_000_000
		print(f"Estimated cost: ${input_cost + output_cost:.2f}", file=sys.stderr)
	
	print(f"\nOutput: {output_dir}", file=sys.stderr)
	print(f"Manifest: {output_dir / 'manifest.json'}", file=sys.stderr)


if __name__ == "__main__":
	main()
