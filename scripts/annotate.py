#!/usr/bin/env python3
"""
Annotation pipeline for the isosceles corpus.

Backends:
  stanza      - Stanza for tokenization and parsing (baseline)
  corenlp     - CoreNLP for everything (requires CORENLP_HOME)
  spacy       - CoreNLP segmentation + spaCy parsing
  spacy+llm   - CoreNLP segmentation + spaCy parsing + LLM corrections

Examples:
  python annotate.py data/maupassant/fr/txt data/maupassant/fr/conllu -l fr -b stanza
  python annotate.py data/maupassant/fr/txt data/maupassant/fr/conllu -l fr -b spacy
  python annotate.py data/maupassant/fr/txt data/maupassant/fr/conllu -l fr -b spacy+llm
  python annotate.py data/maupassant/fr/txt data/maupassant/fr/conllu -l fr -b spacy+llm -m gpt-4o
  python annotate.py data/poe/en/txt data/poe/en/conllu -l en -b spacy
  
  # Test on 5 random documents first
  python annotate.py data/maupassant/fr/txt output/fr -l fr -b spacy+llm --limit 5
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path


# -----------------------------------------------------------------------------
# LLM correction system prompt
# -----------------------------------------------------------------------------

llm_system_prompt = """
You are an expert linguist reviewing Universal Dependencies (UD v2) annotations.

You will receive a French sentence and its CoNLL-U parse.

Your task is to identify annotation ERRORS and output only the necessary corrections as JSON.

CRITICAL RULES:
- Make the MINIMUM number of corrections needed.
- Only modify tokens listed in the corrections output.
- Use only valid UD v2 POS tags, dependency labels, and subtypes listed below.
- Do not invent labels or subtypes.
- If unsure, do NOT change the annotation.

PRIORITY (resolve in this order):
1. Tree validity (single ROOT, no cycles, all tokens connected)
2. Wrong HEAD attachment
3. Wrong DEPREL
4. Wrong UPOS
5. Wrong lemma

You may output multiple corrections for the same token if needed.

=== ERROR CATEGORIES TO CHECK ===

STRUCTURE:
- Wrong HEAD attachment (especially PP attachment, coordination scope, relative clause attachment)
- Tree validity: exactly one ROOT, no cycles, all tokens reachable from ROOT

DEPENDENCY LABELS:
- appos vs flat:name
- obj vs iobj
- advmod vs obl
- cop vs aux
- mark vs case
- nmod vs obl
- acl vs advcl
- dislocated for left/right dislocation
- fixed / flat / compound for multiword expressions

FRENCH-SPECIFIC:
- Clitics and elision: j'→je, c'→ce, l'→le/la, m'→me, t'→te, s'→se, d'→de, n'→ne, qu'→que
- Reflexive/pronominal verbs: use expl:pv or expl:comp
- Causative constructions: faire + infinitive
- Relative pronouns: qui (nsubj/obj), que (obj/mark), dont (nmod/obl), où (advmod/obl)
- Auxiliary selection: être vs avoir in compound tenses
- Clitic doubling vs true argument pronouns
- Past participle: adjectival (ADJ) vs verbal (VERB) use

POS TAGS:
- ADJ vs VERB for participles
- ADV vs ADP
- AUX vs VERB
- DET vs PRON

LEMMA:
- Irregular verb lemmas
- Clitic normalization after elision

=== VALID UD V2 UNIVERSAL POS TAGS ===

ADJ (adjective), ADP (adposition), ADV (adverb), AUX (auxiliary), CCONJ (coordinating conjunction), DET (determiner), INTJ (interjection), NOUN (noun), NUM (numeral), PART (particle), PRON (pronoun), PROPN (proper noun), PUNCT (punctuation), SCONJ (subordinating conjunction), SYM (symbol), VERB (verb), X (other)

=== VALID UD V2 DEPENDENCY RELATIONS ===

Core arguments:
- nsubj: nominal subject
- obj: direct object
- iobj: indirect object
- csubj: clausal subject
- ccomp: clausal complement
- xcomp: open clausal complement

Non-core dependents:
- obl: oblique nominal (adjuncts, agents in passive)
- vocative: vocative
- expl: expletive
- dislocated: dislocated elements
- advcl: adverbial clause modifier
- advmod: adverbial modifier
- discourse: discourse element
- aux: auxiliary
- cop: copula
- mark: marker (subordinating conjunction)
- nmod: nominal modifier
- appos: appositional modifier
- nummod: numeric modifier
- acl: adnominal clause
- amod: adjectival modifier
- det: determiner
- clf: classifier
- case: case marking (prepositions)

Coordination:
- conj: conjunct
- cc: coordinating conjunction

MWE and special:
- fixed: fixed multiword expression
- flat: flat multiword expression (names, etc.)
- compound: compound

Loose joining:
- list: list
- parataxis: parataxis
- orphan: orphan in ellipsis
- goeswith: goes with (orthographic errors)
- reparandum: overridden disfluency

Other:
- punct: punctuation
- root: root of sentence
- dep: unspecified dependency

=== FRENCH-SPECIFIC SUBTYPES ===

- nsubj:pass, nsubj:caus (passive/causative subjects)
- obj:agent, obj:lvc (agent, light verb construction)
- iobj:agent (indirect object agent)
- obl:agent, obl:arg, obl:mod (oblique subtypes)
- expl:comp, expl:subj, expl:pass, expl:pv (expletive subtypes)
- aux:pass, aux:caus, aux:tense (auxiliary subtypes)
- acl:relcl (relative clause)
- advcl:cleft (cleft construction)
- flat:name, flat:foreign (flat subtypes)
- compound:prt (particle verbs, rare in French)

=== OUTPUT FORMAT ===

Output a JSON object with a "corrections" array:

{
  "corrections": [
    {"id": <token_id>, "field": "head", "value": <new_head_id>},
    {"id": <token_id>, "field": "deprel", "value": "<new_deprel>"},
    {"id": <token_id>, "field": "upos", "value": "<new_upos>"},
    {"id": <token_id>, "field": "lemma", "value": "<new_lemma>"}
  ]
}

If no errors are found:
{"corrections": []}

Output ONLY valid JSON. No commentary.
"""


# -----------------------------------------------------------------------------
# Text chunking (for CoreNLP's 10k character limit)
# -----------------------------------------------------------------------------

def chunk_text(text, max_chars=9000):
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
			print(f"  Skipping {skipped} files with existing output (use --overwrite to replace)", file=sys.stderr)
	else:
		files = list(all_files)
	
	if limit and limit < len(files):
		random.shuffle(files)
		files = files[:limit]
		files = sorted(files)
	
	return files


# -----------------------------------------------------------------------------
# Output formatters
# -----------------------------------------------------------------------------

def tokens_to_conllu(tokens, sent_id, sent_text):
	lines = [
		f"# sent_id = {sent_id}",
		f"# text = {sent_text}",
	]
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
	return "\n".join(lines)


def tokens_to_json_sentence(tokens, sent_text):
	deps = []
	for t in tokens:
		gov_tok = next((x for x in tokens if x["id"] == t["head"]), None)
		deps.append({
			"dependent": t["id"],
			"dependentGloss": t["form"],
			"dependentLemma": t["lemma"],
			"governor": t["head"],
			"governorGloss": gov_tok["form"] if gov_tok else "ROOT",
			"governorLemma": gov_tok["lemma"] if gov_tok else "ROOT",
			"dep": t["deprel"],
		})
	return {
		"text": sent_text,
		"tokens": tokens,
		"basicDependencies": deps,
	}


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
	
	print("Downloading Stanza models...", file=sys.stderr)
	stanza.download(lang, processors="tokenize,pos,lemma,depparse", verbose=False)
	nlp = stanza.Pipeline(lang, processors="tokenize,pos,lemma,depparse", verbose=False)
	
	ext = ".json" if fmt == "json" else ".conllu"
	
	for filepath in files:
		print(f"  {filepath.name}", file=sys.stderr)
		text = filepath.read_text(encoding="utf-8")
		doc = nlp(text)
		
		sentences = []
		for sent_idx, sent in enumerate(doc.sentences):
			tokens = []
			for word in sent.words:
				tokens.append({
					"id": word.id,
					"form": word.text,
					"lemma": word.lemma or "_",
					"upos": word.upos or "_",
					"xpos": word.xpos or "_",
					"feats": "_",
					"head": word.head,
					"deprel": word.deprel or "_",
					"deps": "_",
					"misc": "_",
				})
			sentences.append((f"{filepath.stem}-{sent_idx + 1}", sent.text, tokens))
		
		out_path = output_dir / (filepath.stem + ext)
		write_output(out_path, filepath.stem, sentences, fmt)


# -----------------------------------------------------------------------------
# CoreNLP backend
# -----------------------------------------------------------------------------

def process_corenlp(input_dir, output_dir, lang, fmt, limit=None, overwrite=False):
	from stanza.server import CoreNLPClient
	
	input_dir = Path(input_dir)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	
	files = select_files(input_dir, output_dir, fmt, limit, overwrite)
	if not files:
		print("  No files to process", file=sys.stderr)
		return
	
	ext = ".json" if fmt == "json" else ".conllu"
	
	with CoreNLPClient(
		annotators=["tokenize", "ssplit", "pos", "lemma", "depparse"],
		properties={"pipelineLanguage": lang},
		be_quiet=True
	) as client:
		for filepath in files:
			print(f"  {filepath.name}", file=sys.stderr)
			text = filepath.read_text(encoding="utf-8")
			
			sentences = []
			sent_idx = 0
			
			for chunk in chunk_text(text):
				doc = client.annotate(chunk, properties={"pipelineLanguage": lang})
				
				for sent in doc.sentence:
					sent_idx += 1
					sent_text = "".join(t.word + t.after for t in sent.token).strip()
					sent_text = " ".join(sent_text.split())  # normalize whitespace
					
					deps = {d.target: (d.source, d.dep) for d in sent.basicDependencies.edge}
					tokens = []
					for tok in sent.token:
						head, deprel = deps.get(tok.tokenEndIndex, (0, "root"))
						tokens.append({
							"id": tok.tokenEndIndex,
							"form": tok.word,
							"lemma": tok.lemma or "_",
							"upos": tok.pos or "_",
							"xpos": tok.pos or "_",
							"feats": "_",
							"head": head,
							"deprel": deprel,
							"deps": "_",
							"misc": "_",
						})
					sentences.append((f"{filepath.stem}-{sent_idx}", sent_text, tokens))
			
			out_path = output_dir / (filepath.stem + ext)
			write_output(out_path, filepath.stem, sentences, fmt)


# -----------------------------------------------------------------------------
# spaCy backend (CoreNLP segmentation + spaCy parsing)
# -----------------------------------------------------------------------------

def process_spacy(input_dir, output_dir, lang, fmt, limit=None, overwrite=False):
	import spacy
	from spacy.language import Language
	from stanza.server import CoreNLPClient
	
	@Language.component("single_sentence")
	def single_sentence(doc):
		for token in doc:
			token.is_sent_start = False
		doc[0].is_sent_start = True
		return doc
	
	input_dir = Path(input_dir)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	
	files = select_files(input_dir, output_dir, fmt, limit, overwrite)
	if not files:
		print("  No files to process", file=sys.stderr)
		return
	
	print("Loading spaCy model...", file=sys.stderr)
	model_name = "fr_dep_news_trf" if lang == "fr" else "en_core_web_trf"
	nlp = spacy.load(model_name, disable=["ner"])
	nlp.add_pipe("single_sentence", before="parser")
	
	ext = ".json" if fmt == "json" else ".conllu"
	
	with CoreNLPClient(
		annotators=["tokenize", "ssplit"],
		properties={"pipelineLanguage": lang},
		be_quiet=True
	) as client:
		for filepath in files:
			print(f"  {filepath.name}", file=sys.stderr)
			text = filepath.read_text(encoding="utf-8")
			
			sentences = []
			sent_idx = 0
			
			for chunk in chunk_text(text):
				doc = client.annotate(chunk, properties={"pipelineLanguage": lang})
				
				for sent in doc.sentence:
					sent_idx += 1
					sent_text = "".join(t.word + t.after for t in sent.token).strip()
					sent_text = " ".join(sent_text.split())  # normalize whitespace
					
					if not sent_text:
						continue
					
					spacy_doc = nlp(sent_text)
					tokens = spacy_doc_to_tokens(spacy_doc)
					sentences.append((f"{filepath.stem}-{sent_idx}", sent_text, tokens))
			
			out_path = output_dir / (filepath.stem + ext)
			write_output(out_path, filepath.stem, sentences, fmt)


def spacy_doc_to_tokens(doc):
	tokens = []
	non_space = [(i, tok) for i, tok in enumerate(doc) if tok.pos_ != "SPACE"]
	old_to_new = {old_i: new_i for new_i, (old_i, _) in enumerate(non_space, start=1)}
	
	for new_i, (old_i, tok) in enumerate(non_space, start=1):
		if tok.head == tok:
			head = 0
		else:
			head_old = tok.head.i
			head = old_to_new.get(head_old, 0)
		
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


# -----------------------------------------------------------------------------
# spaCy + LLM backend (CoreNLP seg + spaCy parse + LLM corrections)
# -----------------------------------------------------------------------------

def process_spacy_llm(input_dir, output_dir, lang, fmt, model, limit=None, overwrite=False):
	import spacy
	from spacy.language import Language
	from stanza.server import CoreNLPClient
	
	@Language.component("single_sentence_llm")
	def single_sentence_llm(doc):
		for token in doc:
			token.is_sent_start = False
		doc[0].is_sent_start = True
		return doc
	
	input_dir = Path(input_dir)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	
	files = select_files(input_dir, output_dir, fmt, limit, overwrite)
	if not files:
		print("  No files to process", file=sys.stderr)
		return
	
	print("Loading spaCy model...", file=sys.stderr)
	spacy_model = "fr_dep_news_trf" if lang == "fr" else "en_core_web_trf"
	nlp = spacy.load(spacy_model, disable=["ner"])
	nlp.add_pipe("single_sentence_llm", before="parser")
	
	# Initialize LLM client based on model prefix
	if model.startswith("claude-"):
		import anthropic
		llm_client = anthropic.Anthropic()
	else:
		import openai
		llm_client = openai.OpenAI()
	
	ext = ".json" if fmt == "json" else ".conllu"
	
	total_input = 0
	total_output = 0
	total_corrections = 0
	
	with CoreNLPClient(
		annotators=["tokenize", "ssplit"],
		properties={"pipelineLanguage": lang},
		be_quiet=True
	) as client:
		for filepath in files:
			print(f"  {filepath.name}", file=sys.stderr)
			text = filepath.read_text(encoding="utf-8")
			
			sentences = []
			sent_idx = 0
			
			for chunk in chunk_text(text):
				doc = client.annotate(chunk, properties={"pipelineLanguage": lang})
				
				for sent in doc.sentence:
					sent_idx += 1
					sent_text = "".join(t.word + t.after for t in sent.token).strip()
					sent_text = " ".join(sent_text.split())
					
					if not sent_text:
						continue
					
					spacy_doc = nlp(sent_text)
					tokens = spacy_doc_to_tokens(spacy_doc)
					
					try:
						corrections, input_toks, output_toks = get_llm_corrections(
							llm_client, sent_text, tokens, lang, model
						)
						
						total_input += input_toks
						total_output += output_toks
						
						if corrections:
							total_corrections += len(corrections)
							tokens = apply_corrections(tokens, corrections)
							print(f"    [{sent_idx}] {len(corrections)} corrections", file=sys.stderr)
						
					except Exception as e:
						print(f"    [{sent_idx}] ERROR: {e}", file=sys.stderr)
					
					sentences.append((f"{filepath.stem}-{sent_idx}", sent_text, tokens))
			
			out_path = output_dir / (filepath.stem + ext)
			write_output(out_path, filepath.stem, sentences, fmt)
	
	print(f"\n=== Token Usage ===", file=sys.stderr)
	print(f"Input tokens:  {total_input:,}", file=sys.stderr)
	print(f"Output tokens: {total_output:,}", file=sys.stderr)
	print(f"Total corrections: {total_corrections}", file=sys.stderr)


def get_llm_corrections(client, sent_text, tokens, lang, model):
	"""Dispatch to appropriate provider based on model prefix."""
	if model.startswith("claude-"):
		return _anthropic_corrections(client, sent_text, tokens, lang, model)
	else:
		return _openai_corrections(client, sent_text, tokens, lang, model)


def _format_parse_for_llm(tokens, lang):
	"""Format tokens as simplified parse string for LLM input."""
	lang_name = "French" if lang == "fr" else "English"
	parse_lines = []
	for t in tokens:
		parse_lines.append(f"{t['id']}\t{t['form']}\t{t['lemma']}\t{t['upos']}\t{t['head']}\t{t['deprel']}")
	parse_str = "\n".join(parse_lines)
	return lang_name, parse_str


def _parse_corrections_json(response_text):
	"""Extract corrections from LLM response text."""
	match = re.search(r"\{[\s\S]*\}", response_text)
	if match:
		try:
			data = json.loads(match.group())
			return data.get("corrections", [])
		except json.JSONDecodeError:
			return []
	return []


def _anthropic_corrections(client, sent_text, tokens, lang, model):
	"""Get corrections using Anthropic API."""
	lang_name, parse_str = _format_parse_for_llm(tokens, lang)
	
	response = client.messages.create(
		model=model,
		max_tokens=1024,
		system=[
			{
				"type": "text",
				"text": llm_system_prompt,
				"cache_control": {"type": "ephemeral"}
			}
		],
		messages=[
			{
				"role": "user",
				"content": f"{lang_name} sentence:\n{sent_text}\n\nParse:\n{parse_str}"
			}
		]
	)
	
	response_text = response.content[0].text
	corrections = _parse_corrections_json(response_text)
	
	return corrections, response.usage.input_tokens, response.usage.output_tokens


def _openai_corrections(client, sent_text, tokens, lang, model):
	"""Get corrections using OpenAI API."""
	lang_name, parse_str = _format_parse_for_llm(tokens, lang)
	
	response = client.chat.completions.create(
		model=model,
		max_tokens=1024,
		messages=[
			{
				"role": "system",
				"content": llm_system_prompt
			},
			{
				"role": "user",
				"content": f"{lang_name} sentence:\n{sent_text}\n\nParse:\n{parse_str}"
			}
		]
	)
	
	response_text = response.choices[0].message.content
	corrections = _parse_corrections_json(response_text)
	
	return corrections, response.usage.prompt_tokens, response.usage.completion_tokens


def apply_corrections(tokens, corrections):
	token_map = {t["id"]: t for t in tokens}
	
	for c in corrections:
		tok_id = c.get("id")
		field = c.get("field")
		value = c.get("value")
		
		if tok_id in token_map and field in ("head", "deprel", "upos", "lemma"):
			token_map[tok_id][field] = value
	
	return tokens


# -----------------------------------------------------------------------------
# Output writer
# -----------------------------------------------------------------------------

def write_output(out_path, doc_id, sentences, fmt):
	if fmt == "json":
		result = {
			"doc_id": doc_id,
			"sentences": [tokens_to_json_sentence(tokens, text) for _, text, tokens in sentences]
		}
		out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
	else:
		lines = []
		for sent_id, sent_text, tokens in sentences:
			lines.append(tokens_to_conllu(tokens, sent_id, sent_text))
		out_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
	ap = argparse.ArgumentParser(
		description="Annotation pipeline for the isosceles corpus",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Backends:
  stanza      Stanza tokenization + parsing (baseline)
  corenlp     CoreNLP for everything (requires CORENLP_HOME)
  spacy       CoreNLP segmentation + spaCy parsing
  spacy+llm   CoreNLP segmentation + spaCy parsing + LLM corrections

Examples:
  %(prog)s data/maupassant/fr/txt output/fr -l fr -b stanza
  %(prog)s data/maupassant/fr/txt output/fr -l fr -b spacy
  %(prog)s data/maupassant/fr/txt output/fr -l fr -b spacy+llm
  %(prog)s data/maupassant/fr/txt output/fr -l fr -b spacy+llm -m gpt-4o
  %(prog)s data/poe/en/txt output/en -l en -b spacy
  
  # Test on 5 random documents first
  %(prog)s data/maupassant/fr/txt output/fr -l fr -b spacy+llm --limit 5
"""
	)
	ap.add_argument("input_dir", help="Directory containing .txt files")
	ap.add_argument("output_dir", help="Output directory for annotations")
	ap.add_argument("--lang", "-l", choices=["fr", "en"], required=True, help="Language")
	ap.add_argument("--format", "-f", choices=["json", "conllu"], default="conllu", help="Output format")
	ap.add_argument(
		"--backend", "-b",
		choices=["stanza", "corenlp", "spacy", "spacy+llm"],
		default="spacy",
		help="Annotation backend"
	)
	ap.add_argument(
		"--model", "-m",
		default=None,
		help="LLM model for corrections (only with spacy+llm). Supports claude-* and gpt-*/o1-*/o3-*"
	)
	ap.add_argument(
		"--limit", "-n",
		type=int,
		default=None,
		help="Process only N documents (random selection for sampling)"
	)
	ap.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing output files (default: skip existing)"
	)
	args = ap.parse_args()
	
	if args.backend in ("corenlp", "spacy", "spacy+llm"):
		if not os.environ.get("CORENLP_HOME"):
			print("Error: CORENLP_HOME environment variable not set", file=sys.stderr)
			print("CoreNLP is required for sentence segmentation.", file=sys.stderr)
			sys.exit(1)
	
	if args.backend == "spacy+llm":
		if args.model.startswith("claude-"):
			if not os.environ.get("ANTHROPIC_API_KEY"):
				print("Error: ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
				sys.exit(1)
		elif args.model.startswith(("gpt-", "o1", "o3")):
			if not os.environ.get("OPENAI_API_KEY"):
				print("Error: OPENAI_API_KEY environment variable not set", file=sys.stderr)
				sys.exit(1)
		else:
			print(f"Error: Unknown model provider for '{args.model}'", file=sys.stderr)
			print("Model must start with 'claude-', 'gpt-', 'o1', or 'o3'", file=sys.stderr)
			sys.exit(1)
	
	print(f"Backend: {args.backend}", file=sys.stderr)
	print(f"Language: {args.lang}", file=sys.stderr)
	print(f"Format: {args.format}", file=sys.stderr)
	print(f"Input: {args.input_dir}", file=sys.stderr)
	print(f"Output: {args.output_dir}", file=sys.stderr)
	if args.limit:
		print(f"Limit: {args.limit} documents (random)", file=sys.stderr)
	if not args.overwrite:
		print(f"Overwrite: off (use --overwrite to replace existing)", file=sys.stderr)
	print("", file=sys.stderr)
	
	if args.backend == "stanza":
		process_stanza(args.input_dir, args.output_dir, args.lang, args.format, args.limit, args.overwrite)
	elif args.backend == "corenlp":
		process_corenlp(args.input_dir, args.output_dir, args.lang, args.format, args.limit, args.overwrite)
	elif args.backend == "spacy":
		process_spacy(args.input_dir, args.output_dir, args.lang, args.format, args.limit, args.overwrite)
	elif args.backend == "spacy+llm":
		process_spacy_llm(args.input_dir, args.output_dir, args.lang, args.format, args.model, args.limit, args.overwrite)
	
	print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
	main()
