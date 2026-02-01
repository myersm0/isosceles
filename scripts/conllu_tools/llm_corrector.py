#!/usr/bin/env python3
"""
LLM-based correction for UD parses.
"""

import copy
import json
import re
import sys
from typing import Optional

try:
	from .validator import validate_tree, apply_deterministic_fixes
except ImportError:
	from validator import validate_tree, apply_deterministic_fixes


def get_llm_client(model: str):
	"""Initialize appropriate LLM client based on model name."""
	if model.startswith("claude-"):
		import anthropic
		return anthropic.Anthropic()
	else:
		import openai
		return openai.OpenAI()


def format_parse_for_llm(tokens: list[dict], lang: str) -> tuple[str, str]:
	"""Format tokens for LLM input. Returns (lang_name, parse_string)."""
	lang_name = "French" if lang == "fr" else "English"
	lines = [
		f"{t['id']}\t{t['form']}\t{t['lemma']}\t{t['upos']}\t{t['head']}\t{t['deprel']}"
		for t in tokens
	]
	return lang_name, "\n".join(lines)


def parse_corrections_json(response_text: str) -> list[dict]:
	"""Extract corrections from LLM response JSON."""
	match = re.search(r"\{[\s\S]*\}", response_text)
	if match:
		try:
			return json.loads(match.group()).get("corrections", [])
		except json.JSONDecodeError:
			pass
	return []


def call_anthropic(client, sent_text: str, tokens: list[dict], lang: str, model: str, prompt: str) -> tuple[list[dict], int, int]:
	"""Call Anthropic API for corrections."""
	lang_name, parse_str = format_parse_for_llm(tokens, lang)
	
	response = client.messages.create(
		model=model,
		max_tokens=1024,
		system=[{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}],
		messages=[{"role": "user", "content": f"{lang_name} sentence:\n{sent_text}\n\nParse:\n{parse_str}"}]
	)
	
	corrections = parse_corrections_json(response.content[0].text)
	return corrections, response.usage.input_tokens, response.usage.output_tokens


def call_openai(client, sent_text: str, tokens: list[dict], lang: str, model: str, prompt: str) -> tuple[list[dict], int, int]:
	"""Call OpenAI API for corrections."""
	lang_name, parse_str = format_parse_for_llm(tokens, lang)
	
	response = client.chat.completions.create(
		model=model,
		max_completion_tokens=1024,
		messages=[
			{"role": "system", "content": prompt},
			{"role": "user", "content": f"{lang_name} sentence:\n{sent_text}\n\nParse:\n{parse_str}"}
		]
	)
	
	corrections = parse_corrections_json(response.choices[0].message.content)
	return corrections, response.usage.prompt_tokens, response.usage.completion_tokens


def get_llm_corrections(client, sent_text: str, tokens: list[dict], lang: str, model: str, prompt: str) -> tuple[list[dict], int, int]:
	"""Get corrections from LLM. Returns (corrections, input_tokens, output_tokens)."""
	if model.startswith("claude-"):
		return call_anthropic(client, sent_text, tokens, lang, model, prompt)
	else:
		return call_openai(client, sent_text, tokens, lang, model, prompt)


def apply_corrections(tokens: list[dict], corrections: list[dict], verbose: bool = False) -> tuple[list[dict], int]:
	"""
	Apply corrections to token list.
	
	Returns: (corrected_tokens, n_applied)
	
	IMPORTANT: This makes a deep copy and handles type conversion for token IDs.
	"""
	tokens = copy.deepcopy(tokens)
	token_map = {t["id"]: t for t in tokens}
	n_applied = 0
	
	for c in corrections:
		tok_id = c.get("id")
		field = c.get("field")
		value = c.get("value")
		
		if tok_id is None or field is None or value is None:
			if verbose:
				print(f"    Skipping malformed correction: {c}", file=sys.stderr)
			continue
		
		try:
			tok_id = int(tok_id)
		except (ValueError, TypeError):
			if verbose:
				print(f"    Skipping invalid token ID: {tok_id}", file=sys.stderr)
			continue
		
		if tok_id not in token_map:
			if verbose:
				print(f"    Skipping non-existent token ID: {tok_id}", file=sys.stderr)
			continue
		
		if field not in ("head", "deprel", "upos", "lemma", "xpos", "feats"):
			if verbose:
				print(f"    Skipping unknown field: {field}", file=sys.stderr)
			continue
		
		tok = token_map[tok_id]
		
		if field == "head":
			try:
				value = int(value)
			except (ValueError, TypeError):
				if verbose:
					print(f"    Skipping invalid head value: {value}", file=sys.stderr)
				continue
		
		tok[field] = value
		n_applied += 1
	
	return tokens, n_applied


def correct_sentence(
	client,
	sent_text: str,
	tokens: list[dict],
	lang: str,
	model: str,
	prompt: str,
	validate: bool = True,
	max_retries: int = 1,
	verbose: bool = False
) -> tuple[list[dict], dict]:
	"""
	Apply LLM corrections to a single sentence with optional validation.
	
	Returns: (corrected_tokens, stats_dict)
	
	Stats dict contains:
		- input_tokens: API input tokens
		- output_tokens: API output tokens
		- corrections_suggested: total corrections from LLM
		- corrections_applied: corrections actually applied
		- retries: number of retries due to validation failure
		- outcome: "accepted" | "rejected_cycles" | "error"
	"""
	stats = {
		"input_tokens": 0,
		"output_tokens": 0,
		"corrections_suggested": 0,
		"corrections_applied": 0,
		"retries": 0,
		"outcome": "accepted",
	}
	
	original_tokens = copy.deepcopy(tokens)
	current_tokens = tokens
	
	for attempt in range(1 + max_retries):
		try:
			corrections, in_toks, out_toks = get_llm_corrections(
				client, sent_text, current_tokens, lang, model, prompt
			)
			
			stats["input_tokens"] += in_toks
			stats["output_tokens"] += out_toks
			stats["corrections_suggested"] += len(corrections)
			
			if not corrections:
				break
			
			corrected, n_applied = apply_corrections(current_tokens, corrections, verbose=verbose)
			stats["corrections_applied"] += n_applied
			
			if validate:
				validation = validate_tree(corrected)
				
				if not validation["valid"]:
					if verbose:
						print(f"    Validation failed: {validation}", file=sys.stderr)
					
					if attempt < max_retries:
						stats["retries"] += 1
						continue
					else:
						stats["outcome"] = "rejected_cycles"
						corrected, _ = apply_deterministic_fixes(original_tokens)
						return corrected, stats
			
			current_tokens = corrected
			break
			
		except Exception as e:
			if verbose:
				print(f"    LLM error: {e}", file=sys.stderr)
			stats["outcome"] = "error"
			current_tokens, _ = apply_deterministic_fixes(original_tokens)
			return current_tokens, stats
	
	current_tokens, _ = apply_deterministic_fixes(current_tokens)
	return current_tokens, stats


def correct_sentences(
	sentences: list[tuple[str, str, list[dict]]],
	client,
	model: str,
	prompt: str,
	lang: str,
	validate: bool = True,
	verbose: bool = False
) -> tuple[list[tuple[str, str, list[dict]]], dict]:
	"""
	Apply LLM corrections to multiple sentences.
	
	Returns: (corrected_sentences, aggregate_stats)
	"""
	corrected = []
	total_stats = {
		"input_tokens": 0,
		"output_tokens": 0,
		"corrections_suggested": 0,
		"corrections_applied": 0,
		"retries": 0,
		"accepted": 0,
		"rejected": 0,
		"errors": 0,
	}
	
	for sent_id, sent_text, tokens in sentences:
		result_tokens, stats = correct_sentence(
			client, sent_text, tokens, lang, model, prompt,
			validate=validate, verbose=verbose
		)
		
		total_stats["input_tokens"] += stats["input_tokens"]
		total_stats["output_tokens"] += stats["output_tokens"]
		total_stats["corrections_suggested"] += stats["corrections_suggested"]
		total_stats["corrections_applied"] += stats["corrections_applied"]
		total_stats["retries"] += stats["retries"]
		
		if stats["outcome"] == "accepted":
			total_stats["accepted"] += 1
		elif stats["outcome"] == "error":
			total_stats["errors"] += 1
		else:
			total_stats["rejected"] += 1
		
		corrected.append((sent_id, sent_text, result_tokens))
	
	return corrected, total_stats
