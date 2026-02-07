#!/usr/bin/env python3
"""
Tests for conllu_tools.

Run with: python -m pytest test_conllu_tools.py -v
Or standalone: python test_conllu_tools.py
"""

import sys
from pathlib import Path

try:
	from conllu_tools.validator import (
		parse_conllu_block,
		tokens_to_conllu,
		detect_cycles,
		detect_multiple_roots,
		detect_self_loops,
		detect_invalid_heads,
		validate_tree,
		apply_deterministic_fixes,
	)
	from conllu_tools.llm_corrector import (
		apply_corrections,
		format_parse_for_llm,
	)
except ImportError:
	sys.path.insert(0, str(Path(__file__).parent))
	from validator import (
		parse_conllu_block,
		tokens_to_conllu,
		detect_cycles,
		detect_multiple_roots,
		detect_self_loops,
		detect_invalid_heads,
		validate_tree,
		apply_deterministic_fixes,
	)
	from llm_corrector import (
		apply_corrections,
		format_parse_for_llm,
	)


sample_conllu = """# sent_id = test-1
# text = Elle était bonne.
1	Elle	lui	PRON	PRON	Gender=Fem|Number=Sing|Person=3	3	nsubj	_	_
2	était	être	AUX	AUX	Mood=Ind|Number=Sing|Person=3|Tense=Imp|VerbForm=Fin	3	cop	_	_
3	bonne	bon	ADJ	ADJ	Gender=Fem|Number=Sing	0	ROOT	_	_
4	.	.	PUNCT	PUNCT	_	3	punct	_	_"""


def make_tokens(heads):
	"""Helper to create minimal tokens with given head structure."""
	return [{"id": i+1, "form": f"t{i+1}", "lemma": f"t{i+1}", "head": h, "deprel": "dep"} 
	        for i, h in enumerate(heads)]


class TestParseConllu:
	def test_basic_parse(self):
		sent_id, sent_text, tokens = parse_conllu_block(sample_conllu)
		
		assert sent_id == "test-1"
		assert sent_text == "Elle était bonne."
		assert len(tokens) == 4
		
		assert tokens[0]["id"] == 1
		assert tokens[0]["form"] == "Elle"
		assert tokens[0]["lemma"] == "lui"
		assert tokens[0]["upos"] == "PRON"
		assert tokens[0]["head"] == 3
		assert tokens[0]["deprel"] == "nsubj"
	
	def test_roundtrip(self):
		sent_id, sent_text, tokens = parse_conllu_block(sample_conllu)
		output = tokens_to_conllu(tokens, sent_id, sent_text)
		
		sent_id2, sent_text2, tokens2 = parse_conllu_block(output)
		
		assert sent_id == sent_id2
		assert sent_text == sent_text2
		assert len(tokens) == len(tokens2)
		
		for t1, t2 in zip(tokens, tokens2):
			assert t1["id"] == t2["id"]
			assert t1["form"] == t2["form"]
			assert t1["lemma"] == t2["lemma"]
			assert t1["head"] == t2["head"]


class TestCycleDetection:
	def test_no_cycles(self):
		tokens = make_tokens([2, 0, 2])
		assert detect_cycles(tokens) == []
	
	def test_simple_cycle(self):
		tokens = make_tokens([2, 1])
		cycles = detect_cycles(tokens)
		assert len(cycles) == 1
		assert set(cycles[0]) == {1, 2}
	
	def test_self_loop_is_cycle(self):
		"""A self-loop (token pointing to itself) is a degenerate cycle."""
		tokens = make_tokens([1, 0])
		cycles = detect_cycles(tokens)
		assert len(cycles) == 1
		assert cycles[0] == [1]
	
	def test_three_node_cycle(self):
		tokens = make_tokens([2, 3, 1])
		cycles = detect_cycles(tokens)
		assert len(cycles) == 1
		assert set(cycles[0]) == {1, 2, 3}
	
	def test_valid_tree(self):
		tokens = make_tokens([3, 3, 0, 3])
		assert detect_cycles(tokens) == []


class TestMultipleRoots:
	def test_single_root(self):
		tokens = make_tokens([2, 0, 2])
		assert detect_multiple_roots(tokens) == []
	
	def test_multiple_roots(self):
		tokens = make_tokens([0, 0, 1])
		roots = detect_multiple_roots(tokens)
		assert set(roots) == {1, 2}


class TestSelfLoops:
	def test_no_self_loops(self):
		tokens = make_tokens([2, 0, 2])
		assert detect_self_loops(tokens) == []
	
	def test_has_self_loop(self):
		tokens = make_tokens([1, 0, 2])
		assert detect_self_loops(tokens) == [1]


class TestInvalidHeads:
	def test_valid_heads(self):
		tokens = make_tokens([2, 0, 2])
		assert detect_invalid_heads(tokens) == []
	
	def test_invalid_head(self):
		tokens = make_tokens([2, 0, 99])
		invalid = detect_invalid_heads(tokens)
		assert len(invalid) == 1
		assert invalid[0]["id"] == 3
		assert invalid[0]["head"] == 99


class TestValidateTree:
	def test_valid_tree(self):
		tokens = make_tokens([3, 3, 0, 3])
		result = validate_tree(tokens)
		assert result["valid"] == True
	
	def test_invalid_with_cycle(self):
		tokens = make_tokens([2, 1])
		result = validate_tree(tokens)
		assert result["valid"] == False
		assert len(result["cycles"]) > 0


class TestDeterministicFixes:
	def test_elle_lui_fix(self):
		tokens = [{"id": 1, "form": "Elle", "lemma": "lui", "head": 0, "deprel": "root"}]
		fixed, changes = apply_deterministic_fixes(tokens)
		assert fixed[0]["lemma"] == "elle"
		assert len(changes) == 1
	
	def test_no_fix_needed(self):
		tokens = [{"id": 1, "form": "Elle", "lemma": "elle", "head": 0, "deprel": "root"}]
		fixed, changes = apply_deterministic_fixes(tokens)
		assert fixed[0]["lemma"] == "elle"
		assert len(changes) == 0
	
	def test_preserves_correct_lemmas(self):
		tokens = [{"id": 1, "form": "lui", "lemma": "lui", "head": 0, "deprel": "root"}]
		fixed, changes = apply_deterministic_fixes(tokens)
		assert fixed[0]["lemma"] == "lui"
		assert len(changes) == 0


class TestApplyCorrections:
	"""Tests for the correction application logic, including bug fixes."""
	
	def test_basic_correction(self):
		tokens = [
			{"id": 1, "form": "Elle", "lemma": "lui", "upos": "PRON", "head": 2, "deprel": "nsubj"},
			{"id": 2, "form": "dort", "lemma": "dormir", "upos": "VERB", "head": 0, "deprel": "root"},
		]
		corrections = [{"id": 1, "field": "lemma", "value": "elle"}]
		
		result, n = apply_corrections(tokens, corrections)
		
		assert n == 1
		assert result[0]["lemma"] == "elle"
		assert tokens[0]["lemma"] == "lui"
	
	def test_string_token_id_converted(self):
		"""BUG FIX: Token IDs as strings should be converted to int."""
		tokens = [
			{"id": 1, "form": "Elle", "lemma": "lui", "upos": "PRON", "head": 2, "deprel": "nsubj"},
		]
		corrections = [{"id": "1", "field": "lemma", "value": "elle"}]
		
		result, n = apply_corrections(tokens, corrections)
		
		assert n == 1
		assert result[0]["lemma"] == "elle"
	
	def test_head_value_converted(self):
		"""BUG FIX: Head values as strings should be converted to int."""
		tokens = [
			{"id": 1, "form": "test", "lemma": "test", "upos": "NOUN", "head": 0, "deprel": "root"},
		]
		corrections = [{"id": 1, "field": "head", "value": "2"}]
		
		result, n = apply_corrections(tokens, corrections)
		
		assert n == 1
		assert result[0]["head"] == 2
		assert isinstance(result[0]["head"], int)
	
	def test_nonexistent_token_skipped(self):
		tokens = [{"id": 1, "form": "test", "lemma": "test", "upos": "NOUN", "head": 0, "deprel": "root"}]
		corrections = [{"id": 99, "field": "lemma", "value": "changed"}]
		
		result, n = apply_corrections(tokens, corrections)
		
		assert n == 0
		assert result[0]["lemma"] == "test"
	
	def test_unknown_field_skipped(self):
		tokens = [{"id": 1, "form": "test", "lemma": "test", "upos": "NOUN", "head": 0, "deprel": "root"}]
		corrections = [{"id": 1, "field": "unknown_field", "value": "changed"}]
		
		result, n = apply_corrections(tokens, corrections)
		
		assert n == 0
	
	def test_malformed_correction_skipped(self):
		tokens = [{"id": 1, "form": "test", "lemma": "test", "upos": "NOUN", "head": 0, "deprel": "root"}]
		corrections = [{"id": 1}]
		
		result, n = apply_corrections(tokens, corrections)
		
		assert n == 0
	
	def test_multiple_corrections(self):
		tokens = [
			{"id": 1, "form": "Elle", "lemma": "lui", "upos": "PRON", "head": 2, "deprel": "nsubj"},
			{"id": 2, "form": "dort", "lemma": "dormir", "upos": "VERB", "head": 0, "deprel": "root"},
		]
		corrections = [
			{"id": 1, "field": "lemma", "value": "elle"},
			{"id": 1, "field": "head", "value": 2},
			{"id": 2, "field": "deprel", "value": "ROOT"},
		]
		
		result, n = apply_corrections(tokens, corrections)
		
		assert n == 3
		assert result[0]["lemma"] == "elle"
		assert result[0]["head"] == 2
		assert result[1]["deprel"] == "ROOT"
	
	def test_original_not_mutated(self):
		"""Ensure original tokens are not modified."""
		tokens = [{"id": 1, "form": "test", "lemma": "old", "upos": "NOUN", "head": 0, "deprel": "root"}]
		corrections = [{"id": 1, "field": "lemma", "value": "new"}]
		
		result, n = apply_corrections(tokens, corrections)
		
		assert result[0]["lemma"] == "new"
		assert tokens[0]["lemma"] == "old"


class TestFormatParseForLLM:
	def test_french(self):
		tokens = [
			{"id": 1, "form": "Elle", "lemma": "elle", "upos": "PRON", "head": 2, "deprel": "nsubj"},
			{"id": 2, "form": "dort", "lemma": "dormir", "upos": "VERB", "head": 0, "deprel": "root"},
		]
		
		lang_name, parse_str = format_parse_for_llm(tokens, "fr")
		
		assert lang_name == "French"
		assert "1\tElle\telle\tPRON\t2\tnsubj" in parse_str
		assert "2\tdort\tdormir\tVERB\t0\troot" in parse_str
	
	def test_english(self):
		tokens = [{"id": 1, "form": "test", "lemma": "test", "upos": "NOUN", "head": 0, "deprel": "root"}]
		
		lang_name, _ = format_parse_for_llm(tokens, "en")
		
		assert lang_name == "English"


def run_tests():
	"""Run all tests and report results."""
	import traceback
	
	test_classes = [
		TestParseConllu,
		TestCycleDetection,
		TestMultipleRoots,
		TestSelfLoops,
		TestInvalidHeads,
		TestValidateTree,
		TestDeterministicFixes,
		TestApplyCorrections,
		TestFormatParseForLLM,
	]
	
	total = 0
	passed = 0
	failed = []
	
	for test_class in test_classes:
		instance = test_class()
		methods = [m for m in dir(instance) if m.startswith("test_")]
		
		for method_name in methods:
			total += 1
			test_name = f"{test_class.__name__}.{method_name}"
			
			try:
				getattr(instance, method_name)()
				passed += 1
				print(f"  ✓ {test_name}")
			except AssertionError as e:
				failed.append((test_name, e))
				print(f"  ✗ {test_name}")
				traceback.print_exc()
			except Exception as e:
				failed.append((test_name, e))
				print(f"  ✗ {test_name} (exception)")
				traceback.print_exc()
	
	print(f"\n{'='*50}")
	print(f"Results: {passed}/{total} passed")
	
	if failed:
		print(f"\nFailed tests:")
		for name, error in failed:
			print(f"  - {name}: {error}")
		return 1
	
	return 0


if __name__ == "__main__":
	sys.exit(run_tests())
