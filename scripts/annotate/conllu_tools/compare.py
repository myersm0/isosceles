"""Compare CoNLL-U annotations across multiple pipelines.

Aligns sentences by sent_id and tokens by position, producing
structured comparison data. Handles MWT mismatches by skipping
affected tokens with warnings.

Usage:
    from conllu_tools.compare import compare_files, compare_parsed

    result = compare_files(
        stanza="chunk_stanza.conllu",
        corenlp="chunk_corenlp.conllu",
        spacy="chunk_spacy.conllu",
    )

    # Stanza outliers: field value differs from both others
    for d in result.stanza_outliers("lemma"):
        print(d)

    # All disagreements on a field
    for d in result.disagreements("upos"):
        print(d)

    # Per-field agreement summary
    result.print_summary()
"""

from dataclasses import dataclass, field
from typing import Optional

try:
	from .validator import parse_conllu_block
except ImportError:
	from validator import parse_conllu_block


@dataclass
class TokenComparison:
	sent_id: str
	token_id: int
	form: str
	pipelines: dict  # {pipeline_name: token_dict}
	skipped: bool = False
	skip_reason: str = ""

	def values(self, field_name):
		"""Get {pipeline: value} for a field."""
		return {
			name: tok.get(field_name, "_")
			for name, tok in self.pipelines.items()
		}

	def agreement(self, field_name):
		"""Classify agreement on a field.

		Returns one of:
		  'all_agree' — all pipelines have same value
		  '<name>_outlier' — one pipeline differs from all others
		  'mixed' — no clear pattern
		"""
		vals = self.values(field_name)
		unique = set(vals.values())
		if len(unique) == 1:
			return "all_agree"
		names = list(vals.keys())
		values = list(vals.values())
		for i, name in enumerate(names):
			others = [v for j, v in enumerate(values) if j != i]
			if len(set(others)) == 1 and values[i] != others[0]:
				return f"{name}_outlier"
		return "mixed"


@dataclass
class CompareResult:
	sentences: list  # list of list[TokenComparison]
	pipeline_names: list
	warnings: list = field(default_factory=list)

	def all_comparisons(self):
		for sentence in self.sentences:
			yield from (tc for tc in sentence if not tc.skipped)

	def disagreements(self, field_name):
		"""All token comparisons where not all pipelines agree."""
		return [
			tc for tc in self.all_comparisons()
			if tc.agreement(field_name) != "all_agree"
		]

	def stanza_outliers(self, field_name):
		"""Tokens where stanza differs from all other pipelines."""
		return [
			tc for tc in self.all_comparisons()
			if tc.agreement(field_name) == "stanza_outlier"
		]

	def outliers(self, pipeline_name, field_name):
		"""Tokens where a specific pipeline differs from all others."""
		return [
			tc for tc in self.all_comparisons()
			if tc.agreement(field_name) == f"{pipeline_name}_outlier"
		]

	def agreement_counts(self, field_name):
		"""Count agreement categories for a field."""
		counts = {}
		for tc in self.all_comparisons():
			cat = tc.agreement(field_name)
			counts[cat] = counts.get(cat, 0) + 1
		return counts

	def print_summary(self, fields=None):
		if fields is None:
			fields = ["lemma", "upos", "feats"]
		total = sum(1 for _ in self.all_comparisons())
		print(f"Compared {total} tokens across {len(self.sentences)} sentences")
		print(f"Pipelines: {', '.join(self.pipeline_names)}")
		if self.warnings:
			print(f"Warnings: {len(self.warnings)}")
		print()
		for f in fields:
			counts = self.agreement_counts(f)
			print(f"  {f}:")
			for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
				print(f"    {cat}: {n}")
			print()

	def field_counts(self, form, field_name, case_sensitive=False):
		"""Count field values per pipeline for a given surface form.

		Returns: {pipeline: {value: count}}

		Example:
		    result.field_counts("prie", "lemma")
		    → {'stanza': {'prier': 8, 'prendre': 1}, 'spacy': {'prier': 9}}
		"""
		counts = {name: {} for name in self.pipeline_names}
		for tc in self.all_comparisons():
			match = tc.form == form if case_sensitive else tc.form.lower() == form.lower()
			if not match:
				continue
			for name, tok in tc.pipelines.items():
				val = tok.get(field_name, "_")
				counts[name][val] = counts[name].get(val, 0) + 1
		return counts

	def cross_tabulate(self, form, field_name, when_pipeline, when_value,
					   case_sensitive=False):
		"""When one pipeline maps a form to a specific value, what do others say?

		Returns: {pipeline: {value: count}} (excluding when_pipeline)

		Example:
		    result.cross_tabulate("prie", "lemma", "stanza", "prendre")
		    → {'spacy': {'prier': 1}}
		"""
		counts = {
			name: {} for name in self.pipeline_names
			if name != when_pipeline
		}
		for tc in self.all_comparisons():
			match = tc.form == form if case_sensitive else tc.form.lower() == form.lower()
			if not match:
				continue
			when_tok = tc.pipelines.get(when_pipeline)
			if not when_tok or when_tok.get(field_name, "_") != when_value:
				continue
			for name, tok in tc.pipelines.items():
				if name == when_pipeline:
					continue
				val = tok.get(field_name, "_")
				counts[name][val] = counts[name].get(val, 0) + 1
		return counts


def _read_file(path):
	"""Read CoNLL-U file into {sent_id: (sent_text, tokens)} dict."""
	sentences = {}
	with open(path, encoding="utf-8") as f:
		content = f.read()

	blocks = content.strip().split("\n\n")
	for block in blocks:
		block = block.strip()
		if not block:
			continue
		sent_id, sent_text, tokens = parse_conllu_block(block)
		if sent_id and tokens:
			sentences[sent_id] = (sent_text, tokens)

	return sentences


def _has_mwt(block_text):
	"""Check if a raw block has MWT range lines."""
	for line in block_text.split("\n"):
		if line and not line.startswith("#"):
			parts = line.split("\t")
			if parts and "-" in parts[0]:
				return True
	return False


def compare_files(order=None, **pipeline_paths):
	"""Compare CoNLL-U files from multiple pipelines.

	Args:
		order: optional list specifying pipeline display order
		**pipeline_paths: name=path pairs, e.g. stanza="file.conllu"

	Returns: CompareResult
	"""
	pipeline_data = {}
	for name, path in pipeline_paths.items():
		pipeline_data[name] = _read_file(path)

	return compare_parsed(order=order, **pipeline_data)


def compare_parsed(order=None, **pipeline_sentences):
	"""Compare pre-parsed pipeline data.

	Args:
		order: optional list specifying pipeline display order
		**pipeline_sentences: name={sent_id: (sent_text, tokens)} pairs

	Returns: CompareResult
	"""
	names = order or list(pipeline_sentences.keys())
	warnings = []

	# Find common sent_ids
	id_sets = [set(pipeline_sentences[n].keys()) for n in names]
	common_ids = id_sets[0]
	for s in id_sets[1:]:
		common_ids = common_ids & s

	for name, data in pipeline_sentences.items():
		missing = set(data.keys()) - common_ids
		if missing:
			warnings.append(
				f"{name}: {len(missing)} sentences not in all pipelines")

	all_sentences = []

	for sent_id in sorted(common_ids):
		token_lists = {}
		for name in names:
			_, tokens = pipeline_sentences[name][sent_id]
			token_lists[name] = tokens

		lengths = {name: len(toks) for name, toks in token_lists.items()}
		if len(set(lengths.values())) > 1:
			warnings.append(
				f"{sent_id}: token count mismatch: "
				+ ", ".join(f"{n}={l}" for n, l in lengths.items()))
			# Still align what we can
			min_len = min(lengths.values())
		else:
			min_len = list(lengths.values())[0]

		sentence_comparisons = []
		for i in range(min_len):
			tokens_at_i = {name: token_lists[name][i] for name in names}

			# Check form consistency
			forms = {name: tok["form"] for name, tok in tokens_at_i.items()}
			unique_forms = set(forms.values())

			if len(unique_forms) > 1:
				# MWT or tokenization mismatch
				tc = TokenComparison(
					sent_id=sent_id,
					token_id=tokens_at_i[names[0]]["id"],
					form=forms[names[0]],
					pipelines=tokens_at_i,
					skipped=True,
					skip_reason=f"form mismatch: "
						+ ", ".join(f"{n}={f}" for n, f in forms.items()),
				)
				warnings.append(f"{sent_id} tok {tc.token_id}: {tc.skip_reason}")
			else:
				tc = TokenComparison(
					sent_id=sent_id,
					token_id=tokens_at_i[names[0]]["id"],
					form=list(unique_forms)[0],
					pipelines=tokens_at_i,
				)

			sentence_comparisons.append(tc)

		all_sentences.append(sentence_comparisons)

	return CompareResult(
		sentences=all_sentences,
		pipeline_names=names,
		warnings=warnings,
	)
