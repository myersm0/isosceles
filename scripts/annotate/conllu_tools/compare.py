"""Compare CoNLL-U annotations across multiple pipelines.

Aligns sentences by sent_id and tokens by surface form, handling
MWT expansion differences between pipelines. Stanza expands
contractions (au → à + le) while spaCy may not; the alignment
groups MWT-expanded tokens and matches by surface form.

Usage:
    from conllu_tools.compare import compare_files, compare_parsed

    result = compare_files(
        stanza="chunk_stanza.conllu",
        spacy="chunk_spacy.conllu",
    )

    for d in result.stanza_outliers("lemma"):
        print(d)

    result.print_summary()
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TokenComparison:
	sent_id: str
	token_id: int
	form: str
	pipelines: dict  # {pipeline_name: token_dict}
	skipped: bool = False
	skip_reason: str = ""

	def values(self, field_name):
		return {
			name: tok.get(field_name, "_")
			for name, tok in self.pipelines.items()
		}

	def agreement(self, field_name):
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
	sentences: list
	pipeline_names: list
	warnings: list = field(default_factory=list)

	def all_comparisons(self):
		for sentence in self.sentences:
			yield from (tc for tc in sentence if not tc.skipped)

	def disagreements(self, field_name):
		return [
			tc for tc in self.all_comparisons()
			if tc.agreement(field_name) != "all_agree"
		]

	def stanza_outliers(self, field_name):
		return [
			tc for tc in self.all_comparisons()
			if tc.agreement(field_name) == "stanza_outlier"
		]

	def spacy_outliers(self, field_name):
		return [
			tc for tc in self.all_comparisons()
			if tc.agreement(field_name) == "spacy_outlier"
		]

	def outliers(self, pipeline_name, field_name):
		return [
			tc for tc in self.all_comparisons()
			if tc.agreement(field_name) == f"{pipeline_name}_outlier"
		]

	def agreement_counts(self, field_name):
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


# --- File reading with MWT awareness ---

def _parse_block_with_mwt(text):
	sent_id = None
	sent_text = None
	tokens = []
	mwt_ranges = {}

	for line in text.strip().split("\n"):
		if line.startswith("# sent_id = "):
			sent_id = line[12:]
		elif line.startswith("# text = "):
			sent_text = line[9:]
		elif line.startswith("#") or not line.strip():
			continue
		else:
			parts = line.split("\t")
			if len(parts) != 10:
				continue
			if "-" in parts[0]:
				halves = parts[0].split("-")
				try:
					mwt_ranges[int(halves[0])] = (int(halves[1]), parts[1])
				except ValueError:
					pass
				continue
			if "." in parts[0]:
				continue
			try:
				tok_id = int(parts[0])
			except ValueError:
				continue
			tokens.append({
				"id": tok_id,
				"form": parts[1],
				"lemma": parts[2],
				"upos": parts[3],
				"xpos": parts[4],
				"feats": parts[5],
				"head": int(parts[6]) if parts[6] != "_" else 0,
				"deprel": parts[7],
				"deps": parts[8],
				"misc": parts[9],
			})

	return sent_id, sent_text, tokens, mwt_ranges


def _read_file(path):
	sentences = {}
	with open(path, encoding="utf-8") as f:
		content = f.read()

	for block in content.strip().split("\n\n"):
		block = block.strip()
		if not block:
			continue
		sent_id, sent_text, tokens, mwt_ranges = _parse_block_with_mwt(block)
		if sent_id and tokens:
			sentences[sent_id] = (sent_text, tokens, mwt_ranges)

	return sentences


# --- MWT-aware alignment ---

@dataclass
class _AlignUnit:
	surface: str
	tokens: list
	is_mwt: bool = False


def _build_align_units(tokens, mwt_ranges):
	mwt_members = set()
	for start_id, (end_id, _surface) in mwt_ranges.items():
		for tok_id in range(start_id, end_id + 1):
			mwt_members.add(tok_id)

	units = []
	i = 0
	while i < len(tokens):
		tok = tokens[i]
		tok_id = tok["id"]

		if tok_id in mwt_members:
			for start_id, (end_id, surface) in mwt_ranges.items():
				if tok_id == start_id:
					mwt_tokens = []
					while i < len(tokens) and tokens[i]["id"] <= end_id:
						mwt_tokens.append(tokens[i])
						i += 1
					units.append(_AlignUnit(
						surface=surface,
						tokens=mwt_tokens,
						is_mwt=True,
					))
					break
			else:
				units.append(_AlignUnit(surface=tok["form"], tokens=[tok]))
				i += 1
		else:
			units.append(_AlignUnit(surface=tok["form"], tokens=[tok]))
			i += 1

	return units


def _align_two(units_a, units_b, name_a, name_b, sent_id, warnings):
	comparisons = []
	i, j = 0, 0
	max_look = 4

	while i < len(units_a) and j < len(units_b):
		ua, ub = units_a[i], units_b[j]

		if ua.surface.lower() == ub.surface.lower():
			if not ua.is_mwt and not ub.is_mwt:
				comparisons.append(TokenComparison(
					sent_id=sent_id,
					token_id=ua.tokens[0]["id"],
					form=ua.surface,
					pipelines={name_a: ua.tokens[0], name_b: ub.tokens[0]},
				))
			elif ua.is_mwt and ub.is_mwt and len(ua.tokens) == len(ub.tokens):
				for ta, tb in zip(ua.tokens, ub.tokens):
					comparisons.append(TokenComparison(
						sent_id=sent_id,
						token_id=ta["id"],
						form=ta["form"],
						pipelines={name_a: ta, name_b: tb},
					))
			else:
				warnings.append(
					f"{sent_id}: MWT structure mismatch at '{ua.surface}'"
				)
			i += 1
			j += 1
			continue

		found = False
		for look in range(1, min(max_look, len(units_a) - i)):
			if units_a[i + look].surface.lower() == ub.surface.lower():
				for skip in range(look):
					warnings.append(
						f"{sent_id}: unmatched {name_a} token "
						f"'{units_a[i + skip].surface}'"
					)
				i += look
				found = True
				break

		if not found:
			for look in range(1, min(max_look, len(units_b) - j)):
				if units_b[j + look].surface.lower() == ua.surface.lower():
					for skip in range(look):
						warnings.append(
							f"{sent_id}: unmatched {name_b} token "
							f"'{units_b[j + skip].surface}'"
						)
					j += look
					found = True
					break

		if not found:
			warnings.append(
				f"{sent_id}: can't align '{ua.surface}' ({name_a}) "
				f"vs '{ub.surface}' ({name_b})"
			)
			i += 1
			j += 1

	return comparisons


def _align_positional(units_by_name, names, sent_id, warnings):
	lengths = {name: len(units) for name, units in units_by_name.items()}
	if len(set(lengths.values())) > 1:
		warnings.append(
			f"{sent_id}: unit count mismatch: "
			+ ", ".join(f"{n}={l}" for n, l in lengths.items())
		)
	min_len = min(lengths.values())

	comparisons = []
	for i in range(min_len):
		units_at_i = {name: units_by_name[name][i] for name in names}
		surfaces = {name: u.surface for name, u in units_at_i.items()}
		unique_surfaces = set(s.lower() for s in surfaces.values())

		if len(unique_surfaces) > 1:
			warnings.append(
				f"{sent_id}: surface mismatch at position {i}: "
				+ ", ".join(f"{n}={s}" for n, s in surfaces.items())
			)
			continue

		any_mwt = any(u.is_mwt for u in units_at_i.values())
		all_single = all(len(u.tokens) == 1 for u in units_at_i.values())

		if all_single and not any_mwt:
			comparisons.append(TokenComparison(
				sent_id=sent_id,
				token_id=units_at_i[names[0]].tokens[0]["id"],
				form=list(surfaces.values())[0],
				pipelines={
					name: units_at_i[name].tokens[0]
					for name in names
				},
			))
		else:
			warnings.append(
				f"{sent_id}: MWT structure mismatch at "
				f"'{list(surfaces.values())[0]}'"
			)

	return comparisons


# --- Public API ---

def compare_files(order=None, **pipeline_paths):
	pipeline_data = {}
	for name, path in pipeline_paths.items():
		pipeline_data[name] = _read_file(path)
	return compare_parsed(order=order, **pipeline_data)


def compare_parsed(order=None, **pipeline_sentences):
	"""Compare pre-parsed pipeline data.

	Accepts values as either:
	  {sent_id: (sent_text, tokens)} — legacy, no MWT info
	  {sent_id: (sent_text, tokens, mwt_ranges)} — MWT-aware
	"""
	names = order or list(pipeline_sentences.keys())
	warnings = []

	normalized = {}
	for name, data in pipeline_sentences.items():
		normalized[name] = {}
		for sent_id, value in data.items():
			if isinstance(value, (list, tuple)) and len(value) == 3:
				sent_text, tokens, mwt_ranges = value
			else:
				sent_text, tokens = value
				mwt_ranges = {}
			normalized[name][sent_id] = (sent_text, tokens, mwt_ranges)

	id_sets = [set(normalized[n].keys()) for n in names]
	common_ids = id_sets[0]
	for s in id_sets[1:]:
		common_ids = common_ids & s

	for name, data in normalized.items():
		missing = set(data.keys()) - common_ids
		if missing:
			warnings.append(
				f"{name}: {len(missing)} sentences not in all pipelines"
			)

	all_sentences = []

	for sent_id in sorted(common_ids):
		units_by_name = {}
		for name in names:
			_text, tokens, mwt_ranges = normalized[name][sent_id]
			units_by_name[name] = _build_align_units(tokens, mwt_ranges)

		if len(names) == 2:
			comparisons = _align_two(
				units_by_name[names[0]],
				units_by_name[names[1]],
				names[0], names[1],
				sent_id, warnings,
			)
		else:
			comparisons = _align_positional(
				units_by_name, names, sent_id, warnings,
			)

		all_sentences.append(comparisons)

	return CompareResult(
		sentences=all_sentences,
		pipeline_names=names,
		warnings=warnings,
	)
