#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import stanza

def to_conllu(doc, doc_id):
	lines = []
	for sent_idx, sent in enumerate(doc.sentences):
		lines.append(f"# sent_id = {doc_id}-{sent_idx + 1}")
		lines.append(f"# text = {sent.text}")
		for word in sent.words:
			row = [
				str(word.id),
				word.text,
				word.lemma or "_",
				word.upos or "_",
				word.xpos or "_",
				"_",
				str(word.head),
				word.deprel or "_",
				"_",
				"_",
			]
			lines.append("\t".join(row))
		lines.append("")
	return "\n".join(lines)

def to_json(doc, doc_id):
	result = {"doc_id": doc_id, "sentences": []}
	for sent in doc.sentences:
		tokens = []
		deps = []
		for word in sent.words:
			tokens.append({
				"id": word.id,
				"form": word.text,
				"lemma": word.lemma,
				"upos": word.upos,
				"xpos": word.xpos,
			})
			deps.append({
				"dependent": word.id,
				"dependentGloss": word.text,
				"dependentLemma": word.lemma,
				"governor": word.head,
				"governorGloss": sent.words[word.head - 1].text if word.head > 0 else "ROOT",
				"governorLemma": sent.words[word.head - 1].lemma if word.head > 0 else "ROOT",
				"dep": word.deprel,
			})
		result["sentences"].append({"tokens": tokens, "basicDependencies": deps})
	return result

def process_corpus(input_dir, output_dir, lang, format):
	input_dir = Path(input_dir)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	
	stanza.download(lang, processors="tokenize,pos,lemma,depparse")
	nlp = stanza.Pipeline(lang, processors="tokenize,pos,lemma,depparse")

	ext = ".json" if format == "json" else ".conllu"
	for filepath in sorted(input_dir.glob("*.txt")):
		print(f"  {filepath.name}")
		doc = nlp(filepath.read_text(encoding="utf-8"))
		out_path = output_dir / (filepath.stem + ext)
		if format == "json":
			out_path.write_text(json.dumps(to_json(doc, filepath.stem), ensure_ascii=False, indent=2), encoding="utf-8")
		else:
			out_path.write_text(to_conllu(doc, filepath.stem), encoding="utf-8")
	

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("input_dir")
	ap.add_argument("output_dir")
	ap.add_argument("--lang", "-l", choices=["fr", "en"], required=True)
	ap.add_argument("--format", "-f", choices=["json", "conllu"], default="conllu")
	args = ap.parse_args()
	process_corpus(args.input_dir, args.output_dir, args.lang, args.format)

if __name__ == "__main__":
	main()

