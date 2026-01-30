# isosceles
A small corpus for computational analysis of literary translation, focusing on 19th-century French-English pairs for short fiction by Maupassant and Poe.

## Structure
Three components:
1. **Poe-Baudelaire**: Baudelaire's French translations of Poe's tales (1856-1865), with Poe originals from 1845 (Wiley & Putnam) and 1850 (Lowell & Griswold) editions
2. **Maupassant-McMaster**: McMaster et al. public domain English translations (1903) of Maupassant's short stories, and the French originals digitized from the authoritative Pléiade edition of Maupassant (Forestier 1987).
3. **Reference corpora**: ELTeC-fra and ELTeC-eng provide period-appropriate baselines (~100 novels each, 1840-1920) for comparison with native prose norms. See [ELTeC](github.com/COST-ELTeC) for more details.

## Directory structure

```
data/
├── maupassant/
│   ├── en/
│   │   ├── txt/          # one file per story
│   │   └── metadata.tsv
│   ├── fr/
│   │   ├── txt/
│   │   └── metadata.tsv
│   └── index.tsv         # parallel alignment with word counts
├── poe/
│   ├── en/
│   │   ├── 1845/
│   │   │   └── txt/
│   │   └── 1850/
│   │       └── txt/
│   ├── fr/
│   │   ├── histoires_extraordinaires/
│   │   │   └── txt/
│   │   ├── nouvelles_histoires/
│   │   │   └── txt/
│   │   ├── grotesques_serieuses/
│   │   │   └── txt/
│   │   └── arthur_gordon_pym/
│   │       └── txt/
│   └── index.tsv         # parallel alignment
└── eltec/                # not in repo; see note below
    ├── en/
    │   ├── xml/
    │   └── txt/
    ├── fr/
    │   ├── xml/
    │   └── txt/
    └── manifest.tsv
```

The `eltec` folder is omitted from this repository, due to its size, but you can generate its contents locally with `scripts/eltec.py` to download and clean texts from the official ELTeC project; see usage below.

## Setup
```bash
# clone the repository:
git clone https://github.com/YOUR_USERNAME/corpus-isosceles.git
cd corpus-isosceles

# install Python dependencies:
pip install -r requirements.txt

# download Stanza models (if running annotation):
python -c "import stanza; stanza.download('fr'); stanza.download('en')"
```

## Scripts

```bash
# ELTeC reference corpora
python scripts/eltec.py download fra
python scripts/eltec.py download eng
python scripts/eltec.py extract fra
python scripts/eltec.py extract eng

# Maupassant French (from maupassant.free.fr)
python scripts/maupassant.py list
python scripts/maupassant.py download

# Maupassant English (from Project Gutenberg)
python scripts/maupassant_en.py download
python scripts/maupassant_en.py toc
python scripts/maupassant_en.py list
python scripts/maupassant_en.py segment

# Poe English (from WikiSource)
python scripts/poe.py list
python scripts/poe.py download 1845
python scripts/poe.py download 1850

# Baudelaire-Poe French (from WikiSource)
python scripts/baudelaire.py list
python scripts/baudelaire.py download

# build parallel index for Maupassant:
python scripts/parallel_index.py check
python scripts/parallel_index.py build
```

## Linguistic annotation
The `annotate.py` script provides tokenization, POS tagging, lemmatization, and dependency parsing with multiple backend options.

### Backends

| Backend | Segmentation | Parsing | Quality | Use case |
|---------|--------------|---------|---------|----------|
| `stanza` | Stanza | Stanza | Baseline | Legacy compatibility |
| `corenlp` | CoreNLP | CoreNLP | Poor | Not recommended |
| `spacy` | CoreNLP | spaCy | Good | ELTeC, large corpora |
| `spacy+llm` | CoreNLP | spaCy + Claude corrections | Best | Maupassant, Poe-Baudelaire |

**Recommended approach:**
- French: `spacy+llm` is necessary for high quality annotation
- English: `spacy` alone is sufficient

The `spacy` backend uses CoreNLP for sentence segmentation (more reliable for literary text than Stanza or spaCy alone) and spaCy transformer models (`fr_dep_news_trf`, `en_core_web_trf`) for parsing.

The `spacy+llm` backend adds a correction pass using Claude Sonnet, which fixes many errors in lemmatization, dependency relations, and tree validity.

### Usage
```bash
# Set environment variables
export CORENLP_HOME=/path/to/stanford-corenlp-4.5.x
export ANTHROPIC_API_KEY=sk-...  # only needed if using spacy+llm

# Maupassant French — both versions for different analyses
python scripts/annotate.py data/maupassant/fr/txt data/maupassant/fr/conllu -l fr -b spacy
python scripts/annotate.py data/maupassant/fr/txt data/maupassant/fr/conllu_claude -l fr -b spacy+claude

# Maupassant English
python scripts/annotate.py data/maupassant/en/txt data/maupassant/en/conllu -l en -b spacy

# Poe-Baudelaire French
python scripts/annotate.py data/poe/fr/histoires_extraordinaires/txt data/poe/fr/histoires_extraordinaires/conllu -l fr -b spacy
python scripts/annotate.py data/poe/fr/histoires_extraordinaires/txt data/poe/fr/histoires_extraordinaires/conllu_claude -l fr -b spacy+claude

# ELTeC reference corpora (spacy only, due to size)
python scripts/annotate.py data/eltec/fr/txt data/eltec/fr/conllu -l fr -b spacy
python scripts/annotate.py data/eltec/en/txt data/eltec/en/conllu -l en -b spacy
```

## Index files
Each author directory has an `index.tsv` for parallel text alignment:

**maupassant/index.tsv**:
```
fr_file  en_file  fr_words  en_words
```

**poe/index.tsv**:
```
en_file  fr_file  en_title  fr_title  fr_volume  en_edition  en_words  fr_words
```

For Poe, the `en_edition` column indicates whether the English text is from the 1845 edition (Wiley & Putnam), the 1850 edition (Lowell & Griswold), or both.

## Notes on the texts
### Poe

- **Footnotes**: Manually removed footnotes, to focus on body text.
- **Footnote markers**: Manually removed (e.g., `12]` artifacts from WikiSource).
- **Editions**: The 1845 and 1850 texts show ~5% divergence (LCS analysis). Both are retained since we don't yet know which Baudelaire used for each story.
- **Duplicates**: "Le Mystère de Marie Roget" appears in both `histoires_extraordinaires/` and `grotesques_serieuses/`; these appear to be the same translation.
- **OCR errors**: The Novalis epigraph in "Marie Roget" has errors in both WikiSource editions ("gewohulich" for "gewöhnlich"). Other OCR artifacts likely remain. We consider these errors outside of the scope of our cleanup.
- **Zero-width spaces**: Manually removed. These were present in WikiSource transcriptions:
```bash
  perl -pi -e 's/\xe2\x80\x8b//g' data/poe/*/*/txt/*.txt
```

### Maupassant

- **McMaster segmentation**: The PG volumes have TOC/body title mismatches requiring a variant lookup table in `maupassant_en.py`.
- **Partial coverage**: McMaster et al. translated ~180 of Maupassant's ~300 stories. The parallel index shows which have matches.
- **Missing French sources**: Four McMaster et al. stories have no matching French text in the Pléiade-based corpus: "The Englishmen of Etretat", "The Lancer's Wife," "The Legion of Honor," and "The Thief." These may be misattributions.
- **Partial translations**: A comparison of word counts between the French and English versions will reveal at least a few stories that are significantly truncated or redacted on the English side, including:
    - "Les Dimanches d'un bourgeois de Paris" is truncated in English — the concluding section X is absent
    - "Une Farce" has an extended framing prologue in French that is omitted in the English translation
    - "Monsieur Parent" is evidently missing material somewhere in the middle of the English translation


### General

This corpus is a work in progress. Additional cleanup will be applied as issues are discovered. See the commit history for details.

## Sources and licensing

| Corpus | Source | License |
|--------|--------|---------|
| ELTeC | [github.com/COST-ELTeC](https://github.com/COST-ELTeC) | CC-BY 4.0 |
| Maupassant (fr) | [maupassant.free.fr](http://maupassant.free.fr) | Public domain |
| Maupassant (en) | [gutenberg.org](https://gutenberg.org) | Public domain |
| Poe (en) | [en.wikisource.org](https://en.wikisource.org) | Public domain |
| Poe (fr) | [fr.wikisource.org](https://fr.wikisource.org) | Public domain |

Maupassant French texts follow the Pléiade edition (Forestier 1987).

McMaster et al. translations are the "Complete Short Stories" series (PG #3077-3089).

Poe 1845 = *Tales* (published in Poe's lifetime). Poe 1850 = Griswold edition (posthumous, more complete).

## Citation
If you use this corpus in your research, please cite:

> Myers, Michael J. (2026). *Isosceles*: A parallel corpus for computational analysis of literary translation [Data set]. GitHub. https://github.com/myersm0/isosceles

A `CITATION.cff` file is included for automated citation tools.

## Contributing

Found an error? Have a suggestion? Please [open an issue](../../issues) on GitHub.


## License

This project is licensed under [CC BY 4.0](LICENSE). You are free to use, adapt, and redistribute the data with attribution.
