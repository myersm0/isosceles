# corpus-isosceles
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
│   │   ├── raw/          # original PG volumes
│   │   ├── txt/          # segmented stories
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
└── eltec/
    ├── en/
    │   ├── xml/
    │   └── txt/
    ├── fr/
    │   ├── xml/
    │   └── txt/
    └── manifest.tsv
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
The `annotate.py` script runs tokenization, POS tagging, lemmatization, and dependency parsing via Stanford CoreNLP with [Stanza](https://stanfordnlp.github.io/stanza/).

```bash
# use a French language model to annotate Maupassant's French:
python scripts/annotate.py data/maupassant/fr/txt data/maupassant/fr/json -l fr -f json

# use an English language model to annotate the McMaster et al. transation of Maupassant:
python scripts/stanza_annotate.py data/maupassant/en/txt data/maupassant/en/json -l en -f json
```

**Output formats**:
- `-f json`: CoreNLP-style JSON with denormalized dependency info
- `-f conllu`: Standard CoNLL-U for compatibility with other NLP tools

JSON output includes `basicDependencies` with governor/dependent lemmas inline:
```json
{
  "dependent": 3,
  "dependentGloss": "vieux",
  "dependentLemma": "vieux",
  "governor": 4,
  "governorGloss": "bibelots",
  "governorLemma": "bibelot",
  "dep": "amod"
}
```

## Index files
Each author directory has an `index.tsv` for parallel text alignment:

**maupassant/index.tsv**:
```
fr_file  en_file  fr_title  en_title  collection  fr_words  en_words
```

**poe/index.tsv**:
```
en_file  fr_file  en_title  fr_title  fr_volume  en_edition  en_words  fr_words
```

For Poe, the `en_edition` column indicates whether the English text is from the 1845 edition (Wiley & Putnam), the 1850 edition (Lowell & Griswold), or both.

## Cleanup notes
Zero-width spaces in WikiSource texts can be cleaned up with :
```bash
sed -i 's/\xe2\x80\x8b//g' data/poe/en/*/txt/*.txt
```

Author-specific details of post-processing cleanup that has been done:
- Poe
    - **Footnotes**: Excluded by manual cleanup, to focus on the body text and reduce noise.
    - **Editions**: The 1845 and 1850 texts show ~5% divergence (LCS analysis). Both are retained since we don't yet know which Baudelaire used for each story.
    - **Duplicates**: "Le Mystère de Marie Roget" appears in both `histoires_extraordinaires/` and `grotesques_serieuses/`; preliminarily these appear to be the same translation.
    - **OCR errors**: The Novalis quote in "Marie Roget," for example, has OCR/printing errors in both editions ("gewohulich" for "gewöhnlich", etc.); there are likely to be other cases as well.

- Maupassant
    - **McMaster segmentation**: The PG volumes have TOC/body title mismatches requiring a variant lookup table in `maupassant_en.py`.
    - **Partial coverage**: McMaster translated ~180 of Maupassant's ~300 stories. The parallel index shows which have matches.

## Sources and licensing

| Corpus | Source | License |
|--------|--------|---------|
| ELTeC | github.com/COST-ELTeC | CC-BY 4.0 |
| Maupassant (fr) | maupassant.free.fr | Public domain |
| Maupassant (en) | gutenberg.org | Public domain |
| Poe (en) | en.wikisource.org | Public domain |
| Poe (fr) | fr.wikisource.org | Public domain |

Maupassant French texts follow the Pléiade edition (Forestier 1987).

McMaster et al. translations are the "Complete Short Stories" series (PG #3077-3089).

Poe 1845 = *Tales* (published in Poe's lifetime). Poe 1850 = Griswold edition (posthumous, more complete).

## License

This project is licensed under CC BY 4.0. You are free to use, adapt, and redistribute the data with attribution.
