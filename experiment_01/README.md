# Experiment 01 — Authorship Fingerprinting

Baseline pipeline for attributing authorship of Czech Constitutional Court
dissenting opinions using stylometric features (function words, surface features,
character n-grams, POS n-grams). Achieves **69.7% accuracy** across 19 judges
using leave-one-out cross-validation.

See [methodology_and_results.md](methodology_and_results.md) for the full research
report including literature review, detailed methodology, and result analysis.

---

## Prerequisites

- **Python ≥ 3.11**
- **Poetry** (dependency manager)
- **Git LFS** (the dataset `subset_disent2.csv` is tracked via LFS)

## Setup

```bash
# 1. Ensure Git LFS data is pulled (from repo root)
git lfs pull

# 2. Install Python dependencies
cd experiment_01
poetry install
```

### UDPipe Czech Model

The pipeline requires the UDPipe 1 Czech-PDT model for tokenization, POS tagging,
and lemmatization.

1. Download `czech-pdt-ud-2.5-191206.udpipe` from
   [LINDAT](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3131)
   (direct link: look for the "Czech" section under "Universal Dependencies 2.5
   models").
2. Place it at:
   ```
   models/czech-pdt-ud-2.5-191206.udpipe   # relative to repo root
   ```
   Or pass a custom path via `--model-path`.

## Running the Pipeline

The pipeline has 4 steps: **load → udpipe → features → evaluate**. Each step saves
its output to `outputs/`, so you can resume from any step.

```bash
cd experiment_01

# Full pipeline (~8 min on first run; UDPipe processing takes ~5 min)
poetry run python scripts/run_baseline.py --min-dissents 5

# Skip UDPipe (reuse cached documents) and re-extract features + evaluate
poetry run python scripts/run_baseline.py --from-step features

# Only re-run evaluation (fastest, ~2 min)
poetry run python scripts/run_baseline.py --from-step evaluate

# Use a subset of feature sets
poetry run python scripts/run_baseline.py --from-step features --features function_words surface

# Custom UDPipe model path
poetry run python scripts/run_baseline.py --model-path /path/to/model.udpipe

# Stricter author threshold (fewer authors, more samples each)
poetry run python scripts/run_baseline.py --min-dissents 9
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--min-dissents` | `5` | Minimum dissents per author to be included |
| `--model-path` | `../models/czech-pdt-ud-2.5-191206.udpipe` | Path to UDPipe model |
| `--features` | all four | Feature sets: `function_words`, `surface`, `char_ngrams`, `pos_ngrams` |
| `--from-step` | `load` | Resume from: `load`, `udpipe`, `features`, or `evaluate` |

## Outputs

After a successful run, `outputs/` contains:

| File | Step | Description |
|------|------|-------------|
| `corpus.pkl` | load | Filtered DataFrame (pickle) |
| `documents.pkl` | udpipe | UDPipe-processed document objects (pickle) |
| `features.pkl` | features | Feature matrix as NumPy arrays (pickle) |
| `feature_matrix.csv` | features | Feature matrix as CSV (for inspection) |
| `results_summary.txt` | evaluate | Full evaluation report with per-author metrics |
| `confusion_matrices.json` | evaluate | Per-method confusion matrices (JSON) |
| `eval_results.pkl` | evaluate | Serialized result dicts (pickle) |

## Project Structure

```
experiment_01/
├── methodology_and_results.md  # Research report (literature, methodology, results)
├── pyproject.toml              # Poetry config (Python ≥3.11)
├── README.md                   # This file
├── outputs/                    # Pipeline outputs (auto-generated)
├── scripts/
│   └── run_baseline.py         # CLI pipeline runner (4 steps)
└── src/
    └── fingerprint/            # Python package
        ├── __init__.py
        ├── data_loader.py      # CSV loading, author filtering, summary stats
        ├── preprocessing.py    # Text cleaning, UDPipe tokenization/POS/lemma
        ├── features/
        │   ├── __init__.py
        │   ├── function_words.py   # 263 Czech function word frequencies
        │   ├── ngrams.py           # Character 3-grams (200) + POS bigrams (78)
        │   └── surface.py          # Sentence/word length, punctuation, TTR (22)
        ├── classifiers.py      # Burrows'/Cosine Delta, SGD SVM, SGD logistic, kNN
        └── evaluation.py       # LOO-CV, metrics, confusion matrix
```

## Data

Uses `subset_disent2.csv` from the repository root (314 single-author dissents,
35 judges). With the default `--min-dissents 5` threshold: **19 authors, 277 texts,
563 features**.

## Quick Results

| Method | Accuracy |
|--------|----------|
| Burrows' Delta | 63.2% (rank-3: 80.1%) |
| Cosine Delta | 64.3% (rank-3: 82.7%) |
| Linear SVM | 68.6% |
| **Logistic Regression** | **69.7%** |
| k-NN (k=3) | 41.9% |

Random baseline: 5.3% (1/19 authors). See `outputs/results_summary.txt` for full
per-author breakdown or [methodology_and_results.md](methodology_and_results.md)
for analysis.
