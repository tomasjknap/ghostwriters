# Experiment 01 — Methodology and Results

> **Authorship Fingerprinting for Czech Constitutional Court Dissenting Opinions**
>
> This document serves as a comprehensive research report for Experiment 01. It
> describes the motivation, literature background, methodology, implementation,
> and baseline results. Colleagues unfamiliar with the codebase should be able to
> understand the full approach by reading this document alone.

---

## 1. Research Question

Can we build reliable "author fingerprints" from single-authored dissenting opinions
of Czech Constitutional Court judges using stylometric features? If so, these
fingerprints can later be applied to majority opinions to estimate the probability
that the attributed judge actually wrote the text — i.e., to detect ghostwriting.

### 1.1 Hypothesis

Dissenting opinions are written by the judge they are attributed to. Majority
opinions, on the other hand, are often drafted by judicial assistants
(ghostwriters). If a stylometric model trained on dissents can reliably distinguish
between judges, the same model can be used to test whether the stylistic profile of
a majority opinion matches the attributed judge.

---

## 2. Literature Review

### 2.1 Rosenthal & Yoon (2011a) — *Judicial Ghostwriting*

**Paper**: [Cornell Law Review, Vol. 96, No. 6](https://scholarship.law.cornell.edu/clr/vol96/iss6/11/)

Legal framing of the ghostwriting problem in the U.S. Supreme Court context. Law
clerks draft most opinions, but the extent of their influence on the final text is
unknown. The paper argues that statistical stylometry can reveal which justices rely
more heavily on clerks.

**Key takeaway for our project**: The assumption that *personally authored* texts
(dissents, concurrences) are more stylistically authentic than majority opinions
directly mirrors our hypothesis for the Czech Constitutional Court.

### 2.2 Rosenthal & Yoon (2011b) — *Detecting Multiple Authorship Using Function Words*

**Paper**: [Annals of Applied Statistics, Vol. 5, No. 1](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-5/issue-1/Detecting-multiple-authorship-of-United-States-Supreme-Court-legal-decisions/10.1214/10-AOAS378.full) · [arXiv:1104.2974](https://arxiv.org/abs/1104.2974)

- **Features**: 63 English function words. Their frequency is hypothesized to be
  *topic-invariant*, reflecting writing style rather than subject matter.
- **Method**: χ² statistic on bootstrapped function-word counts. High within-author
  variance suggests multiple authors (justice + clerk collaboration).
- **Data**: ~2,400 SCOTUS majority opinions spanning ~30 years.
- **Strengths**: Simple, interpretable, grounded in the Mosteller–Wallace tradition.
- **Limitations**: Language-specific word lists; ignores phrase-level patterns.

A replication by [Hampton (GitHub)](https://github.com/jphampton/capstone-SC-NLP)
found that **TF-IDF n-gram features outperform function words** for pairwise
attribution, suggesting phrase-level features capture additional authorial signal.

### 2.3 Avraham et al. (2025) — *Identifying Authorship in Unsigned Opinions*

**Paper**: [Journal of Legal Analysis, Vol. 17, No. 1](https://academic.oup.com/jla/article/17/1/2/8098035)

- Fine-tuned a **large language model** on 4,069 signed SCOTUS opinions from 17
  justices. Achieved 91% accuracy.
- Tool publicly available: [SCOTUS_AID](https://raminass.github.io/SCOTUS_AI/).
- **Relevance**: We have far fewer texts per author (max 32 vs. ~240). Full LLM
  fine-tuning would likely overfit. Using a pre-trained Czech model (RobeCzech)
  as a **feature extractor** is a more viable path for future experiments.

---

## 3. Data

### 3.1 Source

The dataset (`subset_disent2.csv`) was prepared by project collaborators from the
[Czech Constitutional Court dataset](https://github.com/stepanpaulik/ccc_dataset).
It contains 314 single-author dissenting opinions from 35 judges.

**Columns**: `doc_id`, `text`, `date_decision`, `date_submission`, `type_decision`,
`separate_opinion` (judge name), `formation`, `length_proceeding`,
`separate_opinion_extracted` (the actual dissent text, extracted from HTML).

### 3.2 Author filtering

We include only judges with ≥ 5 dissenting opinions, yielding **19 authors and 277
texts**. The distribution is highly skewed:

| Author | Dissents | Avg words |
|--------|----------|-----------|
| Ivana Janů | 32 | ~2,300 |
| Jan Filip | 31 | ~4,700 |
| Radovan Suchánek | 30 | ~3,800 |
| Jan Musil | 23 | ~2,500 |
| Eliška Wagnerová | 22 | ~3,900 |
| Josef Fiala | 21 | ~3,800 |
| Stanislav Balík | 17 | ~3,100 |
| Pavel Varvařovský | 14 | ~2,800 |
| Jiří Zemánek | 12 | ~3,400 |
| Ludvík David | 11 | ~2,500 |
| Vladimír Kůrka | 10 | ~3,100 |
| Miloslav Výborný | 9 | ~3,500 |
| Vojtěch Šimíček | 9 | ~2,200 |
| Iva Brožová | 9 | ~1,900 |
| David Uhlíř | 6 | ~2,300 |
| Kateřina Šimáčková | 6 | ~4,000 |
| Vladimír Sládeček | 5 | ~2,000 |
| Michaela Židlická | 5 | ~1,400 |
| Jiří Nykodým | 5 | ~1,200 |

This class imbalance (5–32 samples per author) is a key challenge.

---

## 4. Methodology

### 4.1 Authorship tasks

| Task | Description | Status in Experiment 01 |
|------|-------------|------------------------|
| **Closed-set attribution** | Pick the most likely author from 19 candidates | ✅ Implemented, evaluated |
| **Authorship verification** | Estimate P(author wrote this) for a given judge | Planned for future experiments |
| **Open-set attribution** | Rank authors + allow "none of the above" | Rank-3 accuracy reported |
| **Multiple authorship detection** | Detect collaborative writing | Future work |

### 4.2 Preprocessing pipeline

Implemented in `src/fingerprint/preprocessing.py`:

1. **Text cleaning** (`clean_text()`): Normalize whitespace, strip residual HTML
   entities (`&amp;` etc.), normalize dashes (en-dash, em-dash → hyphen).
2. **Tokenization, POS tagging, lemmatization**: UDPipe 1 with the
   `czech-pdt-ud-2.5-191206` model. The pipeline runs in `tokenize` mode and
   outputs CoNLL-U format, which is parsed into `Document → Sentence → Token`
   dataclasses.
3. **Token representation**: Each token carries `form`, `lemma`, `upos` (Universal
   POS), `xpos` (Czech-specific POS), and `feats` (morphological features).

No text chunking or length normalization was applied in this baseline — feature
vectors are computed over the entire document and normalized by document length
(relative frequencies).

### 4.3 Feature sets

We implemented four feature sets totalling **563 features**:

#### 4.3.1 Czech function words (263 features)

File: `src/fingerprint/features/function_words.py`

A curated list of 263 Czech function words organized into 8 categories:
- **Personal pronouns** (40): já, ty, on, ona, mě, mi, ho, mu, ...
- **Reflexive pronouns** (5): se, si, sebe, sobě, sebou
- **Demonstrative pronouns** (21): ten, ta, to, tím, tohoto, tento, ...
- **Relative pronouns** (27): který, která, jenž, jehož, co, kdo, ...
- **Indefinite pronouns** (17): něco, někdo, každý, všechen, žádný, ...
- **Prepositions** (35): v, ve, na, do, z, ze, s, k, o, u, po, za, od, ...
- **Conjunctions** (40): a, ale, nebo, že, aby, protože, když, pokud, ...
- **Auxiliary verbs** (23): být, je, jsou, byl, jsem, bude, by, bych, ...
- **Particles** (30): asi, jen, již, ještě, také, pouze, zejména, ...
- **Other** (17): tak, pak, tam, zde, nyní, dále, ne, ano, ...

Features are **relative frequencies**: for each word, `count(word) / total_tokens`.
Matching is performed on **lemmas** (lowercased) to handle Czech inflection.

Some words overlap between categories (e.g., "se" is both a reflexive pronoun and
a preposition). After deduplication, the flat sorted set of unique lemmas defines
the feature vector.

#### 4.3.2 Surface features (22 features)

File: `src/fingerprint/features/surface.py`

- **Sentence length** (6): mean, std, median, min, max, count
- **Word length** (3): mean, std, median (excluding punctuation tokens)
- **Punctuation frequency** (11): per-1000-token rate for: `, . ; : ! ? - ( ) " …`
- **Vocabulary richness** (3): type-token ratio on forms, type-token ratio on
  lemmas, hapax legomena ratio

#### 4.3.3 Character trigrams (200 features)

File: `src/fingerprint/features/ngrams.py`

Character 3-grams extracted from the concatenation of lowercased token forms
(space-separated to preserve word boundaries). A **corpus-wide vocabulary** of the
top 200 most frequent trigrams is built first, then each document is represented
as relative frequencies over this fixed vocabulary.

Character n-grams are especially effective for morphologically rich languages like
Czech, as they capture sub-word patterns (suffixes, prefixes, common morphemes)
without requiring explicit morphological analysis.

#### 4.3.4 POS bigrams (78 features)

File: `src/fingerprint/features/ngrams.py`

Universal POS tag bigrams (e.g., `NOUN_VERB`, `ADP_NOUN`) extracted per sentence.
Top 100 most frequent bigrams across the corpus form the vocabulary; each document
is represented as relative frequencies. (78 features after pruning to actually
observed bigrams.)

POS n-grams capture syntactic style independent of vocabulary — a judge who
consistently uses particular sentence structures will have a distinctive POS
bigram profile regardless of the legal topic.

### 4.4 Classification methods

Implemented in `src/fingerprint/classifiers.py`:

| Method | Implementation | Details |
|--------|---------------|---------|
| **Burrows' Delta** | Custom (NumPy) | `mean(\|z_test - z_author\|)` on z-scored features. Rank by ascending distance. |
| **Cosine Delta** | Custom (NumPy) | Cosine distance on z-scored features. Smith & Aldridge variant. |
| **Linear SVM** | `SGDClassifier(loss="hinge")` | Stochastic gradient descent with hinge loss (equivalent to linear SVM). `max_iter=1000`, `tol=1e-3`. Features StandardScaled per LOO fold. |
| **Logistic regression** | `SGDClassifier(loss="log_loss")` | SGD with log loss. Same parameters as SVM. StandardScaled. |
| **k-NN** | `KNeighborsClassifier(k=3)` | 3 nearest neighbors. StandardScaled. |

**Note on SGDClassifier**: We initially used `LinearSVC` and `LogisticRegression`
from scikit-learn, but they were prohibitively slow for LOO-CV (277 iterations × 563
features). SGDClassifier with equivalent loss functions completes in seconds.

### 4.5 Evaluation

Implemented in `src/fingerprint/evaluation.py`:

- **Leave-one-out cross-validation (LOO-CV)**: Each of the 277 documents is held out
  once as the test sample; the remaining 276 are used for training. This maximizes
  the training set size, which is critical given the small per-author counts.
- **Feature scaling**: For sklearn classifiers, `StandardScaler` is fit on each
  training fold and applied to the test sample (no data leakage).
- **Delta methods**: Author profiles (mean feature vectors) are recomputed on each
  training fold, excluding the held-out sample.

**Metrics reported**:
- Rank-1 accuracy (overall accuracy)
- Rank-3 accuracy (true author in top 3 predictions — Delta methods only)
- Per-author precision, recall, F1
- Macro and weighted averages
- Confusion matrix

---

## 5. Results

### 5.1 Summary

| Method | Accuracy | Rank-3 Acc | Macro F1 | Weighted F1 |
|--------|----------|------------|----------|-------------|
| Burrows' Delta | 63.2% | 80.1% | 0.57 | 0.62 |
| Cosine Delta | 64.3% | 82.7% | 0.55 | 0.63 |
| Linear SVM (SGD hinge) | 68.6% | — | 0.60 | 0.67 |
| **Logistic (SGD log loss)** | **69.7%** | — | **0.60** | **0.68** |
| k-NN (k=3) | 41.9% | — | 0.31 | 0.39 |

**Best method**: Logistic regression (SGD) at **69.7% accuracy** across 19 authors.
A random baseline would yield ~5.3% (1/19), so even the worst method (kNN at 41.9%)
is far above chance.

**Rank-3 accuracy** for Delta methods reaches **82.7%** (Cosine Delta), meaning the
true author is among the top 3 candidates in over 4 out of 5 cases.

### 5.2 Per-author results (best method: logistic regression)

| Author | Dissents | Precision | Recall | F1 |
|--------|----------|-----------|--------|-----|
| Jan Filip | 31 | 0.91 | 1.00 | **0.95** |
| Jan Musil | 23 | 0.82 | 1.00 | **0.90** |
| Josef Fiala | 21 | 0.82 | 0.86 | **0.84** |
| Vojtěch Šimíček | 9 | 0.88 | 0.78 | **0.82** |
| Miloslav Výborný | 9 | 1.00 | 0.67 | **0.80** |
| Pavel Varvařovský | 14 | 0.83 | 0.71 | **0.77** |
| Jiří Zemánek | 12 | 0.89 | 0.67 | **0.76** |
| Iva Brožová | 9 | 1.00 | 0.56 | 0.71 |
| Radovan Suchánek | 30 | 0.61 | 0.83 | 0.70 |
| Eliška Wagnerová | 22 | 0.58 | 0.64 | 0.61 |
| Ludvík David | 11 | 0.67 | 0.55 | 0.60 |
| Michaela Židlická | 5 | 1.00 | 0.40 | 0.57 |
| Ivana Janů | 32 | 0.45 | 0.69 | 0.54 |
| Vladimír Kůrka | 10 | 0.80 | 0.40 | 0.53 |
| Stanislav Balík | 17 | 0.50 | 0.53 | 0.51 |
| Kateřina Šimáčková | 6 | 1.00 | 0.33 | 0.50 |
| Vladimír Sládeček | 5 | 0.50 | 0.20 | 0.29 |
| David Uhlíř | 6 | 0.00 | 0.00 | 0.00 |
| Jiří Nykodým | 5 | 0.00 | 0.00 | 0.00 |

### 5.3 Analysis

**Well-classified authors** (F1 ≥ 0.75):
- **Jan Filip** (31 texts, F1 = 0.95): The most distinctive writer in the corpus.
  Perfect recall — every Filip dissent is correctly attributed. His writing style is
  highly recognizable.
- **Jan Musil** (23 texts, F1 = 0.90): Near-perfect. All of his texts are correctly
  recalled; only a few false positives from other authors.
- **Josef Fiala** (21 texts, F1 = 0.84): Strong performance with a larger sample.
- **Vojtěch Šimíček** (9 texts, F1 = 0.82): Good performance even with a small
  sample, suggesting a distinctive style.

**Poorly classified authors** (F1 < 0.30):
- **David Uhlíř** (6 texts, F1 = 0.00): Zero recall. His dissents are systematically
  misclassified as other authors, likely due to too few training samples and/or a
  less distinctive style.
- **Jiří Nykodým** (5 texts, F1 = 0.00): Same issue — the minimum sample size
  combined with possible stylistic overlap.

**Observations**:
1. **Sample size matters**: Authors with ≥ 20 dissents generally perform well. Below
   ~9 samples, performance drops sharply.
2. **Class imbalance**: Authors with large corpora (Janů: 32, Suchánek: 30) attract
   false positives from small-sample authors, inflating their recall but reducing
   precision.
3. **Ivana Janů anomaly**: Despite having the most dissents (32), her F1 is only
   0.54. This could indicate genuinely high within-author stylistic variance — a
   possible signal of ghostwriting even in dissents, or simply a less distinctive
   personal style.
4. **kNN underperformance**: k-NN at 41.9% is far below the linear methods,
   suggesting the feature space is high-dimensional and sparse — distance-based
   methods with few neighbors struggle in 563 dimensions.

---

## 6. Implementation

### 6.1 Project structure

```
experiment_01/
├── methodology_and_results.md  # This document
├── pyproject.toml              # Poetry project config (Python ≥3.11)
├── README.md                   # Setup & replication instructions
├── outputs/                    # Pipeline outputs (auto-generated)
│   ├── corpus.pkl              # Filtered DataFrame (step: load)
│   ├── documents.pkl           # UDPipe-processed documents (step: udpipe)
│   ├── features.pkl            # Feature matrix as NumPy arrays (step: features)
│   ├── feature_matrix.csv      # Feature matrix as CSV (step: features)
│   ├── results_summary.txt     # Full evaluation report (step: evaluate)
│   ├── confusion_matrices.json # Per-method confusion matrices (step: evaluate)
│   └── eval_results.pkl        # Serialized result dicts (step: evaluate)
├── scripts/
│   └── run_baseline.py         # CLI pipeline runner (4 steps)
└── src/
    └── fingerprint/            # Python package
        ├── __init__.py
        ├── data_loader.py      # CSV loading, author filtering, summary stats
        ├── preprocessing.py    # Text cleaning, UDPipe wrapper
        ├── features/
        │   ├── __init__.py
        │   ├── function_words.py   # Czech function word frequencies (263 features)
        │   ├── ngrams.py           # Character 3-grams (200) + POS bigrams (78)
        │   └── surface.py          # Sentence/word length, punctuation, TTR (22)
        ├── classifiers.py      # Delta methods + sklearn wrappers
        └── evaluation.py       # LOO-CV, metrics, confusion matrix
```

### 6.2 Pipeline

The pipeline (`scripts/run_baseline.py`) consists of 4 steps, each saving its output
to `outputs/`. The `--from-step` flag allows resuming from any step:

| Step | Name | Input | Output | Duration |
|------|------|-------|--------|----------|
| 1 | `load` | `subset_disent2.csv` | `outputs/corpus.pkl` | ~2 s |
| 2 | `udpipe` | corpus.pkl + UDPipe model | `outputs/documents.pkl` | ~5 min |
| 3 | `features` | documents.pkl | `outputs/features.pkl`, `feature_matrix.csv` | ~5 s |
| 4 | `evaluate` | features.pkl | `results_summary.txt`, `confusion_matrices.json` | ~2 min |

Example invocations:

```bash
# Full pipeline (first run, ~8 min)
poetry run python scripts/run_baseline.py --min-dissents 5

# Re-evaluate with existing features (skip UDPipe, ~2 min)
poetry run python scripts/run_baseline.py --from-step evaluate

# Re-extract features with different feature sets, then evaluate
poetry run python scripts/run_baseline.py --from-step features --features function_words surface

# Custom minimum dissent threshold
poetry run python scripts/run_baseline.py --min-dissents 9
```

### 6.3 Dependencies

Managed via Poetry (`pyproject.toml`):

| Package | Purpose |
|---------|---------|
| `ufal.udpipe` | Tokenization, POS tagging, lemmatization |
| `numpy` | Feature matrix operations |
| `pandas` | Data loading and manipulation |
| `scikit-learn` | Classifiers (SGDClassifier, KNeighborsClassifier), StandardScaler, metrics |

Python ≥ 3.11 is required (scikit-learn dependency).

### 6.4 External model

**UDPipe Czech PDT model** (`czech-pdt-ud-2.5-191206.udpipe`): A pre-trained model
for Czech from the [LINDAT repository](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3131).
Provides tokenization, sentence segmentation, POS tagging, lemmatization, and
dependency parsing. Placed in `models/` at the repository root (shared across
experiments).

---

## 7. Limitations and Future Work

### 7.1 Current limitations

- **Small sample sizes**: Authors with < 10 dissents are poorly classified. The
  minimum threshold of 5 may be too low for reliable attribution.
- **No embedding features**: Layer 3 (RobeCzech embeddings) was not implemented in
  this baseline. Deep contextual features may capture authorial patterns that
  surface features miss.
- **No text chunking**: Long dissents (some exceed 10,000 words) contribute
  disproportionately to feature statistics. Chunking into fixed-size segments
  could provide more balanced representation and more training samples.
- **No feature selection**: All 563 features are used. Ablation studies and
  feature importance analysis could identify the most discriminative features.
- **kNN sensitivity**: The poor kNN performance suggests the feature space may
  benefit from dimensionality reduction (PCA) before distance-based methods.

### 7.2 Planned improvements (future experiments)

1. **RobeCzech embeddings**: Extract mean-pooled token embeddings as dense feature
   vectors. May substantially improve accuracy.
2. **Feature ablation**: Test each feature set independently and in combinations to
   understand their individual contributions.
3. **Higher minimum threshold**: Test with ≥ 9 or ≥ 10 dissents per author to
   improve per-author reliability at the cost of fewer candidate authors.
4. **Text chunking**: Split long documents into ~2,000-word segments to increase
   sample count and reduce length effects.
5. **Authorship verification**: Train one-vs-rest models per judge. Apply to
   majority opinions from `subset.csv` to detect ghostwriting.
6. **Within-author variance analysis**: Following Rosenthal & Yoon, compute
   bootstrapped χ² variance for each judge's function-word usage to identify
   judges with suspiciously high stylistic variation (possible ghostwriting signal).

---

## 8. References

1. Rosenthal, J. S. & Yoon, A. H. (2011). *Judicial Ghostwriting: Authorship on
   the Supreme Court*. Cornell Law Review, 96(6), 1307.
2. Rosenthal, J. S. & Yoon, A. H. (2011). *Detecting Multiple Authorship of United
   States Supreme Court Legal Decisions Using Function Words*. Annals of Applied
   Statistics, 5(1), 283–308.
3. Avraham, R., Nasser, R., Kohn, I., Kricheli-Katz, T. & Sharan, R. (2025).
   *Lifting the American Supreme Court Veil: Identifying Authorship in Unsigned
   Opinions*. Journal of Legal Analysis, 17(1), 2–13.
4. Mosteller, F. & Wallace, D. L. (1963). *Inference in an Authorship Problem*.
   Journal of the American Statistical Association, 58(302), 275–309.
5. Hampton, J. P. *NLP on Supreme Court Opinions: Authorship Attribution,
   Multiauthor Detection, and Document Similarity*.
   [GitHub](https://github.com/jphampton/capstone-SC-NLP).
6. Burrows, J. (2002). *'Delta': A Measure of Stylistic Difference and a Guide to
   Likely Authorship*. Literary and Linguistic Computing, 17(3), 267–287.
