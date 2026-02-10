## Sprint 1 — 2026-02-09 → 2026-02-16

### Goal

Build a proof-of-concept **author fingerprint** pipeline: extract stylometric
features from single-authored dissenting opinions and evaluate whether they can
distinguish between authors. This is a stepping stone toward detecting
ghostwriting in majority opinions.

### Hypothesis

Dissenting opinions are written by the judge they are attributed to. Majority
opinions are often written by assistants (ghostwriters). If we can build
reliable author fingerprints from dissents, we can later test majority opinions
against those fingerprints to estimate the probability of true authorship.

### Context

- **Data**: `subset_disent2.csv` (314 single-author dissents, 35 judges)
- **Target authors**: all judges with ≥ 5 dissents (19 authors, 277 texts after filtering)
- **Stack**: Python 3.11 + Poetry, each experiment in its own `experiment_xx/` subfolder
- **NLP tools**: ÚFAL ecosystem — UDPipe (Czech-PDT 2.5 model)

### Tasks

1. **Literature review** → `experiment_01/methodology_and_results.md` §2 ✅
   - Rosenthal & Yoon (2011a) — *Judicial Ghostwriting* (legal framing)
   - Rosenthal & Yoon (2011b) — *Detecting Multiple Authorship* (function words + χ² variance)
   - Avraham et al. (2025) — LLM-based authorship attribution (91% on SCOTUS)
   - Hampton replication — TF-IDF n-grams outperform function words

2. **Data loading & EDA** ✅
   - Loaded `subset_disent2.csv`, profiled all 35 authors
   - Minimum-dissent threshold: ≥ 5 → 19 authors, 277 texts
   - Word count range: ~1,200–16,000 words per dissent

3. **Feature extraction** → 563 features ✅
   - Czech function word frequencies (263 features, lemma-matched)
   - Surface features (22): sentence/word length, punctuation, vocabulary richness
   - Character trigrams (200): corpus-wide top-200 vocabulary
   - POS bigrams (78): Universal POS tag bigrams via UDPipe

4. **Baseline evaluation** → LOO-CV ✅
   - **Best result: 69.7% accuracy** (logistic regression, 19-way classification)
   - Burrows' Delta: 63.2%, Cosine Delta: 64.3% (rank-3: 82.7%)
   - Linear SVM (SGD): 68.6%, Logistic (SGD): 69.7%, kNN: 41.9%
   - Random baseline: 5.3%

### Deliverables

- ✅ `experiment_01/methodology_and_results.md` — full research report (literature, methodology, implementation, results)
- ✅ `experiment_01/` — Python package (Poetry) with modular pipeline (`--from-step` for resuming)
- ✅ `experiment_01/outputs/` — results summary, confusion matrices, feature matrix
- ✅ `experiment_01/README.md` — replication instructions

### Key findings

- Jan Filip (31 texts) and Jan Musil (23 texts) are the most distinctive writers (F1 > 0.90).
- Authors with ≤ 6 dissents are poorly classified (F1 ≈ 0).
- Ivana Janů (32 texts, F1 = 0.54) shows unexpectedly high stylistic variance — possible ghostwriting signal even in dissents.
- kNN struggles in the high-dimensional feature space (563 dims); linear methods dominate.

### Next steps (Sprint 2)

- [ ] Feature ablation: test each feature set independently
- [ ] RobeCzech embeddings as additional feature layer
- [ ] Text chunking for long documents
- [ ] Higher minimum threshold experiments (≥ 9, ≥ 10)
- [ ] Authorship verification: apply fingerprints to majority opinions
- [ ] Pull `subset.csv` via Git LFS for majority opinion analysis

### Resolved questions

- **Q1 (author selection)**: Broadened to all authors with ≥ 5 dissents (19 authors).
- **Q2 (multi-author)**: Multi-author closed-set classification with 19-way LOO-CV. Rank-3 accuracy also reported.
- **Q3 (Czech NLP)**: Using UDPipe with Czech-PDT 2.5 model for tokenization, POS tagging, and lemmatization.
- **Q4 (preprocessing)**: Built a preprocessing module: text cleaning → UDPipe → Document/Sentence/Token dataclasses.
- **Q5 (end goal)**: Majority opinion ghostwriting detection. `subset.csv` needs Git LFS pull.
