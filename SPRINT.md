## Backlog

Prioritized list of topics for future work. Items near the top are higher
priority (foundations first, speculative ideas last).

### High priority — foundations

- [ ] **Feature ablation study**: Run LOO-CV with each of the 4 feature sets
  independently and in combinations. Understand what we have before adding more.
  Already supported via `--features` flag. (~0.5 day)
- [ ] **Text chunking / length normalization**: Split long dissents into
  fixed-size segments (~2,000 words). Benefits: more training samples, removes
  document length as a confound. Could substantially help small-author
  classification. (~1 day)
- [ ] **Probability output + confidence calibration**: Extract `predict_proba()`
  from logistic regression, convert Delta distances to probabilities via softmax.
  Add Platt scaling / isotonic regression for calibration. Critical for any real
  ghostwriting claims. (~1 day)
- [ ] **Within-author variance analysis**: Rosenthal & Yoon bootstrapped χ²
  approach on function-word usage per judge. High variance = possible ghostwriting
  even in dissents. Directly addresses the Janů anomaly (32 texts, F1=0.54).
  (~0.5 day)

### Medium priority — easy wins

- [ ] **XGBoost classifier**: Gradient boosting handles feature interactions and
  class imbalance better than linear models. Needs careful regularization with
  small n. (~0.5 day)
- [ ] **Feature importance / author signatures**: Per-author discriminative
  features. Options: logistic regression weights, permutation importance, SHAP
  values with XGBoost. Key for interpretability. (~1–2 days)
- [ ] **Save fingerprints for application**: Serialize trained models + scaler +
  feature vocabulary. Build `predict.py` to score new texts (majority opinions).
  Bridge to the actual ghostwriting detection goal. (~0.5 day)
- [ ] **Readability / comprehensibility metrics**: Czech-adapted readability
  index, dependency tree depth (from UDPipe), subordinate clause ratio, passive
  voice frequency. Genuine style markers, low-hanging fruit. (~1 day)
- [ ] **Higher minimum threshold experiments**: Test with ≥9 or ≥10 dissents per
  author. Fewer authors but more reliable per-author estimates. (~0.5 day)
- [ ] **Hyperparameter tuning**: Nested CV (inner loop for selection, outer LOO
  for evaluation). Especially important for XGBoost. (~1 day)

### Lower priority — exploratory

- [ ] **RobeCzech embeddings**: Mean-pooled [CLS] embeddings as 768-dim feature
  vector. High potential but overfitting risk with 277 samples — PCA first. GPU
  recommended. (~1.5 days)
- [ ] **Legal-domain specific features**: Citation patterns, formulaic phrase
  usage, legal terminology density. Risk of topic contamination if judges handle
  different case types. Needs careful scoping. (~2–4 days)
- [ ] **Topic-semantic analysis**: Latin quotes, judicate references, European
  Court citations. Similar topic-contamination risk as above. Latin quotes
  specifically may be a genuine style marker. (~2 days)
- [ ] **Speech acts / rhetorical features**: Hedging expressions, assertiveness
  markers, rhetorical questions, imperative constructions. Theoretically
  appealing but limited Czech tooling. Scope to lexicon-based approach first.
  (~2–5 days)

---

## Sprint 2 — 2026-02-19 → 2026-02-26

### Goal

Strengthen the baseline with ablation analysis, improve handling of small-sample
authors via text chunking, and prepare the pipeline for ghostwriting detection by
adding probability output and saving trained fingerprints.

### Tasks

1. **Feature ablation study**
   - Run LOO-CV with each feature set alone: function_words, surface,
     char_ngrams, pos_ngrams
   - Run key combinations (e.g., function_words + surface, all minus one)
   - Document which features contribute most to overall accuracy
   - Output: ablation results table in methodology_and_results.md

2. **Text chunking**
   - Implement document splitting into ~2,000-word segments
   - Re-run pipeline with chunked texts → more training samples
   - Compare accuracy with/without chunking
   - Handle edge case: chunks inherit the parent document's author label

3. **Probability output + calibration**
   - Expose `predict_proba()` from logistic regression
   - Convert Delta distances to probabilities via softmax
   - Add confidence calibration (Platt scaling)
   - Output: per-document probability distributions over all 19 authors

4. **Save fingerprints**
   - Serialize trained model, scaler, and feature vocabulary
   - Build `predict.py` script to load fingerprint and score new texts
   - Test on held-out dissents as sanity check

5. **Within-author variance analysis**
   - Implement bootstrapped χ² variance per judge (Rosenthal & Yoon method)
   - Rank judges by stylistic consistency
   - Flag high-variance authors for further investigation

### Deliverables

- Updated `methodology_and_results.md` with ablation results and variance analysis
- Chunking implementation in `preprocessing.py`
- `predict.py` script for scoring new texts
- Probability-calibrated output per document

---

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
