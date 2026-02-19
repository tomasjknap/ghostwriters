#!/usr/bin/env python3
"""Run the baseline authorship attribution pipeline.

The pipeline has 4 steps, each saving its output to outputs/:

  1. load       Load & filter CSV             -> outputs/corpus.pkl
  2. udpipe     Tokenize/tag with UDPipe      -> outputs/documents.pkl
  3. features   Extract feature matrix        -> outputs/feature_matrix.csv
  4. evaluate   LOO-CV with all classifiers   -> outputs/results_summary.txt
                                                 outputs/confusion_matrices.json

Use --from-step to resume from any step (earlier outputs are loaded from disk).

Examples:
    poetry run python scripts/run_baseline.py                  # full run
    poetry run python scripts/run_baseline.py --from-step evaluate
    poetry run python scripts/run_baseline.py --from-step features --features function_words surface
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fingerprint.classifiers import build_sklearn_classifier
from fingerprint.data_loader import author_summary, load_dissents
from fingerprint.evaluation import (
    loo_delta_evaluation,
    loo_sklearn_evaluation,
    print_results,
)
from fingerprint.features.function_words import (
    function_word_feature_names,
    function_word_frequencies,
)
from fingerprint.features.ngrams import (
    build_ngram_vocab_from_corpus,
    character_ngram_profile,
    pos_ngram_profile,
)
from fingerprint.features.surface import all_surface_features
from fingerprint.preprocessing import UDPipeProcessor, clean_text

STEPS = ["load", "udpipe", "features", "evaluate"]
OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline authorship attribution pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--min-dissents", type=int, default=5,
        help="Minimum dissents per author (default: 5)",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to UDPipe Czech model (.udpipe file)",
    )
    parser.add_argument(
        "--features", type=str, nargs="+",
        default=["function_words", "surface", "char_ngrams", "pos_ngrams"],
        choices=["function_words", "surface", "char_ngrams", "pos_ngrams"],
        help="Feature sets to use",
    )
    parser.add_argument(
        "--from-step", type=str, default="load",
        choices=STEPS,
        help="Resume pipeline from this step (default: load = full run)",
    )
    return parser.parse_args()


# ── Helpers ──────────────────────────────────────────────────

def step_header(name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"Step: {name}")
    print("=" * 60)


def save_pickle(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  -> Saved {path.name}")


def load_pickle(path: Path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"  <- Loaded {path.name}")
    return obj


# ── Step implementations ─────────────────────────────────────

def step_load(args) -> pd.DataFrame:
    """Step 1: Load and filter the CSV dataset."""
    step_header("load")
    df = load_dissents(min_dissents=args.min_dissents)
    print(f"  {len(df)} dissents from {df['separate_opinion'].nunique()} authors\n")
    print(author_summary(df).to_string())
    save_pickle(df, OUTPUTS_DIR / "corpus.pkl")
    return df


def step_udpipe(args, df: pd.DataFrame) -> list:
    """Step 2: Process all texts with UDPipe."""
    step_header("udpipe")
    model_path = Path(args.model_path) if args.model_path else None
    try:
        processor = UDPipeProcessor(model_path=model_path)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)
    print("  UDPipe model loaded.")

    documents = []
    n = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        text = clean_text(str(row["separate_opinion_extracted"]))
        doc = processor.process(text, doc_id=str(row["doc_id"]))
        documents.append(doc)
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  Processed {i + 1}/{n}", flush=True)

    save_pickle(documents, OUTPUTS_DIR / "documents.pkl")
    return documents


def step_features(args, df: pd.DataFrame, documents: list) -> tuple:
    """Step 3: Extract feature matrix from processed documents."""
    step_header("features")
    feature_names: list[str] = []
    feature_matrix: list[list[float]] = []

    char_vocab = None
    pos_vocab = None
    if "char_ngrams" in args.features:
        print("  Building character 3-gram vocabulary...")
        char_vocab = build_ngram_vocab_from_corpus(
            documents, character_ngram_profile, n=3, top_k=200,
        )
    if "pos_ngrams" in args.features:
        print("  Building POS 2-gram vocabulary...")
        pos_vocab = build_ngram_vocab_from_corpus(
            documents, pos_ngram_profile, n=2, top_k=100,
        )

    for doc in documents:
        vec: list[float] = []
        names: list[str] = []

        if "function_words" in args.features:
            fw = function_word_frequencies(doc)
            vec.extend(fw.tolist())
            if not feature_names:
                names.extend(function_word_feature_names())

        if "surface" in args.features:
            sf = all_surface_features(doc)
            vec.extend(sf.values())
            if not feature_names:
                names.extend(sf.keys())

        if "char_ngrams" in args.features and char_vocab:
            cng = character_ngram_profile(doc, n=3, vocab=char_vocab)
            vec.extend(cng.values())
            if not feature_names:
                names.extend([f"char3_{ng}" for ng in char_vocab])

        if "pos_ngrams" in args.features and pos_vocab:
            png = pos_ngram_profile(doc, n=2, vocab=pos_vocab)
            vec.extend(png.values())
            if not feature_names:
                names.extend([f"pos2_{ng}" for ng in pos_vocab])

        feature_matrix.append(vec)
        if not feature_names:
            feature_names = names

    X = np.array(feature_matrix)
    y = df["separate_opinion"].values
    print(f"  Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"  Feature sets: {args.features}")

    # Save as CSV for inspection
    feat_df = pd.DataFrame(X, columns=feature_names)
    feat_df.insert(0, "doc_id", df["doc_id"].values)
    feat_df.insert(1, "author", y)
    feat_df.to_csv(OUTPUTS_DIR / "feature_matrix.csv", index=False)
    print(f"  -> Saved feature_matrix.csv")

    # Also pickle the numpy arrays for fast reload
    save_pickle({"X": X, "y": y, "feature_names": feature_names}, OUTPUTS_DIR / "features.pkl")
    return X, y, feature_names


def step_evaluate(args, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
    """Step 4: Run LOO-CV evaluation with all classifiers."""
    step_header("evaluate")
    all_results = []

    # Delta methods
    for method in ["burrows", "cosine"]:
        results = loo_delta_evaluation(X, y, method=method)
        print("-" * 40)
        print_results(results)
        print()
        all_results.append(results)

    # sklearn classifiers
    for clf_name in ["svm", "logistic", "knn"]:
        print("-" * 40)
        results = loo_sklearn_evaluation(
            X, y,
            classifier_factory=lambda name=clf_name: build_sklearn_classifier(name),
        )
        print_results(results)
        print()
        all_results.append(results)

    # Save summary
    summary_path = OUTPUTS_DIR / "results_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Baseline authorship attribution results\n")
        f.write(f"Authors: {len(np.unique(y))} (min {args.min_dissents} dissents)\n")
        f.write(f"Samples: {X.shape[0]}, Features: {X.shape[1]}\n")
        f.write(f"Feature sets: {args.features}\n")
        f.write(f"{'=' * 60}\n\n")
        for r in all_results:
            f.write(f"Method: {r['method']}\n")
            f.write(f"Accuracy: {r['accuracy']:.3f}\n")
            if "rank3_accuracy" in r:
                f.write(f"Rank-3 accuracy: {r['rank3_accuracy']:.3f}\n")
            f.write(f"\n{r['classification_report']}\n")
            f.write(f"{'-' * 40}\n\n")
    print(f"  -> Saved results_summary.txt")

    # Save confusion matrices
    labels = list(np.unique(y))
    cm_data = {}
    for r in all_results:
        cm_data[r["method"]] = {
            "labels": labels,
            "matrix": r["confusion_matrix"].tolist(),
            "accuracy": r["accuracy"],
        }
    with open(OUTPUTS_DIR / "confusion_matrices.json", "w") as f:
        json.dump(cm_data, f, indent=2, ensure_ascii=False)
    print(f"  -> Saved confusion_matrices.json")

    # Save per-method pickle for later analysis
    save_pickle(all_results, OUTPUTS_DIR / "eval_results.pkl")


# ── Main ─────────────────────────────────────────────────────

def main():
    args = parse_args()
    OUTPUTS_DIR.mkdir(exist_ok=True)

    start_idx = STEPS.index(args.from_step)

    # ── Step 1: load ──
    if start_idx <= 0:
        df = step_load(args)
    else:
        print(f"\n  Skipping step 'load' — loading from outputs/corpus.pkl")
        df = load_pickle(OUTPUTS_DIR / "corpus.pkl")

    # ── Step 2: udpipe ──
    if start_idx <= 1:
        documents = step_udpipe(args, df)
    else:
        print(f"  Skipping step 'udpipe' — loading from outputs/documents.pkl")
        documents = load_pickle(OUTPUTS_DIR / "documents.pkl")

    # ── Step 3: features ──
    if start_idx <= 2:
        X, y, feature_names = step_features(args, df, documents)
    else:
        print(f"  Skipping step 'features' — loading from outputs/features.pkl")
        data = load_pickle(OUTPUTS_DIR / "features.pkl")
        X, y, feature_names = data["X"], data["y"], data["feature_names"]

    # ── Step 4: evaluate ──
    step_evaluate(args, X, y, feature_names)

    print(f"\n{'=' * 60}")
    print("Done. All outputs in outputs/")


if __name__ == "__main__":
    main()
