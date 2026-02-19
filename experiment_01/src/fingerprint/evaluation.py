"""Evaluation: leave-one-out cross-validation and metrics."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from fingerprint.classifiers import (
    burrows_delta,
    compute_author_profiles,
    cosine_delta,
)


def loo_delta_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "burrows",
) -> dict:
    """Leave-one-out evaluation using Burrows' or Cosine Delta.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    method : str
        "burrows" or "cosine".

    Returns
    -------
    dict with keys: accuracy, rank1_acc, rank3_acc, predictions, true_labels,
    classification_report, confusion_matrix
    """
    delta_fn = burrows_delta if method == "burrows" else cosine_delta
    loo = LeaveOneOut()
    predictions = []
    true_labels = []
    rank3_correct = 0

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        author_means, corpus_mean, corpus_std = compute_author_profiles(
            X_train, y_train
        )
        scores = delta_fn(X_test[0], author_means, corpus_std)

        # Rank authors by score (ascending = more similar)
        ranked = sorted(scores.items(), key=lambda x: x[1])
        predicted = ranked[0][0]
        predictions.append(predicted)
        true_labels.append(y_test[0])

        top3_authors = [a for a, _ in ranked[:3]]
        if y_test[0] in top3_authors:
            rank3_correct += 1

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    n = len(true_labels)

    return {
        "method": f"{method}_delta",
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "rank3_accuracy": rank3_correct / n if n > 0 else 0.0,
        "predictions": predictions,
        "true_labels": true_labels,
        "classification_report": classification_report(
            true_labels, predictions, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            true_labels, predictions, labels=np.unique(y)
        ),
    }


def loo_sklearn_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    classifier_factory: Callable,
    scale: bool = True,
) -> dict:
    """Leave-one-out evaluation with a scikit-learn classifier.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    classifier_factory : callable
        A function that returns an unfitted sklearn estimator.
    scale : bool
        Whether to StandardScale features.

    Returns
    -------
    dict with evaluation results.
    """
    loo = LeaveOneOut()
    predictions = []
    true_labels = []

    n_total = X.shape[0]
    for i, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        clf = classifier_factory()
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        predictions.append(pred[0])
        true_labels.append(y_test[0])

        if (i + 1) % 50 == 0 or i == n_total - 1:
            print(f"  LOO-CV: {i + 1}/{n_total}", flush=True)

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    return {
        "method": str(classifier_factory()),
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "predictions": predictions,
        "true_labels": true_labels,
        "classification_report": classification_report(
            true_labels, predictions, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            true_labels, predictions, labels=np.unique(y)
        ),
    }


def print_results(results: dict) -> None:
    """Pretty-print evaluation results."""
    print(f"Method: {results['method']}")
    print(f"Accuracy: {results['accuracy']:.3f}")
    if "rank3_accuracy" in results:
        print(f"Rank-3 accuracy: {results['rank3_accuracy']:.3f}")
    print()
    print(results["classification_report"])
