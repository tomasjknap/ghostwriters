"""Authorship attribution classifiers."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def burrows_delta(
    test_vec: np.ndarray,
    author_means: dict[str, np.ndarray],
    corpus_std: np.ndarray,
) -> dict[str, float]:
    """Compute Burrows' Delta between a test vector and each author's mean profile.

    Delta(test, author) = mean(|z_test - z_author|) where z = (x - corpus_mean) / corpus_std.

    Parameters
    ----------
    test_vec : np.ndarray
        Feature vector of the test document.
    author_means : dict[str, np.ndarray]
        Mean feature vectors per author.
    corpus_std : np.ndarray
        Standard deviation of each feature across the entire corpus.

    Returns
    -------
    dict[str, float]
        Delta score per author (lower = more similar).
    """
    # Avoid division by zero
    safe_std = np.where(corpus_std > 0, corpus_std, 1.0)
    z_test = test_vec / safe_std

    scores = {}
    for author, mean_vec in author_means.items():
        z_author = mean_vec / safe_std
        scores[author] = float(np.mean(np.abs(z_test - z_author)))
    return scores


def cosine_delta(
    test_vec: np.ndarray,
    author_means: dict[str, np.ndarray],
    corpus_std: np.ndarray,
) -> dict[str, float]:
    """Compute Cosine Delta (Smith & Aldridge variant).

    Like Burrows' Delta but uses cosine distance on z-scored vectors.

    Returns
    -------
    dict[str, float]
        Cosine distance per author (lower = more similar).
    """
    safe_std = np.where(corpus_std > 0, corpus_std, 1.0)
    z_test = test_vec / safe_std
    norm_test = np.linalg.norm(z_test)
    if norm_test == 0:
        return {author: 1.0 for author in author_means}

    z_test_normed = z_test / norm_test

    scores = {}
    for author, mean_vec in author_means.items():
        z_author = mean_vec / safe_std
        norm_author = np.linalg.norm(z_author)
        if norm_author == 0:
            scores[author] = 1.0
        else:
            cos_sim = float(np.dot(z_test_normed, z_author / norm_author))
            scores[author] = 1.0 - cos_sim  # distance
    return scores


def compute_author_profiles(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Compute per-author mean vectors and corpus statistics.

    Returns
    -------
    author_means : dict[str, np.ndarray]
    corpus_mean : np.ndarray
    corpus_std : np.ndarray
    """
    corpus_mean = X.mean(axis=0)
    corpus_std = X.std(axis=0)
    authors = np.unique(y)
    author_means = {}
    for a in authors:
        mask = y == a
        author_means[a] = X[mask].mean(axis=0)
    return author_means, corpus_mean, corpus_std


def build_sklearn_classifier(
    method: str = "svm",
    **kwargs,
):
    """Build a scikit-learn classifier.

    Parameters
    ----------
    method : str
        One of "svm", "logistic", "knn".

    Returns
    -------
    sklearn estimator (unfitted)
    """
    if method == "svm":
        return SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3, random_state=42, **kwargs)
    elif method == "logistic":
        return SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42, **kwargs)
    elif method == "knn":
        k = kwargs.pop("n_neighbors", 3)
        return KNeighborsClassifier(n_neighbors=k, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'svm', 'logistic', or 'knn'.")
