"""Evaluation metrics, implemented once and tested against hand-computed values."""
from __future__ import annotations

import numpy as np


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann–Whitney AUC with average ranks for ties. NaN if a class is empty."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    hits = labels[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


def reciprocal_rank(ranked: list[int], gold: int, k: int = 10) -> float:
    for i, item in enumerate(ranked[:k]):
        if item == gold:
            return 1.0 / (i + 1)
    return 0.0


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(probs) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & ((probs < hi) if hi < 1 else (probs <= hi))
        if mask.any():
            ece += mask.mean() * abs(probs[mask].mean() - labels[mask].mean())
    return float(ece)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, reps: int = 400,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean — used so the self-model can say
    'MRR 0.41 (0.36–0.46)' instead of a bare point estimate."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    means = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_sign_flip_p(diffs: np.ndarray, rng: np.random.Generator, reps: int = 5000) -> float:
    """One-sided p-value that the mean paired difference is above zero.

    Under the null the two systems are exchangeable item by item, so each
    difference is as likely to have either sign; flipping signs at random gives
    the null distribution of the mean. Pairing is what makes a small holdout set
    usable: items both systems rank identically contribute zero and no noise.
    The +1 in numerator and denominator keeps the estimate a valid p-value.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    observed = d.mean()
    signs = rng.choice((-1.0, 1.0), size=(reps, len(d)))
    null = (signs * d).mean(axis=1)
    return float((1 + np.count_nonzero(null >= observed - 1e-12)) / (reps + 1))
