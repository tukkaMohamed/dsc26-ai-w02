
"""Decision metrics for a mule-risk score.

You write this file. Keep the signatures exactly as they are.
"""
from __future__ import annotations

import math


def _validate_inputs(y_true, scores):
    """Validate labels and scores shared by all metric functions."""
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")

    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty")

    for label in y_true:
        if label not in (0, 1):
            raise ValueError("Labels must be 0 or 1")

    for score in scores:
        if math.isnan(float(score)):
            raise ValueError("Scores must not contain NaN")


def confusion_at(y_true, scores, threshold: float) -> dict:
    """Return {"tp": int, "fp": int, "tn": int, "fn": int} at threshold."""
    _validate_inputs(y_true, scores)

    tp = fp = tn = fn = 0

    for actual, score in zip(y_true, scores):
        flagged = score >= threshold

        if actual == 1 and flagged:
            tp += 1
        elif actual == 0 and flagged:
            fp += 1
        elif actual == 0 and not flagged:
            tn += 1
        else:
            fn += 1

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_recall_at(
    y_true, scores, threshold: float
) -> tuple[float, float]:
    """Return (precision, recall) at threshold."""
    counts = confusion_at(y_true, scores, threshold)

    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return (precision, recall)


def precision_at_k(y_true, scores, k: int) -> float:
    """Share of positives among the k highest scores."""
    _validate_inputs(y_true, scores)

    if k <= 0:
        raise ValueError("k must be greater than 0")

    # Python's sorting is stable, so tied scores keep their original order.
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    k = min(k, len(scores))
    top_indices = ranked_indices[:k]

    positives = sum(y_true[i] == 1 for i in top_indices)

    return positives / k


def expected_cost(
    y_true,
    scores,
    threshold: float,
    *,
    cost_fp: float,
    cost_fn: float,
) -> float:
    """Return fp * cost_fp + fn * cost_fn at threshold."""
    counts = confusion_at(y_true, scores, threshold)

    return (
        counts["fp"] * cost_fp
        + counts["fn"] * cost_fn
    )


def best_threshold(
    y_true,
    scores,
    *,
    cost_fp: float,
    cost_fn: float,
) -> float:
    """Choose the threshold with the lowest expected cost.

    On a cost tie, return the highest threshold.
    """
    _validate_inputs(y_true, scores)

    candidates = set(scores)
    candidates.add(math.inf)

    best_t = None
    best_cost = None

    for threshold in candidates:
        cost = expected_cost(
            y_true,
            scores,
            threshold,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
        )

        if (
            best_cost is None
            or cost < best_cost
            or (cost == best_cost and threshold > best_t)
        ):
            best_cost = cost
            best_t = threshold

    return best_t


def threshold_for_precision(
    y_true, scores, min_precision: float
) -> float | None:
    """Return the lowest distinct score meeting min_precision, or None."""
    _validate_inputs(y_true, scores)

    for threshold in sorted(set(scores)):
        precision, _ = precision_recall_at(
            y_true, scores, threshold
        )

        if precision >= min_precision:
            return threshold

    return None
