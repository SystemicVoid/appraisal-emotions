"""Pure statistical kernels shared by the reveal-RPE surfaces.

Extracted from functional-valence-validity src/fv_validity/core/stats.py @ 10c4662.

Kept: the four kernels the reveal-RPE estimator reads — ``midrank_auroc``, ``add_one_p``,
``paired_projection_auroc``, ``cosine_between``.

Dropped: ``lower_median_int``, ``split_extraction_heldout``, the d-prime family, and the
statistic-record comparator (all belong to surfaces this scaffold does not port).
"""

from __future__ import annotations

import math
from collections.abc import Iterable


def midrank_auroc(positive_scores: Iterable[float], negative_scores: Iterable[float]) -> float:
    """Two-sample AUROC with midrank (0.5) tie handling.

    Orientation: the probability that a positive-class score exceeds a
    negative-class score; ties count one half.
    """

    positives = list(positive_scores)
    negatives = list(negative_scores)
    if not positives or not negatives:
        raise ValueError("midrank AUROC requires at least one score per class")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def add_one_p(exceedances: int, n_permutations: int) -> float:
    """Add-one Monte-Carlo permutation p-value: ``(1 + exceedances) / (1 + n_permutations)``.

    The conservative finite-draw estimator (Phipson & Smyth 2010): the observed
    statistic counts as its own permutation, so the estimate is never exactly zero
    and the test stays valid at the draw counts these batteries actually run.
    """

    return (1 + exceedances) / (1 + n_permutations)


def paired_projection_auroc(difference_projections: Iterable[float]) -> float:
    """Paired midrank AUROC of difference projections against zero.

    The fraction of pairs whose difference projects strictly positively onto the predictor
    direction, with ties at exactly zero counting one half. Equal to the two-sample midrank
    AUROC of the differences against a zero baseline, but immune to between-cell variance
    because each pair's baseline is subtracted within the pair.
    """

    differences = [float(value) for value in difference_projections]
    if not differences:
        raise ValueError("paired AUROC requires at least one difference projection")
    wins = sum(1.0 if value > 0.0 else 0.5 if value == 0.0 else 0.0 for value in differences)
    return wins / len(differences)


def cosine_between(a: Iterable[float], b: Iterable[float]) -> float:
    """Cosine similarity between two equal-length nonzero vectors."""

    left = [float(value) for value in a]
    right = [float(value) for value in b]
    if not left or len(left) != len(right):
        raise ValueError("cosine requires two nonempty vectors of equal length")
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    if norm_left == 0.0 or norm_right == 0.0:
        raise ValueError("cosine is undefined for a zero-norm vector")
    dot = sum(lhs * rhs for lhs, rhs in zip(left, right, strict=True))
    return dot / (norm_left * norm_right)
