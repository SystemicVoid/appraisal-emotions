"""The per-block valence-residual estimands: P2, P4, P5a, P5c and their two nulls.

NEW module (no parent counterpart). Split out of ``emotion_mapping`` along the seam between the
ESTIMANDS (what is computed at one block, from one pair of artifacts) and the RUN (which blocks,
which valence scale, which report). This module never opens a file and never decides a headline
block; it takes a block's geometry and returns that block's readouts.

The named object is CONTEXT.md's minted **valence-residual alignment**: ``cos(v_appraisal, e_j)``
after regressing that cosine on affect-concept valence across the word set — the residual
alignment an emotion word has with an appraisal direction beyond what its valence predicts. Every
estimator here is a statement about that residual, never about the raw cosine (which is
near-certain and uninformative: any positive-value direction has a positive-valence component).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from appraisal_emotions.analysis.direction_stats import random_direction_floor
from appraisal_emotions.analysis.emotion_vectors import EmotionVectors, valence_oriented_pc_axes
from appraisal_emotions.analysis.reveal_rpe import DIRECTION_FAMILIES, RevealRpeDirections
from appraisal_emotions.core.schema import StrictModel
from appraisal_emotions.core.stats import add_one_p
from appraisal_emotions.stimuli.emotion_stories import STYLE_CONTROL_LABEL, EmotionWordSet

__all__ = [
    "FLOOR_QUANTILE",
    "BlockGeometry",
    "P4Result",
    "P5aResult",
    "P5cStyleResult",
    "PairResult",
    "block_geometry",
    "cosines",
    "label_shuffled_pair_floor",
    "norm_fraction",
    "ols_residuals",
    "p2_matched_pairs",
    "p4_contrast",
    "p4_surprise_contrast",
    "p5a_discriminant_null",
    "p5c_style_control",
    "pair_statistics",
    "random_direction_pair_floor",
    "spearman",
]

FLOOR_QUANTILE = 0.95

# --------------------------------------------------------------------------------------
# Small numeric kernels
# --------------------------------------------------------------------------------------


def cosines(rows: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Cosine of every row of ``rows`` (n, hidden) with ``vector`` (hidden,)."""

    row_norms = np.linalg.norm(rows, axis=1)
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm == 0.0 or not np.all(row_norms > 0.0):
        raise ValueError("cosine is undefined for a zero-norm vector")
    return (rows @ vector) / (row_norms * vector_norm)


def ols_residuals(y: np.ndarray, design: np.ndarray) -> np.ndarray:
    return y - design @ np.linalg.lstsq(design, y, rcond=None)[0]


def _predict_at(y: np.ndarray, design: np.ndarray, row: np.ndarray) -> float:
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(row @ coef)


def _holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values in input order (monotone non-decreasing, clipped at 1)."""

    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    result = spearmanr(x, y)
    return (
        float(np.nan_to_num(result.statistic, nan=0.0)),
        float(np.nan_to_num(result.pvalue, nan=1.0)),
    )


def norm_fraction(vector: np.ndarray, basis: np.ndarray) -> float:
    """Fraction of ``||vector||`` that lies in the row space of ``basis`` (least squares)."""

    coef = np.linalg.lstsq(basis.T, vector, rcond=None)[0]
    return float(np.linalg.norm(basis.T @ coef) / np.linalg.norm(vector))


# --------------------------------------------------------------------------------------
# Per-block geometry
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockGeometry:
    """One block's contribution: word cosines per family, the style cosines, and the PC axes."""

    block: int
    cos: dict[str, np.ndarray]
    cos_style: dict[str, float]
    family_vectors: dict[str, np.ndarray]
    components: np.ndarray
    scores: np.ndarray
    centered_words: np.ndarray


def block_geometry(
    directions: RevealRpeDirections, emotion: EmotionVectors, block: int
) -> BlockGeometry:
    words = emotion.word_vectors[:, block, :]
    center = words.mean(axis=0)  # per-block mean-centring (anisotropy discipline, §6)
    centered = words - center
    style = emotion.row(STYLE_CONTROL_LABEL)[block] - center
    components, scores = valence_oriented_pc_axes(words, emotion.word_valence)
    cos: dict[str, np.ndarray] = {}
    cos_style: dict[str, float] = {}
    family_vectors: dict[str, np.ndarray] = {}
    for index, family in enumerate(DIRECTION_FAMILIES):
        direction = directions.directions[index, block, :]
        family_vectors[family] = direction
        cos[family] = cosines(centered, direction)
        cos_style[family] = float(cosines(style[None, :], direction)[0])
    return BlockGeometry(block, cos, cos_style, family_vectors, components, scores, centered)


# --------------------------------------------------------------------------------------
# P2 — the headline: valence-residual matched pairs
# --------------------------------------------------------------------------------------


class PairResult(StrictModel):
    """One pre-registered P2 matched pair tested on valence residuals (one-sided high > low)."""

    high: str
    low: str
    residual_high: float
    residual_low: float
    statistic: float
    p_value: float
    p_holm: float
    n_valence_matched: int
    passed: bool


def _pair_permutation_p(
    residuals: np.ndarray,
    matched: np.ndarray,
    positions: tuple[int, int],
    *,
    rng: np.random.Generator,
    n_permutations: int,
) -> tuple[float, float]:
    """One-sided permutation p for ``high > low``, shuffling residuals within the matched set.

    The exchangeability unit is the valence-matched set — the words the §5 pairing declares
    interchangeable up to prospect-disconfirmation — so the null asks: among words of this
    valence, how often does a random pair split this far in the predicted direction?
    """

    values = residuals[matched]
    observed = float(values[positions[0]] - values[positions[1]])
    exceedances = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(values)
        if float(shuffled[positions[0]] - shuffled[positions[1]]) >= observed:
            exceedances += 1
    return observed, add_one_p(exceedances, n_permutations)


def _matched_positions(
    binary_valence: np.ndarray, index: dict[str, int], pair: tuple[str, str]
) -> tuple[np.ndarray, tuple[int, int]]:
    if binary_valence[index[pair[0]]] != binary_valence[index[pair[1]]]:
        raise ValueError(f"pre-registered pair {pair} is not valence-matched in the word file")
    matched = np.flatnonzero(binary_valence == binary_valence[index[pair[0]]])
    lookup = {row: position for position, row in enumerate(matched.tolist())}
    return matched, (lookup[index[pair[0]]], lookup[index[pair[1]]])


def pair_statistics(
    residuals: np.ndarray,
    pairs: tuple[tuple[str, str], ...],
    binary_valence: np.ndarray,
    index: dict[str, int],
) -> list[float]:
    """The three observed pair differences (no permutation) — the block-sweep / null statistic."""

    stats: list[float] = []
    for pair in pairs:
        matched, positions = _matched_positions(binary_valence, index, pair)
        values = residuals[matched]
        stats.append(float(values[positions[0]] - values[positions[1]]))
    return stats


def p2_matched_pairs(
    geometry: BlockGeometry,
    pairs: tuple[tuple[str, str], ...],
    index: dict[str, int],
    design: np.ndarray,
    binary_valence: np.ndarray,
    *,
    rng: np.random.Generator,
    n_permutations: int,
    alpha: float,
) -> tuple[tuple[PairResult, ...], np.ndarray]:
    residuals = ols_residuals(geometry.cos["v_rpe"], design)
    raw: list[tuple[tuple[str, str], float, float, int]] = []
    for pair in pairs:
        matched, positions = _matched_positions(binary_valence, index, pair)
        statistic, p_value = _pair_permutation_p(
            residuals, matched, positions, rng=rng, n_permutations=n_permutations
        )
        raw.append((pair, statistic, p_value, int(matched.size)))
    holm = _holm([entry[2] for entry in raw])
    results = tuple(
        PairResult(
            high=pair[0],
            low=pair[1],
            residual_high=float(residuals[index[pair[0]]]),
            residual_low=float(residuals[index[pair[1]]]),
            statistic=statistic,
            p_value=p_value,
            p_holm=adjusted,
            n_valence_matched=n_matched,
            passed=adjusted < alpha and statistic > 0.0,
        )
        for (pair, statistic, p_value, n_matched), adjusted in zip(raw, holm, strict=True)
    )
    return results, residuals


# --------------------------------------------------------------------------------------
# P4 / P5a / P5c
# --------------------------------------------------------------------------------------


class P4Result(StrictModel):
    """``v_absrpe`` vs the surprise family, against arousal-matched valenced controls."""

    surprise_words: tuple[str, ...]
    arousal_matched_words: tuple[str, ...]
    mean_cos_surprise: float
    mean_cos_arousal_matched: float
    contrast: float
    p_value: float
    passed: bool
    cos_absrpe_pc1: float
    cos_absrpe_pc2: float
    cos_rpe_pc1: float
    cos_rpe_pc2: float
    loads_arousal_pc: bool
    orthogonality_caveat: str


_P4_CAVEAT = (
    "v_RPE ⊥ v_absrpe holds BY CONSTRUCTION of the reveal design matrix; the informative half of "
    "P4 is the positive claim about which words v_absrpe picks out, never the orthogonality."
)


def p4_contrast(cos: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    return float(cos[left].mean() - cos[right].mean())


def p4_surprise_contrast(
    geometry: BlockGeometry,
    words: EmotionWordSet,
    index: dict[str, int],
    *,
    rng: np.random.Generator,
    n_permutations: int,
    alpha: float,
) -> P4Result:
    surprise = words.confirmatory_set("surprise")
    controls = words.confirmatory_set("arousal_matched")
    cos = geometry.cos["v_absrpe"]
    left = np.asarray([index[word] for word in surprise])
    right = np.asarray([index[word] for word in controls])
    observed = p4_contrast(cos, left, right)
    pooled = np.concatenate([cos[left], cos[right]])
    exceedances = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        if float(shuffled[: left.size].mean() - shuffled[left.size :].mean()) >= observed:
            exceedances += 1
    p_value = add_one_p(exceedances, n_permutations)
    loading = {
        family: cosines(geometry.components[:2], geometry.family_vectors[family])
        for family in ("v_absrpe", "v_rpe")
    }
    return P4Result(
        surprise_words=surprise,
        arousal_matched_words=controls,
        mean_cos_surprise=float(cos[left].mean()),
        mean_cos_arousal_matched=float(cos[right].mean()),
        contrast=observed,
        p_value=p_value,
        passed=p_value < alpha and observed > 0.0,
        cos_absrpe_pc1=float(loading["v_absrpe"][0]),
        cos_absrpe_pc2=float(loading["v_absrpe"][1]),
        cos_rpe_pc1=float(loading["v_rpe"][0]),
        cos_rpe_pc2=float(loading["v_rpe"][1]),
        loads_arousal_pc=abs(loading["v_absrpe"][1]) > abs(loading["v_absrpe"][0]),
        orthogonality_caveat=_P4_CAVEAT,
    )


class P5aResult(StrictModel):
    """The discriminant null: ``sad`` should carry NO RPE excess once valence is partialled."""

    word: str
    residual: float
    ci_low: float
    ci_high: float
    n_bootstrap: int
    includes_zero: bool
    note: str


_P5A_NOTE = (
    "P5a is a PREDICTION OF ABSENCE: it passes by the CI covering zero, and it only carries "
    "information while P2 is positive. A null everywhere is not a P5a pass, it is no signal."
)


def p5a_discriminant_null(
    geometry: BlockGeometry,
    word: str,
    index: dict[str, int],
    design: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> P5aResult:
    """Word-resampling bootstrap CI on the target word's valence residual."""

    y = geometry.cos["v_rpe"]
    observed = float(ols_residuals(y, design)[index[word]])
    target = index[word]
    others = np.asarray([row for row in range(len(y)) if row != target])
    draws: list[float] = []
    for _ in range(n_bootstrap):
        resampled = rng.choice(others, size=others.size, replace=True)
        rows = np.concatenate([np.asarray([target]), resampled])
        residual = y[rows] - design[rows] @ np.linalg.lstsq(design[rows], y[rows], rcond=None)[0]
        draws.append(float(residual[0]))
    low, high = np.quantile(draws, [0.025, 0.975])
    return P5aResult(
        word=word,
        residual=observed,
        ci_low=float(low),
        ci_high=float(high),
        n_bootstrap=n_bootstrap,
        includes_zero=bool(low <= 0.0 <= high),
        note=_P5A_NOTE,
    )


class P5cStyleResult(StrictModel):
    """The scale control: one appraisal family vs the valence-free ``style_control`` vector."""

    family: str
    cos_style: float
    predicted_from_valence: float
    residual: float
    word_residual_p95: float
    passed: bool


def p5c_style_control(
    geometry: BlockGeometry, binary_design: np.ndarray
) -> tuple[P5cStyleResult, ...]:
    """Style-control residual per family, judged against the word residuals' own spread.

    The prediction is evaluated at binary valence 0 — the style vector is valence-free, and the
    binary scale (unlike a 1-9 norm scale) has a defined zero, so P5c always uses the binary
    design even when P2 upgraded to numeric norms.
    """

    neutral = np.zeros(binary_design.shape[1])
    neutral[0] = 1.0
    results: list[P5cStyleResult] = []
    for family in DIRECTION_FAMILIES:
        y = geometry.cos[family]
        predicted = _predict_at(y, binary_design, neutral)
        residual = geometry.cos_style[family] - predicted
        p95 = float(np.quantile(np.abs(ols_residuals(y, binary_design)), FLOOR_QUANTILE))
        results.append(
            P5cStyleResult(
                family=family,
                cos_style=geometry.cos_style[family],
                predicted_from_valence=predicted,
                residual=float(residual),
                word_residual_p95=p95,
                passed=bool(abs(residual) <= p95),
            )
        )
    return tuple(results)


# --------------------------------------------------------------------------------------
# Nulls (anisotropy discipline, §6)
# --------------------------------------------------------------------------------------


def label_shuffled_pair_floor(
    geometry: BlockGeometry,
    pairs: tuple[tuple[str, str], ...],
    index: dict[str, int],
    design: np.ndarray,
    binary_valence: np.ndarray,
    *,
    rng: np.random.Generator,
    n_draws: int,
) -> float:
    """95th percentile of the max pair statistic when emotion vectors are re-labelled at random."""

    cos = geometry.cos["v_rpe"]
    draws: list[float] = []
    for _ in range(n_draws):
        shuffled = rng.permutation(cos)
        residuals = ols_residuals(shuffled, design)
        draws.append(max(pair_statistics(residuals, pairs, binary_valence, index)))
    return float(np.quantile(draws, FLOOR_QUANTILE, method="linear"))


def random_direction_pair_floor(
    geometry: BlockGeometry,
    pairs: tuple[tuple[str, str], ...],
    index: dict[str, int],
    design: np.ndarray,
    binary_valence: np.ndarray,
    *,
    rng: np.random.Generator,
    n_draws: int,
) -> float:
    """95th percentile of the max pair statistic against norm-matched random directions."""

    def score(direction: np.ndarray) -> float:
        residuals = ols_residuals(cosines(geometry.centered_words, direction), design)
        return max(pair_statistics(residuals, pairs, binary_valence, index))

    return random_direction_floor(
        score,
        rng=rng,
        hidden_size=geometry.centered_words.shape[1],
        n_directions=n_draws,
        quantile=FLOOR_QUANTILE,
        method="linear",  # numpy's default, stated explicitly
    )
