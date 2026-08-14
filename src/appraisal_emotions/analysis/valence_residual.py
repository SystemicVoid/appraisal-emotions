"""The per-block valence-residual estimands: the §5 expectation readouts, P4, P5a, P5c, floors.

NEW module (no parent counterpart). Split out of ``emotion_mapping`` along the seam between the
ESTIMANDS (what is computed at one block, from one pair of artifacts) and the RUN (which blocks,
which valence scale, which report). This module never opens a file and never decides a headline
block; it takes a block's geometry and returns that block's readouts.

The named object is CONTEXT.md's minted **valence-residual alignment**: ``cos(v_appraisal, e_j)``
after regressing that cosine on affect-concept valence across the word set — the residual
alignment an emotion word has with an appraisal direction beyond what its valence predicts. Every
estimator here is a statement about that residual, never about the raw cosine (which is
near-certain and uninformative: any positive-value direction has a positive-valence component).

Every word's residual is reported. The §5 *recorded expectations* — the named pairs, the two
family contrasts, the three-level outcome ordering — additionally get an effect size and a cheap
one-sided permutation p in the recorded direction. There is no multiple-comparison correction and
no confirmatory/exploratory caste (operator decision, 2026-08-13): effect sizes lead, p-values
follow, and a readout the expectations did not anticipate is still reported.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from appraisal_emotions.analysis.direction_stats import random_direction_floor
from appraisal_emotions.analysis.emotion_vectors import EmotionVectors, valence_oriented_pc_axes
from appraisal_emotions.analysis.reveal_rpe import DIRECTION_FAMILIES, RevealRpeDirections
from appraisal_emotions.core.schema import StrictModel
from appraisal_emotions.core.stats import add_one_p
from appraisal_emotions.stimuli.emotion_stories import (
    STYLE_CONTROL_LABEL,
    EmotionWordSet,
    ExpectedFamilyContrast,
    ExpectedPair,
)

__all__ = [
    "FLOOR_QUANTILE",
    "BlockGeometry",
    "ExpectedPairResult",
    "FamilyContrastResult",
    "OutcomeOrderingResult",
    "P4Result",
    "P5aResult",
    "P5cStyleResult",
    "ResidualStatistic",
    "block_geometry",
    "cosines",
    "expected_pair_results",
    "family_contrast_results",
    "family_contrast_statistics",
    "family_rows",
    "label_shuffled_floor",
    "norm_fraction",
    "ols_residuals",
    "outcome_ordering_result",
    "p4_surprise_contrast",
    "p5a_discriminant_null",
    "pair_pool_rows",
    "p5c_style_control",
    "random_direction_residual_floor",
    "spearman",
    "two_sample_permutation_p",
    "two_sample_statistic",
]

FLOOR_QUANTILE = 0.95

#: A scalar read off one block's residual vector — what the two null floors resample.
ResidualStatistic = Callable[[np.ndarray], float]

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


def family_rows(words: EmotionWordSet, index: dict[str, int]) -> dict[str, np.ndarray]:
    """Row indices of each §5 family, in word-file order."""

    rows: dict[str, list[int]] = {}
    for word in words.words:
        rows.setdefault(word.category, []).append(index[word.word])
    return {family: np.asarray(members) for family, members in rows.items()}


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
    centered_words: np.ndarray


def block_geometry(
    directions: RevealRpeDirections, emotion: EmotionVectors, block: int
) -> BlockGeometry:
    words = emotion.word_vectors[:, block, :]
    center = words.mean(axis=0)  # per-block mean-centring (anisotropy discipline, §6)
    centered = words - center
    style = emotion.row(STYLE_CONTROL_LABEL)[block] - center
    components, _scores = valence_oriented_pc_axes(words, emotion.word_valence)
    cos: dict[str, np.ndarray] = {}
    cos_style: dict[str, float] = {}
    family_vectors: dict[str, np.ndarray] = {}
    for index, family in enumerate(DIRECTION_FAMILIES):
        direction = directions.directions[index, block, :]
        family_vectors[family] = direction
        cos[family] = cosines(centered, direction)
        cos_style[family] = float(cosines(style[None, :], direction)[0])
    return BlockGeometry(block, cos, cos_style, family_vectors, components, centered)


# --------------------------------------------------------------------------------------
# Recorded-expectation readouts (design §5): named pairs, family contrasts, outcome ordering
# --------------------------------------------------------------------------------------


class ExpectedPairResult(StrictModel):
    """One §5 named pair on the valence residuals, read in its recorded direction.

    ``statistic = expected_sign * (residual_outcome - residual_control)``, so a positive statistic
    always means the recorded split, whichever pole the outcome word sits on. The permutation is
    one-sided in that direction and shuffles residuals WITHIN the §5 pool: the same-pole words of
    the outcome / non-outcome / confirmation families, NOT every word of that valence.
    """

    outcome: str
    control: str
    expected_sign: int
    statistic: float
    p_value: float
    n_pool: int


def pair_pool_rows(words: EmotionWordSet, rows: dict[str, np.ndarray]) -> np.ndarray:
    """Row indices of the §5 pair pool — the families the recorded expectations already name.

    The §5 resolution table permutes a named pair inside its *same-pole* outcome/non-outcome pool,
    not inside every word sharing its binary valence: `hopeful` and `surprised` are not words the
    pairing declares interchangeable with `elated`. The families are derived from the expectations
    block (the two family contrasts plus the three-level ordering) rather than listed again here,
    so the pool cannot drift from the contrasts it belongs to.
    """

    return np.sort(
        np.concatenate([rows[family] for family in words.pair_pool_families() if family in rows])
    )


def _matched_positions(
    binary_valence: np.ndarray, index: dict[str, int], pair: ExpectedPair, pool: np.ndarray
) -> tuple[np.ndarray, tuple[int, int]]:
    valence = binary_valence[index[pair.outcome]]
    if binary_valence[index[pair.control]] != valence:
        raise ValueError(f"recorded pair {pair} is not valence-matched in the word file")
    matched = pool[binary_valence[pool] == valence]
    lookup = {row: position for position, row in enumerate(matched.tolist())}
    missing = [word for word in (pair.outcome, pair.control) if index[word] not in lookup]
    if missing:
        raise ValueError(
            f"recorded pair {pair} names {missing}, which is outside the §5 pair pool "
            "(the outcome / non-outcome / confirmation families); the permutation has no null"
        )
    return matched, (lookup[index[pair.outcome]], lookup[index[pair.control]])


def expected_pair_results(
    residuals: np.ndarray,
    pairs: tuple[ExpectedPair, ...],
    binary_valence: np.ndarray,
    index: dict[str, int],
    pool: np.ndarray,
    *,
    rng: np.random.Generator,
    n_permutations: int,
) -> tuple[ExpectedPairResult, ...]:
    """Effect size + one-sided within-pool permutation p for each recorded pair."""

    results: list[ExpectedPairResult] = []
    for pair in pairs:
        matched, (left, right) = _matched_positions(binary_valence, index, pair, pool)
        values = residuals[matched]
        observed = pair.expected_sign * float(values[left] - values[right])
        exceedances = 0
        for _ in range(n_permutations):
            shuffled = rng.permutation(values)
            if pair.expected_sign * float(shuffled[left] - shuffled[right]) >= observed:
                exceedances += 1
        results.append(
            ExpectedPairResult(
                outcome=pair.outcome,
                control=pair.control,
                expected_sign=pair.expected_sign,
                statistic=observed,
                p_value=add_one_p(exceedances, n_permutations),
                n_pool=int(matched.size),
            )
        )
    return tuple(results)


class FamilyContrastResult(StrictModel):
    """One pole's family contrast: outcome-disconfirmation words vs valence-matched controls.

    The design's powered statistic (§4 E1 P2). ``statistic = expected_sign * (mean_outcome -
    mean_control)`` on the valence residuals, with a one-sided two-sample label permutation over
    the pooled same-pole words.
    """

    pole: str
    outcome_family: str
    control_family: str
    expected_sign: int
    n_outcome: int
    n_control: int
    mean_residual_outcome: float
    mean_residual_control: float
    statistic: float
    p_value: float


def two_sample_statistic(
    values: np.ndarray, left: np.ndarray, right: np.ndarray, sign: int
) -> float:
    """``sign * (mean(values[left]) - mean(values[right]))`` — positive = the recorded direction."""

    return sign * float(values[left].mean() - values[right].mean())


def two_sample_permutation_p(
    values: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    sign: int,
    observed: float,
    *,
    rng: np.random.Generator,
    n_permutations: int,
) -> float:
    """One-sided label-permutation p for :func:`two_sample_statistic` over the pooled two sets.

    The one kernel behind every group contrast here (family contrasts, the outcome-ordering trend,
    P4): pool the two sets, reshuffle the group labels, count draws at least as extreme in the
    recorded direction.
    """

    pooled = np.concatenate([values[left], values[right]])
    exceedances = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        draw = sign * float(shuffled[: left.size].mean() - shuffled[left.size :].mean())
        if draw >= observed:
            exceedances += 1
    return add_one_p(exceedances, n_permutations)


def family_contrast_statistics(
    residuals: np.ndarray,
    contrasts: tuple[ExpectedFamilyContrast, ...],
    rows: dict[str, np.ndarray],
) -> list[float]:
    """The signed family-contrast effect sizes — the statistic the null floors resample."""

    return [
        two_sample_statistic(
            residuals,
            rows[contrast.outcome_family],
            rows[contrast.control_family],
            contrast.expected_sign,
        )
        for contrast in contrasts
    ]


def family_contrast_results(
    residuals: np.ndarray,
    contrasts: tuple[ExpectedFamilyContrast, ...],
    rows: dict[str, np.ndarray],
    *,
    rng: np.random.Generator,
    n_permutations: int,
) -> tuple[FamilyContrastResult, ...]:
    observed = family_contrast_statistics(residuals, contrasts, rows)
    results: list[FamilyContrastResult] = []
    for contrast, statistic in zip(contrasts, observed, strict=True):
        left = rows[contrast.outcome_family]
        right = rows[contrast.control_family]
        results.append(
            FamilyContrastResult(
                pole=contrast.pole,
                outcome_family=contrast.outcome_family,
                control_family=contrast.control_family,
                expected_sign=contrast.expected_sign,
                n_outcome=int(left.size),
                n_control=int(right.size),
                mean_residual_outcome=float(residuals[left].mean()),
                mean_residual_control=float(residuals[right].mean()),
                statistic=statistic,
                p_value=two_sample_permutation_p(
                    residuals,
                    left,
                    right,
                    contrast.expected_sign,
                    statistic,
                    rng=rng,
                    n_permutations=n_permutations,
                ),
            )
        )
    return tuple(results)


class OutcomeOrderingResult(StrictModel):
    """The OCC three-level ordering on the SIGNED residual (§5 ``outcome_ordering``).

    Recorded expectation: positive-disconfirmation > confirmation > negative-disconfirmation.
    ``ordering_holds`` reads the full three-level structure; the permuted trend statistic is the
    end-to-end contrast (first family mean − last family mean), which is where the power is.
    """

    families: tuple[str, ...]
    mean_residuals: tuple[float, ...]
    ordering_holds: bool
    trend_statistic: float
    p_value: float


def outcome_ordering_result(
    residuals: np.ndarray,
    families: tuple[str, ...],
    rows: dict[str, np.ndarray],
    *,
    rng: np.random.Generator,
    n_permutations: int,
) -> OutcomeOrderingResult:
    members = [rows[family] for family in families]
    means = [float(residuals[block].mean()) for block in members]
    observed = means[0] - means[-1]
    return OutcomeOrderingResult(
        families=families,
        mean_residuals=tuple(means),
        ordering_holds=all(left > right for left, right in zip(means, means[1:], strict=False)),
        trend_statistic=observed,
        p_value=two_sample_permutation_p(
            residuals, members[0], members[-1], 1, observed, rng=rng, n_permutations=n_permutations
        ),
    )


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


def p4_surprise_contrast(
    geometry: BlockGeometry,
    words: EmotionWordSet,
    index: dict[str, int],
    *,
    rng: np.random.Generator,
    n_permutations: int,
) -> P4Result:
    surprise = words.p4_words("surprise")
    controls = words.p4_words("arousal_matched")
    cos = geometry.cos["v_absrpe"]
    left = np.asarray([index[word] for word in surprise])
    right = np.asarray([index[word] for word in controls])
    observed = two_sample_statistic(cos, left, right, 1)
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
        p_value=two_sample_permutation_p(
            cos, left, right, 1, observed, rng=rng, n_permutations=n_permutations
        ),
        cos_absrpe_pc1=float(loading["v_absrpe"][0]),
        cos_absrpe_pc2=float(loading["v_absrpe"][1]),
        cos_rpe_pc1=float(loading["v_rpe"][0]),
        cos_rpe_pc2=float(loading["v_rpe"][1]),
        loads_arousal_pc=abs(loading["v_absrpe"][1]) > abs(loading["v_absrpe"][0]),
        orthogonality_caveat=_P4_CAVEAT,
    )


class P5aResult(StrictModel):
    """The discriminant expectation of absence: ``sad`` carries NO RPE excess after partialling."""

    word: str
    residual: float
    word_residual_p95: float
    abs_rank: int
    n_words: int
    passed: bool
    note: str


_P5A_NOTE = (
    "P5a is an EXPECTATION OF ABSENCE: it holds by the word's residual sitting INSIDE the word "
    "residuals' own spread (the same p95 band P5c uses), and it only carries information while "
    "the outcome-family contrasts are positive. A null everywhere is not a P5a result, it is no "
    "signal. It is a within-set comparison, not a confidence interval: there is one measurement "
    "per word, so nothing here estimates this word's own sampling error."
)


def p5a_discriminant_null(
    geometry: BlockGeometry,
    word: str,
    index: dict[str, int],
    design: np.ndarray,
) -> P5aResult:
    """The target word's valence residual judged against the word residuals' own spread.

    Same convention as :func:`p5c_style_control`, and for the same reason: the word set gives ONE
    residual per word, so the only calibrated yardstick available is how far the other words'
    residuals scatter. The earlier word-resampling bootstrap was replaced (2026-08-13) because it
    resampled the OTHER words while pinning the target, so the interval never contained the
    target's own measurement noise and under-covered badly.
    """

    residuals = ols_residuals(geometry.cos["v_rpe"], design)
    observed = float(residuals[index[word]])
    magnitudes = np.abs(residuals)
    p95 = float(np.quantile(magnitudes, FLOOR_QUANTILE))
    return P5aResult(
        word=word,
        residual=observed,
        word_residual_p95=p95,
        abs_rank=int((magnitudes > abs(observed)).sum()) + 1,
        n_words=int(residuals.size),
        passed=bool(abs(observed) <= p95),
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
    design even when the headline regression upgraded to numeric norms.
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
# Nulls (anisotropy discipline, §6). Both floors resample the SAME headline statistic.
# --------------------------------------------------------------------------------------


def label_shuffled_floor(
    geometry: BlockGeometry,
    design: np.ndarray,
    statistic: ResidualStatistic,
    *,
    rng: np.random.Generator,
    n_draws: int,
) -> float:
    """95th percentile of ``statistic`` when the emotion vectors are re-labelled at random.

    RESIDUALIZE FIRST, then permute (fixed 2026-08-13). Permuting the raw cosines and residualizing
    afterwards puts the floor on the RAW cosine scale while the observed statistic lives on the
    post-residual scale — with any valence component in ``v_rpe`` the raw spread is much the larger
    of the two, and the floor rejects true effects on scale alone (a reviewer reproduction measured
    a ~4.4x inflated floor rejecting 40/40 planted effects). Shuffling the residuals themselves is
    the same null — "this word's residual could have belonged to any word" — on the right scale.
    """

    residuals = ols_residuals(geometry.cos["v_rpe"], design)
    draws = [statistic(rng.permutation(residuals)) for _ in range(n_draws)]
    return float(np.quantile(draws, FLOOR_QUANTILE, method="linear"))


def random_direction_residual_floor(
    geometry: BlockGeometry,
    design: np.ndarray,
    statistic: ResidualStatistic,
    *,
    rng: np.random.Generator,
    n_draws: int,
) -> float:
    """95th percentile of ``statistic`` against norm-matched random directions."""

    def score(direction: np.ndarray) -> float:
        return statistic(ols_residuals(cosines(geometry.centered_words, direction), design))

    return random_direction_floor(
        score,
        rng=rng,
        hidden_size=geometry.centered_words.shape[1],
        n_directions=n_draws,
        quantile=FLOOR_QUANTILE,
        method="linear",  # numpy's default, stated explicitly
    )
