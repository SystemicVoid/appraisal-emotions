"""P1's math: what fraction of E1's word-level noise would more stories remove?

E1's family contrast sits at roughly half the smallest effect its test can detect. That is an
instrument report, not an absence — but it is silent on the one question that decides what to do
next, because E0 stores only each word's story MEAN. With one number per word the residual spread
`s` cannot be split, and `s` is a mixture of two things with opposite implications:

    s^2  =  sigma^2_tau,resid  +  sigma^2_w / k
            ^ between-word,        ^ within-word,
              no k removes it        shrinks as 1/k

If the second term dominates, a larger capture is worth renting a GPU for and P1 says how large.
If the first does, no story count rescues the design and the next move is a wider word set. That
is the whole decision, and `docs/design/p1-prereg.md` fixes its thresholds before any number here
exists.

Everything below is a pure function over arrays, so the recovery of KNOWN planted components is
testable without a capture (`tests/test_story_reliability.py`) — which is what lets the analysis of
record be dry-run and frozen before the GPU is rented, as the pre-registration requires.

Two choices are load-bearing and both come from the advisory ruling recorded in the prereg:

* **Nothing here rescales E1's statistic.** The attenuation `lambda` is real but approximately
  common across words, so it cancels between the observed contrast and its permutation null.
  `lambda` therefore enters the PROPHECY — what the contrast would read at a larger k — and never
  the test. P1 recomputes no p-value.
* **Split-half is exact, not delta-method.** The linear projection `u_ij` has an exact word-mean
  identity, but the statistic E1 actually reads is a COSINE, whose denominator is not linear in the
  stories. The stored within-word Gram supplies `||mean_A y||` exactly, so the half-sample cosines
  below are the real statistic rather than a first-order stand-in for it. That matters here more
  than usual: reliability IS the measurement, so approximating it would defeat the capture.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist

import numpy as np

from appraisal_emotions.analysis.story_projections import StoryProjections
from appraisal_emotions.analysis.valence_residual import ols_residuals

# z_.95 + z_.80 = 2.4865, the one part of the MDE80 coefficient that is a property of the test
# rather than of the word set.
MDE80_Z_SUM = NormalDist().inv_cdf(0.95) + NormalDist().inv_cdf(0.80)


def mde80_coefficient(*, n_outcome: int, n_control: int) -> float:
    """The analytic two-sample MDE80 coefficient at THIS contrast's family sizes.

    Was a module constant baked at 9 vs 10 — E1's family widths at the time. That is a property of
    ``data/emotion_words.json``, not of the test, and the widened word set moves it: at 11 vs 15 it
    falls from 1.1425 to 0.9750, so a stale constant would report the old geometry's floors and
    prophecy against the new data without anything contradicting it. Callers pass the sizes off the
    contrast they are actually scoring.
    """

    if n_outcome < 1 or n_control < 1:
        raise ValueError(f"MDE80 needs a word on each side, got {n_outcome} and {n_control}")
    return MDE80_Z_SUM * sqrt(1 / n_outcome + 1 / n_control)


@dataclass(frozen=True)
class WordProjectionStats:
    """One block, one direction: the per-word quantities every downstream number is built from.

    ``mean_u`` is E0's own word vector projected onto the direction — the capture's faithfulness
    gate is exactly the claim that this equals `e_j . d`. ``word_norm`` and ``within_scatter`` come
    from the within-word Gram, which is the only reason the cosine-scale results below are exact.
    """

    labels: tuple[str, ...]
    counts: np.ndarray  # k_j
    mean_u: np.ndarray  # mean_i u_ij  ==  e_j . d
    word_norm: np.ndarray  # ||e_j - c||
    within_var_u: np.ndarray  # Var_i(u_ij), ddof=1
    within_scatter: np.ndarray  # S_j^2 = (1/(k-1)) sum_i ||y_ij - e_j||^2

    @property
    def cos(self) -> np.ndarray:
        """E1's readout, reconstructed from story-level scalars: cos(e_j - c, d)."""

        return self.mean_u / self.word_norm

    def select(self, index: np.ndarray) -> WordProjectionStats:
        """The same statistics over a subset (or bootstrap resample) of the words."""

        return WordProjectionStats(
            labels=tuple(self.labels[position] for position in index),
            counts=self.counts[index],
            mean_u=self.mean_u[index],
            word_norm=self.word_norm[index],
            within_var_u=self.within_var_u[index],
            within_scatter=self.within_scatter[index],
        )


def word_projection_stats(
    artifact: StoryProjections, labels: tuple[str, ...], *, block: int, direction: str
) -> WordProjectionStats:
    """Reduce the per-story payload to the per-word quantities, for one block and direction."""

    column = artifact.metadata.direction_names.index(direction)
    counts, mean_u, word_norm, within_var, scatter = [], [], [], [], []
    for label in labels:
        rows = artifact.rows_for(label)
        u = artifact.projections[rows, block, column]
        gram = artifact.gram_for(label)[block]
        k = rows.size
        # ||mean_i y||^2 is the mean of ALL Gram entries; the within-word scatter is what the
        # diagonal has left over after that mean is removed. Both exact, neither reachable from
        # per-story scalars alone.
        mean_norm_sq = float(gram.mean())
        counts.append(k)
        mean_u.append(float(u.mean()))
        word_norm.append(sqrt(max(mean_norm_sq, 0.0)))
        within_var.append(float(u.var(ddof=1)))
        scatter.append(float((np.trace(gram) - k * mean_norm_sq) / (k - 1)))
    return WordProjectionStats(
        labels=tuple(labels),
        counts=np.asarray(counts, dtype=float),
        mean_u=np.asarray(mean_u),
        word_norm=np.asarray(word_norm),
        within_var_u=np.asarray(within_var),
        within_scatter=np.asarray(scatter),
    )


def half_sample_cosines(
    artifact: StoryProjections,
    labels: tuple[str, ...],
    *,
    block: int,
    direction: str,
    rng: np.random.Generator,
    n_splits: int,
) -> np.ndarray:
    """``(n_splits, n_words, 2)`` EXACT half-sample word cosines, via the within-word Gram.

    The numerator is linear so the projections carry it; the denominator ``||mean_A y||`` is not,
    and that is precisely what the Gram supplies. Halves are drawn independently per word per
    split, so a topic that happens to sit in one half of one word does not correlate across words.
    """

    column = artifact.metadata.direction_names.index(direction)
    out = np.empty((n_splits, len(labels), 2))
    for word_index, label in enumerate(labels):
        rows = artifact.rows_for(label)
        u = artifact.projections[rows, block, column]
        gram = artifact.gram_for(label)[block]
        cut = rows.size // 2
        for split in range(n_splits):
            order = rng.permutation(rows.size)
            for half, picks in enumerate((order[:cut], order[cut:])):
                norm = sqrt(max(float(gram[np.ix_(picks, picks)].mean()), 0.0))
                out[split, word_index, half] = float(u[picks].mean()) / norm
    return out


@dataclass(frozen=True)
class VarianceComponents:
    """The decomposition the whole rung exists to produce, on the valence-residual cosine scale."""

    mean_k: float
    observed_residual_sd: float
    within_word_variance_cos: float
    between_word_variance_resid: float
    icc_1k_resid: float
    icc_1k_raw: float


def _pooled_within_cos(stats: WordProjectionStats) -> float:
    """Within-word variance on the COSINE scale, pooled by df.

    ``Var_i(u) / ||e_j - c||^2`` is the first-order transfer of the projection's within-word
    variance onto the cosine. It is an approximation, and it is the one place P1 uses one — which
    is exactly why the exact split-half runs beside it as a concordance check.
    """

    weights = stats.counts - 1.0
    return float(np.sum(weights * stats.within_var_u / stats.word_norm**2) / np.sum(weights))


def variance_components(
    stats: WordProjectionStats, design: np.ndarray
) -> tuple[VarianceComponents, np.ndarray]:
    """Split the observed word-level residual spread into its within- and between-word parts."""

    residuals = ols_residuals(stats.cos, design)
    observed_var = float(residuals.var(ddof=1))
    raw_var = float(stats.cos.var(ddof=1))
    mean_k = float(stats.counts.mean())
    within_cos = _pooled_within_cos(stats)
    # The word MEAN carries sigma^2_w / k of sampling noise; whatever the observed spread has left
    # over is the between-word signal. A negative remainder means the observed spread is entirely
    # explained by within-word noise, which is the `recipe is the finding` route, not an error.
    between = observed_var - within_cos / mean_k
    components = VarianceComponents(
        mean_k=mean_k,
        observed_residual_sd=sqrt(observed_var),
        within_word_variance_cos=within_cos,
        between_word_variance_resid=between,
        icc_1k_resid=between / observed_var,
        icc_1k_raw=(raw_var - within_cos / mean_k) / raw_var,
    )
    return components, residuals


def split_half_reliability(halves: np.ndarray, design: np.ndarray) -> dict[str, float]:
    """Spearman-Brown-corrected split-half reliability of the word-level cosine, raw and residual.

    Each half is residualized on the same design as the analysis of record before correlating, so
    the reported number is the reliability of the quantity E1 actually tests rather than of the
    valence shadow that dominates the raw cosine.
    """

    def _corrected(values: np.ndarray) -> tuple[float, float]:
        per_split = [float(np.corrcoef(row[:, 0], row[:, 1])[0, 1]) for row in values]
        half = float(np.mean(per_split))
        return half, 2.0 * half / (1.0 + half)

    raw_half, raw_full = _corrected(halves)
    residualized = np.stack(
        [
            np.column_stack([ols_residuals(row[:, side], design) for side in (0, 1)])
            for row in halves
        ]
    )
    resid_half, resid_full = _corrected(residualized)
    return {
        "n_splits": int(halves.shape[0]),
        "raw_half_length_r": raw_half,
        "raw_spearman_brown": raw_full,
        "residual_half_length_r": resid_half,
        "residual_spearman_brown": resid_full,
    }


def attenuation(stats: WordProjectionStats) -> dict[str, float]:
    """The cosine's denominator inflates by ``S_j^2 / k``; lambda is how much that shrinks it.

    Reported, never applied to the test: lambda is approximately common across words, so it
    multiplies the observed statistic and every permutation draw alike and cancels out of the p.
    Its only job is the k-projection below.
    """

    true_norm_sq = np.maximum(stats.word_norm**2 - stats.within_scatter / stats.counts, 0.0)
    lam = np.sqrt(true_norm_sq / stats.word_norm**2)
    return {
        "lambda_mean_at_observed_k": float(lam.mean()),
        "lambda_min": float(lam.min()),
        "lambda_max": float(lam.max()),
        "n_words_with_norm_fully_explained_by_noise": int(np.sum(true_norm_sq <= 0.0)),
        "mean_true_norm": float(np.sqrt(true_norm_sq).mean()),
        "mean_within_scatter": float(stats.within_scatter.mean()),
        # Every word, not the interesting tail: a writeup that names its least reliable words has
        # to be checkable against the full table, or naming four of them is a selection.
        "per_word_lambda": {
            label: float(value) for label, value in zip(stats.labels, lam, strict=True)
        },
    }


def lambda_at(stats: WordProjectionStats, k: float) -> float:
    """Mean attenuation the same words would carry at ``k`` stories each, or NaN if undefined.

    Averaged over the words whose observed norm still exceeds what within-word noise explains.
    A word where it does not has an estimated true norm of zero and therefore lambda = 0 at EVERY
    k, so including it would drag the mean toward zero without carrying information about how
    attenuation relaxes. If no word survives, the whole de-attenuation is undefined and this
    returns NaN rather than a number the caller would divide by — that regime is the
    pre-registered "reliability so low the ceiling is uninformative" route, and it must report
    itself rather than crash or, worse, return 0.
    """

    true_norm_sq = stats.word_norm**2 - stats.within_scatter / stats.counts
    usable = true_norm_sq > 0.0
    if not np.any(usable):
        return float("nan")
    surviving = true_norm_sq[usable]
    return float(np.mean(np.sqrt(surviving / (surviving + stats.within_scatter[usable] / k))))


def residual_sd_at_k(components: VarianceComponents, k: float) -> float:
    """Word-level residual spread a capture of ``k`` stories per word would produce."""

    between = max(components.between_word_variance_resid, 0.0)
    return sqrt(between + components.within_word_variance_cos / k)


def prophecy(
    stats: WordProjectionStats,
    components: VarianceComponents,
    *,
    observed_effect: float,
    max_k: int,
    coefficient: float,
) -> dict[str, object]:
    """Does ANY affordable capture size bring MDE80(k) below the effect it would then measure?

    Both sides move with k, which is why this is not a one-line power calculation: the detection
    threshold falls as `1/sqrt(k)` while the measured effect RISES, because de-attenuation lifts it
    toward its true value. `k_star` is the first k where they cross, or None if they never do.
    """

    floor = coefficient * sqrt(max(components.between_word_variance_resid, 0.0))
    lam_observed = lambda_at(stats, components.mean_k)
    if not np.isfinite(lam_observed) or lam_observed <= 0.0:
        return {
            "mde80_at_observed_k": coefficient * components.observed_residual_sd,
            "mde80_floor_k_infinite": floor,
            "projected_effect_k_infinite": None,
            "detectable_at_any_k": False,
            "k_star": None,
            "max_k_searched": max_k,
            "de_attenuation_undefined": True,
            "note": (
                "Every word's observed norm is fully explained by within-word noise, so there is "
                "no estimated true norm to de-attenuate toward. No k is projected: the capture "
                "cannot see the words at all at this story count."
            ),
            "curve": [],
        }
    curve = []
    k_star = None
    for k in range(2, max_k + 1):
        mde = coefficient * residual_sd_at_k(components, k)
        effect = observed_effect * lambda_at(stats, k) / lam_observed
        if k_star is None and mde <= effect:
            k_star = k
        curve.append(
            {"k": k, "mde80": mde, "projected_effect": effect, "detectable": mde <= effect}
        )
    return {
        "mde80_at_observed_k": coefficient * components.observed_residual_sd,
        "mde80_floor_k_infinite": floor,
        "projected_effect_k_infinite": observed_effect / lam_observed,
        "detectable_at_any_k": floor <= observed_effect / lam_observed,
        "k_star": k_star,
        "max_k_searched": max_k,
        "de_attenuation_undefined": False,
        "curve": [row for row in curve if row["k"] in (6, 12, 24, 36, 48, max_k)],
    }


def floor_bootstrap(
    stats: WordProjectionStats,
    design: np.ndarray,
    *,
    observed_effect: float,
    rng: np.random.Generator,
    n_resamples: int,
    coefficient: float,
) -> dict[str, object]:
    """How firm is `k -> infinity` never suffices? Resample the WORDS and ask again.

    `prophecy` compares two point estimates, and the losing side of that comparison — the MDE80
    floor `coefficient * sqrt(sigma^2_tau,resid)` — is a variance component estimated from the word
    set's own words. Its
    sampling error is large enough to matter at a gap this narrow, so reporting `detectable_at_any_k
    = False` without it would state a lean as a proof.

    Words are the resampling unit because they are the exchangeable ones: the design row (valence)
    travels with the word, and both the floor and the de-attenuation are recomputed inside each
    draw. What does NOT vary is E1's contrast — P1 re-tests nothing, so `observed_effect` is held
    fixed and this quantifies uncertainty in the THRESHOLD it is compared against, not in itself.
    Read `p_floor_below_effect` as: the fraction of word-sets like this one in which infinite
    stories WOULD suffice.
    """

    n_words = len(stats.labels)
    floors, effects = [], []
    for _ in range(n_resamples):
        index = rng.integers(0, n_words, n_words)
        drawn = stats.select(index)
        components, _ = variance_components(drawn, design[index])
        lam = lambda_at(drawn, components.mean_k)
        if not np.isfinite(lam) or lam <= 0.0:
            continue  # de-attenuation undefined for this word-set; it carries no comparison
        floors.append(coefficient * sqrt(max(components.between_word_variance_resid, 0.0)))
        effects.append(observed_effect / lam)
    floor_draws = np.asarray(floors)
    effect_draws = np.asarray(effects)
    return {
        "n_resamples": n_resamples,
        "n_usable": int(floor_draws.size),
        "resampling_unit": "word",
        "mde80_floor_ci95": [
            float(np.quantile(floor_draws, 0.025)),
            float(np.quantile(floor_draws, 0.975)),
        ],
        "de_attenuated_effect_ci95": [
            float(np.quantile(effect_draws, 0.025)),
            float(np.quantile(effect_draws, 0.975)),
        ],
        "p_floor_below_effect": float(np.mean(floor_draws <= effect_draws)),
        "note": (
            "Post-hoc uncertainty on the prophecy's floor, computed after the pre-registered "
            "verdict was recorded and routing nothing. E1's contrast is held fixed: this is the "
            "error bar on the threshold, not a re-test."
        ),
    }


def topic_adjusted_word_means(
    artifact: StoryProjections,
    labels: tuple[str, ...],
    stats: WordProjectionStats,
    *,
    block: int,
    direction: str,
) -> np.ndarray:
    """Word cosines with story topic swept out as a FIXED effect — the pre-committed secondary.

    Fixed rather than random: 24 dummies against 1,017 rows is trivial df, and a random intercept
    would assume topic effects are uncorrelated with family membership, which a topic x valence
    interaction violates outright. Inference never comes from this — the p stays E1's word-level
    permutation, which is robust to any within-word dependence. This produces a nuisance-adjusted
    point estimate and nothing else.

    Only the numerator is adjusted: the denominator ``||e_j - c||`` is a property of the word's
    geometry, not of which topics it drew, and rescaling it would mix the nuisance back in.
    """

    column = artifact.metadata.direction_names.index(direction)
    rows = np.concatenate([artifact.rows_for(label) for label in labels])
    u = artifact.projections[rows, block, column]
    topics = [artifact.metadata.story_topics[row] for row in rows]
    levels = sorted(set(topics))
    # Word dummies alongside topic dummies: without them the topic effects would absorb the
    # between-word signal this exists to preserve. Drop-first on topics only, so the word columns
    # carry the intercept.
    word_of = np.concatenate(
        [np.full(artifact.rows_for(label).size, i) for i, label in enumerate(labels)]
    )
    design = np.zeros((rows.size, len(labels) + len(levels) - 1))
    design[np.arange(rows.size), word_of] = 1.0
    for column_index, level in enumerate(levels[1:]):
        design[:, len(labels) + column_index] = [1.0 if topic == level else 0.0 for topic in topics]
    coefficients = np.linalg.lstsq(design, u, rcond=None)[0]
    return coefficients[: len(labels)] / stats.word_norm
