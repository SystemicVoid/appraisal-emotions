"""P1's reliability math, checked against PLANTED components rather than against itself.

The pre-registration requires the analysis of record to be frozen and dry-run before any GPU is
rented, and a dry run that only proves the code executes proves nothing. So the fixtures here plant
a known within-word noise scale on top of known word vectors and ask whether the estimator recovers
the decomposition — with the ground truth measured across INDEPENDENT replicate captures of the
same words, so the check is not the estimator agreeing with its own formula.

That distinction is the whole value of this file. `sigma^2_w` is estimated from the spread of
stories WITHIN one capture; the quantity it claims to be is the spread of the word MEAN ACROSS
captures that never happened. Those coincide only if the sampling model is right, and the replicate
fixture is the only thing here that can tell.

Everything is synthetic; the hidden dimension is small and no model is involved.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from appraisal_emotions.analysis.story_projections import (
    StoryProjections,
    StoryProjectionsMetadata,
)
from appraisal_emotions.analysis.story_reliability import (
    MDE80_COEFFICIENT,
    attenuation,
    floor_bootstrap,
    half_sample_cosines,
    lambda_at,
    prophecy,
    residual_sd_at_k,
    split_half_reliability,
    topic_adjusted_word_means,
    variance_components,
    word_projection_stats,
)
from appraisal_emotions.core.util import states_sha256

N_WORDS = 120
K = 12
HIDDEN = 48
N_BLOCKS = 1
DIRECTION = "v_rpe"
LABELS = tuple(f"w{index:03d}" for index in range(N_WORDS))
TOPICS = tuple(f"t{index}" for index in range(5))


def _artifact(
    states: np.ndarray, topics: tuple[str, ...], direction: np.ndarray
) -> StoryProjections:
    """Wrap planted per-story vectors (n_words, k, hidden) as a valid P1 artifact.

    Built directly rather than through a backend: the capture path has its own tests, and what
    needs exercising here is the arithmetic downstream of a capture.
    """

    flat = states.reshape(N_WORDS * K, HIDDEN)
    directions = np.zeros((5, HIDDEN))
    directions[0] = direction
    projections = np.ascontiguousarray(
        np.einsum("nh,dh->nd", flat, directions)[:, None, :], dtype=np.float64
    )
    norms = np.linalg.norm(flat, axis=1)[:, None]
    grams = np.full((N_WORDS, N_BLOCKS, K, K), np.nan)
    for word in range(N_WORDS):
        grams[word, 0] = states[word] @ states[word].T
    labels = tuple(label for label in LABELS for _ in range(K))
    metadata = StoryProjectionsMetadata(
        model_key="fake",
        backend="fake",
        model_id="fake",
        revision=None,
        tokenizer_id="fake",
        dtype=None,
        min_token=0,
        emotion_vectors_sha256="0" * 64,
        emotion_vectors_selected_block=0,
        directions_sha256="0" * 64,
        directions_selected_block=0,
        words_file_sha256="0" * 64,
        story_labels=labels,
        story_topics=topics,
        story_indices=tuple(range(K)) * N_WORDS,
        n_stories=N_WORDS * K,
        n_blocks=N_BLOCKS,
        hidden_size=HIDDEN,
        direction_names=("v_rpe", "v_ev", "v_absrpe", "pc1", "pc2"),
        stories_per_label=dict.fromkeys(LABELS, K),
        gram_labels=LABELS,
        gram_max_stories_per_label=K,
        gate_max_relative_deviation=0.0,
        gate_max_relative_deviation_threshold=1e-2,
        gate_word_mean_correlation=1.0,
        gate_verdict="pass",
        projections_dtype=str(projections.dtype),
        projections_sha256=states_sha256(projections),
    )
    return StoryProjections(
        metadata=metadata, projections=projections, norms=norms, word_grams=grams
    )


def _planted(seed: int, *, noise: float, tau: np.ndarray, direction: np.ndarray):
    """One capture: the same word vectors, a fresh draw of within-word noise."""

    rng = np.random.default_rng(seed)
    states = tau[:, None, :] + noise * rng.standard_normal((N_WORDS, K, HIDDEN))
    topics = tuple(TOPICS[index % len(TOPICS)] for _ in range(N_WORDS) for index in range(K))
    return _artifact(states, topics, direction)


@pytest.fixture(scope="module")
def planted():
    rng = np.random.default_rng(11)
    direction = rng.standard_normal(HIDDEN)
    direction /= np.linalg.norm(direction)
    # Word vectors with a real spread along the direction, on a base norm large enough that the
    # cosine's denominator is dominated by signal — the regime E0 is actually in.
    tau = (
        1.0 * rng.standard_normal((N_WORDS, HIDDEN))
        + 0.25 * rng.standard_normal((N_WORDS, 1)) * direction
    )
    return direction, tau


@pytest.fixture(scope="module")
def artifact(planted):
    direction, tau = planted
    return _planted(0, noise=0.6, tau=tau, direction=direction)


@pytest.fixture
def design():
    return np.ones((N_WORDS, 1))


def test_word_statistics_are_exact_reductions_of_the_stories(artifact, planted):
    """mean_u, the word norm and the within-word scatter must equal their direct definitions."""

    stats = word_projection_stats(artifact, LABELS, block=0, direction=DIRECTION)
    for index, label in enumerate(LABELS[:20]):
        rows = artifact.rows_for(label)
        u = artifact.projections[rows, 0, 0]
        gram = artifact.gram_for(label)[0]
        assert stats.mean_u[index] == pytest.approx(u.mean(), rel=1e-12)
        # ||mean_i y||^2 from the Gram, against the same thing from the norms and inner products.
        assert stats.word_norm[index] ** 2 == pytest.approx(gram.mean(), rel=1e-12)
        direct = (np.diag(gram) - gram.mean()).sum() / (K - 1)
        assert stats.within_scatter[index] == pytest.approx(direct, rel=1e-9)


def test_within_word_variance_predicts_the_spread_across_replicate_captures(planted, design):
    """The estimator's claim, tested against captures that actually happened.

    `sigma^2_w / k` says how much a word's mean cosine would move if the same word were captured
    again with fresh stories. Here it IS captured again, 60 times, and the two numbers must agree.
    A within-capture spread that failed to predict the across-capture spread would mean the
    sampling model is wrong and every ICC below is decoration.
    """

    direction, tau = planted
    replicates = [_planted(seed, noise=0.6, tau=tau, direction=direction) for seed in range(60)]
    residualized = np.stack(
        [
            variance_components(
                word_projection_stats(rep, LABELS, block=0, direction=DIRECTION), design
            )[1]
            for rep in replicates
        ]
    )
    measured = float(residualized.var(axis=0, ddof=1).mean())

    stats = word_projection_stats(replicates[0], LABELS, block=0, direction=DIRECTION)
    components, _ = variance_components(stats, design)
    predicted = components.within_word_variance_cos / components.mean_k
    assert predicted == pytest.approx(measured, rel=0.12)


def test_icc_recovers_the_planted_reliability(planted, design):
    """ICC = 1 - (within share of the observed word-level spread), against the replicate truth."""

    direction, tau = planted
    replicates = [_planted(seed, noise=0.6, tau=tau, direction=direction) for seed in range(60)]
    residualized = np.stack(
        [
            variance_components(
                word_projection_stats(rep, LABELS, block=0, direction=DIRECTION), design
            )[1]
            for rep in replicates
        ]
    )
    observed_var = float(residualized[0].var(ddof=1))
    truth = 1.0 - float(residualized.var(axis=0, ddof=1).mean()) / observed_var

    stats = word_projection_stats(replicates[0], LABELS, block=0, direction=DIRECTION)
    components, _ = variance_components(stats, design)
    assert components.icc_1k_resid == pytest.approx(truth, abs=0.05)
    assert 0.0 < components.icc_1k_resid < 1.0


def test_icc_falls_as_the_planted_noise_grows(planted, design):
    """The direction of the whole rung: more within-word noise, less reliable word means."""

    direction, tau = planted
    values = []
    for noise in (0.3, 0.9, 2.0):
        stats = word_projection_stats(
            _planted(7, noise=noise, tau=tau, direction=direction),
            LABELS,
            block=0,
            direction=DIRECTION,
        )
        values.append(variance_components(stats, design)[0].icc_1k_resid)
    assert values[0] > values[1] > values[2]


def test_exact_split_half_concordance_with_the_icc(artifact, design):
    """The prereg expects concordance; discordance would mean the norm nonlinearity matters."""

    stats = word_projection_stats(artifact, LABELS, block=0, direction=DIRECTION)
    components, _ = variance_components(stats, design)
    halves = half_sample_cosines(
        artifact,
        LABELS,
        block=0,
        direction=DIRECTION,
        rng=np.random.default_rng(5),
        n_splits=50,
    )
    reliability = split_half_reliability(halves, design)
    assert reliability["residual_spearman_brown"] == pytest.approx(
        components.icc_1k_resid, abs=0.08
    )


def test_split_half_uses_the_gram_denominator_not_the_word_level_one(artifact):
    """A half-sample cosine must differ from the full-sample one; equality would mean it is fake."""

    stats = word_projection_stats(artifact, LABELS, block=0, direction=DIRECTION)
    halves = half_sample_cosines(
        artifact, LABELS, block=0, direction=DIRECTION, rng=np.random.default_rng(1), n_splits=2
    )
    assert not np.allclose(halves[0, :, 0], stats.cos)
    # ...but the two halves must average to something close to it, or the split is biased.
    assert float(np.mean(halves[:, :, :].mean(axis=(0, 2)) - stats.cos)) == pytest.approx(
        0, abs=0.02
    )


def test_de_attenuation_reports_itself_undefined_when_noise_explains_every_word(planted, design):
    """The regime the prereg calls `the ceiling is uninformative` must not divide by zero.

    Found by the pre-capture dry run, which is the whole reason the prereg requires one. The
    construction is degenerate on purpose: the within-word deviations are centred, so each word's
    MEAN carries none of the noise its stories carry, and `||e_j||^2 - S_j^2/k` goes negative for
    every word at once. Real captures sit far from here — the estimate is unbiased, so it is
    negative only by sampling accident — but the estimator must survive the corner rather than
    return a zero the caller divides by.
    """

    direction, tau = planted
    rng = np.random.default_rng(4)
    eps = rng.standard_normal((N_WORDS, K, HIDDEN))
    states = 0.05 * tau[:, None, :] + 3.0 * (eps - eps.mean(axis=1, keepdims=True))
    stats = word_projection_stats(
        _artifact(
            states,
            tuple(TOPICS[i % len(TOPICS)] for _ in range(N_WORDS) for i in range(K)),
            direction,
        ),
        LABELS,
        block=0,
        direction=DIRECTION,
    )
    components, _ = variance_components(stats, design)
    assert not np.isfinite(lambda_at(stats, components.mean_k))
    forecast = prophecy(stats, components, observed_effect=0.02, max_k=100)
    assert forecast["de_attenuation_undefined"]
    assert forecast["k_star"] is None
    assert not forecast["detectable_at_any_k"]


def test_attenuation_is_below_one_and_relaxes_with_k(artifact):
    """lambda is the cosine denominator's inflation; it must vanish as stories accumulate."""

    stats = word_projection_stats(artifact, LABELS, block=0, direction=DIRECTION)
    summary = attenuation(stats)
    assert 0.0 < summary["lambda_mean_at_observed_k"] < 1.0
    curve = [lambda_at(stats, k) for k in (6, 12, 48, 10_000)]
    assert curve == sorted(curve)
    assert curve[-1] == pytest.approx(1.0, abs=1e-3)


def test_prophecy_finds_a_crossing_only_when_one_exists(artifact, design):
    """k* must exist for an effect above the irreducible floor and never for one below it."""

    stats = word_projection_stats(artifact, LABELS, block=0, direction=DIRECTION)
    components, _ = variance_components(stats, design)
    floor = MDE80_COEFFICIENT * np.sqrt(max(components.between_word_variance_resid, 0.0))

    reachable = prophecy(stats, components, observed_effect=floor * 3.0, max_k=200)
    assert reachable["k_star"] is not None
    assert reachable["detectable_at_any_k"]

    # Below the floor no capture size helps: the between-word term never shrinks.
    hopeless = prophecy(stats, components, observed_effect=floor * 0.3, max_k=200)
    assert hopeless["k_star"] is None
    assert not hopeless["detectable_at_any_k"]

    # And the detection threshold must fall monotonically with k toward that floor.
    ladder = [residual_sd_at_k(components, k) for k in (6, 12, 24, 48, 10_000)]
    assert ladder == sorted(ladder, reverse=True)


def test_topic_adjustment_removes_a_planted_topic_shift(planted, design):
    """A topic that shifts every story it touches must not survive into the adjusted word means.

    Unadjusted, an imbalanced topic effect contaminates the word means it lands on; the fixed-effect
    design exists to sweep exactly that out, and this is the check that it does.
    """

    direction, tau = planted
    rng = np.random.default_rng(21)
    states = tau[:, None, :] + 0.6 * rng.standard_normal((N_WORDS, K, HIDDEN))
    # An imbalanced assignment: early words draw topic0 far more often than late ones.
    topics_by_word = [
        [TOPICS[0] if index < K - word // 12 else TOPICS[1 + index % 4] for index in range(K)]
        for word in range(N_WORDS)
    ]
    for word in range(N_WORDS):
        for index in range(K):
            if topics_by_word[word][index] == TOPICS[0]:
                states[word, index] += 3.0 * direction
    topics = tuple(topic for row in topics_by_word for topic in row)
    artifact = _artifact(states, topics, direction)

    stats = word_projection_stats(artifact, LABELS, block=0, direction=DIRECTION)
    adjusted = topic_adjusted_word_means(artifact, LABELS, stats, block=0, direction=DIRECTION)
    share = np.asarray([row.count(TOPICS[0]) / K for row in topics_by_word])

    # The planted shift correlates with topic share in the raw means and must not in the adjusted.
    raw_correlation = abs(float(np.corrcoef(stats.cos, share)[0, 1]))
    adjusted_correlation = abs(float(np.corrcoef(adjusted, share)[0, 1]))
    assert raw_correlation > 0.3
    assert adjusted_correlation < raw_correlation / 3


def test_floor_bootstrap_brackets_the_point_estimate_and_prices_the_comparison(
    artifact, design
) -> None:
    """The bootstrap must price the FLOOR's sampling error, not restate the point estimate.

    Three properties pin it: the interval brackets the floor computed on the full word set, an
    effect far above every draw is called detectable with probability 1, and an effect of zero
    with probability 0. Without the last two, a constant would pass.
    """

    stats = word_projection_stats(artifact, LABELS, block=0, direction=DIRECTION)
    components, _ = variance_components(stats, design)
    floor = MDE80_COEFFICIENT * math.sqrt(max(components.between_word_variance_resid, 0.0))

    priced = floor_bootstrap(
        stats, design, observed_effect=floor, rng=np.random.default_rng(0), n_resamples=400
    )
    low, high = priced["mde80_floor_ci95"]
    assert low <= floor <= high
    assert priced["n_usable"] == 400
    # The effect sits AT the point-estimate floor, so about half the word-sets fall each side.
    assert 0.2 < priced["p_floor_below_effect"] < 0.8

    certain = floor_bootstrap(
        stats, design, observed_effect=1e6, rng=np.random.default_rng(0), n_resamples=200
    )
    impossible = floor_bootstrap(
        stats, design, observed_effect=0.0, rng=np.random.default_rng(0), n_resamples=200
    )
    assert certain["p_floor_below_effect"] == 1.0
    assert impossible["p_floor_below_effect"] == 0.0


def test_per_word_lambda_table_agrees_with_its_own_summary(artifact) -> None:
    """The report names its least reliable words from this table, so it must match min/max exactly."""

    stats = word_projection_stats(artifact, LABELS, block=0, direction=DIRECTION)
    summary = attenuation(stats)
    per_word = summary["per_word_lambda"]
    assert tuple(per_word) == LABELS
    assert min(per_word.values()) == pytest.approx(summary["lambda_min"])
    assert max(per_word.values()) == pytest.approx(summary["lambda_max"])
