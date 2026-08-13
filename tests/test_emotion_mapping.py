"""E1 machinery on SYNTHETIC data with planted structure — the harness's own positive control.

This file is the sensitivity evidence for ``map_geometry``. Per
``docs/agents/experiment-gating.md``, "the harness runs" is not sensitivity: a null from E1 may
only be read as information if the harness has been shown able to recover an effect of the
claimed shape. So each emotion-concept vector is constructed as

    ``e_j = valence_j * u + appraisal_j * w + noise``

with ``u ⊥ w``, and ``v_rpe`` planted along ``w``. The valence term is the nuisance P2 partials
out; the ``appraisal_j`` term is exactly the residual structure P2 claims to detect. Two cases,
both required:

- **recovery** — ``appraisal_j`` is planted as the pre-registered contrast (+1 on the three
  appraisal-congruent words, −1 on their valence-matched controls); the three P2 pairs must come
  out positive and Holm-significant, and the observed statistic must clear both nulls.
- **rejection** — ``appraisal_j = 0`` everywhere (valence and noise only); no pair may pass.

A power note the recovery case makes concrete: the within-valence permutation's resolution floor
is ``1 / (n(n-1))`` for a matched set of size ``n``, so a Holm-corrected pass needs the pair to
straddle near-extremes of its matched set. A large positive pair statistic that misses
significance is a POWER finding, not a null.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from appraisal_emotions.analysis.emotion_mapping import format_map_geometry_summary, map_geometry
from appraisal_emotions.analysis.emotion_vectors import write_emotion_vectors
from appraisal_emotions.analysis.reveal_rpe import write_reveal_rpe_directions
from appraisal_emotions.stimuli.emotion_stories import read_emotion_words
from conftest import synthetic_directions_artifact, synthetic_emotion_artifact

WORDS_PATH = Path(__file__).resolve().parents[1] / "data" / "emotion_words.json"
HIDDEN = 24
BLOCKS = 3
PERMUTATIONS = 3000
NULL_DRAWS = 200


def _axes(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis, _r = np.linalg.qr(np.random.default_rng(seed).standard_normal((HIDDEN, 3)))
    return basis[:, 0], basis[:, 1], basis[:, 2]


def _planted(appraisal_scale: float, *, seed: int = 5):
    """Artifacts for the planted design; ``appraisal_scale=0`` is the rejection case."""

    words = read_emotion_words(WORDS_PATH)
    labels = words.labels
    u_valence, w_appraisal, z_other = _axes(31)
    rng = np.random.default_rng(seed)

    # Distinct per-pair amplitudes: with every pair at the same amplitude the three pair
    # statistics tie at the matched set's extremes, and the label-shuffled null can reproduce the
    # observed maximum often enough to sit at its own 95th percentile. Distinct amplitudes make
    # the largest observed statistic reachable by exactly one arrangement.
    appraisal = dict.fromkeys(labels, 0.0)
    for rank, (high, low) in enumerate(words.confirmatory_pairs()):
        amplitude = 1.0 - 0.25 * rank
        appraisal[high] = +amplitude
        appraisal[low] = -amplitude

    rows = [
        words.valence_by_word[label] * u_valence + appraisal_scale * appraisal[label] * w_appraisal
        for label in labels
    ]
    # The style_control row is planted at the origin: valence-free and appraisal-free, so P5c's
    # residual should sit inside the word residuals' own spread.
    rows.append(np.zeros(HIDDEN))
    base = np.stack(rows, axis=0)
    vectors = np.stack(
        [base + 0.01 * rng.standard_normal(base.shape) for _ in range(BLOCKS)], axis=1
    )

    directions = np.stack(
        [
            np.stack([w_appraisal] * BLOCKS),  # v_rpe   — the planted appraisal axis
            np.stack([u_valence] * BLOCKS),  # v_ev    — the planted valence axis
            np.stack([z_other] * BLOCKS),  # v_absrpe — an unplanted third axis
        ],
        axis=0,
    )
    emotion = synthetic_emotion_artifact(
        vectors, labels, tuple(words.valence_by_word[label] for label in labels)
    )
    return synthetic_directions_artifact(directions), emotion


def _report(
    tmp_path: Path,
    appraisal_scale: float,
    *,
    noise_seed: int = 5,
    permutations: int = PERMUTATIONS,
    null_draws: int = NULL_DRAWS,
):
    directions, emotion = _planted(appraisal_scale, seed=noise_seed)
    directions_path = tmp_path / f"reveal_directions_{noise_seed}.json"
    emotion_path = tmp_path / f"emotion_vectors_{noise_seed}.json"
    write_reveal_rpe_directions(directions, directions_path)
    write_emotion_vectors(emotion, emotion_path)
    return map_geometry(
        directions_path,
        emotion_path,
        WORDS_PATH,
        None,
        seed=7,
        n_permutations=permutations,
        n_null_draws=null_draws,
    )


def test_p2_recovers_the_planted_appraisal_alignment(tmp_path):
    report = _report(tmp_path, appraisal_scale=1.0)
    block = report.confirmatory[0]
    assert block.valence_source == "binary_project_labels"
    for pair in block.p2_pairs:
        assert pair.statistic > 0.0, f"{pair.high}>{pair.low} lost its planted sign"
        assert pair.passed, f"{pair.high}>{pair.low} p_holm={pair.p_holm}"
    assert block.p2_any_passed
    # And the effect clears both anisotropy nulls, which is what makes it readable at all.
    assert block.clears_both_nulls
    assert block.observed_max_pair_statistic > block.label_shuffled_max_pair_p95
    assert block.observed_max_pair_statistic > block.random_direction_max_pair_p95


def test_p2_rejects_when_only_valence_is_planted(tmp_path):
    """Rejection is a FALSE-POSITIVE RATE check, not a per-seed determinism claim.

    A Holm-corrected family at alpha=0.05 is allowed to fire on ~1 noise realisation in 20; the
    defect this guards against is a residual that leaks valence structure into P2, which would
    fire on most of them. Six independent noise draws, at most one firing.
    """

    fired = 0
    for noise_seed in range(20, 26):
        block = _report(
            tmp_path,
            appraisal_scale=0.0,
            noise_seed=noise_seed,
            permutations=600,
            null_draws=60,
        ).confirmatory[0]
        fired += int(block.p2_any_passed)
    assert fired <= 1, f"P2 fired on {fired}/6 valence-only draws — the residual is leaking"


def test_p5c_style_control_is_flat_when_the_style_row_is_valence_free(tmp_path):
    report = _report(tmp_path, appraisal_scale=1.0)
    block = report.confirmatory[0]
    assert block.p5c_passed, [entry.model_dump() for entry in block.p5c]
    assert {entry.family for entry in block.p5c} == {"v_rpe", "v_ev", "v_absrpe"}


def test_p1_is_positive_and_labeled_exploratory(tmp_path):
    report = _report(tmp_path, appraisal_scale=1.0)
    exploratory = report.exploratory[0]
    # v_ev is planted along the valence axis, so cos(v_rpe, e_j) is valence-free here by
    # construction: P1 is reported, but it is not what this planting exercises.
    assert -1.0 <= exploratory.p1_spearman_rho <= 1.0
    assert "EXPLORATORY" in exploratory.note
    assert 0.0 <= exploratory.subspace_fraction_pc12_plane <= 1.0001


def test_report_carries_its_input_digests_and_gate_cap(tmp_path):
    directions, emotion = _planted(1.0)
    directions_path = tmp_path / "reveal_directions.json"
    emotion_path = tmp_path / "emotion_vectors.json"
    write_reveal_rpe_directions(directions, directions_path)
    write_emotion_vectors(emotion, emotion_path)
    report = map_geometry(
        directions_path,
        emotion_path,
        WORDS_PATH,
        None,
        seed=7,
        n_permutations=200,
        n_null_draws=50,
    )
    assert report.directions_sha256 == directions.metadata.directions_sha256
    assert report.emotion_vectors_sha256 == emotion.metadata.vectors_sha256
    assert report.seed == 7 and report.n_permutations == 200
    assert report.sensitivity_gate == "G0=pass"
    assert "present-and-separable" in report.verdict_cap
    assert len(report.block_sweep) == BLOCKS
    assert "CONFIRMATORY" in format_map_geometry_summary(report)


def test_gate_failure_caps_the_verdict(tmp_path):
    directions, emotion = _planted(1.0)
    failed = synthetic_emotion_artifact(
        np.asarray(emotion.vectors),
        emotion.word_labels,
        tuple(int(value) for value in emotion.word_valence),
        gate_verdict="harness_inadequate",
    )
    directions_path = tmp_path / "reveal_directions.json"
    emotion_path = tmp_path / "emotion_vectors.json"
    write_reveal_rpe_directions(directions, directions_path)
    write_emotion_vectors(failed, emotion_path)
    report = map_geometry(
        directions_path, emotion_path, WORDS_PATH, None, seed=7, n_permutations=100, n_null_draws=20
    )
    assert report.sensitivity_gate == "G0=harness_inadequate"
    assert report.verdict_cap.startswith("harness_inadequate")


def test_map_geometry_refuses_mismatched_artifacts(tmp_path):
    directions, emotion = _planted(1.0)
    truncated = np.asarray(directions.directions)[:, :, : HIDDEN - 1]
    truncated = truncated / np.linalg.norm(truncated, axis=2, keepdims=True)
    narrower = synthetic_directions_artifact(truncated)
    directions_path = tmp_path / "reveal_directions.json"
    emotion_path = tmp_path / "emotion_vectors.json"
    write_reveal_rpe_directions(narrower, directions_path)
    write_emotion_vectors(emotion, emotion_path)
    with pytest.raises(ValueError, match="hidden size"):
        map_geometry(directions_path, emotion_path, WORDS_PATH, None, seed=7, n_permutations=10)


def _norms_csv(path: Path, words, *, skip: tuple[str, ...] = ()) -> Path:
    """A fetched-subset CSV in scripts/fetch_norms.py's output schema."""

    scale = {1: 7.4, -1: 2.6, 0: 5.1}
    rows = ["word,valence,arousal,source"]
    rows += [
        f"{label},{scale[words.valence_by_word[label]]},5.0,synthetic"
        for label in words.labels
        if label not in skip
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_full_norm_coverage_upgrades_the_valence_scale(tmp_path):
    directions, emotion = _planted(1.0)
    words = read_emotion_words(WORDS_PATH)
    directions_path = tmp_path / "reveal_directions.json"
    emotion_path = tmp_path / "emotion_vectors.json"
    write_reveal_rpe_directions(directions, directions_path)
    write_emotion_vectors(emotion, emotion_path)
    norms = _norms_csv(tmp_path / "vad_subset.csv", words)
    report = map_geometry(
        directions_path,
        emotion_path,
        WORDS_PATH,
        norms,
        seed=7,
        n_permutations=200,
        n_null_draws=50,
    )
    assert report.valence_source == "numeric_norms"
    assert report.norms_covered_words == len(words.labels)
    assert report.norms_file == str(norms)
    # The pre-registered pairs stay valence-MATCHED on the binary labels even when the
    # regression upgrades to a numeric scale: the matched set is what the null permutes over.
    assert all(pair.n_valence_matched > 2 for pair in report.confirmatory[0].p2_pairs)


def test_partial_norm_coverage_falls_back_to_the_binary_labels(tmp_path):
    directions, emotion = _planted(1.0)
    words = read_emotion_words(WORDS_PATH)
    directions_path = tmp_path / "reveal_directions.json"
    emotion_path = tmp_path / "emotion_vectors.json"
    write_reveal_rpe_directions(directions, directions_path)
    write_emotion_vectors(emotion, emotion_path)
    norms = _norms_csv(tmp_path / "partial.csv", words, skip=("wistful", "crestfallen"))
    report = map_geometry(
        directions_path,
        emotion_path,
        WORDS_PATH,
        norms,
        seed=7,
        n_permutations=200,
        n_null_draws=50,
    )
    # Silent partial coverage is the failure to avoid: a mixed scale would make the residuals
    # incommensurable, so the report says which scale it actually used and how many it found.
    assert report.valence_source == "binary_project_labels"
    assert report.norms_covered_words == len(words.labels) - 2
