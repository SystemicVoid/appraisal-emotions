"""E2 machinery on SYNTHETIC reveal states — the harness's own positive control.

Same discipline as ``test_emotion_mapping``: before an E2 null may be read as adjudicating the
situational-context rival, the harness must be shown able to recover the effect it claims to
detect. Two plantings over the REAL reveal battery, both read at the same block:

- **expectation-tracking** — the reveal state carries ``signed_rpe`` along the emotion basis's
  PC1. Within reward-matched cells the outcome is held fixed and only the stated EV moves, so a
  readout that tracks expectation must produce a positive within-cell slope. The test requires
  recovery on both axes.
- **pure outcome-tracking** — the state carries realised ``reward`` instead. ``reward`` is
  constant inside a reward-matched cell by construction, so the within-cell slope must vanish:
  that is exactly the null the Peiris situational rival predicts, and a harness that "recovers"
  a slope here would be reading between-cell variance.
"""

from __future__ import annotations

import numpy as np
import pytest

from appraisal_emotions.analysis.emotion_vectors import write_emotion_vectors
from appraisal_emotions.analysis.expectation_control import (
    PAIR_AXIS,
    PC1_AXIS,
    expectation_control,
    format_expectation_control_summary,
)
from appraisal_emotions.analysis.reveal_rpe import (
    REVEAL_RPE_STATES_CONTRACT_VERSION,
    build_reveal_rpe_states,
    write_reveal_rpe_states,
)
from appraisal_emotions.core.util import write_json
from conftest import MODEL_SPEC, synthetic_emotion_artifact

HIDDEN = 16
BLOCKS = 2
LABELS = ("elated", "disappointed", "calm", "sad", "surprised")
VALENCE = (1, -1, 1, -1, 0)
PERMUTATIONS = 2000


def _axes() -> tuple[np.ndarray, np.ndarray]:
    basis, _r = np.linalg.qr(np.random.default_rng(3).standard_normal((HIDDEN, 2)))
    return basis[:, 0], basis[:, 1]


def _emotion_artifact():
    """A basis whose PC1 is the planted valence axis, with ``elated``/``disappointed`` on it."""

    u_valence, w_other = _axes()
    rows = [float(value) * u_valence for value in VALENCE]
    rows.append(np.zeros(HIDDEN))  # style_control
    base = np.stack(rows, axis=0)
    rng = np.random.default_rng(4)
    # A small second component keeps PC2 defined without moving PC1 off the valence axis.
    base = base + 0.05 * np.outer(np.arange(len(rows)) - 2.0, w_other)
    vectors = np.stack([base + 0.005 * rng.standard_normal(base.shape) for _ in range(BLOCKS)], 1)
    return synthetic_emotion_artifact(vectors, LABELS, VALENCE), u_valence


def _states(reveals, axis: np.ndarray, *, regressor: str, scale: float) -> np.ndarray:
    rng = np.random.default_rng(9)
    base = np.stack(
        [scale * float(comparison.metadata[regressor]) * axis for comparison in reveals], axis=0
    )
    return np.stack([base + 0.05 * rng.standard_normal(base.shape) for _ in range(BLOCKS)], axis=1)


def _run(tmp_path, reveals, battery, regressor: str):
    emotion, axis = _emotion_artifact()
    states = _states(reveals, axis, regressor=regressor, scale=0.02)
    artifact = build_reveal_rpe_states(
        states,
        reveals,
        spec=MODEL_SPEC,
        seed=7,
        battery_contract_version=REVEAL_RPE_STATES_CONTRACT_VERSION,
    )
    states_path = tmp_path / f"{regressor}_states.json"
    battery_path = tmp_path / f"{regressor}_battery.json"
    emotion_path = tmp_path / f"{regressor}_emotions.json"
    write_reveal_rpe_states(artifact, states_path)
    write_json(battery_path, battery)
    write_emotion_vectors(emotion, emotion_path)
    return expectation_control(
        states_path, battery_path, emotion_path, seed=7, n_permutations=PERMUTATIONS
    )


def test_planted_expectation_tracking_is_recovered(tmp_path, reveals, battery):
    report = _run(tmp_path, reveals, battery, "signed_rpe")
    assert {axis.axis for axis in report.axes} == {PC1_AXIS, PAIR_AXIS}
    for axis in report.axes:
        assert axis.pooled_within_cell_slope > 0.0, f"{axis.axis} lost the planted sign"
        assert axis.reads_expectation, f"{axis.axis} p={axis.p_value}"
        assert axis.n_cells >= 2 and axis.n_rows > axis.n_cells
    assert "expectation" in format_expectation_control_summary(report)


def test_pure_outcome_tracking_gives_a_within_cell_null(tmp_path, reveals, battery):
    report = _run(tmp_path, reveals, battery, "reward")
    for axis in report.axes:
        # reward is constant inside a reward-matched cell, so a reward-only code has no
        # within-cell contrast to read: the slope collapses and the permutation says so.
        assert abs(axis.pooled_within_cell_slope) < 1e-3
        assert not axis.reads_expectation


def test_report_binds_its_inputs_and_carries_the_gate_cap(tmp_path, reveals, battery):
    report = _run(tmp_path, reveals, battery, "signed_rpe")
    assert len(report.states_sha256) == 64
    assert len(report.battery_sha256) == 64
    assert len(report.emotion_vectors_sha256) == 64
    assert report.seed == 7 and report.n_permutations == PERMUTATIONS
    assert report.sensitivity_gate == "G0=pass"
    assert "present-and-separable" in report.verdict_cap
    assert report.block == report.emotion_selected_block


def test_refuses_a_battery_from_a_different_run(tmp_path, reveals, battery):
    emotion, axis = _emotion_artifact()
    states = _states(reveals, axis, regressor="signed_rpe", scale=0.02)
    artifact = build_reveal_rpe_states(
        states,
        reveals,
        spec=MODEL_SPEC,
        seed=7,
        battery_contract_version=REVEAL_RPE_STATES_CONTRACT_VERSION,
    )
    states_path = tmp_path / "states.json"
    battery_path = tmp_path / "short_battery.json"
    emotion_path = tmp_path / "emotions.json"
    write_reveal_rpe_states(artifact, states_path)
    write_json(battery_path, battery.model_copy(update={"reveals": battery.reveals[:5]}))
    write_emotion_vectors(emotion, emotion_path)
    with pytest.raises(ValueError, match="absent from the battery"):
        expectation_control(states_path, battery_path, emotion_path, seed=7, n_permutations=10)
