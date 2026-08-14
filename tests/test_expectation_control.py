"""E2/E2b machinery on SYNTHETIC reveal states — the harness's own positive control.

Same discipline as ``test_emotion_mapping``: before a null may be read as adjudicating a rival,
the harness must be shown able to recover the effect it claims to detect. Here that bar is higher
than it was for E2 alone, because the rung's whole point is that ONE arm reading positive does
not distinguish three codes. So the planting is done three times over the REAL reveal battery,
once per rival, and each planting must produce its OWN signature:

- **comparison** — the state carries ``signed_rpe``. Both matched families have a within-cell
  contrast on it, so both arms read and ``comparison_signature.holds``.
- **outcome tracking** — the state carries realised ``reward``, which is constant inside a
  reward-matched cell by construction: that arm must collapse (the null the Peiris situational
  rival predicts) while the EV-matched arm, where the outcome is what moves, reads.
- **expectation tracking** — the state carries ``ev``, constant inside an EV-matched cell: the
  mirror image. This is the rival E2 alone could never exclude, and a harness that reported
  "comparison" here would be the failure the rung exists to prevent.

The sign bookkeeping for the third planting is worth stating once. With ``projection = s·ev`` and
``signed_rpe = reward − ev``, holding reward fixed gives ``d(projection)/d(signed_rpe) = −s``, so
a *positive* reward-matched slope — the sign E2 published — is planted with ``s < 0``.
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


def _run(tmp_path, reveals, battery, regressor: str, *, scale: float = 0.02):
    emotion, axis = _emotion_artifact()
    states = _states(reveals, axis, regressor=regressor, scale=scale)
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


def _arm(report, family: str):
    (arm,) = [entry for entry in report.arms if entry.cell_family == family]
    return {axis.axis: axis for axis in arm.axes}


def test_planted_comparison_code_lights_both_arms(tmp_path, reveals, battery):
    report = _run(tmp_path, reveals, battery, "signed_rpe")
    assert {arm.cell_family for arm in report.arms} == {"reward_matched", "ev_matched"}
    for family in ("reward_matched", "ev_matched"):
        axes = _arm(report, family)
        assert set(axes) == {PC1_AXIS, PAIR_AXIS}
        for name, axis in axes.items():
            assert axis.pooled_within_cell_slope > 0.0, f"{family}/{name} lost the planted sign"
            assert axis.reads_expectation, f"{family}/{name} p={axis.p_value}"
            assert axis.n_cells >= 2 and axis.n_rows > axis.n_cells
    for signature in report.comparison_signature:
        assert signature.holds, signature.reading
        assert "COMPARISON SIGNATURE" in signature.reading
    assert "comparison signature" in format_expectation_control_summary(report)


def test_planted_outcome_code_reads_only_where_the_outcome_moves(tmp_path, reveals, battery):
    report = _run(tmp_path, reveals, battery, "reward")
    for name, axis in _arm(report, "reward_matched").items():
        # reward is constant inside a reward-matched cell, so a reward-only code has no
        # within-cell contrast to read: the slope collapses and the permutation says so.
        assert abs(axis.pooled_within_cell_slope) < 1e-3, name
        assert not axis.reads_expectation
    for name, axis in _arm(report, "ev_matched").items():
        assert axis.pooled_within_cell_slope > 0.0, name
        assert axis.reads_expectation
    for signature in report.comparison_signature:
        assert not signature.holds
        assert "OUTCOME TRACKER NOT EXCLUDED" in signature.reading


def test_planted_expectation_code_is_not_reported_as_a_comparison(tmp_path, reveals, battery):
    """The rival E2 alone cannot exclude — and the reason the EV-matched arm exists."""

    report = _run(tmp_path, reveals, battery, "ev", scale=-0.02)
    for name, axis in _arm(report, "reward_matched").items():
        # Sign-flipped by the identity signed_rpe = reward - ev: a negative EV code shows up as
        # the SAME positive reward-matched slope E2 published, which is the whole problem.
        assert axis.pooled_within_cell_slope > 0.0, name
        assert axis.reads_expectation
    for name, axis in _arm(report, "ev_matched").items():
        assert abs(axis.pooled_within_cell_slope) < 1e-3, name
        assert not axis.reads_expectation
    for signature in report.comparison_signature:
        assert not signature.holds
        assert "EXPECTATION TRACKER NOT EXCLUDED" in signature.reading


def test_the_two_arms_differ_only_in_their_grouping(tmp_path, reveals, battery):
    report = _run(tmp_path, reveals, battery, "signed_rpe")
    keys = {arm.cell_family: arm.cell_key for arm in report.arms}
    assert keys == {"reward_matched": "reward_cell_id", "ev_matched": "ev_cell_id"}
    scopes = {arm.cell_family: arm.scope_note for arm in report.arms}
    # The EV-matched arm carries its own weaker scope note: its surface is NOT outcome-matched.
    assert "symbol" in scopes["ev_matched"].lower()
    assert scopes["reward_matched"] != scopes["ev_matched"]


def test_report_binds_its_inputs_and_carries_the_gate_cap(tmp_path, reveals, battery):
    report = _run(tmp_path, reveals, battery, "signed_rpe")
    assert len(report.states_sha256) == 64
    assert len(report.battery_sha256) == 64
    assert len(report.emotion_vectors_sha256) == 64
    assert report.seed == 7 and report.n_permutations == PERMUTATIONS
    assert report.sensitivity_gate == "G0=pass"
    assert "present-and-separable" in report.verdict_cap
    assert "CONJUNCTION" in report.verdict_cap
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
