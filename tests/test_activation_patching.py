"""E3 patching on SYNTHETIC reveal states — the harness's own positive control.

Same discipline as ``test_expectation_control``: before an E3 readout may be believed, the harness
must be shown able to move it when the structure is there and to leave it alone when it is not.
The reveal-token state carries ``signed_rpe`` along the emotion basis's PC1, so donor and
recipient inside a reward-matched cell differ on the readout axis BY CONSTRUCTION (same realised
reward, opposite signed RPE) — a planted expectation difference the patch has to transfer.

Four things the arms must show, all planted:

- ``full_residual`` transfers the whole difference (1.0 in state mode — the wiring check, not a
  result);
- ``v_rpe_component`` transfers most of it, because the planted expectation code IS that axis;
- ``random_component`` transfers ~nothing — the floor, now a quantile over draws;
- ``same_condition_donor`` moves the readout ~nothing: a different rendering of the recipient's
  own condition carries the same expectation, so the no-op control must be flat.

Both modes are covered. ``state`` is arithmetic; ``forward`` runs the fake backend's
``patched_forward``, whose synthetic semantics propagate an injected vector to downstream layers
and leave a self-patch exactly inert — enough to prove the plumbing carries a patch through the
model seam and that the no-op does not, which is the only thing a fake backend can establish.

Plus the design's literal wiring check: a self-patch (donor is recipient) moves nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from appraisal_emotions.analysis.activation_patching import (
    ARMS,
    _aggregate,
    activation_patching,
    build_patch_pairs,
    format_activation_patching_summary,
    patch_state,
)
from appraisal_emotions.analysis.emotion_vectors import write_emotion_vectors
from appraisal_emotions.analysis.expectation_control import PAIR_AXIS, PC1_AXIS
from appraisal_emotions.analysis.reveal_rpe import (
    REVEAL_RPE_STATES_CONTRACT_VERSION,
    build_reveal_rpe_states,
    write_reveal_rpe_directions,
    write_reveal_rpe_states,
)
from appraisal_emotions.backends.fake import FakeBackend
from appraisal_emotions.core.util import write_json
from conftest import MODEL_SPEC, synthetic_directions_artifact, synthetic_emotion_artifact

# The fake backend's synthetic hidden width; forward mode injects real captured rows into it.
HIDDEN = 8
BLOCKS = 4
LABELS = ("elated", "disappointed", "calm", "sad", "surprised")
VALENCE = (1, -1, 1, -1, 0)


def _axes() -> tuple[np.ndarray, np.ndarray]:
    basis, _r = np.linalg.qr(np.random.default_rng(3).standard_normal((HIDDEN, 2)))
    return basis[:, 0], basis[:, 1]


def _artifacts(tmp_path, reveals, battery, *, noise: float = 0.0):
    """States carrying signed RPE along the emotion basis's PC1, plus matching directions."""

    u_valence, w_other = _axes()
    rows = [float(value) * u_valence for value in VALENCE]
    rows.append(np.zeros(HIDDEN))  # style_control
    base = np.stack(rows, axis=0) + 0.05 * np.outer(np.arange(len(rows)) - 2.0, w_other)
    rng = np.random.default_rng(4)
    vectors = np.stack([base + 0.005 * rng.standard_normal(base.shape) for _ in range(BLOCKS)], 1)
    emotion = synthetic_emotion_artifact(vectors, LABELS, VALENCE)

    state_rng = np.random.default_rng(9)
    planted = np.stack(
        [0.02 * float(c.metadata["signed_rpe"]) * u_valence for c in reveals], axis=0
    )
    states = np.stack(
        [planted + noise * state_rng.standard_normal(planted.shape) for _ in range(BLOCKS)], axis=1
    )
    artifact = build_reveal_rpe_states(
        states,
        reveals,
        spec=MODEL_SPEC,
        seed=7,
        battery_contract_version=REVEAL_RPE_STATES_CONTRACT_VERSION,
    )
    directions = synthetic_directions_artifact(
        np.stack(
            [
                np.stack([u_valence] * BLOCKS),  # v_rpe — the planted expectation axis
                np.stack([w_other] * BLOCKS),  # v_ev
                np.stack([w_other] * BLOCKS),  # v_absrpe
            ],
            axis=0,
        )
    )
    paths = {name: tmp_path / f"{name}.json" for name in ("states", "battery", "emo", "dirs")}
    write_reveal_rpe_states(artifact, paths["states"])
    write_json(paths["battery"], battery)
    write_emotion_vectors(emotion, paths["emo"])
    write_reveal_rpe_directions(directions, paths["dirs"])
    return paths


def _report(tmp_path, reveals, battery, *, mode="state", artifact_kwargs=None, **kwargs):
    paths = _artifacts(tmp_path, reveals, battery, **(artifact_kwargs or {}))
    backend = FakeBackend(MODEL_SPEC, decoder_block_count=BLOCKS) if mode == "forward" else None
    return activation_patching(
        paths["states"],
        paths["battery"],
        paths["dirs"],
        paths["emo"],
        mode=mode,
        backend=backend,
        seed=7,
        **kwargs,
    )


def test_pairs_are_reward_matched_opposite_sign_and_surface_matched(reveals):
    pairs, symbol_matched = build_patch_pairs(reveals)
    assert pairs, "the certified battery yields no donor/recipient pair"
    assert symbol_matched, "expected symbol-matched pairs on a battery that renders both orders"
    for pair in pairs:
        donor, recipient = reveals[pair.donor_row], reveals[pair.recipient_row]
        assert donor.metadata["reward_cell_id"] == recipient.metadata["reward_cell_id"]
        assert donor.metadata["reward"] == recipient.metadata["reward"]
        assert donor.metadata["abs_rpe"] == recipient.metadata["abs_rpe"]
        assert donor.metadata["rpe_sign"] == 1 and recipient.metadata["rpe_sign"] == -1
        assert donor.metadata["template_family"] == recipient.metadata["template_family"]
        assert donor.metadata["realised_symbol"] == recipient.metadata["realised_symbol"]
        if pair.same_condition_row is not None:
            no_op = reveals[pair.same_condition_row]
            assert pair.same_condition_row != pair.recipient_row
            assert no_op.metadata["ev_cell_id"] == recipient.metadata["ev_cell_id"]
            assert no_op.metadata["rpe_sign"] == recipient.metadata["rpe_sign"]


def test_pair_count_is_capped_deterministically(reveals):
    capped, _matched = build_patch_pairs(reveals, max_pairs=5)
    full, _again = build_patch_pairs(reveals)
    assert len(capped) == 5
    assert capped == full[:5]


def test_patched_forward_honours_the_index_convention_and_the_self_patch_invariant():
    """The backend contract E3 reads, pinned on the executable backend.

    Both properties are the ones ``backends.base.PatchedForwardBackend`` documents and the HF hook
    implements, so this pins the convention rather than the fake: the patch site is
    ``hidden_states[block + 1]`` (reading it back returns the replacement — a wiring check, not a
    measurement), and a self-patch is inert at EVERY requested layer, which is what lets the
    unpatched baseline be measured through the same call instead of assumed.
    """

    backend = FakeBackend(MODEL_SPEC, decoder_block_count=BLOCKS)
    prompt = "a reveal prompt with several tokens SIL"
    block = 1
    layers = tuple(range(BLOCKS + 1))
    replacement = np.arange(1.0, HIDDEN + 1.0)

    patched = backend.patched_forward(
        prompt, block=block, position="last", replacement=replacement, layers=layers
    )
    # The patch site returns exactly what was put in.
    site = layers.index(block + 1)
    assert np.allclose(patched.states[site], replacement)
    # Blocks upstream of the patch are untouched; blocks downstream of it have moved.
    unpatched_at_site = patched.states[site]
    baseline = backend.patched_forward(
        prompt, block=block, position="last", replacement=unpatched_at_site, layers=layers
    )
    assert np.allclose(baseline.states[site], unpatched_at_site)

    # A self-patch: replace the position's own unpatched value at block + 1 with itself.
    own = backend.hidden_states((prompt,), layers=(block + 1,), position="last").states[0][0]
    self_patched = backend.patched_forward(
        prompt, block=block, position="last", replacement=own, layers=layers
    )
    reference = backend.hidden_states((prompt,), layers=layers, position="last").states[0]
    assert np.allclose(self_patched.states, reference), "a self-patch must be inert at every layer"
    # ...and it is only inert because it IS a self-patch: an arbitrary vector moves downstream.
    downstream = layers.index(block + 2)
    assert not np.allclose(patched.states[downstream], reference[downstream])


def test_a_self_patch_moves_nothing():
    """Design §4 E3's wiring check — under either substitution form."""

    rng = np.random.default_rng(0)
    state = rng.standard_normal(HIDDEN)
    direction = state / np.linalg.norm(state)
    assert np.allclose(patch_state(state, state), state)
    assert np.allclose(patch_state(state, state, direction), state)
    # And a component substitution only moves the recipient along that direction.
    donor = rng.standard_normal(HIDDEN)
    moved = patch_state(state, donor, direction) - state
    assert np.allclose(moved, float(moved @ direction) * direction)


def test_state_mode_transfers_the_planted_expectation_difference(tmp_path, reveals, battery):
    report = _report(tmp_path, reveals, battery)
    assert report.mode == "state" and report.readout_blocks == (report.block,)
    assert report.n_pairs > 0 and report.n_reward_cells > 0
    assert report.symbol_matched and report.symbol_confound is None
    by_arm = {(arm.axis, arm.arm): arm for arm in report.arms}
    assert set(by_arm) == {(axis, arm) for axis in (PC1_AXIS, PAIR_AXIS) for arm in ARMS}
    for axis in (PC1_AXIS, PAIR_AXIS):
        full = by_arm[(axis, "full_residual")]
        v_rpe = by_arm[(axis, "v_rpe_component")]
        random = by_arm[(axis, "random_component")]
        # The ceiling is 1.0 by construction at the state level — a wiring check.
        assert full.transfer_fraction == pytest.approx(1.0, abs=1e-9)
        # The planted expectation code IS v_rpe, so that component carries essentially all of it.
        assert v_rpe.transfer_fraction > 0.9, f"{axis}: v_rpe carried {v_rpe.transfer_fraction}"
        # A random direction carries almost none of it — and the floor is a quantile over many
        # draws, not one lucky direction.
        assert random.n_random_draws > 1
        assert abs(random.transfer_fraction) < 0.5 * v_rpe.transfer_fraction


def test_the_same_condition_donor_is_a_no_op(tmp_path, reveals, battery):
    report = _report(tmp_path, reveals, battery)
    assert report.n_no_op_pairs > 0, "the battery yields no same-condition donor to control with"
    for axis in (PC1_AXIS, PAIR_AXIS):
        no_op = next(
            arm for arm in report.arms if arm.axis == axis and arm.arm == "same_condition_donor"
        )
        full = next(arm for arm in report.arms if arm.axis == axis and arm.arm == "full_residual")
        # A different rendering of the recipient's OWN condition carries the same expectation, so
        # substituting it must not move the readout the way a real donor does.
        assert abs(no_op.mean_shift) < 0.05 * abs(full.mean_shift)
        assert abs(no_op.transfer_fraction) < 0.05


def test_state_mode_caps_the_claim_and_names_the_missing_tier(tmp_path, reveals, battery):
    report = _report(tmp_path, reveals, battery)
    assert len(report.states_sha256) == 64
    assert len(report.battery_sha256) == 64
    assert len(report.directions_sha256) == 64
    assert len(report.emotion_vectors_sha256) == 64
    assert report.sensitivity_gate == "G0=pass"
    assert report.verdict_cap.startswith("present-and-separable")
    assert "it runs no forward" in report.verdict_cap
    assert "mode=forward" in report.verdict_cap  # names the tier it is NOT claiming
    assert "EV and signed RPE are perfectly anti-correlated" in report.identification_limit
    assert set(report.arm_notes) == set(ARMS)
    assert report.continuations == ()
    summary = format_activation_patching_summary(report)
    assert "STATE-LEVEL PREVIEW" in summary
    assert all(arm in summary for arm in ARMS)


def test_forward_mode_propagates_the_patch_downstream(tmp_path, reveals, battery):
    """The causal tier on the fake backend: the plumbing carries a patch through the model seam."""

    report = _report(tmp_path, reveals, battery, mode="forward", max_pairs=6, n_random_draws=3)
    assert report.mode == "forward"
    # Readouts at the patch block AND deeper, where the substitution has actually propagated.
    assert report.block in report.readout_blocks
    downstream = max(report.readout_blocks)
    assert downstream > report.block
    by_arm = {(arm.readout_block, arm.axis, arm.arm): arm for arm in report.arms}
    for axis in (PC1_AXIS, PAIR_AXIS):
        full = by_arm[(downstream, axis, "full_residual")]
        v_rpe = by_arm[(downstream, axis, "v_rpe_component")]
        no_op = by_arm[(downstream, axis, "same_condition_donor")]
        # A real donor moves the downstream readout in the donor's direction...
        assert full.transfer_fraction > 0.3, f"{axis}: forward transfer {full.transfer_fraction}"
        assert v_rpe.transfer_fraction > 0.0
        # ...and a different rendering of the recipient's OWN condition does not.
        assert abs(no_op.mean_shift) < 0.5 * abs(full.mean_shift)
    assert "functionally-used" in report.verdict_cap


def test_forward_mode_stores_raw_unscored_continuations(tmp_path, reveals, battery):
    report = _report(tmp_path, reveals, battery, mode="forward", max_pairs=2, n_random_draws=1)
    assert report.continuations, "forward mode must store continuations for reading"
    arms = {entry.arm for entry in report.continuations}
    assert "unpatched_baseline" in arms and "full_residual" in arms
    assert all(entry.text for entry in report.continuations)
    # No grader, no score, no verdict field anywhere on a continuation: rails.md requires a
    # reality sample of this surface before anything parses it.
    assert set(report.continuations[0].model_dump()) == {
        "reward_cell_id",
        "arm",
        "donor_row",
        "recipient_row",
        "text",
    }
    assert "UNPARSED" in format_activation_patching_summary(report)


def test_forward_mode_marks_the_patch_site_row_as_a_wiring_check(tmp_path, reveals, battery):
    report = _report(tmp_path, reveals, battery, mode="forward", max_pairs=3, n_random_draws=2)
    for arm in report.arms:
        assert arm.wiring_check == (arm.readout_block == report.block)
    assert any(arm.wiring_check for arm in report.arms)
    assert any(not arm.wiring_check for arm in report.arms)
    # The functionally-used cap must point the reader at the non-tautological rows.
    assert "wiring_check=false" in report.verdict_cap
    assert "WIRING CHECK" in format_activation_patching_summary(report)


def test_forward_mode_refuses_a_patch_block_with_nothing_downstream(tmp_path, reveals, battery):
    """Patching the deepest block leaves only the tautological patch-site row to report."""

    paths = _artifacts(tmp_path, reveals, battery)
    with pytest.raises(ValueError, match="no downstream readout block exists"):
        activation_patching(
            paths["states"],
            paths["battery"],
            paths["dirs"],
            paths["emo"],
            mode="forward",
            backend=FakeBackend(MODEL_SPEC, decoder_block_count=BLOCKS),
            block=BLOCKS - 1,
        )


def test_state_mode_with_no_random_draws_drops_the_floor_arm(tmp_path, reveals, battery):
    report = _report(tmp_path, reveals, battery, n_random_draws=0)
    assert report.n_random_draws == 0
    assert report.arms, "the other arms must still be reported"
    assert not any(arm.arm == "random_component" for arm in report.arms)


def test_transfer_fraction_is_null_when_there_is_no_gap_to_normalize_against():
    """A zero denominator is 'no scale', which must not read as a measured zero transfer."""

    result = _aggregate(
        "state", "full_residual", PC1_AXIS, 0, 0, np.asarray([0.4, -0.2]), np.zeros(2)
    )
    assert result.transfer_fraction is None
    assert result.mean_shift != 0.0
    nonzero = _aggregate(
        "state", "full_residual", PC1_AXIS, 0, 0, np.asarray([0.4, -0.2]), np.asarray([1.0, -1.0])
    )
    assert nonzero.transfer_fraction is not None


def test_forward_mode_requires_a_backend(tmp_path, reveals, battery):
    paths = _artifacts(tmp_path, reveals, battery)
    with pytest.raises(ValueError, match="needs a patching backend"):
        activation_patching(
            paths["states"], paths["battery"], paths["dirs"], paths["emo"], mode="forward"
        )


def test_refuses_a_battery_from_a_different_run(tmp_path, reveals, battery):
    paths = _artifacts(tmp_path, reveals, battery)
    write_json(paths["battery"], battery.model_copy(update={"reveals": battery.reveals[:5]}))
    with pytest.raises(ValueError, match="absent from the battery"):
        activation_patching(paths["states"], paths["battery"], paths["dirs"], paths["emo"])
