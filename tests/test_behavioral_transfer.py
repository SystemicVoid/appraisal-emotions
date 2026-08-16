"""E4 on a PLANTED behavioural surface — the harness's own positive and negative controls.

Same discipline as ``test_activation_patching`` and ``test_expectation_control``: before an E4
number may be believed, the harness must be shown able to find a behavioural transfer that IS
there, to return a readable refusal when the window cannot move at all, and to keep its floors and
its corruption controls flat while it does.

The planted world has two orthogonal codes. The EXPECTATION code lives on ``v_rpe`` and drives the
choice window's logit margin; that is deliberately the friendliest possible version of the
hypothesis — a model that reads exactly the certified direction out of the patched position and
lets it move a later token — because a harness that cannot detect the friendly case has nothing to
say about the real one. The BOOKKEEPING code lives on ``v_reward`` and drives only the windows that
ask for a point total, which is what gives the reachability control something real to find: inside
a reward-matched cell it cannot move (donor and recipient agree on the reward, so the corruption
windows stay flat for the right reason), and across cells it must.

What the plant does NOT do is make the arms pass for free. The random floor now injects the
certified arm's own magnitude along a random direction, so it is a floor that can rise rather than
one that vanishes by construction; the same-condition donor carries the recipient's own expectation
and must score ~0; and the corruption windows are inert unless a test asks for a corrupting
backend.

Every backend here composes ``FakeBackend`` rather than reimplementing it, so the token-prefix
invariant the patch site depends on is the real one throughout.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from appraisal_emotions.analysis.activation_patching import build_patch_pairs
from appraisal_emotions.analysis.behavioral_transfer import (
    CORRUPTION_TOLERANCE,
    GATE_MDE_FRACTION,
    behavioral_arms,
    behavioral_transfer,
    corruption_controls,
    corruption_reading,
    format_behavioral_transfer_summary,
    reachability_control,
    read,
    sensitivity_gate,
    verdict_cap,
)
from appraisal_emotions.analysis.emotion_vectors import write_emotion_vectors
from appraisal_emotions.analysis.reveal_rpe import (
    REVEAL_RPE_STATES_CONTRACT_VERSION,
    build_reveal_rpe_states,
    write_reveal_rpe_directions,
    write_reveal_rpe_states,
)
from appraisal_emotions.backends.base import PatchedForwardResult
from appraisal_emotions.backends.fake import FakeBackend
from appraisal_emotions.core.util import write_json
from appraisal_emotions.stimuli.decision_probes import choice_probe, option_renderings
from appraisal_emotions.stimuli.gambles import GambleGridConfig
from appraisal_emotions.stimuli.reveal_probes import build_reveal_battery
from conftest import (
    MODEL_SPEC,
    orthonormal,
    synthetic_directions_artifact,
    synthetic_emotion_artifact,
)

# A realistic width, as in conftest: component substitution along a random unit direction moves
# the readout axis by (r . v_rpe)^2, which is ~1/HIDDEN — at a toy width the "floor" is not one.
HIDDEN = 128
BLOCKS = 4
PATCH_BLOCK = 1
READOUT_BLOCKS = (PATCH_BLOCK, BLOCKS - 1)
POOL = ("A", "B", "C", "D")
LABELS = ("elated", "disappointed", "calm", "sad", "surprised")
VALENCE = (1, -1, 1, -1, 0)
# The planted expectation code: state = SCALE * rpe_sign * v_rpe. 0.02 matches the E3 fixture's
# amplitude, so the two harnesses are exercised on comparably-sized structure.
SCALE = 0.02
# The planted BOOKKEEPING code, on an orthogonal axis: the realised total, linearly. It is what
# makes the reachability control a real control here — a patch that carries a different cell's
# reward has to move a window that asks for the total, or the harness's own positive control fails
# on the harness's own fixture.
REWARD_SCALE = 0.001
PERMUTATIONS = 200


def _axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(v_rpe, spread, v_reward) — three orthonormal directions, one seed, used everywhere."""

    axes = orthonormal(3, 3, dim=HIDDEN)
    return axes[0], axes[1], axes[2]


def _v_rpe() -> np.ndarray:
    return _axes()[0]


def _planted_states(reveals) -> np.ndarray:
    """Reveal-token states carrying the RPE's SIGN along ``v_rpe``, and the total along ``v_reward``.

    Sign rather than magnitude for the RPE, deliberately. Planting ``signed_rpe`` makes each reward
    cell's natural gap proportional to that cell's ``|RPE|``, and the fixture battery's cells span a
    wide enough range that the between-cell SD swamps the mean — the gate then fails on MDE80 for a
    reason that is a fact about the fixture, not about the machinery under test. A homogeneous
    plant isolates the machinery; cell heterogeneity is what the real run's ~30+ cells are for.

    The reward rides on an orthogonal axis, so it is invisible to every choice-window readout and
    to the certified arm, and visible only to the windows that ask for a total.
    """

    v_rpe, _spread, v_reward = _axes()
    rows = np.stack(
        [
            SCALE * float(r.metadata["rpe_sign"]) * v_rpe
            + REWARD_SCALE * float(r.metadata["reward"]) * v_reward
            for r in reveals
        ],
        axis=0,
    )
    return np.stack([rows] * BLOCKS, axis=1)


class PlantedBackend(FakeBackend):
    """A model whose answer-slot margin reads the patched expectation state, and nothing else.

    ``alpha`` is the strength of the planted behavioural pathway; ``alpha=0`` is the inert window
    B0 exists to detect. ``corrupting`` additionally lets the patch move the invariant-answer
    windows, which is the failure W2/W3 exist to catch.

    The margin is built from the REPLACEMENT vector, which is what a patched forward actually
    receives, so a self-patch reproduces the unpatched margin exactly and every baseline in the
    module is measured rather than assumed.
    """

    def __init__(
        self,
        *,
        alpha: float,
        noise: float = 0.0,
        corrupting: bool = False,
        beta: float = 5.0,
    ):
        super().__init__(MODEL_SPEC, decoder_block_count=BLOCKS)
        self.alpha = alpha
        self.beta = beta
        self.noise = noise
        self.corrupting = corrupting
        self.v_rpe, _spread, self.v_reward = _axes()
        self.calls = 0

    def _option_totals(self, prompt: str, logit_tokens: tuple[str, ...]) -> list[float]:
        """The totals the two answer labels stand for, in ANSWER-TOKEN order."""

        totals = []
        for token in logit_tokens:
            line = next(
                line
                for line in prompt.splitlines()
                if line.startswith(f" {token.strip()} = ") and line.endswith("points")
            )
            totals.append(float(line.split("=", 1)[1].removesuffix("points").strip()))
        return totals

    def _prompt_offset(self, prompt: str) -> float:
        # blake2b, not builtin hash(): Python's string hash is salted per process, which would
        # make every number in this module depend on PYTHONHASHSEED.
        digest = hashlib.blake2b(prompt.encode(), digest_size=8).digest()
        return self.noise * (int.from_bytes(digest, "big") % 1000 / 1000.0 - 0.5)

    def patched_forward(  # noqa: PLR0913
        self,
        prompt: str,
        *,
        block: int,
        position: int | str,
        replacement: np.ndarray,
        layers: tuple[int, ...],
        max_new_tokens: int = 0,
        readout_position: int | str | None = None,
        logit_tokens: tuple[str, ...] = (),
    ) -> PatchedForwardResult:
        self.calls += 1
        carried = float(np.asarray(replacement) @ self.v_rpe)
        # Matched inside the prompt, not at its end: the extension appends the template's own
        # control tokens after the probe, which is the whole point of the concatenation
        # construction.
        if "\nChoice:" in prompt:
            margin = self.alpha * carried
        elif logit_tokens and "What is your point total after the draw?" in prompt:
            # The bookkeeping plant: the model answers with whichever option is nearer the total
            # its PATCHED state carries. Inside a reward-matched cell that is the recipient's own
            # total either way, so the corruption window stays flat for the right reason; across
            # cells it is the donor's, which is what the reachability control predicts.
            total = float(np.asarray(replacement) @ self.v_reward) / REWARD_SCALE
            target, other = self._option_totals(prompt, logit_tokens)
            margin = self.beta * (abs(total - other) - abs(total - target))
        else:
            margin = self.alpha * carried if self.corrupting else 0.0
        margin += self._prompt_offset(prompt)
        # A margin, not two independent logits: the first token carries it and the second is fixed,
        # which is exactly the difference the readout takes.
        logits = (margin, 0.0) if logit_tokens else None
        states = np.stack(
            [np.full(HIDDEN, float(layer)) + carried * self.v_rpe for layer in layers]
        )
        return PatchedForwardResult(
            layers=layers,
            states=states,
            continuation=None,
            generated_token_count=0,
            readout_position=-1,
            logits=None if logits is None else logits[: len(logit_tokens)],
        )


@pytest.fixture(scope="module")
def battery():
    """A battery deep enough that the CLUSTERED null can reach the gate's alpha.

    Overrides the session fixture on purpose. A cell-level sign-flip null has a hard floor of
    ``2 / 2**n_cells`` — the identity assignment and its complement always tie — so at the
    session battery's 5 reward cells the smallest attainable two-sided p is 0.0625 and no gap of
    any size could clear ``GATE_ALPHA``. Testing the gate on a design that cannot pass would
    verify only that it fails.
    """

    return build_reveal_battery(GambleGridConfig(seed=7, single_shot_per_cell=24))


@pytest.fixture(scope="module")
def reveals(battery):
    return battery.reveals


@pytest.fixture(scope="module")
def planted(reveals):
    """A small stratified pair set and the states the plant lives in."""

    pairs, symbol_matched = build_patch_pairs(reveals, max_pairs=12, max_per_cell=1)
    assert symbol_matched
    assert len({pair.reward_cell_id for pair in pairs}) == len(pairs), (
        "max_per_cell=1 must spread the pairs over distinct cells, which is the point of it"
    )
    return pairs, _planted_states(reveals)[:, PATCH_BLOCK, :]


def _gate(reveals, pairs, block_states, backend, **kwargs):
    """The gate alone. Callers that need the baselines it also returns use ``_gate_and_baselines``."""

    return _gate_and_baselines(reveals, pairs, block_states, backend, **kwargs)[0]


def _gate_and_baselines(reveals, pairs, block_states, backend, **kwargs):
    return sensitivity_gate(
        backend,
        reveals,
        pairs,
        block_states,
        patch_block=PATCH_BLOCK,
        readout_blocks=READOUT_BLOCKS,
        answer_pool=POOL,
        seed=7,
        n_permutations=PERMUTATIONS,
        **kwargs,
    )


def _arms(reveals, pairs, block_states, gate, baselines, backend, **kwargs):
    return behavioral_arms(
        backend,
        reveals,
        pairs,
        block_states,
        gate,
        baselines,
        _v_rpe(),
        patch_block=PATCH_BLOCK,
        readout_blocks=READOUT_BLOCKS,
        answer_pool=POOL,
        seed=7,
        n_permutations=PERMUTATIONS,
        n_random_draws=6,
        **kwargs,
    )


# --------------------------------------------------------------------------------------
# B0 — the gate
# --------------------------------------------------------------------------------------


def test_the_gate_measures_a_denominator_and_passes_when_the_window_can_move(reveals, planted):
    pairs, block_states = planted
    gate = _gate(reveals, pairs, block_states, PlantedBackend(alpha=50.0, noise=0.05))

    assert gate.passed, gate.verdict
    assert "B0 PASSED" in gate.verdict
    assert len(gate.per_pair_gap) == len(pairs)
    assert gate.p_value < gate.alpha
    assert gate.mde80 < GATE_MDE_FRACTION * abs(gate.mean_natural_gap)
    # The donor is the +RPE member, so the planted gap has the plant's own sign.
    assert gate.mean_natural_gap > 0.0


def test_the_gate_refuses_and_names_the_reason_when_the_window_cannot_move(reveals, planted):
    """The E3 failure, reproduced deliberately: a window nothing the manipulation does can move.

    This must come back as a REFUSAL with a named cause, not as a transfer fraction of zero. The
    whole design commitment is that an unmeasured denominator produces no readable number.
    """

    pairs, block_states = planted
    gate = _gate(reveals, pairs, block_states, PlantedBackend(alpha=0.0, noise=0.05))

    assert not gate.passed
    assert "B0 FAILED (no natural gap)" in gate.verdict
    assert "harness_inadequate" in gate.verdict


def test_the_patched_leg_will_not_spend_a_forward_behind_a_failed_gate(reveals, planted):
    pairs, block_states = planted
    backend = PlantedBackend(alpha=0.0, noise=0.05)
    gate, baselines = _gate_and_baselines(reveals, pairs, block_states, backend)
    spent = backend.calls

    with pytest.raises(ValueError, match="B0 did not pass"):
        _arms(reveals, pairs, block_states, gate, baselines, backend)
    assert backend.calls == spent, "a failed gate must cost exactly zero patched forwards"


def test_a_design_that_could_never_pass_says_so_instead_of_failing_quietly(reveals, planted):
    """Too few cells and a dead window produce the same failing number; only one is about the model.

    A cell-clustered sign-flip null cannot return a two-sided p below ``2 / 2**n_cells``, so at 4
    cells the smallest attainable p is 0.125 and the 0.01 bar is unreachable by construction. That
    has to read as a design defect, not as evidence that the manipulation does nothing.
    """

    pairs, block_states = planted
    gate = _gate(reveals, pairs[:4], block_states, PlantedBackend(alpha=50.0, noise=0.05))

    assert not gate.passed
    assert "B0 FAILED (design floor)" in gate.verdict
    assert "No natural gap of any size could pass this gate" in gate.verdict
    # ...and the gap it measured is large: the refusal is about the design, not the effect.
    assert abs(gate.mean_natural_gap) > 0.5


def test_titration_selects_the_level_nearest_indifference(reveals, planted):
    """Selected on unpatched data only, so it cannot be an outcome-dependent choice."""

    pairs, block_states = planted

    class Titrating(PlantedBackend):
        def _prompt_offset(self, prompt: str) -> float:
            # A large constant preference that shrinks as the certain amount approaches the EV.
            for certain, offset in ((10, 4.0), (20, 0.25), (30, -3.0)):
                if f" = {certain} points for sure" in prompt:
                    return offset
            return 0.0

    gate = _gate(reveals, pairs, block_states, Titrating(alpha=50.0))
    selected = [level for level in gate.levels if level.selected]
    assert [level.certain for level in selected] == [20]
    assert selected[0].mean_abs_baseline_margin == min(
        level.mean_abs_baseline_margin for level in gate.levels
    )


# --------------------------------------------------------------------------------------
# The patched arms
# --------------------------------------------------------------------------------------


def test_the_planted_transfer_is_carried_by_the_certified_direction_and_not_by_the_floor(
    reveals, planted
):
    pairs, block_states = planted
    backend = PlantedBackend(alpha=50.0, noise=0.05)
    gate, baselines = _gate_and_baselines(reveals, pairs, block_states, backend)
    arms = {arm.arm: arm for arm in _arms(reveals, pairs, block_states, gate, baselines, backend)}

    assert set(arms) >= {"full_residual", "v_rpe_component", "random_component"}
    assert arms["full_residual"].transfer_fraction == pytest.approx(1.0, abs=0.05)
    assert arms["v_rpe_component"].transfer_fraction == pytest.approx(1.0, abs=0.05)
    # The floor is MAGNITUDE-matched, so it is not expected to vanish: it injects the certified
    # arm's own push along a random direction, whose projection back onto v_rpe is ~1/sqrt(HIDDEN).
    # At this fixture's toy width that is 0.09 per draw and ~0.2 at a 95th percentile over six; at
    # the run's 5120 it is 0.014. What must hold at any width is that the certified arm clears it
    # by a wide margin — an old-style floor that vanished by construction would clear it too, and
    # would mean nothing.
    assert abs(arms["random_component"].transfer_fraction) < 0.4
    assert arms["v_rpe_component"].transfer_fraction > 3 * abs(
        arms["random_component"].transfer_fraction
    )
    if "same_condition_donor" in arms:
        assert abs(arms["same_condition_donor"].transfer_fraction) < 0.1
    assert arms["v_rpe_component"].p_value is not None
    assert arms["v_rpe_component"].p_value < 0.05
    assert arms["random_component"].p_value is None, "a floor reports a quantile, not a p-value"
    assert arms["random_component"].n_random_draws == 6


def test_every_arm_is_scored_against_the_gates_own_measured_denominator(reveals, planted):
    pairs, block_states = planted
    backend = PlantedBackend(alpha=50.0, noise=0.05)
    gate, baselines = _gate_and_baselines(reveals, pairs, block_states, backend)
    arms = _arms(reveals, pairs, block_states, gate, baselines, backend)

    for arm in arms:
        assert len(arm.per_pair_shift) == arm.n_pairs
        assert len(arm.per_pair_gap) == arm.n_pairs
        if arm.n_pairs == len(pairs):
            assert arm.per_pair_gap == gate.per_pair_gap


def test_the_free_rider_axis_read_rides_along_without_extra_forwards(reveals, planted):
    pairs, block_states = planted
    backend = PlantedBackend(alpha=50.0, noise=0.05)
    axes = {block: {"pc1": _v_rpe()} for block in READOUT_BLOCKS}
    gate, baselines = _gate_and_baselines(reveals, pairs, block_states, backend, axes_by_block=axes)
    before = backend.calls
    arms = _arms(reveals, pairs, block_states, gate, baselines, backend, axes_by_block=axes)
    spent_with_axes = backend.calls - before

    assert set(gate.free_rider_axis_gap) == {f"pc1@{block}" for block in READOUT_BLOCKS}
    assert all(f"pc1@{block}" in arms[0].free_rider_axis_shift for block in READOUT_BLOCKS)
    assert "validity gate" in gate.free_rider_note

    backend_without = PlantedBackend(alpha=50.0, noise=0.05)
    gate_without, baselines_without = _gate_and_baselines(
        reveals, pairs, block_states, backend_without
    )
    before_without = backend_without.calls
    _arms(reveals, pairs, block_states, gate_without, baselines_without, backend_without)
    assert backend_without.calls - before_without == spent_with_axes


# --------------------------------------------------------------------------------------
# The readout's own wiring, and the positive control that makes every null readable
# --------------------------------------------------------------------------------------


def test_the_readout_asks_for_a_later_position_and_the_contracts_own_layer_indices(reveals):
    """Two properties nothing else in this file can see, because the planted backend ignores both.

    The rung's whole argument is that the readout sits at a LATER token than the patch — a
    same-position readout is reproduced by the residual stream's identity path with no computation.
    And ``layers`` is a raw ``hidden_states`` index, where post-block *l* is index *l+1*; passing
    block numbers straight through reads the block's INPUT, one block upstream of everything the
    free-rider projection then assumes.
    """

    seen: dict[str, object] = {}

    class Recording(PlantedBackend):
        def patched_forward(self, prompt, **kwargs):  # noqa: ANN001, ANN003
            seen.update(kwargs)
            return super().patched_forward(prompt, **kwargs)

    backend = Recording(alpha=1.0)
    probe = choice_probe(
        symbol=str(reveals[0].metadata["realised_symbol"]),
        reward=float(reveals[0].metadata["reward"]),
        rendering=option_renderings("A", "B")[0],
        certain=20,
    )
    read(
        backend,
        reveals[0],
        probe,
        block=PATCH_BLOCK,
        replacement=np.zeros(HIDDEN),
        readout_blocks=READOUT_BLOCKS,
    )
    assert seen["readout_position"] == "last"
    assert seen["position"] != "last", "the patch must sit at an EARLIER token than the readout"
    assert seen["layers"] == tuple(block + 1 for block in READOUT_BLOCKS)


def _reachability(reveals, block_states, backend, **kwargs):
    return reachability_control(
        backend,
        reveals,
        block_states,
        patch_block=PATCH_BLOCK,
        readout_blocks=READOUT_BLOCKS,
        answer_pool=POOL,
        seed=7,
        n_permutations=PERMUTATIONS,
        n_pairs=8,
        **kwargs,
    )


def test_the_reachability_control_detects_that_the_answer_slot_can_be_reached(reveals, planted):
    """A donor from a DIFFERENT reward cell must drag the total window toward its own total."""

    _pairs, block_states = planted
    control = _reachability(reveals, block_states, PlantedBackend(alpha=50.0, noise=0.05))

    assert control.passed, control.verdict
    assert control.mean_shift < 0.0, "the margin must fall toward the DONOR's total"
    assert control.p_value < control.alpha
    assert "REACHABLE" in control.verdict


def test_an_unreachable_answer_slot_is_named_as_such_and_not_read_as_a_model_fact(reveals, planted):
    """The failure this control exists for: a surface where nothing at the reveal token crosses
    positions at all. It must come back as an instrument verdict, not as a transfer of zero."""

    _pairs, block_states = planted
    control = _reachability(reveals, block_states, PlantedBackend(alpha=50.0, noise=0.05, beta=0.0))

    assert not control.passed
    assert "UNREACHABLE" in control.verdict
    assert "harness_inadequate" in control.verdict


def test_the_reachability_pairs_push_in_both_directions(reveals, planted):
    """A readout that simply prefers the larger number would move the same way in both halves.
    Alternating the donor/recipient roles makes that cancel and leaves only donor-following."""

    _pairs, block_states = planted
    control = _reachability(reveals, block_states, PlantedBackend(alpha=50.0))
    assert control.n_pairs % 2 == 0
    first, second = (
        control.per_pair_shift[0::2],
        control.per_pair_shift[1::2],
    )
    assert all(value < 0 for value in first) and all(value < 0 for value in second)


# --------------------------------------------------------------------------------------
# W2/W3 — corruption controls
# --------------------------------------------------------------------------------------


def _controls(reveals, pairs, block_states, backend):
    return corruption_controls(
        backend,
        reveals,
        pairs,
        block_states,
        _v_rpe(),
        patch_block=PATCH_BLOCK,
        readout_blocks=READOUT_BLOCKS,
        answer_pool=POOL,
        choice_natural_gap=1.0,
    )


def test_clean_corruption_controls_read_as_clean(reveals, planted):
    pairs, block_states = planted
    controls = _controls(reveals, pairs, block_states, PlantedBackend(alpha=50.0, noise=0.5))

    assert {control.window for control in controls} == {"outcome_recall", "running_total"}
    assert all(control.within_tolerance for control in controls)
    clean, note = corruption_reading(controls)
    assert clean
    assert "NOT an affect-versus-arithmetic dissociation" in note


def test_a_patch_that_moves_an_invariant_answer_is_caught_and_capped(reveals, planted):
    pairs, block_states = planted
    controls = _controls(
        reveals, pairs, block_states, PlantedBackend(alpha=50.0, noise=0.5, corrupting=True)
    )

    assert any(not control.within_tolerance for control in controls)
    assert max(control.relative_to_gap or 0.0 for control in controls) > CORRUPTION_TOLERANCE
    clean, note = corruption_reading(controls)
    assert not clean
    assert "CORRUPTION" in note


def test_no_controls_at_all_is_not_the_same_as_clean_controls():
    clean, note = corruption_reading(())
    assert not clean
    assert "uncontrolled" in note


# --------------------------------------------------------------------------------------
# Verdict routing and the end-to-end artifact
# --------------------------------------------------------------------------------------


def _arm(name, *, fraction, per_rendering=None, p_value=0.001):
    from appraisal_emotions.analysis.behavioral_transfer import ArmTransfer

    return ArmTransfer(
        arm=name,
        window="choice",
        n_pairs=4,
        n_random_draws=0,
        mean_shift=fraction,
        normalised_shift=fraction,
        transfer_fraction=fraction,
        p_value=p_value,
        per_rendering_fraction=tuple(per_rendering or (fraction,) * 4),
        free_rider_axis_shift={},
        per_pair_shift=(),
        per_pair_gap=(),
        note="",
    )


WINNING_ARMS = (
    _arm("v_rpe_component", fraction=0.8),
    _arm("random_component", fraction=0.05),
    _arm("same_condition_donor", fraction=0.1),
)


def test_verdict_cap_routes_g0_then_reachability_then_b0_then_the_arms():
    assert "G0 sensitivity gate did NOT pass" in verdict_cap(
        gate_passed=True, corruption_clean=True, g0_passed=False, reachable=True, arms=WINNING_ARMS
    )
    assert "reachability control FAILED" in verdict_cap(
        gate_passed=True, corruption_clean=True, g0_passed=True, reachable=False, arms=WINNING_ARMS
    )
    assert "B0 failed" in verdict_cap(
        gate_passed=False, corruption_clean=True, g0_passed=True, reachable=True, arms=WINNING_ARMS
    )
    cap = verdict_cap(
        gate_passed=True, corruption_clean=True, g0_passed=True, reachable=True, arms=WINNING_ARMS
    )
    assert cap.startswith("functionally-used, pilot-suggestive")
    assert "LATER token" in cap
    assert "no welfare, sentience or experience claim" in cap
    assert "functionally-used, CONFOUNDED" in verdict_cap(
        gate_passed=True, corruption_clean=False, g0_passed=True, reachable=True, arms=WINNING_ARMS
    )


def test_an_arm_that_beats_its_controls_by_arithmetic_alone_is_not_a_positive():
    """Three point comparisons with no noise model clear roughly one time in eight on a flat
    surface. The arm's own cell-clustered null was already being computed and read by nothing."""

    noisy = (
        _arm("v_rpe_component", fraction=0.8, p_value=0.40),
        _arm("random_component", fraction=0.05),
        _arm("same_condition_donor", fraction=0.1),
    )
    cap = verdict_cap(
        gate_passed=True, corruption_clean=True, g0_passed=True, reachable=True, arms=noisy
    )
    assert cap.startswith("no transfer")

    unpermuted = (
        _arm("v_rpe_component", fraction=0.8, p_value=None),
        _arm("random_component", fraction=0.05),
        _arm("same_condition_donor", fraction=0.1),
    )
    assert verdict_cap(
        gate_passed=True,
        corruption_clean=True,
        g0_passed=True,
        reachable=True,
        arms=unpermuted,
    ).startswith("no transfer")


def test_a_null_is_only_readable_once_the_ceiling_shows_the_design_could_see_it():
    """Reachability swaps a whole residual across +-160 points; the arms inject a within-cell EV
    difference. A flat table under an unreadable ceiling is an instrument fact, not a null."""

    flat = (
        _arm("full_residual", fraction=0.01),
        _arm("v_rpe_component", fraction=0.0),
        _arm("random_component", fraction=0.05),
        _arm("same_condition_donor", fraction=0.0),
    )
    cap = verdict_cap(
        gate_passed=True,
        corruption_clean=True,
        g0_passed=True,
        reachable=True,
        arms=flat,
        ceiling_readable=False,
        ceiling_note="the full-residual ceiling does not clear the random floor",
    )
    assert cap.startswith("harness_inadequate for the arms")
    assert "claim stays OPEN" in cap
    assert verdict_cap(
        gate_passed=True,
        corruption_clean=True,
        g0_passed=True,
        reachable=True,
        arms=flat,
        ceiling_readable=True,
    ).startswith("no transfer")


def test_a_flat_arm_table_may_not_be_reported_as_a_positive():
    """The defect this exists for: a verdict built from the gate booleans alone announced
    "functionally-used" on a run whose every arm read zero, while its own prose promised a
    comparison against the floor and the no-op that nothing computed."""

    flat = (
        _arm("v_rpe_component", fraction=0.02),
        _arm("random_component", fraction=0.05),
        _arm("same_condition_donor", fraction=0.01),
    )
    cap = verdict_cap(
        gate_passed=True, corruption_clean=True, g0_passed=True, reachable=True, arms=flat
    )
    assert cap.startswith("no transfer")
    assert "readable null" in cap


def test_the_floor_is_compared_against_the_arms_weakest_rendering_not_its_mean():
    """The floor scores ONE rendering cell per draw; the certified arm averages four. Comparing a
    single-cell quantile against a four-cell mean credits the arm with noise it did not have to
    survive, so the comparison uses the arm's weakest cell."""

    lucky_mean = (
        _arm("v_rpe_component", fraction=0.30, per_rendering=(0.05, 0.20, 0.45, 0.50)),
        _arm("random_component", fraction=0.10),
        _arm("same_condition_donor", fraction=0.01),
    )
    cap = verdict_cap(
        gate_passed=True, corruption_clean=True, g0_passed=True, reachable=True, arms=lucky_mean
    )
    assert cap.startswith("no transfer"), "0.05 in the weakest cell does not clear a 0.10 floor"


def _artifacts(tmp_path, reveals, battery):
    # The SAME axes the plant uses. A near-miss v_rpe here would make the certified arm's
    # substitution disturb the orthogonal bookkeeping axis, and the corruption controls would
    # correctly report damage that is an artifact of the fixture.
    v_rpe, other, _v_reward = _axes()
    rows = [float(value) * v_rpe for value in VALENCE]
    rows.append(np.zeros(HIDDEN))  # style_control
    base = np.stack(rows, axis=0) + 0.05 * np.outer(np.arange(len(rows)) - 2.0, other)
    emotion = synthetic_emotion_artifact(
        np.stack([base] * BLOCKS, axis=1), LABELS, VALENCE, selected_block=PATCH_BLOCK
    )
    states = build_reveal_rpe_states(
        _planted_states(reveals),
        reveals,
        spec=MODEL_SPEC,
        seed=7,
        battery_contract_version=REVEAL_RPE_STATES_CONTRACT_VERSION,
    )
    # selected_block on the DIRECTIONS artifact, because that is what the patch block now
    # defaults to -- the direction is certified at a block, and reading it at another block
    # injects an uncertified vector under the certified arm's name.
    directions = synthetic_directions_artifact(
        np.stack([np.stack([v_rpe] * BLOCKS)] * 3, axis=0), selected_block=PATCH_BLOCK
    )
    paths = {name: tmp_path / f"{name}.json" for name in ("states", "battery", "emo", "dirs")}
    write_reveal_rpe_states(states, paths["states"])
    write_json(paths["battery"], battery)
    write_emotion_vectors(emotion, paths["emo"])
    write_reveal_rpe_directions(directions, paths["dirs"])
    return paths


def test_end_to_end_report_reads_as_one_licensed_claim(tmp_path, reveals, battery):
    paths = _artifacts(tmp_path, reveals, battery)
    report = behavioral_transfer(
        paths["states"],
        paths["battery"],
        paths["dirs"],
        paths["emo"],
        backend=PlantedBackend(alpha=50.0, noise=0.05),
        model_key="fake",
        max_pairs=12,
        max_per_cell=1,
        n_random_draws=4,
        # More than the unit tests use: with k cells the clustered null's floor is 2/2**k, so at
        # this fixture's 10 cells the true p sits at ~0.002 and a 200-draw estimate straddles the
        # 0.01 bar on Monte-Carlo noise alone.
        n_permutations=2000,
        answer_candidates=POOL,
    )

    assert report.artifact_contract_version == "behavioral_transfer/v1"
    assert report.patch_block == PATCH_BLOCK == report.direction_block, (
        "the patch block must default to where the direction was CERTIFIED, not to the emotion "
        "artifact's block; on the run of record those differ and the emotion one is out of range"
    )
    assert report.direction_verdict == "separable-signed-rpe"
    assert report.ceiling_readable
    assert report.gate.passed
    assert report.n_cells == report.n_pairs, "stratification must spread pairs across cells"
    assert report.answer_pool == POOL
    assert report.symbol_matched and report.symbol_confound is None
    assert {arm.arm for arm in report.arms} >= {"full_residual", "v_rpe_component"}
    assert report.corruption_clean
    assert report.verdict_cap.startswith("functionally-used, pilot-suggestive")
    assert "identity path" in report.cross_position_note
    assert "NOT 'signed RPE rather than EV transfers'" in report.identification_limit

    summary = format_behavioral_transfer_summary(report)
    assert "B0 sensitivity gate" in summary
    assert "VERDICT CAP:" in summary
    assert "v_rpe_component" in summary


def test_the_report_records_which_wiring_semantics_this_stack_has(tmp_path, reveals, battery):
    """§8's open obligation: E3 documented a patch-site row that verifies nothing on this stack.

    E4 does not assume either semantics — it reads the site back once and records which one it
    got, so a false wiring claim cannot survive in the artifact.
    """

    paths = _artifacts(tmp_path, reveals, battery)
    report = behavioral_transfer(
        paths["states"],
        paths["battery"],
        paths["dirs"],
        paths["emo"],
        backend=PlantedBackend(alpha=50.0, noise=0.05),
        max_pairs=12,
        max_per_cell=1,
        n_random_draws=2,
        n_permutations=2000,
        answer_candidates=POOL,
    )
    assert report.gate.passed, "the wiring check is only spent behind a passing gate"
    assert report.wiring_check.split(":")[0] in {"POST-HOOK", "PRE-HOOK", "UNEXPECTED"}


def test_spending_the_arms_past_a_failed_gate_buys_numbers_but_not_a_verdict(
    tmp_path, reveals, battery
):
    """The override exists so an operator can have the descriptive table. It must not launder it.

    A flat backend fails the gates, so the default run spends nothing. Asking for the arms anyway
    fills the table and changes NOTHING about what the run licensed: ``verdict_cap`` reads the same
    booleans in the same order and still says harness_inadequate. If this test ever goes green on a
    non-``harness_inadequate`` verdict, the flag has become a way to buy a claim with money.
    """

    paths = _artifacts(tmp_path, reveals, battery)
    common = {
        "backend": PlantedBackend(alpha=0.0, noise=0.05),
        "max_pairs": 12,
        "max_per_cell": 1,
        "n_random_draws": 2,
        "n_permutations": 200,
        "answer_candidates": POOL,
    }

    withheld = behavioral_transfer(
        paths["states"], paths["battery"], paths["dirs"], paths["emo"], **common
    )
    assert not withheld.gate.passed or not withheld.reachability.passed, (
        "the fixture must fail a gate for this test to be testing anything"
    )
    assert withheld.arms == ()
    assert withheld.arms_spent_under_failed_gate is None
    assert withheld.verdict_cap.startswith("harness_inadequate")

    spent = behavioral_transfer(
        paths["states"],
        paths["battery"],
        paths["dirs"],
        paths["emo"],
        spend_arms_anyway=True,
        **common,
    )
    assert spent.arms, "the flag's whole job is to fill the arm table"
    assert spent.verdict_cap.startswith("harness_inadequate")
    assert spent.verdict_cap == withheld.verdict_cap, (
        "the override must not reach the routing: same gates in, same verdict out"
    )
    assert "DESCRIPTIVE ONLY" in spent.arms_spent_under_failed_gate
    summary = format_behavioral_transfer_summary(spent)
    # "E3 transfer" appears only in the arm table's header; "mean shift" also appears up in the
    # reachability line, which is not the table this warning is guarding.
    assert summary.index("DESCRIPTIVE ONLY") < summary.index("E3 transfer"), (
        "the warning has to come before the table a reader would otherwise stop at"
    )


# --------------------------------------------------------------------------------------
# The preflight script — the one thing that runs BEFORE the weights are worth renting
# --------------------------------------------------------------------------------------


def test_the_surface_preflight_script_runs_end_to_end(tmp_path, reveals, battery, monkeypatch):
    """``main()`` had never been executed by anything.

    It shipped with ``spec.model_key`` for a field named ``key``, so every invocation died with an
    AttributeError — in the full mode, immediately AFTER the 27B had been downloaded and loaded.
    Python evaluates dict values in order, so the crash preceded every check the script exists to
    perform. The four surface defects §10 credits it with finding were unfindable as shipped.

    This test does the one thing that would have caught it: it calls ``main()``.
    """

    import json as _json
    import sys

    import scripts.e4_surface_preflight as preflight

    paths = _artifacts(tmp_path, reveals, battery)
    out = tmp_path / "preflight.json"
    monkeypatch.setattr(preflight, "create_backend", lambda spec: FakeBackend(MODEL_SPEC))
    monkeypatch.setattr(preflight, "free_backend", lambda backend: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "e4_surface_preflight.py",
            "--states",
            str(paths["states"]),
            "--battery",
            str(paths["battery"]),
            "--config",
            "configs/reveal_rpe_smoke.yaml",
            "--out",
            str(out),
            "--tokenizer-only",
            "--answer-candidates",
            *POOL,
        ],
    )

    preflight.main()

    report = _json.loads(out.read_text())
    assert report["model"] == "fake-functional"
    assert report["blocking"] == []
    assert report["verdict"].startswith("PREFLIGHT CLEAN")
    # --tokenizer-only cannot set the answer form, and the verdict has to SAY so rather than
    # leaving an operator to default it.
    assert report["answer_form"] is None
    assert "Re-run WITHOUT --tokenizer-only" in report["verdict"]
    checks = report["checks"]
    assert checks["answer_pool"]["passed"] and tuple(checks["answer_pool"]["valid"]) == POOL
    assert checks["extension"]["ok"] and checks["extension"]["pinned_is_byte_prefix"]
    assert checks["chat_tail"]
    # Every pool symbol single-token in BOTH forms, which is what the run's own gate requires.
    assert all(
        count == 1
        for per_form in checks["answer_form_token_counts"].values()
        for count in per_form.values()
    )


def test_the_preflight_blocks_when_the_model_does_not_answer_with_an_option():
    """rails.md #1: a contract frozen against outputs nobody read needs a first-contact checkpoint
    that routes to harness_inadequate BEFORE full spend. The reality sample computed
    ``on_option_rate`` and then let a 0/10 run through to a 2,880-forward B0 whose null would have
    been written up as a fact about the model."""

    import scripts.e4_surface_preflight as preflight

    assert preflight.MIN_ON_OPTION_RATE > 0.0
