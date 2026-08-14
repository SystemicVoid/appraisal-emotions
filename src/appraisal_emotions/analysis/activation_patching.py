"""E3 — in-distribution activation patching on the reward-matched cells (design §4 E3).

NEW module (no parent counterpart). Steering is dropped: the parent's steering leg recorded
``harness_inadequate`` (logit movement 0.0013 against a 0.02 floor), and a steering grid answers a
question about an out-of-distribution perturbation nobody asked. Patching asks the causal question
with **real prompts on both sides**, so magnitude and direction are in-distribution by
construction.

**Pairs.** Donor and recipient are two reveals from the same ``reward_cell_id``: identical
realised reward and |RPE|, opposite signed RPE (the cell is built that way — ``reward = ev +
signed_rpe`` with reward fixed). They are additionally matched on **template family and realised
outcome symbol** — possible because the battery renders both symbol orders, so a donor rendering
whose ``symbol_low`` equals the recipient's ``symbol_high`` realises the same surface token — and
the patch therefore cannot smuggle in a different symbol identity. The fallback to
symbol-unmatched pairs fires only when the battery yields ZERO symbol-matched pairs, and then the
report NAMES the confound; the certified battery yields 248 symbol-matched pairs, so on that
surface the fallback is unreachable. The battery and its row alignment are the ones E2 already
reads (``analysis.expectation_control.align_reveals``); the readout axes are the ones E2 already
projects onto (``emotion_axes``). Nothing here re-derives either.

**Arms** (one sweep, four arms, no grid): :data:`ARMS`, one line of rationale each in
``_ARM_NOTES`` — a full-residual ceiling, the certified ``v_rpe`` component, a norm-matched random
component as the floor, and a same-condition donor as the no-op negative control. A literal
self-patch (donor is recipient) moves nothing under :func:`patch_state`: a wiring check, not a
control.

**Two modes, and the claim ladder between them** (:data:`MODE_NOTES`, printed with every report).

- ``state`` — the ZERO-FORWARD PREVIEW. Substitute at the patch block and read the emotion-axis
  projection *of the substituted vector itself*. No model runs, so it measures what the
  substitution PUTS IN, not what the model DOES with it, and ``full_residual`` transfers 1.0 by
  construction (a wiring check). Caps at present-and-separable, and exists because it costs
  nothing and catches pair-selection and axis bugs before any GPU time is spent.
- ``forward`` — the CAUSAL TIER. Re-run the recipient's real prompt with the donor's value
  substituted at ``(block, reveal token)`` through ``backend.patched_forward``, and read the
  emotion axes at downstream blocks, where the substitution has actually propagated. The
  unpatched baseline is a SELF-PATCH (a position replaced by its own captured value), so the
  wiring check and the reference are the same measurement. This is the only tier that can earn
  **functionally-used** (design §7), and only when run on the real model.

The ≤40-token continuation is generated in ``forward`` mode and stored RAW. No grader ships:
``docs/agents/rails.md`` forbids freezing one against outputs nobody has read, so scoring the
continuations is a run-day step after the reality sample — and design §4 E3 drops the behavioral
readout entirely if the continuations turn out non-affective.

*Identification limit, stated up front:* within a reward-matched cell EV and signed RPE are
perfectly anti-correlated (reward is fixed), so E3 identifies "the expectation manipulation
transfers", not "signed RPE rather than EV transfers". That separation is R-A′'s two matched
contests and E2.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal, cast

import numpy as np

from appraisal_emotions.activation.capture import RenderedCaptureBackend
from appraisal_emotions.analysis.direction_stats import seed_int
from appraisal_emotions.analysis.emotion_vectors import read_emotion_vectors
from appraisal_emotions.analysis.expectation_control import align_reveals, emotion_axes
from appraisal_emotions.analysis.reveal_rpe import (
    read_reveal_rpe_directions,
    read_reveal_rpe_states,
    reveal_read_prompt,
)
from appraisal_emotions.backends.base import PatchedForwardBackend
from appraisal_emotions.core.schema import Comparison, StrictModel
from appraisal_emotions.core.util import EXTRACTION_SEED, file_sha256, read_json, unit_vector
from appraisal_emotions.stimuli.reveal_probes import RevealProbeBattery

__all__ = [
    "ACTIVATION_PATCHING_CONTRACT_VERSION",
    "ARMS",
    "DEFAULT_CONTINUATION_TOKENS",
    "DEFAULT_FORWARD_MAX_PAIRS",
    "DEFAULT_MAX_CONTINUATIONS",
    "DEFAULT_MAX_PAIRS",
    "DEFAULT_RANDOM_DRAWS",
    "MODES",
    "MODE_NOTES",
    "ActivationPatchingReport",
    "ArmResult",
    "PatchPair",
    "PatchedContinuation",
    "activation_patching",
    "build_patch_pairs",
    "format_activation_patching_summary",
    "patch_state",
    "substituted_value",
]

ACTIVATION_PATCHING_CONTRACT_VERSION = "activation_patching/v2"
MODES: tuple[str, ...] = ("state", "forward")
DEFAULT_MAX_PAIRS = 480
# Forward mode costs (5 + n_random_draws) patched forwards per pair — two self-patch baselines
# (recipient, donor) plus full_residual, v_rpe_component, same_condition_donor and one per random
# draw — so its pair budget is smaller and its random floor coarser. Both are CLI-overridable and
# both are reported. Continuations add generate passes on top, for the first
# DEFAULT_MAX_CONTINUATIONS pairs only, on the baseline and _CONTINUATION_ARMS.
DEFAULT_FORWARD_MAX_PAIRS = 60
DEFAULT_RANDOM_DRAWS: dict[str, int] = {"state": 200, "forward": 5}
DEFAULT_CONTINUATION_TOKENS = 40
DEFAULT_MAX_CONTINUATIONS = 10
FLOOR_QUANTILE = 0.95
ARMS: tuple[str, ...] = (
    "full_residual",
    "v_rpe_component",
    "random_component",
    "same_condition_donor",
)
# Arms whose continuation is worth storing: the ceiling and the certified-direction arm, against
# the unpatched baseline. A random-direction continuation would be text about nothing.
_CONTINUATION_ARMS = ("full_residual", "v_rpe_component")

MODE_NOTES: dict[str, str] = {
    "state": (
        "STATE-LEVEL PREVIEW (zero forwards). The donor's reveal-token residual is substituted at "
        "the patch block and the emotion-axis projection OF THAT SUBSTITUTED VECTOR is read. "
        "Nothing propagates and nothing is generated, so this measures what the substitution puts "
        "in, not what the model does with it — full_residual transfers 1.0 BY CONSTRUCTION and is "
        "a wiring check, not a result. Caps at present-and-separable. Re-run with mode=forward on "
        "the real model for the causal tier."
    ),
    "forward": (
        "FORWARD-PATCHED (the causal tier). Each recipient's real prompt is re-run with the "
        "donor's value substituted at the reveal token of the patch block, and the emotion axes "
        "are read at downstream blocks, where the substitution has propagated. The unpatched "
        "baseline is a self-patch, so the design's wiring check and the reference are one "
        "measurement. Continuations are stored RAW and UNSCORED: no grader may be frozen before "
        "a reality sample of this surface (docs/agents/rails.md)."
    ),
}
_SYMBOL_CONFOUND = (
    "SYMBOL CONFOUND: the battery yielded NO donor/recipient pair sharing a realised outcome "
    "symbol, so the pairs are template-matched only and the patch also changes the surface token "
    "identity. Any transfer read here is not separable from that."
)
_IDENTIFICATION_LIMIT = (
    "Within a reward-matched cell EV and signed RPE are perfectly anti-correlated (reward is "
    "fixed), so E3 identifies 'the expectation manipulation transfers', NOT 'signed RPE rather "
    "than EV transfers'. That separation is R-A′'s two matched contests and E2. Patching across "
    "EV-matched pairs (same draw, opposite outcome) is the documented complement, not built."
)
_ARM_NOTES: dict[str, str] = {
    "full_residual": "Ceiling: the donor's whole reveal-token state. 1.0 by construction here.",
    "v_rpe_component": "Is the transfer carried by the certified signed-RPE direction?",
    "random_component": (
        "Floor: unit random directions, same substitution, reported at the 95th percentile over "
        "n_random_draws draws — a floor, not a single lucky or unlucky direction."
    ),
    "same_condition_donor": (
        "No-op negative control: a different rendering of the recipient's OWN condition. Its "
        "shift should sit near zero — the transfer fraction is reported against the SAME "
        "donor-minus-recipient denominator so the arms are on one scale."
    ),
}


@dataclass(frozen=True)
class PatchPair:
    """One donor -> recipient substitution inside a reward-matched cell."""

    reward_cell_id: str
    donor_row: int
    recipient_row: int
    same_condition_row: int | None
    template_family: str
    symbol_matched: bool


def _condition(reveal: Comparison) -> tuple[str, int]:
    """A reveal's condition: its draw and which side of it was realised."""

    return str(reveal.metadata["ev_cell_id"]), int(reveal.metadata["rpe_sign"])


def build_patch_pairs(
    reveals: tuple[Comparison, ...], *, max_pairs: int = DEFAULT_MAX_PAIRS
) -> tuple[tuple[PatchPair, ...], bool]:
    """Opposite-signed-RPE pairs within each reward-matched cell, symbol-matched where possible.

    Returns ``(pairs, symbol_matched)``. Symbol- and template-matched pairs are preferred; the
    fallback to symbol-unmatched pairs fires only when the battery yields none, and the caller
    records the confound. Deterministic: rows are visited in the states artifact's own order.
    """

    cells: dict[str, list[int]] = {}
    for row, reveal in enumerate(reveals):
        cells.setdefault(str(reveal.metadata["reward_cell_id"]), []).append(row)

    for symbol_matched in (True, False):
        pairs: list[PatchPair] = []
        for cell_id, rows in cells.items():
            # Group the cell's rows by the surface that must be held fixed across the patch, then
            # cross the +1-sign donors with the -1-sign recipients inside each group.
            groups: dict[tuple[str, str], dict[int, list[int]]] = {}
            for row in rows:
                meta = reveals[row].metadata
                surface = (
                    str(meta["template_family"]),
                    str(meta["realised_symbol"]) if symbol_matched else "",
                )
                groups.setdefault(surface, {}).setdefault(int(meta["rpe_sign"]), []).append(row)
            for (template_family, _symbol), by_sign in groups.items():
                for donor, recipient in product(by_sign.get(1, []), by_sign.get(-1, [])):
                    pairs.append(
                        PatchPair(
                            reward_cell_id=cell_id,
                            donor_row=donor,
                            recipient_row=recipient,
                            same_condition_row=_same_condition(reveals, rows, recipient),
                            template_family=template_family,
                            symbol_matched=symbol_matched,
                        )
                    )
        if pairs:
            return tuple(pairs[:max_pairs]), symbol_matched
    raise ValueError(
        "no reward-matched cell yields a template-matched pair of reveals with opposite signed "
        "RPE; E3 has no in-distribution donor/recipient pair to patch on this battery."
    )


def _same_condition(reveals: tuple[Comparison, ...], rows: list[int], recipient: int) -> int | None:
    """A different rendering of the recipient's own condition, for the no-op arm."""

    target = _condition(reveals[recipient])
    for row in rows:
        if row != recipient and _condition(reveals[row]) == target:
            return row
    return None


def patch_state(
    recipient: np.ndarray, donor: np.ndarray, direction: np.ndarray | None = None
) -> np.ndarray:
    """The recipient's reveal-token state after the donor's value is substituted into it.

    ``direction is None`` substitutes the whole residual; otherwise only the recipient's component
    along that unit direction is replaced by the donor's. A self-patch (donor is recipient) moves
    nothing under either form — the wiring check design §4 E3 names.
    """

    if direction is None:
        return donor
    return recipient + float((donor - recipient) @ direction) * direction


class ArmResult(StrictModel):
    """One (mode, arm, emotion axis, readout block) readout over the matched pairs.

    ``transfer_fraction`` is a SIGN-CORRECTED RATIO OF SUMS —
    ``sum(shift * sign(donor - recipient)) / sum(|donor - recipient|)`` — not a mean of per-pair
    ratios: one well-conditioned denominator per arm, instead of a per-pair division that blows up
    whenever a pair happens to differ barely on that axis. It is ``None`` when that denominator
    sums to zero, which is a "no scale to measure against", NOT a measured zero transfer.

    *It is a normalized shift, not a literal fraction, and 1.0 is not the target.* The denominator
    is the gap between the donor's and the recipient's own downstream states, and those differ for
    reasons the numerator cannot possibly carry — most of all the options block, whose text differs
    between them by construction (EV differs; that is what makes the cell a manipulation). A patch
    that transferred the expectation state perfectly would still not reproduce the donor's reading
    of its own different prompt, so perfect causal transfer does not imply 1.0 downstream. Read it
    as "how far toward the donor did the recipient move, in units of the donor-recipient gap",
    always against the same-condition no-op control and the random-direction floor.

    ``wiring_check`` marks a row read AT the patch site rather than downstream of it. Such a row is
    tautological by construction — the patch site returns the substituted value — so it verifies
    plumbing and licenses nothing. For ``random_component`` the reported numbers are the
    :data:`FLOOR_QUANTILE` quantile over ``n_random_draws`` directions, i.e. a floor rather than a
    point estimate.
    """

    mode: str
    arm: str
    axis: str
    readout_block: int
    wiring_check: bool
    n_pairs: int
    n_random_draws: int
    mean_shift: float
    mean_donor_minus_recipient: float
    transfer_fraction: float | None


class PatchedContinuation(StrictModel):
    """One raw greedy continuation decoded under a patch — stored UNPARSED, on purpose.

    ``docs/agents/rails.md`` forbids freezing a grader against outputs nobody has read, so E3
    ships no lexicon scorer: the run-day step is to read these, take the reality sample, and only
    then decide whether a signed-lexicon readout is even applicable (design §4 E3 says it is
    dropped if the continuations are non-affective).
    """

    reward_cell_id: str
    arm: str
    donor_row: int
    recipient_row: int
    text: str


class ActivationPatchingReport(StrictModel):
    """E3 record: does the reveal-token state carrying the expectation transfer when patched?"""

    artifact_contract_version: Literal["activation_patching/v2"] = (
        ACTIVATION_PATCHING_CONTRACT_VERSION
    )
    mode: str
    seed: int
    block: int
    readout_blocks: tuple[int, ...]
    states_sha256: str
    battery_sha256: str
    directions_sha256: str
    emotion_vectors_sha256: str
    emotion_selected_block: int
    model_key: str | None
    sensitivity_gate: str
    verdict_cap: str
    n_reward_cells: int
    n_pairs: int
    n_no_op_pairs: int
    n_random_draws: int
    symbol_matched: bool
    symbol_confound: str | None
    arms: tuple[ArmResult, ...]
    continuations: tuple[PatchedContinuation, ...]
    arm_notes: dict[str, str]
    mode_note: str
    identification_limit: str


# --------------------------------------------------------------------------------------
# Shared aggregation: both modes reduce to (per-pair shift, per-pair donor-minus-recipient)
# --------------------------------------------------------------------------------------


def _aggregate(
    mode: str,
    arm: str,
    axis: str,
    readout_block: int,
    patch_block: int,
    shifts: np.ndarray,
    denominators: np.ndarray,
    *,
    n_random_draws: int = 0,
) -> ArmResult:
    total = float(np.abs(denominators).sum())
    return ArmResult(
        mode=mode,
        arm=arm,
        axis=axis,
        readout_block=readout_block,
        wiring_check=readout_block == patch_block,
        n_pairs=int(shifts.size),
        n_random_draws=n_random_draws,
        mean_shift=float(shifts.mean()) if shifts.size else 0.0,
        mean_donor_minus_recipient=float(denominators.mean()) if denominators.size else 0.0,
        # None, not 0.0: with no donor-recipient gap on this axis there is no scale to normalize
        # against, which is a different fact from "the patch moved nothing".
        transfer_fraction=(
            float((shifts * np.sign(denominators)).sum() / total) if total > 0.0 else None
        ),
    )


def _random_floor(results: list[ArmResult]) -> ArmResult:
    """Collapse per-draw random-component results into one FLOOR_QUANTILE floor (FIX: was 1 draw).

    A single random direction is a sample, not a floor: whether it happens to align with the
    emotion axis is luck. Each quantity is taken at the same upper quantile the E1 floors use, so
    "the random arm reached X" means "95% of random directions reached less than X".
    """

    template = results[0]
    fractions = [
        entry.transfer_fraction for entry in results if entry.transfer_fraction is not None
    ]
    return template.model_copy(
        update={
            "n_random_draws": len(results),
            "mean_shift": float(
                np.quantile([entry.mean_shift for entry in results], FLOOR_QUANTILE)
            ),
            "transfer_fraction": (
                float(np.quantile(fractions, FLOOR_QUANTILE)) if fractions else None
            ),
        }
    )


def substituted_value(
    arm: str,
    pair: PatchPair,
    block_states: np.ndarray,
    v_rpe: np.ndarray,
    random_direction: np.ndarray,
) -> np.ndarray:
    """The value that replaces the recipient's reveal-token residual, for one arm on one pair."""

    recipient = block_states[pair.recipient_row]
    if arm == "same_condition_donor":
        return patch_state(recipient, block_states[cast(int, pair.same_condition_row)])
    direction = {"full_residual": None, "v_rpe_component": v_rpe}.get(arm, random_direction)
    return patch_state(recipient, block_states[pair.donor_row], direction)


def _scored_pairs(arm: str, pairs: tuple[PatchPair, ...]) -> list[PatchPair]:
    if arm != "same_condition_donor":
        return list(pairs)
    return [pair for pair in pairs if pair.same_condition_row is not None]


# --------------------------------------------------------------------------------------
# Mode 1 — state level: zero forwards, arithmetic at the patch block only
# --------------------------------------------------------------------------------------


def _state_mode_arms(
    pairs: tuple[PatchPair, ...],
    block_states: np.ndarray,
    axes: dict[str, tuple[np.ndarray, str]],
    v_rpe: np.ndarray,
    patch_block: int,
    *,
    rng: np.random.Generator,
    n_random_draws: int,
) -> list[ArmResult]:
    directions = [unit_vector(rng.standard_normal(v_rpe.size)) for _ in range(n_random_draws)]
    results: list[ArmResult] = []
    for name, (axis, _description) in axes.items():
        for arm in ARMS:
            scored = _scored_pairs(arm, pairs)
            denominators = np.asarray(
                [
                    float((block_states[pair.donor_row] - block_states[pair.recipient_row]) @ axis)
                    for pair in scored
                ]
            )
            draws = directions if arm == "random_component" else [v_rpe]
            per_draw = [
                _aggregate(
                    "state",
                    arm,
                    name,
                    patch_block,
                    patch_block,
                    np.asarray(
                        [
                            float(
                                (
                                    substituted_value(arm, pair, block_states, v_rpe, direction)
                                    - block_states[pair.recipient_row]
                                )
                                @ axis
                            )
                            for pair in scored
                        ]
                    ),
                    denominators,
                )
                for direction in draws
            ]
            if not per_draw:
                continue  # --random-draws 0 disables the floor arm rather than crashing on it.
            results.append(_random_floor(per_draw) if arm == "random_component" else per_draw[0])
    return results


# --------------------------------------------------------------------------------------
# Mode 2 — forward: the causal tier. The substituted value propagates through the model.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _ForwardPlan:
    """Everything one forward-mode sweep needs that is not a per-pair argument."""

    backend: PatchedForwardBackend
    reveals: tuple[Comparison, ...]
    block_states: np.ndarray
    patch_block: int
    readout_blocks: tuple[int, ...]
    axes_by_block: dict[int, dict[str, np.ndarray]]
    continuation_tokens: int


def _forward_projections(
    plan: _ForwardPlan, row: int, replacement: np.ndarray, *, continuation: bool
) -> tuple[dict[tuple[int, str], float], str | None]:
    """One patched forward; the emotion-axis projections at every readout block, plus its text."""

    result = plan.backend.patched_forward(
        reveal_read_prompt(cast(RenderedCaptureBackend, plan.backend), plan.reveals[row]),
        block=plan.patch_block,
        position="last",
        replacement=replacement,
        layers=tuple(block + 1 for block in plan.readout_blocks),
        max_new_tokens=plan.continuation_tokens if continuation else 0,
    )
    projections = {
        (block, name): float(result.states[position] @ axis)
        for position, block in enumerate(plan.readout_blocks)
        for name, axis in plan.axes_by_block[block].items()
    }
    return projections, result.continuation


def _forward_mode_arms(
    plan: _ForwardPlan,
    pairs: tuple[PatchPair, ...],
    v_rpe: np.ndarray,
    *,
    rng: np.random.Generator,
    n_random_draws: int,
    max_continuations: int,
) -> tuple[list[ArmResult], list[PatchedContinuation]]:
    directions = [unit_vector(rng.standard_normal(v_rpe.size)) for _ in range(n_random_draws)]
    # Per pair: the recipient's and the donor's own baselines are SELF-PATCHES — replacing a
    # position with its own captured value. That is design §4 E3's wiring check doing double duty
    # as the unpatched reference, so the baseline is measured through the same code path as every
    # arm rather than assumed.
    # Keyed by PAIR INDEX, not appended positionally: the same_condition_donor arm skips pairs
    # that have no third rendering, so a positional zip against the denominators would pair each
    # shift with the wrong pair's baseline gap.
    shifts: dict[tuple[str, int, int, str], dict[int, float]] = {}
    denominators: dict[tuple[int, str], dict[int, float]] = {}
    continuations: list[PatchedContinuation] = []
    for pair_index, pair in enumerate(pairs):
        wants_text = pair_index < max_continuations
        recipient_state = plan.block_states[pair.recipient_row]
        baseline, baseline_text = _forward_projections(
            plan, pair.recipient_row, recipient_state, continuation=wants_text
        )
        donor_baseline, _text = _forward_projections(
            plan, pair.donor_row, plan.block_states[pair.donor_row], continuation=False
        )
        if baseline_text is not None:
            continuations.append(_continuation(pair, "unpatched_baseline", baseline_text))
        for key, value in donor_baseline.items():
            denominators.setdefault(key, {})[pair_index] = value - baseline[key]
        for arm in ARMS:
            if arm == "same_condition_donor" and pair.same_condition_row is None:
                continue
            draws = directions if arm == "random_component" else [v_rpe]
            for draw_index, direction in enumerate(draws):
                patched, text = _forward_projections(
                    plan,
                    pair.recipient_row,
                    substituted_value(arm, pair, plan.block_states, v_rpe, direction),
                    continuation=wants_text and arm in _CONTINUATION_ARMS and draw_index == 0,
                )
                if text is not None:
                    continuations.append(_continuation(pair, arm, text))
                for key, value in patched.items():
                    shifts.setdefault((arm, draw_index, *key), {})[pair_index] = (
                        value - baseline[key]
                    )

    results: list[ArmResult] = []
    for block in plan.readout_blocks:
        for name in plan.axes_by_block[block]:
            gaps = denominators[(block, name)]
            for arm in ARMS:
                per_draw = [
                    _aggregate(
                        "forward",
                        arm,
                        name,
                        block,
                        plan.patch_block,
                        np.asarray([by_pair[pair_index] for pair_index in sorted(by_pair)]),
                        np.asarray([gaps[pair_index] for pair_index in sorted(by_pair)]),
                    )
                    for key, by_pair in sorted(shifts.items())
                    if key[0] == arm and key[2] == block and key[3] == name
                ]
                if not per_draw:
                    continue
                results.append(
                    _random_floor(per_draw) if arm == "random_component" else per_draw[0]
                )
    return results, continuations


def _continuation(pair: PatchPair, arm: str, text: str) -> PatchedContinuation:
    return PatchedContinuation(
        reward_cell_id=pair.reward_cell_id,
        arm=arm,
        donor_row=pair.donor_row,
        recipient_row=pair.recipient_row,
        text=text,
    )


# --------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------


def activation_patching(
    states_artifact: Path,
    battery_file: Path,
    directions_artifact: Path,
    emotion_artifact: Path,
    *,
    mode: str = "state",
    backend: PatchedForwardBackend | None = None,
    model_key: str | None = None,
    block: int | None = None,
    seed: int = EXTRACTION_SEED,
    max_pairs: int | None = None,
    n_random_draws: int | None = None,
    continuation_tokens: int = DEFAULT_CONTINUATION_TOKENS,
    max_continuations: int = DEFAULT_MAX_CONTINUATIONS,
) -> ActivationPatchingReport:
    """Run E3: patch donor reveal-token states into recipients and read the emotion axes.

    ``mode="state"`` is the zero-forward preview (arithmetic at the patch block, no model needed);
    ``mode="forward"`` is the causal tier and requires ``backend``.
    """

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "forward" and backend is None:
        raise ValueError("mode='forward' needs a patching backend; pass --config to the CLI")
    states = read_reveal_rpe_states(Path(states_artifact))
    battery = RevealProbeBattery.model_validate(read_json(Path(battery_file)))
    directions = read_reveal_rpe_directions(Path(directions_artifact))
    emotion = read_emotion_vectors(Path(emotion_artifact))
    if (
        not states.metadata.hidden_size
        == emotion.metadata.hidden_size
        == directions.metadata.hidden_size
    ):
        raise ValueError("states, directions and emotion vectors have different hidden sizes")
    patch_block = emotion.metadata.selected_block if block is None else block
    if not 0 <= patch_block < states.metadata.n_blocks:
        raise ValueError(f"block {patch_block} out of range for {states.metadata.n_blocks}")

    if max_pairs is None:
        max_pairs = DEFAULT_MAX_PAIRS if mode == "state" else DEFAULT_FORWARD_MAX_PAIRS
    if n_random_draws is None:
        n_random_draws = DEFAULT_RANDOM_DRAWS[mode]
    reveals = align_reveals(states, battery)
    pairs, symbol_matched = build_patch_pairs(reveals, max_pairs=max_pairs)
    block_states = np.asarray(states.states[:, patch_block, :])
    v_rpe = unit_vector(directions.directions[0, patch_block, :])
    rng = np.random.default_rng(seed_int(seed, "activation-patching", mode, patch_block))

    # The patch block itself plus the deepest block: enough to say whether the substitution
    # survives depth, without a per-block table nobody reads.
    readout_blocks = tuple(sorted({patch_block, states.metadata.n_blocks - 1}))
    if mode == "forward" and len(readout_blocks) == 1:
        # Patching the deepest block leaves only the patch site to read, and the patch site
        # returns the substituted value by construction. A report of nothing but that tautology
        # must not be emitted under the functionally-used cap.
        raise ValueError(
            f"block {patch_block} is the deepest block ({states.metadata.n_blocks - 1}), so no "
            "downstream readout block exists and forward mode would report only the "
            "patch-site wiring check. Choose a shallower --block."
        )
    continuations: list[PatchedContinuation] = []
    if mode == "state":
        arms = _state_mode_arms(
            pairs,
            block_states,
            emotion_axes(emotion, patch_block),
            v_rpe,
            patch_block,
            rng=rng,
            n_random_draws=n_random_draws,
        )
        readout_blocks = (patch_block,)
    else:
        plan = _ForwardPlan(
            backend=cast(PatchedForwardBackend, backend),
            reveals=reveals,
            block_states=block_states,
            patch_block=patch_block,
            readout_blocks=readout_blocks,
            axes_by_block={
                readout: {
                    name: axis
                    for name, (axis, _description) in emotion_axes(emotion, readout).items()
                }
                for readout in readout_blocks
            },
            continuation_tokens=continuation_tokens,
        )
        arms, continuations = _forward_mode_arms(
            plan,
            pairs,
            v_rpe,
            rng=rng,
            n_random_draws=n_random_draws,
            max_continuations=max_continuations,
        )

    gate = emotion.metadata.gate_verdict
    return ActivationPatchingReport(
        mode=mode,
        seed=seed,
        block=patch_block,
        readout_blocks=readout_blocks,
        states_sha256=states.metadata.states_sha256,
        battery_sha256=file_sha256(Path(battery_file)),
        directions_sha256=directions.metadata.directions_sha256,
        emotion_vectors_sha256=emotion.metadata.vectors_sha256,
        emotion_selected_block=emotion.metadata.selected_block,
        model_key=model_key,
        sensitivity_gate=f"G0={gate}",
        verdict_cap=_verdict_cap(gate, mode),
        n_reward_cells=len({pair.reward_cell_id for pair in pairs}),
        n_pairs=len(pairs),
        n_no_op_pairs=sum(1 for pair in pairs if pair.same_condition_row is not None),
        n_random_draws=n_random_draws,
        symbol_matched=symbol_matched,
        symbol_confound=None if symbol_matched else _SYMBOL_CONFOUND,
        arms=tuple(arms),
        continuations=tuple(continuations),
        arm_notes=_ARM_NOTES,
        mode_note=MODE_NOTES[mode],
        identification_limit=_IDENTIFICATION_LIMIT,
    )


def _verdict_cap(gate: str, mode: str) -> str:
    if gate != "pass":
        return (
            "harness_inadequate — the E0 G0 sensitivity gate did NOT pass, so the emotion axes "
            "read here are not established to carry valence structure. Neither a transfer nor a "
            "null adjudicates anything; the claim stays OPEN."
        )
    if mode == "state":
        return (
            "present-and-separable, pilot-suggestive. The STATE-LEVEL preview says how much of "
            "the donor-recipient difference along an emotion axis the substitution carries; it "
            "runs no forward, so it cannot show the model using it. Re-run with mode=forward on "
            "the real model for the functionally-used tier."
        )
    return (
        "functionally-used, pilot-suggestive — the strongest tier this project licenses, and only "
        "on THIS model/surface/recipe. READ ONLY THE ROWS WITH wiring_check=false: those are the "
        "downstream blocks, where the substituted reveal-token value has actually propagated, and "
        "they are what the claim rests on, against a same-condition no-op control and a "
        "random-direction floor. The wiring_check=true rows sit AT the patch site, which returns "
        "the substituted value by construction; they verify plumbing and license nothing. Still "
        "no welfare / sentience / experience claim: functional use of a representation is not "
        "experience (design §7)."
    )


def format_activation_patching_summary(report: ActivationPatchingReport) -> str:
    """A plain-text summary table for stdout."""

    lines = [
        f"E3 activation-patching — in-distribution donor/recipient transfer (mode={report.mode})",
        f"  patch block={report.block} readout blocks={list(report.readout_blocks)} "
        f"seed={report.seed} pairs={report.n_pairs} cells={report.n_reward_cells} "
        f"no-op donors={report.n_no_op_pairs} random draws={report.n_random_draws} "
        f"symbol-matched={'Y' if report.symbol_matched else 'n'}",
        f"  sensitivity: {report.sensitivity_gate}",
        f"  verdict cap: {report.verdict_cap}",
        "",
        "  blk  axis                              arm                     n   mean shift   "
        "transfer  role",
    ]
    for arm in report.arms:
        transfer = (
            "      n/a" if arm.transfer_fraction is None else f"{arm.transfer_fraction:+9.4f}"
        )
        role = (
            "WIRING CHECK (patch site: reads back the injected vector, no model in between)"
            if arm.wiring_check
            else "downstream"
        )
        lines.append(
            f"  {arm.readout_block:>3}  {arm.axis:<32} {arm.arm:<20} {arm.n_pairs:>4}  "
            f"{arm.mean_shift:+11.5f}  {transfer}  {role}"
        )
    if report.symbol_confound:
        lines.extend(["", f"  {report.symbol_confound}"])
    if report.continuations:
        lines.extend(
            [
                "",
                f"  {len(report.continuations)} raw continuations stored UNPARSED for reading — "
                "no grader exists and none may be written before the reality sample "
                "(docs/agents/rails.md).",
            ]
        )
    lines.extend(["", f"  mode: {report.mode_note}", f"  limit: {report.identification_limit}"])
    return "\n".join(lines)
