"""E4 — does the patched expectation state change what the model DOES? (docs/design/e4-prereg.md)

E3's behavioural leg returned a null that was not a null. Its <=40-token window sat immediately
after the reveal symbol, where the only in-distribution continuation is the model finishing a
ledger line, and all 30 stored samples read ` = 30 points` or ` = 0 points`. The transfer
fraction's denominator — the natural difference in behaviour between the two real conditions —
was never measured, and a ratio with an unmeasured, plausibly-zero denominator cannot return a
readable null in either direction.

Two design commitments follow, and this module is organised around them.

**The denominator is measured first, with no patching.** :func:`sensitivity_gate` runs the extended
surface unpatched over the same pairs the patched leg will use, at each titration level, and reads
the natural gap. It also picks the titration level — the certain amount whose baseline logit
margin sits nearest indifference, where a margin is most sensitive — on unpatched data only, so
the choice cannot be outcome-dependent. If the gate fails, no patched forward is spent and the
window records ``harness_inadequate``.

**The readout lives at a LATER token than the patch.** This is not ergonomics. A linear readout at
the patched position is linear in the injected delta, so the additive residual stream reproduces it
with no computation at all — which is what the passthrough decomposition found for E3, where the
identity path accounts for 80-105% of the published transfer. There is no identity path BETWEEN
positions: a patch at the reveal token can reach an answer slot further along only by being read
through attention. So this is simultaneously the functional-use measurement and the control that
E3's random-direction floor could never be.

The readout itself is a next-token logit margin at a single answer slot with zero decode steps: no
grader, no parser, no LLM judge, and no zero-variance failure mode, since a margin moves
continuously even where the argmax never flips.

Licence: ``functionally-used, pilot-suggestive`` at most, and only when the gate passed and the
corruption controls stayed clean. Emotion words remain concept labels; nothing here is an affect
claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Literal

import numpy as np

from appraisal_emotions.activation.capture import block_layers
from appraisal_emotions.analysis.activation_patching import (
    ARMS,
    FLOOR_QUANTILE,
    PatchPair,
    build_patch_pairs,
    matched_random_value,
    substituted_value,
)
from appraisal_emotions.analysis.behavioral_surface import (
    ChatPatchingBackend,
    ExtendedPrompt,
    Readout,
    chat_tail,
    choice_probe_for,
    corruption_probe_for,
    extend,
    read,
    renderings_for,
    rotating_rendering,
)
from appraisal_emotions.analysis.direction_stats import seed_int
from appraisal_emotions.analysis.emotion_vectors import read_emotion_vectors
from appraisal_emotions.analysis.expectation_control import align_reveals, emotion_axes
from appraisal_emotions.analysis.reveal_rpe import (
    read_reveal_rpe_directions,
    read_reveal_rpe_states,
)
from appraisal_emotions.analysis.symbol_preflight import (
    AnswerPoolPreflight,
    battery_symbols,
    preflight_answer_symbols,
)
from appraisal_emotions.core.schema import Comparison, StrictModel
from appraisal_emotions.core.stats import add_one_p
from appraisal_emotions.core.util import file_sha256, read_json, unit_vector
from appraisal_emotions.stimuli.decision_probes import (
    ANSWER_SYMBOL_CANDIDATES,
    CERTAIN_LEVELS,
    CORRUPTION_WINDOWS,
    DEFAULT_ANSWER_FORM,
    AnswerForm,
    DecisionProbe,
    OptionRendering,
    WindowName,
    answer_symbols,
    option_renderings,
    reachability_probe,
)
from appraisal_emotions.stimuli.reveal_probes import RevealProbeBattery

BEHAVIORAL_TRANSFER_CONTRACT_VERSION = "behavioral_transfer/v1"

# The gate is deliberately stricter than the usual 0.05: a behavioural leg that spends patched
# forwards on a denominator it is not sure about is the failure this whole rung exists to avoid.
GATE_ALPHA = 0.01
# A 50% transfer must be detectable, or the patched leg's null would be unreadable for the same
# reason the continuation window's was.
GATE_MDE_FRACTION = 0.5
GATE_POWER = 0.80
# The arm's own cell-clustered null must clear this before "functionally-used" may be written. The
# usual 0.05 rather than the gate's 0.01: the gate protects a SPEND decision (get it wrong and
# thousands of forwards buy an unreadable table), while this protects a CLAIM that is already capped
# at pilot-suggestive and already has to outrun two controls.
ARM_ALPHA = 0.05
# Mean |shift| on an invariant-answer window must stay under this fraction of the CHOICE window's
# own natural gap. Judged in the units of the thing being claimed, not as a fraction of the
# corruption window's own baseline: both of those answers appear verbatim in the assistant turn
# immediately above, so that baseline is saturated and 10% of it can be larger than any plausible
# choice effect — a tolerance that passes while permitting damage bigger than the result.
CORRUPTION_TOLERANCE = 0.5
CORRUPTION_ARMS: tuple[str, ...] = ("full_residual", "v_rpe_component")
# Raised from E3's 5: at 5 draws a 95th percentile is the maximum of the sample, and a floor that
# IS its own maximum reports the tail rather than bounding it.
DEFAULT_RANDOM_DRAWS = 20
DEFAULT_PERMUTATIONS = 10_000
# ~120 pairs spread over every cell that yields one, instead of E3's 60 drawn from 14 cells in
# artifact order (e4-prereg §6).
DEFAULT_MAX_PAIRS = 120
DEFAULT_MAX_PER_CELL = 4
# The positive control's own pair set: recipients patched from a donor in a DIFFERENT reward cell,
# where a full-residual patch has a point prediction at the answer slot. ~2% of the forward budget,
# and without it a flat arm table cannot be told apart from an unreachable readout.
DEFAULT_REACHABILITY_PAIRS = 30
REACHABILITY_ALPHA = 0.05

SYMBOL_CONFOUND = (
    "SYMBOL CONFOUND: no donor/recipient pair shares a realised outcome symbol, so the patch also "
    "changes the reveal's surface token — and here it changes the assistant turn the probe is "
    "appended to. No transfer read on this run is separable from that."
)

ANSWER_POOL_NOTE = (
    "Round-two option symbols, single-token and affect-neutral on this tokenizer and disjoint "
    "from every symbol the reveal battery rendered, so the answer slot carries no repetition "
    "prior from the surface the patch sits on."
)


# --------------------------------------------------------------------------------------
# Aggregation and inference
# --------------------------------------------------------------------------------------


def _cell_groups(pairs: Sequence[PatchPair]) -> tuple[tuple[int, ...], ...]:
    grouped: dict[str, list[int]] = {}
    for index, pair in enumerate(pairs):
        grouped.setdefault(pair.reward_cell_id, []).append(index)
    return tuple(tuple(rows) for rows in grouped.values())


def _cell_permutation_p(
    values: np.ndarray,
    cells: tuple[tuple[int, ...], ...],
    *,
    rng: np.random.Generator,
    n_permutations: int,
) -> float:
    """Two-sided cluster-aware p on the unweighted mean of REWARD-CELL means.

    Pairs inside a cell share a stimulus and are not independent — E3's 60 pairs came from 14
    cells, a 4:1 pseudoreplication that an unclustered null would have read as real precision. The
    statistic is the mean of cell means rather than the mean of pairs so that it matches the
    estimand :func:`_mde80` powers; a sign flip applies to a whole cell either way, so this is a
    reweighting of the same null, not a different one.
    """

    cell_values = _cell_means(values, cells)
    observed = abs(float(cell_values.mean()))
    flips = rng.integers(0, 2, size=(n_permutations, len(cell_values))) * 2 - 1
    permuted = np.abs((flips * cell_values).mean(axis=1))
    return add_one_p(int((permuted >= observed).sum()), n_permutations)


def _cell_means(values: np.ndarray, cells: tuple[tuple[int, ...], ...]) -> np.ndarray:
    return np.asarray([float(values[list(rows)].mean()) for rows in cells])


def _mde80(cell_values: np.ndarray) -> float:
    """The smallest true mean this clustered design would detect 80% of the time at GATE_ALPHA."""

    if len(cell_values) < 2:
        return float("inf")
    standard_error = float(cell_values.std(ddof=1)) / float(np.sqrt(len(cell_values)))
    normal = NormalDist()
    return (normal.inv_cdf(1.0 - GATE_ALPHA / 2.0) + normal.inv_cdf(GATE_POWER)) * standard_error


def _axis_means(
    state_deltas: np.ndarray,
    axes_by_block: Mapping[int, Mapping[str, np.ndarray]] | None,
    readout_blocks: tuple[int, ...],
) -> dict[str, float]:
    """Mean projection of a (pairs, layers, hidden) state delta on each axis AT its own block.

    Per-block rather than one axis reused everywhere: an emotion axis is fitted at a block, and
    projecting a block-63 state onto a block-35 axis measures the angle between two bases as much
    as anything about the state.
    """

    if not axes_by_block:
        return {}
    projections: dict[str, float] = {}
    for index, block in enumerate(readout_blocks):
        for name, axis in axes_by_block.get(block, {}).items():
            unit = unit_vector(np.asarray(axis, dtype=np.float64))
            projections[f"{name}@{block}"] = float((state_deltas[:, index, :] @ unit).mean())
    return projections


# --------------------------------------------------------------------------------------
# Reported models
# --------------------------------------------------------------------------------------


class TitrationLevel(StrictModel):
    """One certain-amount level, and how the unpatched surface behaved at it."""

    certain: int
    mean_baseline_margin: float
    mean_abs_baseline_margin: float
    selected: bool


class SensitivityGate(StrictModel):
    """B0: does the manipulation move the behaviour at all, before any patch is spent?

    ``passed`` is the conjunction the pre-registration fixes: the natural gap is distinguishable
    from zero under a cell-clustered sign-flip null at ``GATE_ALPHA``, AND the design's MDE80 is
    below ``GATE_MDE_FRACTION`` of that gap, so a half-sized transfer would be detectable. Either
    half failing routes the window to ``harness_inadequate`` and spends nothing further.

    No sign is pre-committed. The decision-affect literature supports carryover in both directions
    (house-money predicts positive-RPE to risk-seeking, mood-maintenance the reverse), so a
    direction recorded here would be a coin flip dressed as a prediction. What IS pre-registered is
    directional consistency downstream: the patched shift carries the sign of the natural gap.
    """

    window: WindowName
    n_pairs: int
    n_cells: int
    levels: tuple[TitrationLevel, ...]
    selected_certain: int
    mean_natural_gap: float
    sd_cell_gap: float
    p_value: float
    mde80: float
    alpha: float
    # The smallest p this design could return at all — the combinatorial sign-flip floor and the
    # Monte-Carlo floor, whichever binds. Recorded so a reader can tell "no effect" from "no test".
    attainable_p: float
    passed: bool
    per_pair_gap: tuple[float, ...]
    free_rider_axis_gap: dict[str, float]
    free_rider_note: str
    verdict: str


class ArmTransfer(StrictModel):
    """One arm's behavioural transfer, per pair and in aggregate.

    ``normalised_shift`` is the HEADLINE and is the ratio of cell-weighted means,
    ``mean(shift) / mean(gap)`` — the same estimand the gate's MDE80 powers and the same one its
    permutation null tests, so the three numbers in the verdict describe one quantity.
    ``transfer_fraction`` is E3's sign-corrected ratio of sums, kept for comparability with the
    published E3 table; where per-pair gaps carry mixed signs the two diverge, and the second is
    attenuated by ``mean(sign(gap))``.

    *Neither is a literal fraction, and 1.0 is not the target* — E3's caveat, which matters more
    here rather than less. The denominator is the whole difference between the donor's and the
    recipient's behaviour on their own different prompts; a patch of ONE token cannot reproduce the
    donor's reading of a prompt whose options block differs by construction. Read it as "how far
    toward the donor did the recipient move, in units of the donor-recipient gap", always against
    the random floor and the same-condition no-op.
    """

    arm: str
    window: WindowName
    n_pairs: int
    n_random_draws: int
    mean_shift: float
    normalised_shift: float
    transfer_fraction: float
    p_value: float | None
    # The certified arm's four per-rendering fractions. The random floor scores ONE rendering cell
    # per draw, so its 95th percentile must be compared against this spread rather than against the
    # four-rendering mean, which carries a quarter of the rendering noise.
    per_rendering_fraction: tuple[float, ...]
    free_rider_axis_shift: dict[str, float]
    # Per-pair, because E3's artifact stored only per-arm means and its passthrough excess could
    # not be given an interval afterwards. That gap does not recur.
    per_pair_shift: tuple[float, ...]
    per_pair_gap: tuple[float, ...]
    note: str


class CorruptionControl(StrictModel):
    """One invariant-answer window under one arm: did the patch garble the bookkeeping?

    The verdict is read off ``mean_abs_shift`` — the mean of per-pair |shift| — and NOT off
    |mean shift|: per-pair damage in opposite directions is still damage, and averaging it first
    cancels exactly the failure mode this control exists to catch. It is scored against the choice
    window's natural gap (``relative_to_gap``), which is the scale the claim is made on;
    ``relative_shift`` against the window's own saturated baseline is kept as a diagnostic only.
    """

    window: WindowName
    arm: str
    n_pairs: int
    mean_baseline_margin: float
    mean_abs_baseline_margin: float
    mean_shift: float
    mean_abs_shift: float
    choice_natural_gap: float
    relative_to_gap: float | None
    relative_shift: float | None
    tolerance: float
    within_tolerance: bool
    per_pair_shift: tuple[float, ...]


class ReachabilityControl(StrictModel):
    """The sensitivity control the whole rung's readability depends on.

    ``docs/agents/experiment-gating.md``: a null is evidence only where sensitivity was shown on
    THIS harness, and "the harness runs" is not sensitivity. Without this row, a flat arm table is
    ambiguous between "the reveal-token state is not behaviourally used" and "nothing written at
    that token reaches the answer slot on this surface" — and only the first is a fact about the
    model. It is run even when B0 fails, because that is exactly the case where the operator needs
    to know which of the two they bought.

    Not a transfer measurement: the donor differs from the recipient in reward, surface symbol and
    expectation at once, so nothing directional about the manipulation follows from it.
    """

    n_pairs: int
    mean_shift: float
    p_value: float
    alpha: float
    passed: bool
    per_pair_shift: tuple[float, ...]
    verdict: str


class BehavioralTransferReport(StrictModel):
    """E4 record: the gate, the arms, and the corruption controls that bound their reading."""

    artifact_contract_version: Literal["behavioral_transfer/v1"] = (
        BEHAVIORAL_TRANSFER_CONTRACT_VERSION
    )
    seed: int
    patch_block: int
    # Where the patched direction was CERTIFIED, and under which verdict, copied from the
    # directions artifact. When ``patch_block`` differs from it the arm labelled
    # ``v_rpe_component`` is injecting an uncertified vector, and a reader cannot tell from
    # ``patch_block`` alone -- so both numbers are recorded rather than one.
    direction_block: int
    direction_verdict: str
    readout_blocks: tuple[int, ...]
    n_pairs: int
    n_cells: int
    n_random_draws: int
    symbol_matched: bool
    symbol_confound: str | None
    answer_pool: tuple[str, ...]
    answer_pool_note: str
    answer_form: AnswerForm
    # The whole preflight, not just the survivors: on a PASS the fact that a candidate was dropped
    # as a collision or as multi-token is otherwise invisible, and it is exactly what a reader
    # needs to judge whether the pool was chosen or merely what was left.
    answer_pool_preflight: AnswerPoolPreflight
    # What fraction of the no-op control's donors match the recipient's rendering surface. The
    # control matches on (ev_cell, rpe_sign) only, so a low rate means the arm also varies symbol
    # and template — read it as a control on full_residual, not on v_rpe_component.
    no_op_surface_match_rate: float
    states_sha256: str
    battery_sha256: str
    directions_sha256: str
    emotion_vectors_sha256: str
    model_key: str | None
    sensitivity_gate: str
    wiring_check: str
    reachability: ReachabilityControl
    gate: SensitivityGate
    arms: tuple[ArmTransfer, ...]
    corruption: tuple[CorruptionControl, ...]
    corruption_clean: bool
    corruption_note: str
    # Whether the full-residual ceiling shows this design could have SEEN a within-cell shift at
    # all. A flat arm table means something different depending on this, so it is a reported field
    # rather than only a phrase inside ``verdict_cap``.
    ceiling_readable: bool
    ceiling_note: str
    cross_position_note: str
    identification_limit: str
    verdict_cap: str


_FREE_RIDER_NOTE = (
    "Off-position emotion-axis read at the answer slot, riding along at zero extra forward cost "
    "(e4-prereg §6). It carries its own validity gate: the axes were built from story-mean states "
    "and their meaning at an appended token is untested, so the UNPATCHED natural gap recorded "
    "here must itself be non-trivial before any patched number at this position is interpretable. "
    "Exploratory otherwise, and never the basis of a claim on its own."
)

CROSS_POSITION_NOTE = (
    "The readout is at a strictly later token than the patch, so the residual stream's identity "
    "path cannot produce it: a value written at the reveal token reaches the answer slot only by "
    "being selected by an attention head. This is what E3's same-position readout could not "
    "establish and what its random-direction floor could not detect "
    "(scripts/e3_passthrough_decomposition.py)."
)

IDENTIFICATION_LIMIT = (
    "Within a reward-matched cell EV and signed RPE are perfectly anti-correlated (reward is "
    "fixed), so E4 identifies 'the expectation manipulation transfers', NOT 'signed RPE rather "
    "than EV transfers'. E2's two matched arms carry that separation; no arm here escapes the "
    "collinearity."
)

_ARM_NOTES: dict[str, str] = {
    "full_residual": (
        "Ceiling: the donor's whole reveal-token state. Unlike the on-position reading, this is "
        "NOT 1.0 by construction — a whole-state substitution still has to be read by attention "
        "to move a later token."
    ),
    "v_rpe_component": "Is the transfer carried by the certified signed-RPE direction?",
    "random_component": (
        "Floor: random directions carrying the SAME injected magnitude as the certified arm "
        "(analysis.activation_patching.matched_random_value), reported at the 95th percentile over "
        "20 draws. Both corrections matter: E3's floor substituted the recipient's own component "
        "along a random direction, which injects ~1/sqrt(d) of what the certified direction does "
        "and so reads near zero whether or not the effect is direction-specific; and at 5 draws a "
        "95th percentile is the sample maximum. The rendering cell rotates with the draw index, so "
        "the floor is compared against the certified arm's PER-RENDERING fractions rather than "
        "against its four-rendering mean -- see ArmTransfer.per_rendering_fraction."
    ),
    "same_condition_donor": (
        "No-op negative control: a different rendering of the recipient's OWN condition, scored "
        "against the same measured denominator so the arms sit on one scale. In E3 this arm's "
        "excess over the identity path was as large as the certified direction's, which is why it "
        "is read as a control on equal footing rather than as a formality."
    ),
}


# --------------------------------------------------------------------------------------
# B0 — the sensitivity gate
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Baselines:
    """The recipients' self-patched choice readouts at the selected level, per rendering cell.

    Measured once, inside B0, and reused by every arm. B0 already runs exactly these forwards — the
    recipient's own state written back at its own position, at every titration level — so paying
    for them a second time would be the harness outspending the run it serves (CLAUDE.md, the
    dual). Kept PER RENDERING rather than averaged, because the random floor scores one rendering
    cell per draw and differencing that against a four-rendering mean would put a rendering effect
    in the floor.
    """

    margins: np.ndarray  # (pairs, renderings)
    states: np.ndarray  # (pairs, renderings, blocks, hidden)


def _choice_margins(
    backend: ChatPatchingBackend,
    reveal: Comparison,
    *,
    renderings: tuple[OptionRendering, ...],
    certain: int,
    patch_block: int,
    replacement: np.ndarray,
    readout_blocks: tuple[int, ...],
    answer_form: AnswerForm,
) -> tuple[np.ndarray, np.ndarray]:
    """One row's choice margin in each of the 2x2 counterbalance cells, plus the states there.

    The caller marginalises. Averaging the four cells rather than picking one puts symbol
    preference and line-order preference in the intercept instead of the slope — the same
    discipline the reveal battery applies to its own symbol pools.
    """

    margins: list[float] = []
    states: list[np.ndarray] = []
    for rendering in renderings:
        readout = read(
            backend,
            reveal,
            choice_probe_for(reveal, rendering=rendering, certain=certain, form=answer_form),
            block=patch_block,
            replacement=replacement,
            readout_blocks=readout_blocks,
        )
        margins.append(readout.margin)
        states.append(readout.states)
    return np.asarray(margins), np.stack(states)


def sensitivity_gate(
    backend: ChatPatchingBackend,
    reveals: tuple[Comparison, ...],
    pairs: tuple[PatchPair, ...],
    block_states: np.ndarray,
    *,
    patch_block: int,
    readout_blocks: tuple[int, ...],
    answer_pool: tuple[str, ...],
    seed: int,
    n_permutations: int,
    axes_by_block: Mapping[int, Mapping[str, np.ndarray]] | None = None,
    certain_levels: tuple[int, ...] = CERTAIN_LEVELS,
    answer_form: AnswerForm = DEFAULT_ANSWER_FORM,
) -> tuple[SensitivityGate, Baselines]:
    """B0 — the natural gap, measured with NO patch, and the titration level it selects.

    Every forward here is a self-patch: the row's own captured value written back at its own
    position, which is inert by contract. That is deliberate — the baseline is measured through the
    same code path the arms use, rather than through a second one that might not agree with it, and
    the selected level's recipient readouts ARE the arms' baselines, returned alongside the gate.

    One weighting, stated once: the gate's effect size, its permutation null and its MDE80 are all
    the unweighted mean of REWARD-CELL means. Cell sizes are unequal on the real battery, so a
    pair-weighted effect size against a cell-clustered SE would be two different estimands wearing
    one inequality — a design where a single large cell carries the gap could pass a bar the
    clustered inference does not support.
    """

    cells = _cell_groups(pairs)
    # Two phases, and the split is a budget decision that changes no reported number. Titration is
    # decided entirely on recipient margins, so phase one reads the recipients at every level;
    # phase two buys the donors at the winning level only, and reuses phase one's recipients there
    # rather than re-reading them.
    recipients = {
        certain: _role_pass(
            backend,
            reveals,
            pairs,
            block_states,
            role="recipient",
            certain=certain,
            patch_block=patch_block,
            readout_blocks=readout_blocks,
            answer_pool=answer_pool,
            answer_form=answer_form,
        )
        for certain in certain_levels
    }

    # Nearest indifference: where a logit margin has the most room to move. Chosen on UNPATCHED
    # data, before any arm runs, so it cannot be an outcome-dependent selection.
    levels = tuple(
        TitrationLevel(
            certain=certain,
            mean_baseline_margin=float(level.margins.mean()),
            mean_abs_baseline_margin=float(np.abs(level.margins.mean(axis=1)).mean()),
            selected=False,
        )
        for certain, level in recipients.items()
    )
    selected = min(levels, key=lambda level: level.mean_abs_baseline_margin)
    levels = tuple(
        level.model_copy(update={"selected": level.certain == selected.certain}) for level in levels
    )

    recipient = recipients[selected.certain]
    donor = _role_pass(
        backend,
        reveals,
        pairs,
        block_states,
        role="donor",
        certain=selected.certain,
        patch_block=patch_block,
        readout_blocks=readout_blocks,
        answer_pool=answer_pool,
        answer_form=answer_form,
    )
    gaps = donor.margins.mean(axis=1) - recipient.margins.mean(axis=1)
    state_gaps = donor.states.mean(axis=1) - recipient.states.mean(axis=1)
    cell_gaps = _cell_means(gaps, cells)
    observed = float(cell_gaps.mean())
    p_value = _cell_permutation_p(
        gaps,
        cells,
        rng=np.random.default_rng(seed_int(seed, "behavioral-gate", "choice", patch_block)),
        n_permutations=n_permutations,
    )
    mde80 = _mde80(cell_gaps)
    floor = _attainable_p(len(cells), n_permutations)
    passed = bool(
        floor <= GATE_ALPHA and p_value < GATE_ALPHA and mde80 < GATE_MDE_FRACTION * abs(observed)
    )
    gate = SensitivityGate(
        window="choice",
        n_pairs=len(pairs),
        n_cells=len(cells),
        levels=levels,
        selected_certain=selected.certain,
        mean_natural_gap=observed,
        sd_cell_gap=float(cell_gaps.std(ddof=1)) if len(cell_gaps) > 1 else 0.0,
        p_value=p_value,
        mde80=mde80,
        alpha=GATE_ALPHA,
        attainable_p=floor,
        passed=passed,
        per_pair_gap=tuple(float(value) for value in gaps),
        free_rider_axis_gap=_axis_means(state_gaps, axes_by_block, readout_blocks),
        free_rider_note=_FREE_RIDER_NOTE,
        verdict=_gate_verdict(passed, p_value, mde80, observed, len(cells), n_permutations),
    )
    return gate, Baselines(margins=recipient.margins, states=recipient.states)


@dataclass(frozen=True)
class _RolePass:
    """One role's unpatched readouts at one titration level, marginalised over nothing yet."""

    margins: np.ndarray  # (pairs, renderings)
    states: np.ndarray  # (pairs, renderings, blocks, hidden)


def _role_pass(
    backend: ChatPatchingBackend,
    reveals: tuple[Comparison, ...],
    pairs: tuple[PatchPair, ...],
    block_states: np.ndarray,
    *,
    role: Literal["donor", "recipient"],
    certain: int,
    patch_block: int,
    readout_blocks: tuple[int, ...],
    answer_pool: tuple[str, ...],
    answer_form: AnswerForm,
) -> _RolePass:
    """Every pair's donor OR recipient, unpatched, in all four counterbalance cells.

    Split by role because the two are wanted at different times. Titration is decided on the
    RECIPIENTS alone -- ``mean_abs_baseline_margin`` is a recipient quantity -- so the recipients
    are read at every level and the donors only at the level that wins. Reading both at every level
    buys 960 forwards whose numbers no field of the report ever contains.

    Every forward here is a self-patch: the row's own captured value written back at its own
    position, inert by contract. Deliberate -- the baseline is measured through the same code path
    the arms use rather than through a second one that might not agree with it.
    """

    margins: list[np.ndarray] = []
    states: list[np.ndarray] = []
    for pair in pairs:
        row = pair.donor_row if role == "donor" else pair.recipient_row
        margin, state = _choice_margins(
            backend,
            reveals[row],
            renderings=renderings_for(reveals, pair, answer_pool),
            certain=certain,
            patch_block=patch_block,
            replacement=block_states[row],
            readout_blocks=readout_blocks,
            answer_form=answer_form,
        )
        margins.append(margin)
        states.append(state)
    return _RolePass(margins=np.stack(margins), states=np.stack(states))


def _attainable_p(n_cells: int, n_permutations: int) -> float:
    """The smallest two-sided p this gate's null can return with this design. TWO floors.

    The combinatorial one: both the identity sign assignment and its complement always tie the
    observed statistic, so no clustered sign-flip null can go below ``2 / 2**k`` with k cells —
    with 5 cells that is 0.0625 and NO gap, however large, clears a 0.01 bar. The Monte-Carlo one:
    the estimator here is ``add_one_p``, whose smallest value is ``1 / (n_permutations + 1)``
    regardless of how many cells there are.

    Both are checked because a design that cannot pass and a manipulation that does nothing produce
    the same failing number, and only one of them is a fact about the model.
    """

    combinatorial = 2.0 ** (1 - n_cells) if n_cells < 64 else 0.0
    return max(combinatorial, 1.0 / (n_permutations + 1))


def _min_cells() -> int:
    """The fewest reward cells whose sign-flip null can reach ``GATE_ALPHA`` at all."""

    cells = 2
    while 2.0 ** (1 - cells) > GATE_ALPHA:
        cells += 1
    return cells


def _gate_verdict(
    passed: bool,
    p_value: float,
    mde80: float,
    observed: float,
    n_cells: int,
    n_permutations: int,
) -> str:
    floor = _attainable_p(n_cells, n_permutations)
    if not passed and floor > GATE_ALPHA:
        return (
            f"B0 FAILED (design floor): {n_cells} reward cells and {n_permutations} permutations "
            f"put the smallest attainable two-sided p at {floor:.4f}, above alpha={GATE_ALPHA}. "
            "No natural gap of any size could pass this gate, so the result says nothing about "
            f"the model. Widen the pair set to >= {_min_cells()} cells, raise the permutation "
            f"count above {int(1 / GATE_ALPHA)}, and re-run; spend no patched forward."
        )
    if passed:
        return (
            "B0 PASSED: the expectation manipulation moves this window unpatched, and a "
            "half-sized transfer would be detectable. The patched leg's transfer fraction now has "
            "a denominator that was measured rather than assumed."
        )
    if p_value >= GATE_ALPHA:
        return (
            "B0 FAILED (no natural gap): the manipulation does not move this window, so the "
            "transfer fraction's denominator is zero and no patch transferring that manipulation "
            "could move it either. harness_inadequate for the window; spend no patched forward. "
            "A strong instruct model computing round 2 from the round-2 numbers is behaving "
            "correctly -- this is a reportable instrument fact, not evidence about functional use."
        )
    return (
        f"B0 FAILED (underpowered): the gap is distinguishable from zero but MDE80={mde80:.4f} is "
        f"not below {GATE_MDE_FRACTION:g} x |gap|={GATE_MDE_FRACTION * abs(observed):.4f}, so a "
        "half-sized transfer would not be detectable and a patched null would be unreadable. "
        "harness_inadequate for the window; widen the pair set before spending."
    )


# --------------------------------------------------------------------------------------
# The patched arms
# --------------------------------------------------------------------------------------


def behavioral_arms(
    backend: ChatPatchingBackend,
    reveals: tuple[Comparison, ...],
    pairs: tuple[PatchPair, ...],
    block_states: np.ndarray,
    gate: SensitivityGate,
    baselines: Baselines,
    v_rpe: np.ndarray,
    *,
    patch_block: int,
    readout_blocks: tuple[int, ...],
    answer_pool: tuple[str, ...],
    seed: int,
    n_permutations: int,
    axes_by_block: Mapping[int, Mapping[str, np.ndarray]] | None = None,
    n_random_draws: int = DEFAULT_RANDOM_DRAWS,
    answer_form: AnswerForm = DEFAULT_ANSWER_FORM,
) -> tuple[ArmTransfer, ...]:
    """The patched choice arms, normalised by B0's own measured per-pair natural gap.

    Refuses to run unless the gate passed: spending patched forwards against an unmeasured
    denominator is the exact failure E4 exists to repair, so it is a runtime error rather than a
    convention a caller in a hurry can skip.
    """

    if not gate.passed:
        raise ValueError(
            "B0 did not pass; the patched leg must not be spent (e4-prereg §5 discard clause)"
        )
    gaps = np.asarray(gate.per_pair_gap)
    if len(gaps) != len(pairs):
        raise ValueError("the gate's per-pair gaps do not correspond one-to-one with the pairs")
    rng = np.random.default_rng(seed_int(seed, "behavioral-arms", patch_block))
    directions = tuple(unit_vector(rng.standard_normal(v_rpe.size)) for _ in range(n_random_draws))
    cells = _cell_groups(pairs)
    results: list[ArmTransfer] = []
    for arm in ARMS:
        kept = tuple(
            index
            for index, pair in enumerate(pairs)
            if arm != "same_condition_donor" or pair.same_condition_row is not None
        )
        if not kept:
            continue
        draws = directions if arm == "random_component" else (v_rpe,)
        per_draw = [
            _arm_pass(
                backend,
                reveals,
                pairs,
                block_states,
                baselines,
                arm=arm,
                kept=kept,
                v_rpe=v_rpe,
                direction=direction,
                # One rendering cell per random draw, rotating with the draw index: the floor's
                # estimand is a distribution over directions, and the counterbalance is a nuisance
                # better marginalised across draws than paid for four times inside each one.
                rendering_index=index if arm == "random_component" else None,
                certain=gate.selected_certain,
                patch_block=patch_block,
                readout_blocks=readout_blocks,
                answer_pool=answer_pool,
                answer_form=answer_form,
            )
            for index, direction in enumerate(draws)
        ]
        results.append(
            _aggregate_arm(
                arm,
                per_draw,
                gaps[list(kept)],
                cells=cells,
                kept=kept,
                axes_by_block=axes_by_block,
                readout_blocks=readout_blocks,
                seed=seed,
                patch_block=patch_block,
                n_permutations=n_permutations,
            )
        )
    return tuple(results)


def _arm_pass(
    backend: ChatPatchingBackend,
    reveals: tuple[Comparison, ...],
    pairs: tuple[PatchPair, ...],
    block_states: np.ndarray,
    baselines: Baselines,
    *,
    arm: str,
    kept: tuple[int, ...],
    v_rpe: np.ndarray,
    direction: np.ndarray,
    rendering_index: int | None,
    certain: int,
    patch_block: int,
    readout_blocks: tuple[int, ...],
    answer_pool: tuple[str, ...],
    answer_form: AnswerForm,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One arm at one direction: per-rendering margin shifts, and the state shift at the slot.

    Both the margin and the state are differenced against the SAME rendering cells that were run,
    never against the four-rendering mean: the random floor scores one cell per draw, and mixing
    the two would put a rendering effect into the floor's state readout.
    """

    shifts: list[np.ndarray] = []
    state_shifts: list[np.ndarray] = []
    for index in kept:
        pair = pairs[index]
        renderings = renderings_for(reveals, pair, answer_pool)
        chosen_cells = (
            (rendering_index % len(renderings),)
            if rendering_index is not None
            else tuple(range(len(renderings)))
        )
        replacement = (
            matched_random_value(pair, block_states, v_rpe, direction)
            if arm == "random_component"
            else substituted_value(arm, pair, block_states, v_rpe, direction)
        )
        margins: list[float] = []
        states: list[np.ndarray] = []
        for cell in chosen_cells:
            readout = read(
                backend,
                reveals[pair.recipient_row],
                choice_probe_for(
                    reveals[pair.recipient_row],
                    rendering=renderings[cell],
                    certain=certain,
                    form=answer_form,
                ),
                block=patch_block,
                replacement=replacement,
                readout_blocks=readout_blocks,
            )
            margins.append(readout.margin - float(baselines.margins[index][cell]))
            states.append(readout.states - baselines.states[index][cell])
        shifts.append(np.asarray(margins))
        state_shifts.append(np.mean(np.stack(states), axis=0))
    per_rendering = np.stack(shifts)
    return per_rendering.mean(axis=1), per_rendering, np.stack(state_shifts)


def _aggregate_arm(
    arm: str,
    per_draw: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    gaps: np.ndarray,
    *,
    cells: tuple[tuple[int, ...], ...],
    kept: tuple[int, ...],
    axes_by_block: Mapping[int, Mapping[str, np.ndarray]] | None,
    readout_blocks: tuple[int, ...],
    seed: int,
    patch_block: int,
    n_permutations: int,
) -> ArmTransfer:
    """Collapse an arm's draws into one row. The floor reports its 95th percentile, not its mean.

    Two normalisations are computed from the same shifts. ``normalised_shift`` is the ratio of
    cell-weighted means and is the headline, because it is the quantity the gate powered and the
    quantity the permutation null below tests. ``transfer_fraction`` is E3's sign-corrected ratio
    of sums, kept so the two experiments' tables can be read against each other.

    The null permutes ``shift * sign(gap)`` rather than the raw shift. Under strict proportionality
    the two are the same test; where the transfer is sign-aligned but not magnitude-proportional —
    and on the real pair set a large minority of per-pair gaps carry the opposite sign to the mean
    — permuting the raw shift tests a statistic the report does not headline.
    """

    cell_index = _remap_cells(cells, kept)
    denominator = float(np.abs(gaps).sum())
    mean_gap = float(_cell_means(gaps, cell_index).mean())
    if denominator <= 0.0 or mean_gap == 0.0:
        raise ValueError("the measured denominator is zero despite a passing gate")
    fractions = [float((shifts * np.sign(gaps)).sum() / denominator) for shifts, _, _ in per_draw]
    if arm == "random_component":
        # Report the direction that IS the 95th percentile, so the stored per-pair shifts and the
        # headline fraction describe the same draw rather than two different ones.
        chosen = int(np.argsort(fractions)[int(round(FLOOR_QUANTILE * (len(fractions) - 1)))])
        fraction, p_value = fractions[chosen], None
    else:
        chosen, fraction = 0, fractions[0]
        p_value = _cell_permutation_p(
            per_draw[0][0] * np.sign(gaps),
            cell_index,
            rng=np.random.default_rng(seed_int(seed, "behavioral-arm", arm, patch_block)),
            n_permutations=n_permutations,
        )
    shifts, per_rendering, state_shifts = per_draw[chosen]
    return ArmTransfer(
        arm=arm,
        window="choice",
        n_pairs=len(kept),
        n_random_draws=len(per_draw) if arm == "random_component" else 0,
        mean_shift=float(_cell_means(shifts, cell_index).mean()),
        normalised_shift=float(_cell_means(shifts, cell_index).mean() / mean_gap),
        transfer_fraction=fraction,
        p_value=p_value,
        per_rendering_fraction=tuple(
            float((per_rendering[:, cell] * np.sign(gaps)).sum() / denominator)
            for cell in range(per_rendering.shape[1])
        ),
        free_rider_axis_shift=_axis_means(state_shifts, axes_by_block, readout_blocks),
        per_pair_shift=tuple(float(value) for value in shifts),
        per_pair_gap=tuple(float(value) for value in gaps),
        note=_ARM_NOTES[arm],
    )


def _remap_cells(
    cells: tuple[tuple[int, ...], ...], kept: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    """Re-index the reward-cell grouping onto an arm's kept subset, dropping emptied cells."""

    position = {index: rank for rank, index in enumerate(kept)}
    remapped = tuple(
        tuple(position[index] for index in rows if index in position) for rows in cells
    )
    return tuple(rows for rows in remapped if rows)


# --------------------------------------------------------------------------------------
# W2/W3 — corruption controls
# --------------------------------------------------------------------------------------


def corruption_controls(
    backend: ChatPatchingBackend,
    reveals: tuple[Comparison, ...],
    pairs: tuple[PatchPair, ...],
    block_states: np.ndarray,
    v_rpe: np.ndarray,
    *,
    patch_block: int,
    readout_blocks: tuple[int, ...],
    answer_pool: tuple[str, ...],
    choice_natural_gap: float,
    answer_form: AnswerForm = DEFAULT_ANSWER_FORM,
) -> tuple[CorruptionControl, ...]:
    """Did the patch garble the numeric bookkeeping the window's answer depends on?

    Both windows have an answer that is identical across a matched pair — the cell pins the reward
    and the target LABEL always carries the correct option — so nothing the manipulation supplies
    can move them, and movement is instrument damage. One rendering cell per pair, rotating with
    the pair index: the counterbalance is marginalised across pairs rather than paid for four times
    each, which is affordable here only because the estimand is a mean over pairs.

    What this can and cannot do is worth stating where the numbers are made. It bounds damage; it
    does not add diagnostic weight to a choice null, because a perfectly-propagating patch and an
    inert one BOTH predict zero movement here.
    """

    controls: list[CorruptionControl] = []
    for window in CORRUPTION_WINDOWS:
        probes = tuple(
            corruption_probe_for(
                reveals[pair.recipient_row],
                window,
                rendering=rotating_rendering(reveals, pair, answer_pool, index),
                form=answer_form,
            )
            for index, pair in enumerate(pairs)
        )
        baselines = np.asarray(
            [
                read(
                    backend,
                    reveals[pair.recipient_row],
                    probes[index],
                    block=patch_block,
                    replacement=block_states[pair.recipient_row],
                    readout_blocks=readout_blocks,
                ).margin
                for index, pair in enumerate(pairs)
            ]
        )
        controls.extend(
            _corruption_arm(
                backend,
                reveals,
                pairs,
                block_states,
                v_rpe,
                probes=probes,
                baselines=baselines,
                window=window,
                arm=arm,
                patch_block=patch_block,
                readout_blocks=readout_blocks,
                choice_natural_gap=choice_natural_gap,
            )
            for arm in CORRUPTION_ARMS
        )
    return tuple(controls)


def _corruption_arm(
    backend: ChatPatchingBackend,
    reveals: tuple[Comparison, ...],
    pairs: tuple[PatchPair, ...],
    block_states: np.ndarray,
    v_rpe: np.ndarray,
    *,
    probes: tuple[DecisionProbe, ...],
    baselines: np.ndarray,
    window: WindowName,
    arm: str,
    patch_block: int,
    readout_blocks: tuple[int, ...],
    choice_natural_gap: float,
) -> CorruptionControl:
    shifts = np.asarray(
        [
            read(
                backend,
                reveals[pair.recipient_row],
                probes[index],
                block=patch_block,
                replacement=substituted_value(arm, pair, block_states, v_rpe, v_rpe),
                readout_blocks=readout_blocks,
            ).margin
            - baselines[index]
            for index, pair in enumerate(pairs)
        ]
    )
    scale = float(np.abs(baselines).mean())
    mean_abs_shift = float(np.abs(shifts).mean())
    gap = abs(choice_natural_gap)
    relative_to_gap = float(mean_abs_shift / gap) if gap > 0.0 else None
    return CorruptionControl(
        window=window,
        arm=arm,
        n_pairs=len(pairs),
        mean_baseline_margin=float(baselines.mean()),
        mean_abs_baseline_margin=scale,
        mean_shift=float(shifts.mean()),
        mean_abs_shift=mean_abs_shift,
        choice_natural_gap=choice_natural_gap,
        relative_to_gap=relative_to_gap,
        relative_shift=float(abs(shifts.mean()) / scale) if scale > 0.0 else None,
        tolerance=CORRUPTION_TOLERANCE,
        within_tolerance=relative_to_gap is not None and relative_to_gap < CORRUPTION_TOLERANCE,
        per_pair_shift=tuple(float(value) for value in shifts),
    )


def corruption_reading(
    controls: tuple[CorruptionControl, ...], arms: tuple[ArmTransfer, ...] = ()
) -> tuple[bool, str]:
    """Are the invariant-answer windows undisturbed enough for the choice result to be read?

    W2/W3 are corruption controls and NOT an affect-versus-arithmetic dissociation: within a
    reward-matched cell the manipulation IS the stated EV, so "the model carries a different
    expectation and reasons from it" is a paraphrase of the claim rather than a rival to it
    (``docs/design/e4-prereg.md`` §6). What these windows can exclude is the patch garbling the
    numeric bookkeeping, which would make any choice shift uninterpretable.
    """

    if not controls:
        return False, "no corruption-control windows were run; the choice reading is uncontrolled"
    # Two bars, and the second is the one the tolerance constant only PROMISED. CORRUPTION_TOLERANCE
    # is a fixed fraction of the natural gap, so at 0.5 it admits damage of half a gap against an
    # effect that is itself a fraction of a gap -- damage larger than the result, passing a control
    # whose whole purpose is to exclude that. So each window is also required to have moved less
    # than the arm it was run under actually moved the choice. Fixed before any data, and it can
    # only ever tighten the gate, never loosen it.
    claimed = {arm.arm: abs(arm.mean_shift) for arm in arms}
    offenders = [
        control
        for control in controls
        if not control.within_tolerance
        or control.mean_abs_shift >= claimed.get(control.arm, float("inf"))
    ]
    if offenders:
        names = ", ".join(f"{control.arm}/{control.window}" for control in offenders)
        return False, (
            f"CORRUPTION: {names} moved a window whose correct answer is identical across the "
            f"pair by more than {CORRUPTION_TOLERANCE:.0%} of the CHOICE window's own natural "
            "gap, or by more than that arm moved the choice itself. The patch is damaging numeric "
            "bookkeeping on the scale of the effect being claimed, so the choice shift is reported "
            "with that confound named and capped."
        )
    return True, (
        "Both invariant-answer windows stayed inside tolerance, so the choice shift is not the "
        "patch garbling the context. This bounds instrument damage and NOTHING else: a patch that "
        "propagates perfectly and a patch that does nothing both predict zero movement here, so a "
        "clean reading adds no weight to a choice null. It is also NOT an affect-versus-arithmetic "
        "dissociation -- see §6."
    )


# --------------------------------------------------------------------------------------
# The positive control: can ANYTHING written at the reveal token reach the answer slot?
# --------------------------------------------------------------------------------------


def _reachability_pairs(
    reveals: tuple[Comparison, ...], n_pairs: int
) -> tuple[tuple[int, int], ...]:
    """(donor_row, recipient_row) drawn from DIFFERENT reward cells, in alternating directions.

    Every other pair set in E4 is reward-matched, which is what makes the corruption windows
    invariant and also what makes them unable to show that anything crosses positions at all. Here
    the reward is deliberately unmatched and maximally so — the extreme rows of the reward
    distribution — because the control needs the largest signal the surface can offer, not a
    representative one.

    The direction alternates: half the pairs push a high-reward donor into a low-reward recipient
    and half the reverse. A readout that simply prefers the larger number, or the second option,
    would move the same way in both halves and cancel; only a shift that follows the donor survives
    averaging with the sign convention applied.
    """

    def reward(row: int) -> float:
        return float(reveals[row].metadata["reward"])

    order = sorted(range(len(reveals)), key=lambda row: (reward(row), row))
    half = min(n_pairs // 2, len(order) // 2)
    if half == 0:
        raise ValueError("too few reveals to build a reachability control")
    low, high = order[:half], order[-half:][::-1]
    pairs: list[tuple[int, int]] = []
    for donor, recipient in zip(high, low, strict=True):
        if reward(donor) == reward(recipient):
            continue
        pairs.append((donor, recipient))
        pairs.append((recipient, donor))
    return tuple(pairs[:n_pairs])


def reachability_control(
    backend: ChatPatchingBackend,
    reveals: tuple[Comparison, ...],
    block_states: np.ndarray,
    *,
    patch_block: int,
    readout_blocks: tuple[int, ...],
    answer_pool: tuple[str, ...],
    seed: int,
    n_permutations: int,
    n_pairs: int = DEFAULT_REACHABILITY_PAIRS,
    answer_form: AnswerForm = DEFAULT_ANSWER_FORM,
) -> ReachabilityControl:
    """Patch a donor from a different reward cell; ask the total; expect the answer to follow it.

    The prediction is a point one and it is signed: the margin is ``logit(recipient's own total) −
    logit(the donor's total)``, so a patch that reaches the answer slot at all makes it FALL.
    """

    symbol_a, symbol_b = answer_symbols(answer_pool, exclude=frozenset())
    renderings = option_renderings(symbol_a, symbol_b)
    shifts: list[float] = []
    for index, (donor_row, recipient_row) in enumerate(_reachability_pairs(reveals, n_pairs)):
        recipient = reveals[recipient_row]
        excluded = {
            str(recipient.metadata["symbol_high"]),
            str(recipient.metadata["symbol_low"]),
            str(reveals[donor_row].metadata["symbol_high"]),
            str(reveals[donor_row].metadata["symbol_low"]),
        }
        if excluded & {symbol_a, symbol_b}:
            continue
        probe = reachability_probe(
            symbol=str(recipient.metadata["realised_symbol"]),
            reward=float(recipient.metadata["reward"]),
            donor_reward=float(reveals[donor_row].metadata["reward"]),
            rendering=renderings[index % len(renderings)],
            form=answer_form,
        )
        baseline = read(
            backend,
            recipient,
            probe,
            block=patch_block,
            replacement=block_states[recipient_row],
            readout_blocks=readout_blocks,
        )
        patched = read(
            backend,
            recipient,
            probe,
            block=patch_block,
            replacement=block_states[donor_row],
            readout_blocks=readout_blocks,
        )
        shifts.append(patched.margin - baseline.margin)
    if not shifts:
        raise ValueError(
            "no reachability pair survived answer-symbol exclusion; the control that makes every "
            "null in this report readable cannot be built, so the run must not proceed"
        )
    values = np.asarray(shifts)
    p_value = _cell_permutation_p(
        values,
        tuple((row,) for row in range(len(values))),
        rng=np.random.default_rng(seed_int(seed, "behavioral-reachability", patch_block)),
        n_permutations=n_permutations,
    )
    mean_shift = float(values.mean())
    passed = bool(mean_shift < 0.0 and p_value < REACHABILITY_ALPHA)
    return ReachabilityControl(
        n_pairs=len(values),
        mean_shift=mean_shift,
        p_value=p_value,
        alpha=REACHABILITY_ALPHA,
        passed=passed,
        per_pair_shift=tuple(float(value) for value in values),
        verdict=(
            "REACHABLE: a full-residual patch from another reward cell moves the answer slot "
            "toward the donor's total, so a value written at the reveal token demonstrably crosses "
            "positions on this surface. A flat arm table below is therefore about the model."
            if passed
            else (
                "UNREACHABLE: patching the reveal token does not move the answer slot even when "
                "the donor's own bookkeeping says it should. Every null in this report is "
                "harness_inadequate -- the instrument, not the model. Do not read the arm table "
                "as evidence about functional use."
            )
        ),
    )


def _arm_beats_its_controls(arms: tuple[ArmTransfer, ...]) -> bool:
    """Does the certified direction outrun its controls AND its own null?

    FOUR conditions, and the fourth is the one an earlier draft was missing. The three point
    comparisons -- positive, above the floor, above the no-op -- are comparisons of estimates with
    no noise model between them, so on a flat surface the certified arm clears all three roughly one
    time in eight by arithmetic alone. ``p_value`` is the arm's own cell-clustered sign-flip null,
    it was already being computed, and nothing read it; requiring it here is what stops
    "functionally-used" resting on three coin flips.

    The floor comparison is deliberately conservative where the comparison would otherwise be
    unfair: the floor scores one rendering cell per draw, so its 95th percentile is compared against
    the certified arm's WEAKEST per-rendering fraction rather than the four-rendering mean, whose
    rendering noise is a quarter as large.

    ``transfer_fraction`` is the comparison quantity throughout, and the arms are only ever compared
    to each other -- so the one requirement is that all four numbers are the same estimand, which
    they are. ``normalised_shift`` is the reported headline and the gate's MDE80 powers it; it is
    not used here because it divides by ``mean(gap)``, which on mixed-sign gaps is a denominator the
    floor and no-op do not share.
    """

    by_arm = {arm.arm: arm for arm in arms}
    certified = by_arm.get("v_rpe_component")
    if certified is None:
        return False
    weakest = min(certified.per_rendering_fraction, default=certified.transfer_fraction)
    floor = by_arm.get("random_component")
    no_op = by_arm.get("same_condition_donor")
    return bool(
        certified.transfer_fraction > 0.0
        and certified.p_value is not None
        and certified.p_value < ARM_ALPHA
        and (floor is None or weakest > floor.transfer_fraction)
        and (no_op is None or certified.transfer_fraction > no_op.transfer_fraction)
    )


def _ceiling_is_readable(
    arms: tuple[ArmTransfer, ...], gate: SensitivityGate, cells: tuple[tuple[int, ...], ...]
) -> tuple[bool, str]:
    """Can this design SEE a within-cell shift at all? Answered by the full-residual arm.

    Without this, a flat arm table is ambiguous in a way the reachability control does not resolve.
    Reachability swaps a whole residual across a +-160-point reward difference; the arms inject a
    within-reward-cell EV difference, which is orders of magnitude smaller by construction because
    the cell pins the reward. So reachability establishes reach at an EXTREME magnitude only, and
    "no transfer" would still be consistent with "the injected magnitude is below the readout's
    resolution" -- the same unmeasured-denominator shape E4 exists to escape, moved one level down.

    ``full_residual`` is the within-design ceiling: the largest shift anything in this pair set can
    produce, at exactly the magnitude the arms work at. The run already buys it. If IT cannot clear
    the random floor, no arm below it could have, and the correct verdict is that the harness could
    not have seen the effect -- not that the effect is absent.
    """

    by_arm = {arm.arm: arm for arm in arms}
    ceiling = by_arm.get("full_residual")
    floor = by_arm.get("random_component")
    if ceiling is None:
        return False, "the full-residual ceiling was not run, so no arm's null is readable"
    if floor is not None and ceiling.transfer_fraction <= floor.transfer_fraction:
        return False, (
            f"the full-residual ceiling ({ceiling.transfer_fraction:+.3f}) does not clear the "
            f"magnitude-matched random floor ({floor.transfer_fraction:+.3f}). Swapping the WHOLE "
            "residual is the largest shift this pair set can produce at the magnitude the arms "
            "work at, so a rank-1 component of it could not have been seen either"
        )
    mde = _mde80(_cell_means(np.asarray(ceiling.per_pair_shift), cells))
    if not mde < GATE_MDE_FRACTION * abs(gate.mean_natural_gap):
        return False, (
            f"the ceiling arm's MDE80 ({mde:.4f}) is not below {GATE_MDE_FRACTION:g} x the natural "
            f"gap ({abs(gate.mean_natural_gap):.4f}), so a half-sized transfer would not have been "
            "detectable in the arms even though B0 passed on the unpatched gap"
        )
    return True, ""


def verdict_cap(
    *,
    gate_passed: bool,
    corruption_clean: bool,
    g0_passed: bool,
    reachable: bool,
    arms: tuple[ArmTransfer, ...],
    ceiling_readable: bool = True,
    ceiling_note: str = "",
) -> str:
    """The one string that says what this run licensed.

    The RULE is written before the numbers arrive; the verdict is not. An earlier draft took only
    the three gate booleans and so wrote "functionally-used, pilot-suggestive" onto a run whose
    every arm read zero — its own prose promised a comparison against the floor and the no-op that
    nothing computed. The comparison is made here, on the arm table.

    A table rather than a chain of ifs, because the ORDER is the design: the first unmet condition
    wins, and it is ordered from "the instrument is not established" down to "the instrument is fine
    and the effect is not there". Reading it top to bottom is the whole routing rule.
    """

    beats_controls = _arm_beats_its_controls(arms)
    routes: tuple[tuple[bool, str], ...] = (
        (
            g0_passed,
            "harness_inadequate — the E0 G0 sensitivity gate did NOT pass, so the emotion basis "
            "the off-position free-rider readout depends on is not established. The behavioural "
            "leg still reads on its own logit margin, but nothing here adjudicates anything about "
            "emotion geometry.",
        ),
        (
            reachable,
            "harness_inadequate — the reachability control FAILED: patching the reveal token does "
            "not move the answer slot even where the donor's own bookkeeping predicts it should. "
            "Nothing below distinguishes 'the expectation state is not used' from 'this surface "
            "carries nothing across positions'. The functional-use claim stays OPEN.",
        ),
        (
            gate_passed,
            "harness_inadequate for the window — B0 failed, so the transfer fraction has no "
            "measured denominator and no patched number below it is readable in either "
            "direction. The functional-use claim stays OPEN.",
        ),
        (
            beats_controls or ceiling_readable,
            "harness_inadequate for the arms — the certified arm does not outrun its controls, but "
            f"neither could anything: {ceiling_note}. Reachability passed at the extreme cross-cell "
            "magnitude, which does not establish resolution at the within-cell magnitude the arms "
            "actually inject. So this is NOT a readable null — it does not distinguish 'the "
            "direction is not used' from 'the design could not have seen it'. The functional-use "
            "claim stays OPEN.",
        ),
        (
            beats_controls,
            "no transfer — the surface is reachable and the window moves, but the certified "
            "signed-RPE arm does not outrun BOTH the magnitude-matched random floor and the "
            "same-condition no-op, or does not clear its own cell-clustered null. This IS a "
            "readable null for the certified direction on this model and recipe: the denominator "
            "was measured, the floor is perturbation-matched, the full-residual ceiling shows the "
            "design can see a shift at the magnitude the arms inject, and the readout sits where "
            "the identity path cannot reach it. It says nothing about whether some other direction "
            "carries the effect.",
        ),
        (
            corruption_clean,
            "functionally-used, CONFOUNDED — a window whose correct answer is invariant across "
            "the pair moved under the patch, so the choice shift may be context corruption "
            "rather than transfer. Report both and claim neither.",
        ),
    )
    for holds, verdict in routes:
        if not holds:
            return verdict
    return (
        "functionally-used, pilot-suggestive — the strongest tier this project licenses, and only "
        "on THIS model/surface/recipe. The certified-direction arm outruns BOTH the random floor "
        "and the same-condition no-op AND clears its own cell-clustered null, and the full-residual "
        "ceiling shows the design could see a shift at the magnitude injected. The readout sits at "
        "a LATER token than the patch, so unlike E3's same-position reading it cannot be produced "
        "by the residual stream's identity path. Identification limit unchanged; no welfare, "
        "sentience or experience claim follows."
    )


# --------------------------------------------------------------------------------------
# Entry point: artifacts in, one report out
# --------------------------------------------------------------------------------------


def behavioral_transfer(
    states_artifact: Path | str,
    battery_file: Path | str,
    directions_artifact: Path | str,
    emotion_artifact: Path | str,
    *,
    backend: ChatPatchingBackend,
    model_key: str | None = None,
    block: int | None = None,
    seed: int = 7,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    max_per_cell: int = DEFAULT_MAX_PER_CELL,
    n_random_draws: int = DEFAULT_RANDOM_DRAWS,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    n_reachability_pairs: int = DEFAULT_REACHABILITY_PAIRS,
    answer_candidates: tuple[str, ...] = ANSWER_SYMBOL_CANDIDATES,
    answer_form: AnswerForm = DEFAULT_ANSWER_FORM,
) -> BehavioralTransferReport:
    """E4 end to end: preflight, show the slot is reachable, gate on B0, then spend the arms.

    The order is the design's and is not an implementation detail.

    Everything that can fail for free fails first: the answer pool is audited on the tokenizer, and
    every probe this run will ever build is constructed and its answer tokens checked, before a
    single forward. Both cost zero GPU and both would otherwise raise hours in, after which
    ``cli_emotions`` never reaches ``write_json`` and the whole run is lost.

    Then the reachability control, which runs even when B0 fails — §5's discard clause forbids
    spending the patched ARMS against an unmeasured denominator, and this is not that. It is the
    60-forward measurement that tells the operator whether a flat run means the model or the
    instrument, which is precisely the question a failed B0 raises.
    """

    states = read_reveal_rpe_states(Path(states_artifact))
    battery = RevealProbeBattery.model_validate(read_json(Path(battery_file)))
    directions = read_reveal_rpe_directions(Path(directions_artifact))
    emotion = read_emotion_vectors(Path(emotion_artifact))
    # The DIRECTIONS artifact's block, not the emotion artifact's. The thing patched here is the
    # certified signed-RPE direction, and "certified" is a property of the block it was selected
    # at (``source_verdict`` is recorded beside it); read it at any other block and the arm injects
    # an uncertified vector under the certified arm's name. The emotion artifact's block answers a
    # different question (where the emotion basis separates) and on the run of record the two
    # disagree -- 35 vs 63 -- with 63 out of range for a patch that must leave a deeper readout.
    patch_block = directions.metadata.selected_block if block is None else block
    n_blocks = states.metadata.n_blocks
    states_sha256 = states.metadata.states_sha256
    if not 0 <= patch_block < n_blocks - 1:
        raise ValueError(
            f"patch block {patch_block} must leave a deeper readout block "
            f"(n_blocks={n_blocks}); patching the deepest block would leave "
            "nothing downstream to read"
        )
    reveals = align_reveals(states, battery)
    pairs, symbol_matched = build_patch_pairs(
        reveals, max_pairs=max_pairs, max_per_cell=max_per_cell
    )
    preflight = _preflight_pool(backend, reveals, answer_candidates)
    answer_pool = preflight.valid
    # ascontiguousarray, not asarray: a basic slice on axis 1 is a VIEW, and ``np.asarray`` on an
    # ndarray is a no-op, so the 81 MB actually wanted would keep all 5.2 GB of ``states.states``
    # alive for the whole run -- beside 27B of weights. Copying breaks the reference; ``states`` is
    # dropped once the two scalars still needed are in locals.
    block_states = np.ascontiguousarray(states.states[:, patch_block, :])
    del states
    v_rpe = unit_vector(directions.directions[0, patch_block, :])
    # The patch block itself and the deepest block. The patch-block row is a free control rather
    # than a readout: the patch sits at an EARLIER position of the SAME block, and no path inside
    # one block moves a later position, so anything but ~0 there means the patch leaked somewhere
    # the design does not model.
    readout_blocks = (patch_block, n_blocks - 1)
    axes_by_block = {
        readout: {
            name: axis for name, (axis, _description) in emotion_axes(emotion, readout).items()
        }
        for readout in readout_blocks
    }

    _preflight_probes(
        backend,
        reveals,
        pairs,
        answer_pool=answer_pool,
        answer_form=answer_form,
        n_reachability_pairs=n_reachability_pairs,
    )
    # ONE forward, and its UNEXPECTED branch declares every number below it unverified -- so it
    # runs before the 2,880 the gate costs, not after them. The §5 discard clause it used to sit
    # behind is about the patched ARMS; a single forward is far under that bar, and the same
    # reasoning already exempts the 60-forward reachability control on the next line.
    wiring = _wiring_check(
        backend,
        reveals,
        pairs,
        block_states,
        patch_block=patch_block,
        answer_pool=answer_pool,
        answer_form=answer_form,
    )
    reachability = reachability_control(
        backend,
        reveals,
        block_states,
        patch_block=patch_block,
        readout_blocks=readout_blocks,
        answer_pool=answer_pool,
        seed=seed,
        n_permutations=n_permutations,
        n_pairs=n_reachability_pairs,
        answer_form=answer_form,
    )
    gate, baselines = sensitivity_gate(
        backend,
        reveals,
        pairs,
        block_states,
        patch_block=patch_block,
        readout_blocks=readout_blocks,
        answer_pool=answer_pool,
        seed=seed,
        n_permutations=n_permutations,
        axes_by_block=axes_by_block,
        answer_form=answer_form,
    )
    arms: tuple[ArmTransfer, ...] = ()
    corruption: tuple[CorruptionControl, ...] = ()
    # Both conditions, because ``verdict_cap`` refuses to READ the arm table under either. A failed
    # B0 means the transfer fraction has no measured denominator; a failed reachability means
    # nothing crosses positions on this surface at all, so a flat arm table would be a fact about
    # the instrument. Spending 4,560 patched forwards to fill in a table the verdict will not look
    # at is the pure form of the dual (CLAUDE.md): harness cost capped by run cost.
    spend_arms = gate.passed and reachability.passed
    unspent = (
        "not run: B0 did not pass, so no patched forward was spent (e4-prereg §5)"
        if not gate.passed
        else "not run: the reachability control found no cross-position path, so an arm table "
        "would describe the instrument rather than the direction (e4-prereg §5)"
    )
    corruption_clean, corruption_note = False, unspent
    ceiling_readable, ceiling_note = False, unspent
    if spend_arms:
        arms = behavioral_arms(
            backend,
            reveals,
            pairs,
            block_states,
            gate,
            baselines,
            v_rpe,
            patch_block=patch_block,
            readout_blocks=readout_blocks,
            answer_pool=answer_pool,
            seed=seed,
            n_permutations=n_permutations,
            axes_by_block=axes_by_block,
            n_random_draws=n_random_draws,
            answer_form=answer_form,
        )
        corruption = corruption_controls(
            backend,
            reveals,
            pairs,
            block_states,
            v_rpe,
            patch_block=patch_block,
            readout_blocks=readout_blocks,
            answer_pool=answer_pool,
            choice_natural_gap=gate.mean_natural_gap,
            answer_form=answer_form,
        )
        corruption_clean, corruption_note = corruption_reading(corruption, arms)
        ceiling_readable, ceiling_note = _ceiling_is_readable(arms, gate, _cell_groups(pairs))
    g0 = emotion.metadata.gate_verdict
    return BehavioralTransferReport(
        seed=seed,
        patch_block=patch_block,
        direction_block=directions.metadata.selected_block,
        direction_verdict=directions.metadata.source_verdict,
        readout_blocks=readout_blocks,
        n_pairs=len(pairs),
        n_cells=len({pair.reward_cell_id for pair in pairs}),
        n_random_draws=n_random_draws if spend_arms else 0,
        symbol_matched=symbol_matched,
        symbol_confound=None if symbol_matched else SYMBOL_CONFOUND,
        answer_pool=answer_pool,
        answer_pool_note=ANSWER_POOL_NOTE,
        answer_form=answer_form,
        answer_pool_preflight=preflight,
        no_op_surface_match_rate=_surface_match_rate(reveals, pairs),
        states_sha256=states_sha256,
        battery_sha256=file_sha256(Path(battery_file)),
        directions_sha256=directions.metadata.directions_sha256,
        emotion_vectors_sha256=emotion.metadata.vectors_sha256,
        model_key=model_key,
        sensitivity_gate=f"G0={g0}",
        wiring_check=wiring,
        reachability=reachability,
        gate=gate,
        arms=arms,
        corruption=corruption,
        corruption_clean=corruption_clean,
        corruption_note=corruption_note,
        cross_position_note=CROSS_POSITION_NOTE,
        identification_limit=IDENTIFICATION_LIMIT,
        ceiling_readable=ceiling_readable,
        ceiling_note=ceiling_note,
        verdict_cap=verdict_cap(
            gate_passed=gate.passed,
            corruption_clean=corruption_clean,
            g0_passed=g0 == "pass",
            reachable=reachability.passed,
            arms=arms,
            ceiling_readable=ceiling_readable,
            ceiling_note=ceiling_note,
        ),
    )


def _preflight_pool(
    backend: ChatPatchingBackend, reveals: tuple[Comparison, ...], candidates: tuple[str, ...]
) -> AnswerPoolPreflight:
    preflight = preflight_answer_symbols(
        backend, candidates=candidates, used_symbols=battery_symbols(reveals)
    )
    if not preflight.passed:
        raise ValueError(preflight.note)
    return preflight


def _preflight_probes(
    backend: ChatPatchingBackend,
    reveals: tuple[Comparison, ...],
    pairs: tuple[PatchPair, ...],
    *,
    answer_pool: tuple[str, ...],
    answer_form: AnswerForm,
    n_reachability_pairs: int,
) -> None:
    """Build every probe this run will ever build, and extend every row it will touch. Zero forwards.

    Constructing the probes is pure string work and each extension is a few tokenizer calls, so the
    whole sweep costs nothing and runs before the first forward. It exists because the alternative
    was measured: a probe that cannot be built, or a prompt that will not concatenate, raises inside
    ``read``, and with the corruption windows scheduled after the gate and every arm that raise
    lands hours in — at which point ``cli_emotions`` never reaches ``write_json`` and the run is
    lost whole (``docs/agents/gotchas.md``).

    The answer tokens are NOT re-checked here. ``_preflight_pool`` has already gated every pool
    symbol as single-token in BOTH forms, and every probe's answer tokens are pool symbols by
    construction, so a check here could not fire — a control that cannot fail is worse than none,
    because it reads as coverage.
    """

    # One probe per distinct row the run will extend -- recipients, donors and the reachability
    # rows. The dict is the work list for the extension sweep at the bottom.
    by_row: dict[int, DecisionProbe] = {}
    for index, pair in enumerate(pairs):
        recipient = reveals[pair.recipient_row]
        renderings = renderings_for(reveals, pair, answer_pool)
        for rendering in renderings:
            for certain in CERTAIN_LEVELS:
                # Constructed for the raise inside ``choice_probe``/``renderings_for``, which is
                # pair-dependent; only one per row is kept for the extension sweep.
                probe = choice_probe_for(
                    recipient, rendering=rendering, certain=certain, form=answer_form
                )
                by_row.setdefault(pair.recipient_row, probe)
        for window in CORRUPTION_WINDOWS:
            corruption_probe_for(
                recipient,
                window,
                rendering=renderings[index % len(renderings)],
                form=answer_form,
            )
        # The donor is extended too, and B0 reads it FIRST -- pair 0's donor is the run's very
        # first forward -- so it cannot be left out of the sweep.
        by_row.setdefault(
            pair.donor_row,
            choice_probe_for(
                reveals[pair.donor_row],
                rendering=renderings[0],
                certain=CERTAIN_LEVELS[0],
                form=answer_form,
            ),
        )
    symbol_a, symbol_b = answer_symbols(answer_pool, exclude=frozenset())
    for index, (donor_row, recipient_row) in enumerate(
        _reachability_pairs(reveals, n_reachability_pairs)
    ):
        by_row.setdefault(
            recipient_row,
            reachability_probe(
                symbol=str(reveals[recipient_row].metadata["realised_symbol"]),
                reward=float(reveals[recipient_row].metadata["reward"]),
                donor_reward=float(reveals[donor_row].metadata["reward"]),
                rendering=option_renderings(symbol_a, symbol_b)[index % 4],
                form=answer_form,
            ),
        )
    # EVERY distinct row, not just the first. The token-prefix guard inside ``extend`` is a
    # per-reveal property: the pinned prompt ends with that row's own symbol and the appended
    # remainder carries that row's own reward, so both sides of the BPE boundary vary across the
    # pair set and a re-merge on ANY row raises inside ``read``. Checking one row would leave the
    # other ~250 to fail up to 1,920 forwards in, with the artifact unwritable. ~250 tokenizer
    # round-trips, no forward.
    for row, probe in by_row.items():
        extend(backend, reveals[row], probe)


def _surface_match_rate(reveals: tuple[Comparison, ...], pairs: tuple[PatchPair, ...]) -> float:
    """How often the no-op control's donor shares the recipient's symbol and template family.

    ``_same_condition`` matches on (ev_cell, rpe_sign) only. Where the rate is low the arm varies
    the rendered surface as well as the condition, which makes it a control on ``full_residual``
    rather than on the rank-1 certified arm — a limit worth reporting rather than discovering.
    """

    eligible = [pair for pair in pairs if pair.same_condition_row is not None]
    if not eligible:
        return 0.0
    matched = sum(
        1
        for pair in eligible
        if (
            reveals[pair.same_condition_row].metadata["realised_symbol"]  # type: ignore[index]
            == reveals[pair.recipient_row].metadata["realised_symbol"]
            and reveals[pair.same_condition_row].metadata["template_family"]  # type: ignore[index]
            == reveals[pair.recipient_row].metadata["template_family"]
        )
    )
    return matched / len(eligible)


def _wiring_check(
    backend: ChatPatchingBackend,
    reveals: tuple[Comparison, ...],
    pairs: tuple[PatchPair, ...],
    block_states: np.ndarray,
    *,
    patch_block: int,
    answer_pool: tuple[str, ...],
    answer_form: AnswerForm,
) -> str:
    """One forward that reads the patch SITE, discharging the §8 obligation E3 left open.

    E3's forward artifact reports ``mean_shift = 0.0`` at the patch site for every arm including
    the full residual, which its own contract says must be the substituted value. The consistent
    reading is that ``output_hidden_states`` on this stack records the module's output BEFORE the
    forward hook replaces it. That does not touch anything downstream, but it means the row
    documented as a wiring check verifies nothing — so E4 measures which of the two it is and says
    so in the artifact instead of leaving a false semantics in place.
    """

    pair = pairs[0]
    recipient = reveals[pair.recipient_row]
    probe = choice_probe_for(
        recipient,
        rendering=option_renderings(answer_pool[0], answer_pool[1])[0],
        certain=CERTAIN_LEVELS[0],
        form=answer_form,
    )
    extended = extend(backend, recipient, probe)
    replacement = block_states[pair.donor_row]
    site = backend.patched_forward(
        extended.text,
        block=patch_block,
        position=extended.patch_position,
        replacement=replacement,
        layers=block_layers((patch_block,)),
        readout_position=extended.patch_position,
    )
    read_back = np.asarray(site.states, dtype=np.float64)[0]

    def matches(reference: np.ndarray) -> bool:
        # Relative, not absolute: late-block residual norms are large and these values have been
        # through bf16, so an absolute 1e-3 would report UNEXPECTED — which makes the report
        # declare every number in it unverified — on a correctly wired stack.
        scale = float(np.linalg.norm(reference))
        return bool(np.linalg.norm(read_back - reference) <= 1e-2 * max(scale, 1e-6))

    if matches(replacement):
        return (
            "POST-HOOK: reading hidden_states at the patch site returns the substituted value, so "
            "the documented semantics hold on this stack and the injection is confirmed."
        )
    if matches(block_states[pair.recipient_row]):
        return (
            "PRE-HOOK: reading hidden_states at the patch site returns the UNPATCHED value, so on "
            "this stack output_hidden_states records the module output before the forward hook's "
            "replacement (docs/agents/gotchas.md). Downstream and cross-position readouts are "
            "unaffected — they are computed from the hooked output — but no row may be reported "
            "as a patch-site wiring check."
        )
    return (
        "UNEXPECTED: the patch site reads back as neither the substituted value nor the recipient's "
        "own, so neither the post-hook nor the pre-hook reading explains it. Treat every number in "
        "this report as unverified until the discrepancy is explained."
    )


def format_behavioral_transfer_summary(report: BehavioralTransferReport) -> str:
    """A plain-text summary for stdout: the gate first, because it decides whether to read on."""

    gate = report.gate
    lines = [
        "E4 behavioural transfer — cross-position logit-margin readout",
        f"  patch block={report.patch_block} readout blocks={list(report.readout_blocks)} "
        f"seed={report.seed} pairs={report.n_pairs} cells={report.n_cells} "
        f"answers={list(report.answer_pool)}",
        f"  {report.sensitivity_gate}   {report.wiring_check.split(':')[0]}   "
        f"answer_form={report.answer_form}",
        "",
        f"Reachability positive control: {report.reachability.verdict.split(':')[0]}  "
        f"n={report.reachability.n_pairs}  mean shift={report.reachability.mean_shift:+.4f}  "
        f"p={report.reachability.p_value:.4f}  (predicted: negative)",
        "",
        f"B0 sensitivity gate (unpatched, certain={gate.selected_certain}):",
        f"  natural gap={gate.mean_natural_gap:+.4f}  p={gate.p_value:.4f}  "
        f"MDE80={gate.mde80:.4f}  bar={GATE_MDE_FRACTION * abs(gate.mean_natural_gap):.4f}  "
        f"passed={gate.passed}",
    ]
    lines += [
        f"    certain={level.certain:>3}  mean z={level.mean_baseline_margin:+.4f}  "
        f"|z|={level.mean_abs_baseline_margin:.4f}{'  <- selected' if level.selected else ''}"
        for level in gate.levels
    ]
    if report.arms:
        lines += [
            "",
            f"{'arm':<22}{'n':>4}{'mean shift':>13}{'norm. shift':>13}{'E3 transfer':>13}{'p':>9}",
        ]
        lines += [
            f"{arm.arm:<22}{arm.n_pairs:>4}{arm.mean_shift:>+13.4f}"
            f"{arm.normalised_shift:>+13.3f}{arm.transfer_fraction:>+13.3f}"
            f"{'' if arm.p_value is None else f'{arm.p_value:>9.4f}'}"
            for arm in report.arms
        ]
    if report.corruption:
        lines += ["", f"{'corruption control':<32}{'mean|shift|/|B0 gap|':>22}{'ok':>6}"]
        lines += [
            f"{control.arm + '/' + control.window:<32}"
            f"{'n/a' if control.relative_to_gap is None else f'{control.relative_to_gap:.3f}':>22}"
            f"{str(control.within_tolerance):>6}"
            for control in report.corruption
        ]
    lines += ["", f"VERDICT CAP: {report.verdict_cap}", f"CORRUPTION: {report.corruption_note}"]
    if report.symbol_confound:
        lines += [f"CONFOUND: {report.symbol_confound}"]
    return "\n".join(lines)


__all__ = [
    "BEHAVIORAL_TRANSFER_CONTRACT_VERSION",
    "CORRUPTION_ARMS",
    "CORRUPTION_TOLERANCE",
    "CROSS_POSITION_NOTE",
    "DEFAULT_MAX_PAIRS",
    "DEFAULT_MAX_PER_CELL",
    "DEFAULT_PERMUTATIONS",
    "DEFAULT_RANDOM_DRAWS",
    "DEFAULT_REACHABILITY_PAIRS",
    "GATE_ALPHA",
    "GATE_MDE_FRACTION",
    "IDENTIFICATION_LIMIT",
    "REACHABILITY_ALPHA",
    "ArmTransfer",
    "Baselines",
    "BehavioralTransferReport",
    "ChatPatchingBackend",
    "CorruptionControl",
    "ExtendedPrompt",
    "ReachabilityControl",
    "Readout",
    "SensitivityGate",
    "TitrationLevel",
    "behavioral_arms",
    "behavioral_transfer",
    "chat_tail",
    "choice_probe_for",
    "corruption_controls",
    "corruption_reading",
    "extend",
    "format_behavioral_transfer_summary",
    "read",
    "reachability_control",
    "sensitivity_gate",
    "verdict_cap",
]
