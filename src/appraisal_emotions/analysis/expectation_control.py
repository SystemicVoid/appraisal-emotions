"""E2 + E2b — expectation vs situation, and expectation vs comparison (design §4 E2;
``docs/design/e2b-prereg.md``).

NEW module (no parent counterpart). Pure CPU over three hash-bound inputs: the reveal-token
states artifact, the battery that produced it, and the E0 emotion basis. Zero new stimuli.

**Two cell families, one estimator.** Write the readout as linear in the regressors that span
the value plane, ``projection ≈ a·reward + b·ev``. Each matched family fixes one of them, so the
within-cell slope on ``signed_rpe = reward − ev`` recovers one coefficient:

- ``reward_cell_id`` (E2) holds the realised outcome fixed and varies stated EV ⇒ recovers ``−b``;
- ``ev_cell_id`` (E2b) holds stated EV fixed and varies the realised outcome ⇒ recovers ``+a``.

E2 alone cannot license the word *comparison*: a readout that represents the stated EV and never
compares it to anything predicts E2's result exactly. The conjunction is the estimand — a
comparison carries ``reward − ev``, hence ``a = −b``, hence BOTH arms positive. One arm positive
and the other null names which rival survives. Both arms run through the same code with the
grouping key as the only difference, so a gap between them cannot be an estimator difference.

The rival this answers is Peiris (arXiv:2604.13466): emotion vectors may be projections of
*situational context* onto human-emotion axes rather than appraisal-tracking states. The
inherited battery adjudicates it for free — inside a ``reward_cell_id`` group the **realised
outcome is held fixed** while the stated EV varies, so the same reveal token carries a different
signed RPE. Project the reveal states onto an emotion axis and regress on ``signed_rpe`` WITHIN
those cells:

- expectation-tracking readout ⇒ a nonzero, sign-congruent within-cell slope;
- situational-affect readout ⇒ the projection follows the outcome text, so the within-cell slope
  is ~0.

*Honest scope note (design §4 E2):* the options block is not byte-identical inside a matched cell
— EV differs by construction. The varying text is numeric point values on a surface that passes
the zero-emotion-lexicon audit, which is exactly the variation an expectation-tracker must use
and a surface-affect detector should ignore.

Two axes, both read at the matched block:

- ``pc1`` — the affect-concept valence axis (PC1 of the emotion basis), oriented toward positive
  valence by :func:`analysis.emotion_vectors.valence_oriented_pc_axes`, so a positive slope on
  ``signed_rpe`` is the sign-congruent direction;
- ``elated_minus_disappointed`` — the design names ``e_disappointed − e_elated``; it is stored
  and reported with the sign FLIPPED to elated-minus-disappointed so that, like PC1, positive
  means better-than-expected. The flip is a labelling choice recorded in the report, not a
  post-hoc orientation: it is fixed here, before any data.

Inference is cluster-aware: the permutation flips the sign of a whole cell's ``signed_rpe`` at
once (a Rademacher draw per cell), because rows inside a cell share a stimulus and are not
independent. Two-sided, since only the situational rival predicts a specific value (zero).

Licence: present-and-separable at most (design §7). E2 adjudicates between two *representational*
readings; it has no causal arm and licenses no welfare / sentience / experience claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from appraisal_emotions.analysis.direction_stats import seed_int
from appraisal_emotions.analysis.emotion_vectors import (
    EmotionVectors,
    read_emotion_vectors,
    valence_oriented_pc_axes,
)
from appraisal_emotions.analysis.reveal_rpe import RevealRpeStates, read_reveal_rpe_states
from appraisal_emotions.core.schema import Comparison, StrictModel
from appraisal_emotions.core.stats import add_one_p
from appraisal_emotions.core.util import EXTRACTION_SEED, file_sha256, read_json, unit_vector
from appraisal_emotions.stimuli.reveal_probes import RevealProbeBattery

__all__ = [
    "CELL_FAMILIES",
    "EXPECTATION_CONTROL_CONTRACT_VERSION",
    "PAIR_AXIS",
    "PC1_AXIS",
    "ArmResult",
    "AxisResult",
    "ExpectationControlReport",
    "align_reveals",
    "emotion_axes",
    "expectation_control",
    "format_expectation_control_summary",
]

EXPECTATION_CONTROL_CONTRACT_VERSION = "expectation_control/v2"
PC1_AXIS = "pc1_affect_concept_valence"
PAIR_AXIS = "elated_minus_disappointed"
_MIN_CELL_ROWS = 2


@dataclass(frozen=True)
class CellFamily:
    """One matched-cell grouping: which regressor it pins, and which coefficient it recovers."""

    name: str
    metadata_key: str
    held_fixed: str
    varies: str
    recovers: str
    scope_note: str


# Order is the reporting order and is fixed here, before data: E2 first because it is the arm
# that has already run and whose numbers the v1 artifact published.
CELL_FAMILIES: tuple[CellFamily, ...] = (
    CellFamily(
        name="reward_matched",
        metadata_key="reward_cell_id",
        held_fixed="realised reward and |RPE|",
        varies="stated EV",
        recovers="-b (the EV coefficient), because reward is pinned inside the cell",
        scope_note=(
            "The options block is NOT byte-identical inside a reward-matched cell — EV differs "
            "by construction. The varying text is numeric point values on a surface that passes "
            "the zero-emotion-lexicon audit: exactly the variation an expectation-tracker must "
            "use and a surface-affect detector should ignore."
        ),
    ),
    CellFamily(
        name="ev_matched",
        metadata_key="ev_cell_id",
        held_fixed="stated EV and |RPE|",
        varies="the realised outcome",
        recovers="+a (the reward coefficient), because EV is pinned inside the cell",
        scope_note=(
            "The realised outcome SYMBOL differs inside an EV-matched cell — that is what makes "
            "the outcome vary — so unlike the reward-matched arm the surface is not "
            "outcome-matched. Symbol identity is put in the intercept rather than the slope by "
            "the battery's balanced rendering (both orders, both strata, four templates at equal "
            "weight) on neutrality-calibrated symbols. This arm is the complement of the "
            "reward-matched one, not a replacement: that arm pins the surface and varies the "
            "expectation, this one pins the expectation and varies the surface with the outcome."
        ),
    ),
)


@dataclass(frozen=True)
class _CellDesign:
    """Within-cell demeaned regressor and the cell membership needed for the cluster null."""

    rows: np.ndarray  # row indices of every scored reveal, grouped by cell
    cells: tuple[np.ndarray, ...]  # per-cell row-index arrays (into ``rows`` positions)
    rpe: np.ndarray  # signed RPE aligned to ``rows``


def align_reveals(states: RevealRpeStates, battery: RevealProbeBattery) -> tuple[Comparison, ...]:
    """The battery reveals in the states artifact's capture row order (fail-closed on a mismatch).

    Shared with E3 patching (``analysis.activation_patching``): both tiers read the same battery
    through the same row alignment, so a mismatch fails the same way in both.
    """

    by_id = {comparison.comparison_id: comparison for comparison in battery.reveals}
    missing = [rid for rid in states.metadata.reveal_ids if rid not in by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} reveal id(s) in the states artifact are absent from the battery "
            "(e.g. " + ", ".join(missing[:3]) + "); these artifacts are from different runs."
        )
    return tuple(by_id[rid] for rid in states.metadata.reveal_ids)


def _cell_design(reveals: tuple[Comparison, ...], family: CellFamily) -> _CellDesign:
    """Group reveals by the family's cell key, keeping cells with >=2 rows AND varying signed RPE."""

    grouped: dict[str, list[int]] = {}
    for row, comparison in enumerate(reveals):
        grouped.setdefault(str(comparison.metadata[family.metadata_key]), []).append(row)
    rows: list[int] = []
    cells: list[np.ndarray] = []
    for members in grouped.values():
        rpe = np.asarray([float(reveals[row].metadata["signed_rpe"]) for row in members])
        if len(members) < _MIN_CELL_ROWS or float(rpe.std()) == 0.0:
            # A cell with one row, or with no EV variation, carries no within-cell contrast.
            continue
        cells.append(np.arange(len(rows), len(rows) + len(members)))
        rows.extend(members)
    if not cells:
        raise ValueError(
            f"no {family.name} cell carries >=2 reveals with varying signed RPE; this arm has "
            "no within-cell contrast to read on this battery."
        )
    row_index = np.asarray(rows)
    rpe = np.asarray([float(reveals[row].metadata["signed_rpe"]) for row in row_index])
    return _CellDesign(rows=row_index, cells=tuple(cells), rpe=rpe)


def _demean_by_cell(values: np.ndarray, cells: tuple[np.ndarray, ...]) -> np.ndarray:
    out = np.array(values, dtype=np.float64, copy=True)
    for cell in cells:
        out[cell] -= out[cell].mean()
    return out


def _pooled_slope(projection: np.ndarray, design: _CellDesign, rpe: np.ndarray) -> float:
    """Within-cell (fixed-effects) pooled slope of ``projection`` on ``rpe``."""

    x = _demean_by_cell(rpe, design.cells)
    y = _demean_by_cell(projection, design.cells)
    denominator = float(np.dot(x, x))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(x, y) / denominator)


def _per_cell_slopes(projection: np.ndarray, design: _CellDesign) -> list[float]:
    slopes: list[float] = []
    for cell in design.cells:
        x = design.rpe[cell] - design.rpe[cell].mean()
        y = projection[cell] - projection[cell].mean()
        denominator = float(np.dot(x, x))
        if denominator > 0.0:
            slopes.append(float(np.dot(x, y) / denominator))
    return slopes


def _cluster_permutation_p(
    projection: np.ndarray,
    design: _CellDesign,
    observed: float,
    *,
    rng: np.random.Generator,
    n_permutations: int,
) -> float:
    """Two-sided cluster-aware p: flip the sign of each cell's whole RPE vector, seeded."""

    exceedances = 0
    for _ in range(n_permutations):
        flips = rng.integers(0, 2, size=len(design.cells)) * 2 - 1
        permuted = design.rpe.copy()
        for cell, flip in zip(design.cells, flips, strict=True):
            permuted[cell] = permuted[cell] * float(flip)
        if abs(_pooled_slope(projection, design, permuted)) >= abs(observed):
            exceedances += 1
    return add_one_p(exceedances, n_permutations)


class AxisResult(StrictModel):
    """One emotion-space axis regressed on signed RPE inside outcome-fixed cells."""

    axis: str
    description: str
    pooled_within_cell_slope: float
    mean_per_cell_slope: float
    n_cells: int
    n_rows: int
    p_value: float
    sign_congruent: bool
    reads_expectation: bool


class ArmResult(StrictModel):
    """One matched-cell family: what it pins, what it recovers, and its per-axis slopes."""

    cell_family: str
    cell_key: str
    held_fixed: str
    varies: str
    recovers: str
    scope_note: str
    axes: tuple[AxisResult, ...]


class ComparisonSignature(StrictModel):
    """Whether one axis shows the conjunction a comparison predicts, and what it says if not.

    ``holds`` is the recorded expectation of ``docs/design/e2b-prereg.md`` §2 — BOTH arms
    sign-congruent and below alpha on this axis. The reading is deliberately spelled out per axis
    rather than left to the writeup, because the whole point of the rung is that a single
    positive arm has already been over-read once.
    """

    axis: str
    reward_matched_slope: float
    ev_matched_slope: float
    reward_matched_reads: bool
    ev_matched_reads: bool
    holds: bool
    reading: str


class ExpectationControlReport(StrictModel):
    """E2+E2b record: does the emotion-probe readout track expectation, and does it compare?"""

    artifact_contract_version: Literal["expectation_control/v2"] = (
        EXPECTATION_CONTROL_CONTRACT_VERSION
    )
    seed: int
    n_permutations: int
    alpha: float
    block: int
    states_sha256: str
    battery_sha256: str
    emotion_vectors_sha256: str
    emotion_selected_block: int
    sensitivity_gate: str
    verdict_cap: str
    arms: tuple[ArmResult, ...]
    comparison_signature: tuple[ComparisonSignature, ...]


def emotion_axes(emotion: EmotionVectors, block: int) -> dict[str, tuple[np.ndarray, str]]:
    """The two emotion-space readout axes at ``block`` — shared by E2 and E3 patching."""

    words = emotion.word_vectors[:, block, :]
    components, _scores = valence_oriented_pc_axes(words, emotion.word_valence)
    pair = emotion.row("elated")[block] - emotion.row("disappointed")[block]
    return {
        PC1_AXIS: (
            unit_vector(components[0]),
            "PC1 of the emotion basis, oriented toward positive affect-concept valence.",
        ),
        PAIR_AXIS: (
            unit_vector(pair),
            "The design's e_disappointed - e_elated contrast, sign-flipped so positive means "
            "better-than-expected (fixed before data; see module docstring).",
        ),
    }


def expectation_control(
    states_artifact: Path,
    battery_file: Path,
    emotion_artifact: Path,
    *,
    block: int | None = None,
    seed: int = EXTRACTION_SEED,
    n_permutations: int = 10_000,
    alpha: float = 0.05,
) -> ExpectationControlReport:
    """Run E2: project reveal states onto emotion axes, regress on signed RPE within cells."""

    states = read_reveal_rpe_states(Path(states_artifact))
    battery = RevealProbeBattery.model_validate(read_json(Path(battery_file)))
    emotion = read_emotion_vectors(Path(emotion_artifact))
    if states.metadata.hidden_size != emotion.metadata.hidden_size:
        raise ValueError("reveal states and emotion vectors have different hidden sizes")
    if states.metadata.n_blocks != emotion.metadata.n_blocks:
        raise ValueError("reveal states and emotion vectors have different block counts")
    matched_block = emotion.metadata.selected_block if block is None else block
    if not 0 <= matched_block < states.metadata.n_blocks:
        raise ValueError(f"block {matched_block} out of range for {states.metadata.n_blocks}")

    reveals = align_reveals(states, battery)
    all_states = np.asarray(states.states[:, matched_block, :])
    axes = emotion_axes(emotion, matched_block)

    arms = tuple(
        _arm(
            family,
            reveals,
            all_states,
            axes,
            block=matched_block,
            seed=seed,
            n_permutations=n_permutations,
            alpha=alpha,
        )
        for family in CELL_FAMILIES
    )

    gate = emotion.metadata.gate_verdict
    return ExpectationControlReport(
        seed=seed,
        n_permutations=n_permutations,
        alpha=alpha,
        block=matched_block,
        states_sha256=states.metadata.states_sha256,
        battery_sha256=file_sha256(Path(battery_file)),
        emotion_vectors_sha256=emotion.metadata.vectors_sha256,
        emotion_selected_block=emotion.metadata.selected_block,
        sensitivity_gate=f"G0={gate}",
        verdict_cap=_verdict_cap(gate),
        arms=arms,
        comparison_signature=_comparison_signature(arms),
    )


def _arm(
    family: CellFamily,
    reveals: tuple[Comparison, ...],
    all_states: np.ndarray,
    axes: dict[str, tuple[np.ndarray, str]],
    *,
    block: int,
    seed: int,
    n_permutations: int,
    alpha: float,
) -> ArmResult:
    """One cell family through the estimator — the ONLY difference between arms is the grouping."""

    design = _cell_design(reveals, family)
    block_states = all_states[design.rows]
    results: list[AxisResult] = []
    for name, (axis, description) in axes.items():
        projection = block_states @ axis
        slope = _pooled_slope(projection, design, design.rpe)
        p_value = _cluster_permutation_p(
            projection,
            design,
            slope,
            rng=np.random.default_rng(
                seed_int(seed, "expectation-control", family.name, name, block)
            ),
            n_permutations=n_permutations,
        )
        per_cell = _per_cell_slopes(projection, design)
        results.append(
            AxisResult(
                axis=name,
                description=description,
                pooled_within_cell_slope=slope,
                mean_per_cell_slope=float(np.mean(per_cell)) if per_cell else 0.0,
                n_cells=len(design.cells),
                n_rows=int(design.rows.size),
                p_value=p_value,
                sign_congruent=slope > 0.0,
                reads_expectation=bool(p_value < alpha and slope > 0.0),
            )
        )
    return ArmResult(
        cell_family=family.name,
        cell_key=family.metadata_key,
        held_fixed=family.held_fixed,
        varies=family.varies,
        recovers=family.recovers,
        scope_note=family.scope_note,
        axes=tuple(results),
    )


_READINGS = {
    (True, True): (
        "COMPARISON SIGNATURE. Both arms are sign-congruent and below alpha, which is what a "
        "readout carrying reward - ev predicts (a = -b). Neither a pure expectation tracker "
        "(predicts the EV-matched arm at zero) nor a pure outcome tracker (predicts the "
        "reward-matched arm at zero) survives both. Representational only: this says the readout "
        "carries the comparison, not that anything uses it."
    ),
    (True, False): (
        "EXPECTATION TRACKER NOT EXCLUDED. The reward-matched arm reads, the EV-matched arm does "
        "not. That is exactly what a readout representing the stated EV, with no comparison to "
        "the outcome at all, predicts. Writeups say 'tracks the stated expectation' and NOT "
        "'compares outcome against expectation' on this axis."
    ),
    (False, True): (
        "OUTCOME TRACKER NOT EXCLUDED. The EV-matched arm reads, the reward-matched arm does not "
        "— the readout may be following the realised outcome. This contradicts the published "
        "reward-matched result and should be debugged before it is reported."
    ),
    (False, False): (
        "NEITHER ARM READS on this axis. Nothing here adjudicates the rivals; with G0 passed this "
        "is a readable null for the emotion-axis readout at this block, and with G0 failed it is "
        "harness_inadequate."
    ),
}


def _comparison_signature(arms: tuple[ArmResult, ...]) -> tuple[ComparisonSignature, ...]:
    """Read the two arms together, per axis — the conjunction is the estimand (e2b-prereg §2)."""

    by_family = {arm.cell_family: {axis.axis: axis for axis in arm.axes} for arm in arms}
    reward, ev_matched = by_family["reward_matched"], by_family["ev_matched"]
    signatures: list[ComparisonSignature] = []
    for name, reward_axis in reward.items():
        ev_axis = ev_matched[name]
        key = (reward_axis.reads_expectation, ev_axis.reads_expectation)
        signatures.append(
            ComparisonSignature(
                axis=name,
                reward_matched_slope=reward_axis.pooled_within_cell_slope,
                ev_matched_slope=ev_axis.pooled_within_cell_slope,
                reward_matched_reads=reward_axis.reads_expectation,
                ev_matched_reads=ev_axis.reads_expectation,
                holds=all(key),
                reading=_READINGS[key],
            )
        )
    return tuple(signatures)


def _verdict_cap(gate: str) -> str:
    if gate != "pass":
        return (
            "harness_inadequate — the E0 G0 sensitivity gate did NOT pass, so the emotion axes "
            "read here are not established to carry valence structure. Neither a null nor a "
            "positive adjudicates the situational rival; the claim stays OPEN."
        )
    return (
        "present-and-separable, pilot-suggestive. A sign-congruent within-cell slope on the "
        "reward-matched arm says the emotion-probe readout tracks the EXPECTATION manipulation "
        "with the outcome text fixed; a null there is what the situational-context rival "
        "predicts. Only the CONJUNCTION with the EV-matched arm licenses the word 'comparison' — "
        "read comparison_signature, not one arm. Nothing here is a causal claim, and none of it "
        "licenses any welfare / sentience / experience claim."
    )


def format_expectation_control_summary(report: ExpectationControlReport) -> str:
    """A plain-text summary table for stdout."""

    lines = [
        "E2 + E2b expectation-control — two matched-cell families, one estimator "
        "(present-and-separable)",
        f"  block={report.block} seed={report.seed} perms={report.n_permutations} "
        f"alpha={report.alpha}",
        f"  sensitivity: {report.sensitivity_gate}",
        f"  verdict cap: {report.verdict_cap}",
    ]
    for arm in report.arms:
        lines.extend(
            [
                "",
                f"  [{arm.cell_family}] cells keyed on {arm.cell_key} — holds {arm.held_fixed} "
                f"fixed, varies {arm.varies}",
                f"    recovers {arm.recovers}",
                "",
                "    axis                              pooled beta   mean cell beta  cells  rows"
                "       p  reads?",
            ]
        )
        for axis in arm.axes:
            lines.append(
                f"    {axis.axis:<32} {axis.pooled_within_cell_slope:+11.5f}   "
                f"{axis.mean_per_cell_slope:+13.5f}  {axis.n_cells:>5}  {axis.n_rows:>4}  "
                f"{axis.p_value:6.4f}  {'YES' if axis.reads_expectation else 'no'}"
            )
        lines.append(f"    scope: {arm.scope_note}")

    lines.extend(["", "  comparison signature (both arms read together — the estimand):"])
    for signature in report.comparison_signature:
        lines.extend(
            [
                f"    {signature.axis}: reward-matched {signature.reward_matched_slope:+.5f} "
                f"({'reads' if signature.reward_matched_reads else 'null'}), "
                f"EV-matched {signature.ev_matched_slope:+.5f} "
                f"({'reads' if signature.ev_matched_reads else 'null'}) "
                f"=> holds={signature.holds}",
                f"      {signature.reading}",
            ]
        )
    return "\n".join(lines)
