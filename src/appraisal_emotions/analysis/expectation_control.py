"""E2 — expectation vs situation on the certified reward-matched cells (design §4 E2).

NEW module (no parent counterpart). Pure CPU over three hash-bound inputs: the reveal-token
states artifact, the battery that produced it, and the E0 emotion basis. Zero new stimuli.

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
    "EXPECTATION_CONTROL_CONTRACT_VERSION",
    "AxisResult",
    "ExpectationControlReport",
    "expectation_control",
    "format_expectation_control_summary",
]

EXPECTATION_CONTROL_CONTRACT_VERSION = "expectation_control/v1"
PC1_AXIS = "pc1_affect_concept_valence"
PAIR_AXIS = "elated_minus_disappointed"
_MIN_CELL_ROWS = 2


@dataclass(frozen=True)
class _CellDesign:
    """Within-cell demeaned regressor and the cell membership needed for the cluster null."""

    rows: np.ndarray  # row indices of every scored reveal, grouped by cell
    cells: tuple[np.ndarray, ...]  # per-cell row-index arrays (into ``rows`` positions)
    rpe: np.ndarray  # signed RPE aligned to ``rows``


def _align_reveals(states: RevealRpeStates, battery: RevealProbeBattery) -> tuple[Comparison, ...]:
    """The battery reveals in the states artifact's capture row order (fail-closed on a mismatch)."""

    by_id = {comparison.comparison_id: comparison for comparison in battery.reveals}
    missing = [rid for rid in states.metadata.reveal_ids if rid not in by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} reveal id(s) in the states artifact are absent from the battery "
            "(e.g. " + ", ".join(missing[:3]) + "); these artifacts are from different runs."
        )
    return tuple(by_id[rid] for rid in states.metadata.reveal_ids)


def _cell_design(reveals: tuple[Comparison, ...]) -> _CellDesign:
    """Group reveals by ``reward_cell_id``, keeping cells with >=2 rows AND varying signed RPE."""

    grouped: dict[str, list[int]] = {}
    for row, comparison in enumerate(reveals):
        grouped.setdefault(str(comparison.metadata["reward_cell_id"]), []).append(row)
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
            "no reward-matched cell carries >=2 reveals with varying signed RPE; E2 has no "
            "within-cell contrast to read on this battery."
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


class ExpectationControlReport(StrictModel):
    """E2 record: does the emotion-probe readout track expectation with the outcome held fixed?"""

    artifact_contract_version: Literal["expectation_control/v1"] = (
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
    axes: tuple[AxisResult, ...]
    scope_note: str


_SCOPE_NOTE = (
    "The options block is NOT byte-identical inside a reward-matched cell — EV differs by "
    "construction. The varying text is numeric point values on a surface that passes the "
    "zero-emotion-lexicon audit: exactly the variation an expectation-tracker must use and a "
    "surface-affect detector should ignore."
)


def _axes(emotion: EmotionVectors, block: int) -> dict[str, tuple[np.ndarray, str]]:
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

    reveals = _align_reveals(states, battery)
    design = _cell_design(reveals)
    block_states = np.asarray(states.states[:, matched_block, :])[design.rows]

    results: list[AxisResult] = []
    for name, (axis, description) in _axes(emotion, matched_block).items():
        projection = block_states @ axis
        slope = _pooled_slope(projection, design, design.rpe)
        p_value = _cluster_permutation_p(
            projection,
            design,
            slope,
            rng=np.random.default_rng(seed_int(seed, "expectation-control", name, matched_block)),
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
        axes=tuple(results),
        scope_note=_SCOPE_NOTE,
    )


def _verdict_cap(gate: str) -> str:
    if gate != "pass":
        return (
            "harness_inadequate — the E0 G0 sensitivity gate did NOT pass, so the emotion axes "
            "read here are not established to carry valence structure. Neither a null nor a "
            "positive adjudicates the situational rival; the claim stays OPEN."
        )
    return (
        "present-and-separable, pilot-suggestive. A sign-congruent within-cell slope says the "
        "emotion-probe readout tracks the EXPECTATION manipulation with the outcome text fixed; "
        "a null within cells is what the situational-context rival predicts. Neither licenses "
        "any welfare / sentience / experience claim."
    )


def format_expectation_control_summary(report: ExpectationControlReport) -> str:
    """A plain-text summary table for stdout."""

    lines = [
        "E2 expectation-control — outcome fixed, expectation varies (present-and-separable)",
        f"  block={report.block} seed={report.seed} perms={report.n_permutations} "
        f"alpha={report.alpha}",
        f"  sensitivity: {report.sensitivity_gate}",
        f"  verdict cap: {report.verdict_cap}",
        "",
        "  axis                              pooled beta   mean cell beta  cells  rows       p  "
        "expectation?",
    ]
    for axis in report.axes:
        lines.append(
            f"  {axis.axis:<32} {axis.pooled_within_cell_slope:+11.5f}   "
            f"{axis.mean_per_cell_slope:+13.5f}  {axis.n_cells:>5}  {axis.n_rows:>4}  "
            f"{axis.p_value:6.4f}  {'YES' if axis.reads_expectation else 'no'}"
        )
    lines.extend(["", f"  scope: {report.scope_note}"])
    return "\n".join(lines)
