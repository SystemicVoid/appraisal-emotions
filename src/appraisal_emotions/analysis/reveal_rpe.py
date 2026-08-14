"""R-A reveal-token signed-RPE estimator (rq8_rpe_valence_spec §4.4; Rung R-A plan).

Extracted from functional-valence-validity src/fv_validity/analysis/reveal_rpe.py @ 10c4662.
The estimator is byte-identical: gate constants, capture convention, the selection-aware sign
null, the cross-fitted matched-sign contests, the orientation signature, the report schema and
the verdict map are all unchanged, so a re-extraction here reproduces the R-A′ recipe exactly.

Kept: ``capture_reveal_states``, the hash-bound states artifact, ``selection_aware_sign_null``,
``matched_sign_contest``, ``orientation_signature``, ``RevealRpeReport``, ``analyze_reveal_rpe``,
``_verdict``, and the directions artifact (npz payload + JSON metadata bound by
``states_sha256`` / ``battery_sha256`` / ``selected_block`` / ``source_verdict``).

Dropped: the provenance-hashed metadata wrapper (``ProvenanceHashedModel``) and its
version-dispatched legacy readers. This module writes the parent's HASHLESS schemas —
``reveal_rpe_states/v1`` and ``reveal_rpe_directions/v2`` — whose field sets are byte-identical
to the parent's frozen legacy readers, which is what ``tests/test_golden_parity.py`` compares
against. Also dropped: the states-metadata-vs-ModelSpec re-authentication loop and the
report/battery count cross-checks (the surviving bindings are the two payload digests, the
reveal-id row order, and the report's own states/battery digests).

One read-only concession to the parent's current format: the readers also accept the
provenance-hashed ``.../v2`` states and ``.../v3`` directions files, ignoring their
``provenance_hash`` field (the only field the wrapper adds). The named obligation is the
drop-in route in ``results/ra_prime_certification.md`` — the certified R-A′ artifact is written
in those versions. What is *not* re-verified on that path is the parent's provenance hash
itself; the bindings this package checks are the npz payload digest plus the states / battery
digests recorded inside the metadata. The sole WRITE authority stays the hashless schema above.
Remove the concession when the certified artifact is no longer a supported input.

The identification is constrained by ``reward = ev + signed_rpe`` exactly, so a design cannot
carry reward, EV, and signed RPE as three separate linear regressors, and "a v_RPE axis
geometrically distinct from value" is vacuous (any linear combination of reward and EV is
trivially findable). Two claims ARE non-trivially identifiable, and the estimator gates on them:

- **Signed-beyond-unsigned (the kill-shot).** A *conjunction* of two matched-sign contests:
  ``reward``-matched cells (fixed reward and ``|RPE|``, sign flipped via EV) exclude pure realised
  reward — the one rival entangled with signed RPE; ``ev``-matched pairs (a draw's high vs low,
  fixed EV and ``|RPE|``) exclude pure EV. The ``|RPE|``-magnitude direction is at chance in both
  (it is sign-free); genuine signed RPE separates in both.
- **Gradient orientation.** Per-block multivariate OLS of the hidden state on
  ``[1, reward, ev, |RPE|, reward×|RPE|]`` gives coefficient vectors ``b_reward`` (the signed-RPE
  slope, ``v_RPE``) and ``b_ev``. Pure signed-RPE coding forces the gradient along ``reward − ev``
  ⇒ ``b_reward`` and ``b_ev`` both substantial and **anti-aligned** (``cos → −1``); pure reward ⇒
  ``b_ev ≈ 0``; pure EV ⇒ ``b_reward ≈ 0``.

Layer selection is held-out (directions from the estimation partition, sign AUROC scored on the
selection partition) and carries the winner's-curse of the block sweep through a selection-aware
sign-permutation null. Direction stability is a split-half cosine; the external floor is a
norm-matched random-direction control.

The licensable outcome is **present-and-separable** only; the report verdict vocabulary is capped
to that — R-A has no causal arm and licenses no functional/welfare/experience claim.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from appraisal_emotions.activation.capture import (
    RenderedCaptureBackend,
    capture_states,
    decoder_layers,
)
from appraisal_emotions.analysis.direction_stats import (
    ABS_RPE_INDEX as _ABS_RPE_INDEX,
)
from appraisal_emotions.analysis.direction_stats import (
    DEFAULT_RANDOM_DIRECTIONS,
    STABILITY_SPLITS,
    RevealArrays,
    StabilityResult,
    abs_rpe_magnitude_signal,
    build_reveal_arrays,
    coefficient_directions,
    fit_block_coefficients,
    held_out_auroc_floor,
    split_half_stability,
)
from appraisal_emotions.analysis.direction_stats import (
    DESIGN_COLUMNS as _DESIGN_COLUMNS,
)
from appraisal_emotions.analysis.direction_stats import (
    EV_INDEX as _EV_INDEX,
)
from appraisal_emotions.analysis.direction_stats import (
    FLOOR_QUANTILE as _FLOOR_QUANTILE,
)
from appraisal_emotions.analysis.direction_stats import (
    MATCHED_FOLDS as _MATCHED_FOLDS,
)
from appraisal_emotions.analysis.direction_stats import (
    REWARD_INDEX as _REWARD_INDEX,
)
from appraisal_emotions.analysis.direction_stats import (
    both_sign_cells as _both_sign_cells,
)
from appraisal_emotions.analysis.direction_stats import (
    crossfit_held_out as _crossfit_held_out,
)
from appraisal_emotions.analysis.direction_stats import (
    paired_cell_auroc as _paired_cell_auroc,
)
from appraisal_emotions.analysis.direction_stats import (
    project_blocks as _project,
)
from appraisal_emotions.analysis.direction_stats import (
    rows_by_group as _cell_to_rows,
)
from appraisal_emotions.analysis.direction_stats import (
    seed_int as _seed_int,
)
from appraisal_emotions.analysis.direction_stats import (
    sign_auroc_curve as _sign_auroc_curve,
)
from appraisal_emotions.core.schema import Comparison, ModelSpec, StrictModel
from appraisal_emotions.core.stats import add_one_p, cosine_between
from appraisal_emotions.core.util import (
    EXTRACTION_SEED,
    ensure_parent,
    stable_hash,
    states_sha256,
    write_json,
)

__all__ = [
    "DEFAULT_PERMUTATIONS",
    "DEFAULT_RANDOM_DIRECTIONS",
    "DIRECTION_FAMILIES",
    "GATE_ALPHA",
    "MatchFamily",
    "MatchedContestResult",
    "ORIENTATION_COS_MAX",
    "OrientationResult",
    "REVEAL_RPE_CONTRACT_VERSION",
    "REVEAL_RPE_DIRECTIONS_CONTRACT_VERSION",
    "REVEAL_RPE_STATES_CONTRACT_VERSION",
    "RevealArrays",
    "RevealRpeDirections",
    "RevealRpeDirectionsMetadata",
    "RevealRpeReport",
    "RevealRpeStates",
    "RevealRpeStatesMetadata",
    "STABILITY_MIN_COSINE",
    "STABILITY_SPLITS",
    "SelectionAwareNull",
    "StabilityResult",
    "Verdict",
    "abs_rpe_magnitude_signal",
    "analyze_reveal_rpe",
    "build_reveal_arrays",
    "build_reveal_rpe_directions",
    "build_reveal_rpe_states",
    "capture_reveal_states",
    "coefficient_directions",
    "fit_block_coefficients",
    "held_out_auroc_floor",
    "matched_sign_contest",
    "orientation_signature",
    "read_reveal_rpe_directions",
    "read_reveal_rpe_states",
    "reveal_read_prompt",
    "reveal_rpe_directions_path",
    "reveal_rpe_states_path",
    "selection_aware_sign_null",
    "signed_rpe_curve",
    "split_half_stability",
    "write_reveal_rpe_directions",
    "write_reveal_rpe_states",
]


REVEAL_RPE_CONTRACT_VERSION = "reveal_rpe/v1"


# Gate constants (spec §4.4 Level-A conventions).
GATE_ALPHA = 0.05
ORIENTATION_COS_MAX = -0.5  # cos(b_reward, b_ev) must be below this (anti-aligned) for RPE.
STABILITY_MIN_COSINE = 0.80
DEFAULT_PERMUTATIONS = 1000

MatchFamily = Literal["reward_matched", "ev_matched"]
Verdict = Literal[
    "separable-signed-rpe",
    "collapses-to-value-or-reward",
    "collapses-to-unsigned-surprise",
    "indeterminate",
]


# --------------------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------------------


def capture_reveal_states(
    backend: RenderedCaptureBackend, reveals: tuple[Comparison, ...], *, fallback_blocks: int = 4
) -> np.ndarray:
    """Residual-stream states at the reveal token for each reveal: ``(n, n_blocks, hidden)``.

    Reads at ``position="last"`` after appending ``metadata['read_prefix']`` (the single
    leading-space outcome symbol) via :func:`reveal_read_prompt`, so capture lands on the
    byte-pinned reveal slot. Read-only: no steering, and no patch (E3's patched forward reads the
    same slot through the same helper).

    Slot placement is the arm-wide, advisor-validated ``_read_prompt`` convention: under the HF
    chat template (``add_generation_prompt=True``) the read symbol is the assistant-side
    ``position="last"`` token, exactly where the validated choicepointer / v_EV capture reads
    (in-context v_EV decodes there at d′≈2.5). The assistant-header prefix is a *constant* across
    every reveal row, so it is absorbed by the design intercept and cannot bias the reward / EV /
    |RPE| slopes the estimator identifies; the read symbol itself is forced (not generated), so
    capture measures outcome representation at that slot, not a generation preference.
    """

    layers = decoder_layers(backend, fallback_blocks=fallback_blocks)  # block l at index l+1.
    resolved: list[tuple[str, int | str]] = [
        (reveal_read_prompt(backend, comparison), "last") for comparison in reveals
    ]
    return capture_states(backend, resolved, layers=layers)


def reveal_read_prompt(backend: RenderedCaptureBackend, comparison: Comparison) -> str:
    """The exact byte-pinned string the reveal token is read at the end of.

    ONE spelling of the read slot, shared by the read-only capture above and by E3 patching
    (``analysis.activation_patching``), which must patch the same position of the same rendered
    surface the states artifact was captured from — two spellings would be two experiments.
    """

    rendered = backend.render_prompt(comparison.prompt).rendered_prompt
    read_prefix = comparison.metadata.get("read_prefix")
    if not isinstance(read_prefix, str) or not read_prefix:
        raise ValueError(
            f"reveal {comparison.comparison_id!r} is missing metadata['read_prefix']; the "
            "byte-pinned reveal-token capture must fail closed, not fall back to "
            "position='last' on the unpinned prompt tail."
        )
    # Assistant-side read slot (the validated _read_prompt convention): the single-token outcome
    # symbol is the natural next token after the chat template's generation header; its constant
    # prefix is absorbed by the design intercept (see capture_reveal_states' docstring).
    return f"{rendered}{read_prefix}"


# --------------------------------------------------------------------------------------
# Hash-bound states artifact: shape/dtype/hash revalidated on construction, so a tampered or
# misaligned artifact is unrepresentable in memory.
# --------------------------------------------------------------------------------------

REVEAL_RPE_STATES_CONTRACT_VERSION = "reveal_rpe_states/v1"
_STATES_SUFFIX = ".states.npz"
_STATES_ARRAY_NAME = "reveal_states"
# The field the parent's provenance-hashed wrapper adds; dropped on read (see module docstring).
_PARENT_PROVENANCE_FIELD = "provenance_hash"


def _read_metadata[T: StrictModel](metadata_path: Path, model: type[T]) -> T:
    """Parse an artifact's metadata JSON, tolerating the parent's provenance-hashed variant."""

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload.pop(_PARENT_PROVENANCE_FIELD, None)
    return model.model_validate(payload)


class RevealRpeStatesMetadata(StrictModel):
    """Model identity plus shape/hash binding for one captured reveal-states artifact."""

    artifact_contract_version: Literal["reveal_rpe_states/v1", "reveal_rpe_states/v2"] = (
        REVEAL_RPE_STATES_CONTRACT_VERSION
    )
    model_key: str
    backend: str
    model_id: str
    revision: str | None
    tokenizer_id: str
    dtype: str | None
    seed: int
    battery_contract_version: str
    reveal_ids: tuple[str, ...]  # capture row order — states row i is reveal reveal_ids[i]
    n_blocks: int
    hidden_size: int
    states_dtype: str
    states_sha256: str


@dataclass(frozen=True)
class RevealRpeStates:
    """Validated reveal states plus their identity metadata (states frozen read-only)."""

    metadata: RevealRpeStatesMetadata
    states: np.ndarray

    def __post_init__(self) -> None:
        array = np.array(self.states, copy=True)
        meta = self.metadata
        if array.ndim != 3:
            raise ValueError("reveal states must be (n_reveals, n_blocks, hidden)")
        rows, blocks, hidden = (int(dim) for dim in array.shape)
        if rows != len(meta.reveal_ids):
            raise ValueError("reveal states row count does not match metadata.reveal_ids")
        if blocks != meta.n_blocks or hidden != meta.hidden_size:
            raise ValueError("reveal states block/hidden dims do not match metadata")
        if str(array.dtype) != meta.states_dtype:
            raise ValueError("reveal states dtype does not match metadata")
        if states_sha256(array) != meta.states_sha256:
            raise ValueError("reveal states hash does not match metadata")
        if not np.all(np.isfinite(array)):
            raise ValueError("reveal states must be finite")
        array.flags.writeable = False
        object.__setattr__(self, "states", array)


def build_reveal_rpe_states(
    states: np.ndarray,
    reveals: tuple[Comparison, ...],
    *,
    spec: ModelSpec,
    seed: int,
    battery_contract_version: str,
) -> RevealRpeStates:
    """Bind a captured states array to its model identity and reveal row order."""

    array = np.ascontiguousarray(states, dtype=np.float64)
    if array.ndim != 3:
        # Guard the shape[1]/shape[2] reads below so a malformed capture raises the contract error,
        # not a raw IndexError (RevealRpeStates.__post_init__ enforces the same invariant on read).
        raise ValueError("reveal states must be (n_reveals, n_blocks, hidden)")
    metadata = RevealRpeStatesMetadata(
        model_key=spec.key,
        backend=spec.backend,
        model_id=spec.model_id,
        revision=spec.revision,
        tokenizer_id=spec.tokenizer_id or spec.model_id,
        dtype=spec.dtype,
        seed=seed,
        battery_contract_version=battery_contract_version,
        reveal_ids=tuple(comparison.comparison_id for comparison in reveals),
        n_blocks=int(array.shape[1]),
        hidden_size=int(array.shape[2]),
        states_dtype=str(array.dtype),
        states_sha256=states_sha256(array),
    )
    return RevealRpeStates(metadata=metadata, states=array)


def reveal_rpe_states_path(metadata_path: Path) -> Path:
    if metadata_path.suffix != ".json":
        raise ValueError("reveal-rpe states metadata path must end in .json")
    return metadata_path.with_name(f"{metadata_path.stem}{_STATES_SUFFIX}")


def write_reveal_rpe_states(artifact: RevealRpeStates, metadata_path: Path) -> tuple[Path, Path]:
    """Write the metadata JSON and the sibling states npz; return both paths."""

    states_path = reveal_rpe_states_path(metadata_path)
    ensure_parent(metadata_path)
    np.savez(states_path, **{_STATES_ARRAY_NAME: np.asarray(artifact.states)})
    write_json(metadata_path, artifact.metadata)
    return metadata_path, states_path


def read_reveal_rpe_states(metadata_path: Path) -> RevealRpeStates:
    """Read and revalidate a reveal-states artifact (metadata + states hash binding)."""

    metadata = _read_metadata(metadata_path, RevealRpeStatesMetadata)
    with np.load(reveal_rpe_states_path(metadata_path), allow_pickle=False) as bundle:
        array = np.array(bundle[_STATES_ARRAY_NAME])
    return RevealRpeStates(metadata=metadata, states=array)


# --------------------------------------------------------------------------------------
# Layer selection + selection-aware sign-permutation null
# --------------------------------------------------------------------------------------


def signed_rpe_curve(
    v_rpe_directions: np.ndarray, states: np.ndarray, signs: np.ndarray
) -> np.ndarray:
    """Per-block held-out sign AUROC of the ``v_RPE`` projection: ``(n_blocks,)``."""

    return np.asarray(_sign_auroc_curve(_project(states, v_rpe_directions), signs))


@dataclass(frozen=True)
class SelectionAwareNull:
    selected_block: int
    observed_auroc: float
    p_value: float
    n_permutations: int
    passed: bool


def selection_aware_sign_null(
    v_rpe_directions: np.ndarray,
    selection_states: np.ndarray,
    selection_signs: np.ndarray,
    *,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = EXTRACTION_SEED,
    alpha: float = GATE_ALPHA,
) -> SelectionAwareNull:
    """Winner's-curse null for the held-out ``v_RPE`` sign separability.

    Directions are fixed from the (unpermuted) estimation fit; only the selection sign labels
    permute. Each permutation re-selects the best block (max sign AUROC) and records it, so the
    null carries the depth-sweep selection bias.
    """

    projections = _project(selection_states, v_rpe_directions)  # (n_sel, n_blocks)
    observed_curve = np.asarray(_sign_auroc_curve(projections, selection_signs))
    selected_block = int(np.argmax(observed_curve))
    observed = float(observed_curve[selected_block])

    rng = random.Random(f"{seed}|reveal-rpe-selection-null")
    permuted = selection_signs.tolist()
    exceedances = 0
    for _ in range(n_permutations):
        rng.shuffle(permuted)
        shuffled = np.asarray(permuted, dtype=np.int64)
        if max(_sign_auroc_curve(projections, shuffled)) >= observed:
            exceedances += 1
    p_value = add_one_p(exceedances, n_permutations)
    return SelectionAwareNull(
        selected_block=selected_block,
        observed_auroc=observed,
        p_value=p_value,
        n_permutations=n_permutations,
        passed=observed > 0.5 and p_value < alpha,
    )


# --------------------------------------------------------------------------------------
# Matched-sign contests (the conjunction kill-shot)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchedContestResult:
    family: MatchFamily
    n_cells_both_signs: int
    n_scored_cells: int
    direction_auroc: float
    magnitude_auroc: float
    p_value: float
    passed: bool


_FAILED_CONTEST_ARGS = (0.5, 0.5, 1.0, False)


def matched_sign_contest(
    arrays: RevealArrays,
    states: np.ndarray,
    v_absrpe_directions: np.ndarray,
    *,
    family: MatchFamily,
    block: int,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = EXTRACTION_SEED,
    alpha: float = GATE_ALPHA,
) -> MatchedContestResult:
    """One matched-sign contest at ``block``: excludes pure reward (reward-matched) or pure EV.

    Cross-fitted over cells: every both-sign cell is scored by a within-cell sign direction fit on
    the OTHER folds' cells, so no cell is discarded to a fit half and each is a held-out test point.
    (The global reveal partition is deliberately NOT used to split here — it assigns partitions per
    reveal, which splits most matched cells across the boundary and guts the contest.) The held-out
    per-cell (positive − negative) projection difference is scored with a paired AUROC against a
    sign-permutation null; the sign-free ``|RPE|`` direction's paired AUROC is reported as the
    magnitude rival but not gated on (``|RPE|`` is constant within a cell, so it cannot separate
    sign there by construction).
    """

    cells_col = arrays.reward_cell if family == "reward_matched" else arrays.ev_cell
    both = _both_sign_cells(cells_col, arrays.signs)
    n_cells = len(both)
    if n_cells < 2:
        return MatchedContestResult(family, n_cells, 0, *_FAILED_CONTEST_ARGS)

    grouped = _cell_to_rows(cells_col)
    block_states = states[:, block, :]

    order = list(both)
    random.Random(f"{seed}|reveal-rpe-{family}-folds").shuffle(order)
    n_folds = min(_MATCHED_FOLDS, n_cells)
    held_out = _crossfit_held_out(block_states, arrays.signs, grouped, order, n_folds)

    scored_cells = [cell for cell in both if not np.isnan(held_out[grouped[cell]]).any()]
    observed = _paired_cell_auroc(held_out, arrays.signs, grouped, scored_cells)
    if observed is None:
        return MatchedContestResult(family, n_cells, len(scored_cells), *_FAILED_CONTEST_ARGS)
    magnitude = _paired_cell_auroc(
        block_states @ v_absrpe_directions[block], arrays.signs, grouped, scored_cells
    )

    # Stratified permutation null: permute signs WITHIN each matched cell (a cross-cell shuffle
    # would change per-cell sign counts and drop cells, scoring a different set than the observed),
    # then REFIT the cross-fit direction under those signs. Holding the true-sign direction fixed
    # and only permuting the scoring labels is anti-conservative — the null must recompute the whole
    # cross-fit end to end so it carries the same direction-fitting optimism as the observed.
    rng = np.random.default_rng(_seed_int(seed, "reveal-rpe", family, "null"))
    exceedances = 0
    for _ in range(n_permutations):
        permuted = arrays.signs.copy()
        for cell in both:
            rows = grouped[cell]
            permuted[rows] = rng.permutation(arrays.signs[rows])
        null_held = _crossfit_held_out(block_states, permuted, grouped, order, n_folds)
        null = _paired_cell_auroc(null_held, permuted, grouped, scored_cells)
        if null is not None and null >= observed:
            exceedances += 1
    p_value = add_one_p(exceedances, n_permutations)
    magnitude_auroc = 0.5 if magnitude is None else magnitude
    passed = observed > 0.5 and p_value < alpha
    return MatchedContestResult(
        family, n_cells, len(scored_cells), observed, magnitude_auroc, p_value, passed
    )


# --------------------------------------------------------------------------------------
# Orientation, stability, random floor
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OrientationResult:
    selected_block: int
    reward_norm: float
    ev_norm: float
    norm_floor: float
    cos_reward_ev: float
    passed: bool


def orientation_signature(
    estimation_states: np.ndarray,
    design: np.ndarray,
    *,
    block: int,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = EXTRACTION_SEED,
) -> OrientationResult:
    """Anti-alignment of the ``b_reward`` and ``b_ev`` coefficient vectors at ``block``.

    RPE coding orients the hidden-state gradient along ``reward − ev`` ⇒ both slopes substantial
    and ``cos(b_reward, b_ev) → −1``. Each slope norm must exceed a row-permutation floor (design
    rows shuffled against states), which distinguishes RPE from pure reward (``b_ev ≈ 0``) and pure
    EV (``b_reward ≈ 0``).
    """

    coef = np.linalg.lstsq(design, estimation_states[:, block, :], rcond=None)[0]
    reward_norm = float(np.linalg.norm(coef[_REWARD_INDEX]))
    ev_norm = float(np.linalg.norm(coef[_EV_INDEX]))
    cos = (
        cosine_between(coef[_REWARD_INDEX].tolist(), coef[_EV_INDEX].tolist())
        if reward_norm > 0.0 and ev_norm > 0.0
        else 0.0
    )

    rng = np.random.default_rng(_seed_int(seed, "reveal-rpe-orientation", block))
    block_states = estimation_states[:, block, :]
    null_norms: list[float] = []
    # NOT a ``direction_stats.random_direction_floor`` site: the draw is a row permutation of the
    # states, not a random direction, and each draw contributes TWO pooled norms to one null.
    for _ in range(n_permutations):
        order = rng.permutation(block_states.shape[0])
        null_coef = np.linalg.lstsq(design, block_states[order], rcond=None)[0]
        null_norms.append(float(np.linalg.norm(null_coef[_REWARD_INDEX])))
        null_norms.append(float(np.linalg.norm(null_coef[_EV_INDEX])))
    norm_floor = float(np.quantile(null_norms, _FLOOR_QUANTILE))
    passed = reward_norm > norm_floor and ev_norm > norm_floor and cos < ORIENTATION_COS_MAX
    return OrientationResult(block, reward_norm, ev_norm, norm_floor, cos, passed)


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


class RevealRpeReport(StrictModel):
    """The R-A representational gate record. Licenses present-and-separable ONLY."""

    artifact_contract_version: Literal["reveal_rpe/v1"] = REVEAL_RPE_CONTRACT_VERSION
    n_reveals: int
    n_estimation: int
    n_selection: int
    n_blocks: int
    hidden_size: int
    # Identity binding to the states artifact this report scored, and to the BATTERY file that
    # supplied the reveal design: the directions refit rebuilds reward/EV/|RPE| regressors from
    # battery metadata, so a same-ids battery with drifted labels must never feed a backfill.
    # The CLI stamps both right after capture.
    states_sha256: str | None = None
    battery_sha256: str | None = None
    selected_block: int
    signed_rpe_null_p: float
    signed_rpe_observed_auroc: float
    signed_rpe_null_passed: bool
    reward_matched_auroc: float
    reward_matched_p: float
    reward_matched_n_cells: int
    reward_matched_n_scored_cells: int
    reward_matched_passed: bool
    ev_matched_auroc: float
    ev_matched_p: float
    ev_matched_n_cells: int
    ev_matched_n_scored_cells: int
    ev_matched_passed: bool
    orientation_cos_reward_ev: float
    orientation_reward_norm: float
    orientation_ev_norm: float
    orientation_norm_floor: float
    orientation_passed: bool
    stability_cosine: float
    stability_cosine_std: float
    stability_cosine_rowsplit_legacy: float
    stability_n_splits: int
    stability_passed: bool
    random_direction_floor_auroc: float
    external_passed: bool
    abs_rpe_magnitude_auroc: float
    abs_rpe_magnitude_floor: float
    abs_rpe_present: bool
    signed_conjunction_passed: bool
    verdict: Verdict


def analyze_reveal_rpe(
    states: np.ndarray,
    reveals: tuple[Comparison, ...],
    *,
    seed: int = EXTRACTION_SEED,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    n_random_directions: int = DEFAULT_RANDOM_DIRECTIONS,
    alpha: float = GATE_ALPHA,
) -> RevealRpeReport:
    """Fit ``v_RPE``, select the block held-out, and evaluate the R-A gates → a capped verdict."""

    if states.ndim != 3 or states.shape[0] != len(reveals):
        raise ValueError("states must be (n_reveals, n_blocks, hidden) aligned to reveals")

    arrays = build_reveal_arrays(reveals)
    est = arrays.partition == "estimation"
    sel = arrays.partition == "selection"
    if int(est.sum()) < 2 or int(sel.sum()) < 2:
        raise ValueError("both estimation and selection partitions need >= 2 reveals")

    estimation_states = states[est]
    selection_states = states[sel]

    coefficients = fit_block_coefficients(estimation_states, arrays.design[est])
    v_rpe = coefficient_directions(coefficients, _REWARD_INDEX)
    v_absrpe = coefficient_directions(coefficients, _ABS_RPE_INDEX)

    null = selection_aware_sign_null(
        v_rpe,
        selection_states,
        arrays.signs[sel],
        n_permutations=n_permutations,
        seed=seed,
        alpha=alpha,
    )
    block = null.selected_block

    reward_matched = matched_sign_contest(
        arrays,
        states,
        v_absrpe,
        family="reward_matched",
        block=block,
        n_permutations=n_permutations,
        seed=seed,
        alpha=alpha,
    )
    ev_matched = matched_sign_contest(
        arrays,
        states,
        v_absrpe,
        family="ev_matched",
        block=block,
        n_permutations=n_permutations,
        seed=seed,
        alpha=alpha,
    )
    orientation = orientation_signature(
        estimation_states,
        arrays.design[est],
        block=block,
        n_permutations=n_permutations,
        seed=seed,
    )
    stability = split_half_stability(
        estimation_states, arrays.design[est], arrays.ev_cell[est], block=block, seed=seed
    )
    floor = held_out_auroc_floor(
        estimation_states,
        arrays.signs[est],
        selection_states,
        arrays.signs[sel],
        block=block,
        n_directions=n_random_directions,
        seed=seed,
    )

    abs_rpe_auroc, abs_rpe_floor = abs_rpe_magnitude_signal(
        v_absrpe,
        estimation_states,
        arrays.abs_rpe[est],
        selection_states,
        arrays.abs_rpe[sel],
        block=block,
        n_directions=n_random_directions,
        seed=seed,
    )

    conjunction_passed = reward_matched.passed and ev_matched.passed
    stability_passed = stability.cosine >= STABILITY_MIN_COSINE
    external_passed = null.observed_auroc > floor
    abs_rpe_present = abs_rpe_auroc > abs_rpe_floor
    verdict = _verdict(
        conjunction_passed=conjunction_passed,
        orientation_passed=orientation.passed,
        stability_passed=stability_passed,
        external_passed=external_passed,
        selection_null_passed=null.passed,
        abs_rpe_present=abs_rpe_present,
        reward_matched_passed=reward_matched.passed,
        ev_matched_passed=ev_matched.passed,
    )

    return RevealRpeReport(
        n_reveals=len(reveals),
        n_estimation=int(est.sum()),
        n_selection=int(sel.sum()),
        n_blocks=states.shape[1],
        hidden_size=states.shape[2],
        selected_block=block,
        signed_rpe_null_p=null.p_value,
        signed_rpe_observed_auroc=null.observed_auroc,
        signed_rpe_null_passed=null.passed,
        reward_matched_auroc=reward_matched.direction_auroc,
        reward_matched_p=reward_matched.p_value,
        reward_matched_n_cells=reward_matched.n_cells_both_signs,
        reward_matched_n_scored_cells=reward_matched.n_scored_cells,
        reward_matched_passed=reward_matched.passed,
        ev_matched_auroc=ev_matched.direction_auroc,
        ev_matched_p=ev_matched.p_value,
        ev_matched_n_cells=ev_matched.n_cells_both_signs,
        ev_matched_n_scored_cells=ev_matched.n_scored_cells,
        ev_matched_passed=ev_matched.passed,
        orientation_cos_reward_ev=orientation.cos_reward_ev,
        orientation_reward_norm=orientation.reward_norm,
        orientation_ev_norm=orientation.ev_norm,
        orientation_norm_floor=orientation.norm_floor,
        orientation_passed=orientation.passed,
        stability_cosine=stability.cosine,
        stability_cosine_std=stability.cosine_std,
        stability_cosine_rowsplit_legacy=stability.rowsplit_legacy,
        stability_n_splits=stability.n_splits,
        stability_passed=stability_passed,
        random_direction_floor_auroc=floor,
        external_passed=external_passed,
        abs_rpe_magnitude_auroc=abs_rpe_auroc,
        abs_rpe_magnitude_floor=abs_rpe_floor,
        abs_rpe_present=abs_rpe_present,
        signed_conjunction_passed=conjunction_passed,
        verdict=verdict,
    )


def _verdict(
    *,
    conjunction_passed: bool,
    orientation_passed: bool,
    stability_passed: bool,
    external_passed: bool,
    selection_null_passed: bool,
    abs_rpe_present: bool,
    reward_matched_passed: bool,
    ev_matched_passed: bool,
) -> Verdict:
    """Map the gate outcomes to a licensed sentence (present-and-separable ONLY).

    Every affirmative verdict requires POSITIVE evidence, never merely the absence of a rival:
    ``separable-signed-rpe`` also gates on the selection-aware sign null (``selection_null_passed``)
    so a non-significant held-out AUROC cannot pass on the other checks alone; the
    ``collapses-to-unsigned-surprise`` collapse requires the |RPE|-magnitude axis to actually
    separate (``abs_rpe_present``) — otherwise a run where nothing separates is ``indeterminate``,
    not a positively-asserted surprise collapse.
    """

    if (
        conjunction_passed
        and orientation_passed
        and stability_passed
        and external_passed
        and selection_null_passed
    ):
        return "separable-signed-rpe"
    # Pure reward / pure EV: exactly ONE matched contest carries sign (the conjunction needs both).
    # reward-matched-only ⇒ expectation-referenced-but-EV-shaped (pure value); ev-matched-only ⇒
    # pure realised reward. Either single-axis pass is a value/reward collapse, not signed RPE — the
    # symmetric case must not silently fall through to indeterminate.
    if reward_matched_passed != ev_matched_passed:
        return "collapses-to-value-or-reward"
    if not conjunction_passed and not orientation_passed and abs_rpe_present:
        # Sign carried by neither matched contest, but the |RPE|-magnitude axis separates ⇒
        # salience / unsigned-surprise shaped (positive evidence, not just an absent signal).
        return "collapses-to-unsigned-surprise"
    return "indeterminate"


# --------------------------------------------------------------------------------------
# Hash-bound directions artifact: persist the fitted per-block unit directions so downstream
# consumers (the emotion-vector layer's geometry comparisons) read ONE artifact instead of
# re-deriving directions. Deterministically recomputable from the states artifact — the binding
# (states_sha256 + battery file hash + selected block) is what makes a comparison against these
# directions auditable, not the fit itself.
# --------------------------------------------------------------------------------------

REVEAL_RPE_DIRECTIONS_CONTRACT_VERSION = "reveal_rpe_directions/v2"
_DIRECTIONS_SUFFIX = ".directions.npz"
_DIRECTIONS_ARRAY_NAME = "reveal_directions"
# Row order of the stacked (family, block, hidden) array. v_rpe is the b_reward slope family
# (the signed-RPE direction under the identification argument in the module docstring); v_ev
# and v_absrpe are the rival value / unsigned-surprise families to compare against.
DIRECTION_FAMILIES: tuple[str, ...] = ("v_rpe", "v_ev", "v_absrpe")
_FAMILY_COLUMN_INDEX: tuple[int, ...] = (_REWARD_INDEX, _EV_INDEX, _ABS_RPE_INDEX)
_UNIT_NORM_ATOL = 1e-6


class RevealRpeDirectionsMetadata(StrictModel):
    """Identity + binding for one persisted reveal-RPE directions artifact.

    ``states_sha256`` binds to the source states artifact, ``battery_sha256`` to the battery
    file the reveal metadata came from, and ``selected_block`` to the gate report's held-out
    block choice — so a consumer can prove which capture and which selection produced these
    directions without re-running either. ``source_verdict`` is carried for context; the
    directions do NOT inherit its licence.
    """

    artifact_contract_version: Literal["reveal_rpe_directions/v2", "reveal_rpe_directions/v3"] = (
        REVEAL_RPE_DIRECTIONS_CONTRACT_VERSION
    )
    model_spec_hash: str
    model_key: str
    backend: str
    model_id: str
    revision: str | None
    tokenizer_id: str
    dtype: str | None
    seed: int
    battery_contract_version: str
    battery_sha256: str
    states_sha256: str
    design_columns: tuple[str, ...]
    direction_families: tuple[str, ...]
    selected_block: int
    source_verdict: Verdict
    n_estimation: int
    n_blocks: int
    hidden_size: int
    directions_dtype: str
    directions_sha256: str


@dataclass(frozen=True)
class RevealRpeDirections:
    """Validated per-block unit directions plus binding metadata (frozen read-only)."""

    metadata: RevealRpeDirectionsMetadata
    directions: np.ndarray  # (n_families, n_blocks, hidden); rows unit-norm or exactly zero

    def __post_init__(self) -> None:
        array = np.array(self.directions, copy=True)
        meta = self.metadata
        if array.ndim != 3:
            raise ValueError("reveal directions must be (n_families, n_blocks, hidden)")
        families, blocks, hidden = (int(dim) for dim in array.shape)
        # The directions hash covers only the ARRAY, so the row/design labels must be pinned to
        # the writer's canonical values — a metadata-only relabel (v_rpe <-> v_ev, or a drifted
        # design column list) would change semantics without breaking the hash.
        if tuple(meta.direction_families) != DIRECTION_FAMILIES:
            raise ValueError(
                "metadata.direction_families must be the canonical DIRECTION_FAMILIES row order"
            )
        if tuple(meta.design_columns) != _DESIGN_COLUMNS:
            raise ValueError(
                "metadata.design_columns must be the canonical estimation design columns"
            )
        if families != len(meta.direction_families):
            raise ValueError("directions family count does not match metadata.direction_families")
        if blocks != meta.n_blocks or hidden != meta.hidden_size:
            raise ValueError("directions block/hidden dims do not match metadata")
        if str(array.dtype) != meta.directions_dtype:
            raise ValueError("directions dtype does not match metadata")
        if states_sha256(array) != meta.directions_sha256:
            raise ValueError("directions hash does not match metadata")
        if not np.all(np.isfinite(array)):
            raise ValueError("reveal directions must be finite")
        norms = np.linalg.norm(array, axis=2)
        unit_or_zero = np.isclose(norms, 1.0, atol=_UNIT_NORM_ATOL) | (norms == 0.0)
        if not bool(np.all(unit_or_zero)):
            raise ValueError("each direction row must be unit-norm or exactly zero")
        if not (0 <= meta.selected_block < meta.n_blocks):
            raise ValueError("metadata.selected_block out of range for n_blocks")
        array.flags.writeable = False
        object.__setattr__(self, "directions", array)


def build_reveal_rpe_directions(
    states_artifact: RevealRpeStates,
    reveals: tuple[Comparison, ...],
    *,
    spec: ModelSpec,
    report: RevealRpeReport,
    battery_sha256: str,
) -> RevealRpeDirections:
    """Refit the estimation-partition OLS directions and bind them to their sources.

    ``reveals`` must be the captured battery in its canonical row order. The fit is byte-for-byte
    the ``analyze_reveal_rpe`` estimation path, so the persisted ``v_rpe`` is exactly the
    direction family the report's gates scored.
    """

    meta = states_artifact.metadata
    canonical_reveal_ids = tuple(comparison.comparison_id for comparison in reveals)
    if tuple(meta.reveal_ids) != canonical_reveal_ids:
        raise ValueError(
            "directions producer states metadata.reveal_ids must match the battery's canonical "
            "order; the refit would bind directions to a different capture's row order."
        )
    if report.states_sha256 is not None and report.states_sha256 != meta.states_sha256:
        raise ValueError(
            "report.states_sha256 does not match the states artifact: this report scored a "
            "DIFFERENT capture; refusing to bind its selected_block/verdict to these states."
        )
    if report.battery_sha256 is not None and report.battery_sha256 != battery_sha256:
        raise ValueError(
            "report.battery_sha256 does not match the battery supplying the reveal design: "
            "the refit rebuilds reward/EV/|RPE| regressors from battery metadata, so a "
            "same-ids battery with drifted labels never feeds these directions."
        )
    arrays = build_reveal_arrays(reveals)
    est = arrays.partition == "estimation"
    n_estimation = int(est.sum())
    if n_estimation < 2:
        raise ValueError("directions refit needs >= 2 estimation reveals")

    coefficients = fit_block_coefficients(states_artifact.states[est], arrays.design[est])
    stacked = np.stack(
        [coefficient_directions(coefficients, index) for index in _FAMILY_COLUMN_INDEX], axis=0
    )
    stacked = np.ascontiguousarray(stacked, dtype=np.float64)
    metadata = RevealRpeDirectionsMetadata(
        model_spec_hash=stable_hash(spec),
        model_key=spec.key,
        backend=spec.backend,
        model_id=spec.model_id,
        revision=spec.revision,
        tokenizer_id=spec.tokenizer_id or spec.model_id,
        dtype=spec.dtype,
        seed=meta.seed,
        battery_contract_version=meta.battery_contract_version,
        battery_sha256=battery_sha256,
        states_sha256=meta.states_sha256,
        design_columns=_DESIGN_COLUMNS,
        direction_families=DIRECTION_FAMILIES,
        selected_block=report.selected_block,
        source_verdict=report.verdict,
        n_estimation=n_estimation,
        n_blocks=meta.n_blocks,
        hidden_size=meta.hidden_size,
        directions_dtype=str(stacked.dtype),
        directions_sha256=states_sha256(stacked),
    )
    return RevealRpeDirections(metadata=metadata, directions=stacked)


def reveal_rpe_directions_path(metadata_path: Path) -> Path:
    if metadata_path.suffix != ".json":
        raise ValueError("reveal-rpe directions metadata path must end in .json")
    return metadata_path.with_name(f"{metadata_path.stem}{_DIRECTIONS_SUFFIX}")


def write_reveal_rpe_directions(
    artifact: RevealRpeDirections, metadata_path: Path
) -> tuple[Path, Path]:
    """Write the metadata JSON and the sibling directions npz; return both paths."""

    directions_path = reveal_rpe_directions_path(metadata_path)
    ensure_parent(metadata_path)
    np.savez(directions_path, **{_DIRECTIONS_ARRAY_NAME: np.asarray(artifact.directions)})
    write_json(metadata_path, artifact.metadata)
    return metadata_path, directions_path


def read_reveal_rpe_directions(metadata_path: Path) -> RevealRpeDirections:
    """Read and revalidate a directions artifact (metadata + directions hash binding)."""

    metadata = _read_metadata(metadata_path, RevealRpeDirectionsMetadata)
    with np.load(reveal_rpe_directions_path(metadata_path), allow_pickle=False) as bundle:
        array = np.array(bundle[_DIRECTIONS_ARRAY_NAME])
    return RevealRpeDirections(metadata=metadata, directions=array)
