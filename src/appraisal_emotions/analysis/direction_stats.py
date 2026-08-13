"""Shared direction-fitting and held-out statistics for the reveal-RPE surface.

Extracted WHOLE (imports retargeted only) from functional-valence-validity
src/fv_validity/analysis/direction_stats.py @ 10c4662. Nothing dropped — every kernel here is
read by ``analysis.reveal_rpe``, and the numeric conventions (design column order, centring,
fold assignment, floor quantile, split-half grouping) are what the R-A′ parity fixtures pin.

The parent module was shared with a bandit surface that is not ported; its notes on which
names stay broad for that second consumer are dropped, and the interface is now single-consumer
by construction.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from appraisal_emotions.core.schema import Comparison
from appraisal_emotions.core.stats import midrank_auroc, paired_projection_auroc
from appraisal_emotions.core.util import EXTRACTION_SEED, unit_vector

__all__ = [
    "ABS_RPE_INDEX",
    "DEFAULT_RANDOM_DIRECTIONS",
    "DESIGN_COLUMNS",
    "EV_INDEX",
    "FLOOR_QUANTILE",
    "MATCHED_FOLDS",
    "REWARD_INDEX",
    "STABILITY_SPLITS",
    "RevealArrays",
    "StabilityResult",
    "abs_rpe_magnitude_signal",
    "both_sign_cells",
    "build_reveal_arrays",
    "coefficient_directions",
    "crossfit_held_out",
    "fit_block_coefficients",
    "grouped_half_indices",
    "held_out_auroc_floor",
    "paired_cell_auroc",
    "project_blocks",
    "random_direction_floor",
    "rows_by_group",
    "seed_int",
    "sign_auroc",
    "split_half_stability",
]


def seed_int(*parts: object) -> int:
    """Stable 64-bit seed from string parts (not Python ``hash()``, which is per-process salted)."""

    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


# Design columns (must match reveal_probes.DESIGN_REGRESSORS + a leading intercept). signed RPE
# (= reward − ev) is DERIVED, never a column — the [1, reward, ev, signed_rpe] design is rank
# deficient, which is exactly why diff-in-means cannot identify RPE.
DESIGN_COLUMNS: tuple[str, ...] = ("intercept", "reward", "ev", "abs_rpe", "reward_x_abs_rpe")
REWARD_INDEX = DESIGN_COLUMNS.index("reward")
EV_INDEX = DESIGN_COLUMNS.index("ev")
ABS_RPE_INDEX = DESIGN_COLUMNS.index("abs_rpe")

STABILITY_SPLITS = 200  # seeded draw-grouped split-half repeats averaged into the stability cosine.
DEFAULT_RANDOM_DIRECTIONS = 200
MATCHED_FOLDS = 5  # cross-fit folds over matched cells (every cell scored held-out)
FLOOR_QUANTILE = 0.95


@dataclass(frozen=True)
class RevealArrays:
    """Per-reveal design + label arrays aligned to a states array's row order."""

    design: np.ndarray  # (n, 5) = [1, reward_c, ev_c, |RPE|_c, (reward×|RPE|)_c] (centred slopes)
    signs: np.ndarray  # (n,) in {+1, -1}
    abs_rpe: np.ndarray  # (n,) raw |RPE| magnitude (the high/low-surprise label source)
    reward_cell: np.ndarray  # (n,) reward-matched cell id
    ev_cell: np.ndarray  # (n,) EV-matched pair (draw) id
    partition: np.ndarray  # (n,) in {"estimation", "selection"}


def build_reveal_arrays(reveals: tuple[Comparison, ...]) -> RevealArrays:
    """Extract the centred design and labels from reveal metadata (row order preserved)."""

    reward = np.asarray([float(c.metadata["reward"]) for c in reveals])
    ev = np.asarray([float(c.metadata["ev"]) for c in reveals])
    abs_rpe = np.asarray([float(c.metadata["abs_rpe"]) for c in reveals])
    interaction = np.asarray([float(c.metadata["reward_x_abs_rpe"]) for c in reveals])
    # Centre the slope columns (intercept absorbs the means; slopes — the directions we read —
    # are unchanged, and conditioning improves).
    core = np.column_stack([reward, ev, abs_rpe, interaction])
    core = core - core.mean(axis=0)
    design = np.column_stack([np.ones(len(reveals)), core])
    signs = np.asarray([int(c.metadata["rpe_sign"]) for c in reveals], dtype=np.int64)
    reward_cell = np.asarray([str(c.metadata["reward_cell_id"]) for c in reveals])
    ev_cell = np.asarray([str(c.metadata["ev_cell_id"]) for c in reveals])
    partition = np.asarray([str(c.metadata["partition"]) for c in reveals])
    return RevealArrays(
        design=design,
        signs=signs,
        abs_rpe=abs_rpe,
        reward_cell=reward_cell,
        ev_cell=ev_cell,
        partition=partition,
    )


def fit_block_coefficients(states: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Per-block multivariate OLS coefficients: ``(n_blocks, n_columns, hidden)``.

    ``states`` is ``(n, n_blocks, hidden)`` and ``design`` is ``(n, n_columns)``; each block's
    hidden vector is regressed on the shared design via least squares.
    """

    n_blocks = states.shape[1]
    coefficients = [
        np.linalg.lstsq(design, states[:, block, :], rcond=None)[0] for block in range(n_blocks)
    ]
    return np.stack(coefficients, axis=0)


def coefficient_directions(coefficients: np.ndarray, column_index: int) -> np.ndarray:
    """Unit direction per block for one design column: ``(n_blocks, hidden)``.

    A zero-norm block coefficient (no dependence on this regressor) yields a zero vector rather
    than raising, so pure-reward / pure-EV codes report a vanishing rival direction.
    """

    raw = coefficients[:, column_index, :]
    directions = np.zeros_like(raw)
    for block in range(raw.shape[0]):
        norm = float(np.linalg.norm(raw[block]))
        if norm > 0.0:
            directions[block] = raw[block] / norm
    return directions


def project_blocks(states: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Project each row's per-block state onto that block's direction: ``(n, n_blocks)``."""

    return np.einsum("nbh,bh->nb", states, directions)


def sign_auroc(projections: np.ndarray, signs: np.ndarray) -> float:
    """Midrank AUROC of positive-sign vs negative-sign projections (>0.5 ⇒ + projects high)."""

    positives = projections[signs > 0]
    negatives = projections[signs < 0]
    if positives.size == 0 or negatives.size == 0:
        return 0.5
    return midrank_auroc(positives.tolist(), negatives.tolist())


def sign_auroc_curve(projections: np.ndarray, signs: np.ndarray) -> list[float]:
    """Per-block sign AUROC of already-projected states: ``[sign_auroc(col, signs) for col]``.

    The single spelling of the per-block curve. ``projections`` is ``(n, n_blocks)`` — project
    first (``project_blocks``) when the caller holds raw states. A plain list is the shared
    shape: the reveal surfaces wrap it in ``np.asarray`` and consume it with ``max`` /
    ``np.argmax``, and the block order (and hence the float ops) is identical to every spelling
    it replaced.
    """

    return [sign_auroc(projections[:, block], signs) for block in range(projections.shape[1])]


def rows_by_group(cell_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Map ids to row indices in first-appearance order.

    The iteration order is load-bearing: the cross-fit fold assignment in
    ``crossfit_held_out`` consumes groups in the input's first-appearance order. Do not sort keys.
    """

    grouped: dict[str, list[int]] = defaultdict(list)
    for row, cell in enumerate(cell_ids):
        grouped[cell].append(row)
    return {cell: np.asarray(rows) for cell, rows in grouped.items()}


def both_sign_cells(cell_ids: np.ndarray, signs: np.ndarray) -> list[str]:
    """Cells (in stimulus order, deduped) that carry BOTH a positive and a negative reveal."""

    present: dict[str, set[int]] = defaultdict(set)
    ordered: list[str] = []
    for cell, sign in zip(cell_ids, signs, strict=True):
        if cell not in present:
            ordered.append(cell)
        present[cell].add(int(sign))
    return [cell for cell in ordered if present[cell] == {1, -1}]


def _within_cell_direction(
    block_states: np.ndarray, signs: np.ndarray, grouped: dict[str, np.ndarray], cells: list[str]
) -> np.ndarray | None:
    """Mean over ``cells`` of the within-cell (positive − negative) mean state.

    Holding the matched variable fixed within each cell removes its between-cell variance; the
    averaged contrast is the matched-sign direction. ``None`` if no listed cell carries both signs.
    """

    diffs: list[np.ndarray] = []
    for cell in cells:
        rows = grouped[cell]
        pos = rows[signs[rows] > 0]
        neg = rows[signs[rows] < 0]
        if pos.size == 0 or neg.size == 0:
            continue
        diffs.append(block_states[pos].mean(axis=0) - block_states[neg].mean(axis=0))
    if not diffs:
        return None
    return np.mean(diffs, axis=0)


def paired_cell_auroc(
    projections: np.ndarray, signs: np.ndarray, grouped: dict[str, np.ndarray], cells: list[str]
) -> float | None:
    """Paired AUROC of per-cell (positive-mean − negative-mean) projection differences.

    Between-cell offset (the held-fixed variable's residual) cancels within each pair, so this
    isolates the sign contrast. ``None`` if no listed cell carries both signs.
    """

    differences: list[float] = []
    for cell in cells:
        rows = grouped[cell]
        pos = rows[signs[rows] > 0]
        neg = rows[signs[rows] < 0]
        if pos.size == 0 or neg.size == 0:
            continue
        differences.append(float(projections[pos].mean() - projections[neg].mean()))
    if not differences:
        return None
    return paired_projection_auroc(differences)


def crossfit_held_out(
    block_states: np.ndarray,
    signs: np.ndarray,
    grouped: dict[str, np.ndarray],
    order: list[str],
    n_folds: int,
) -> np.ndarray:
    """Cross-fitted held-out projection per row (NaN where a fold's direction degenerates).

    Each fold's cells are scored by a within-cell sign direction fit on the OTHER folds' cells, so
    no cell trains on itself. The permutation null recomputes this end to end under permuted signs —
    refitting the direction each time — so the p-value captures the direction-fitting optimism.
    """

    held_out = np.full(len(signs), np.nan)
    for fold in range(n_folds):
        test_cells = order[fold::n_folds]
        train_cells = [cell for cell in order if cell not in set(test_cells)]
        direction_raw = _within_cell_direction(block_states, signs, grouped, train_cells)
        if direction_raw is None or float(np.linalg.norm(direction_raw)) == 0.0:
            continue
        direction = unit_vector(direction_raw)
        for cell in test_cells:
            rows = grouped[cell]
            held_out[rows] = block_states[rows] @ direction
    return held_out


@dataclass(frozen=True)
class StabilityResult:
    """Draw-grouped, K-averaged split-half reliability of the ``v_RPE`` direction."""

    cosine: float  # mean split-half cosine over the K grouped splits — the GATED metric
    cosine_std: float  # std of the cosine across the K splits (dispersion of the estimate)
    n_splits: int  # non-degenerate splits actually scored (== K unless a half degenerated)
    rowsplit_legacy: float  # the pre-fix single row-level split cosine (continuity with 647a8f3)


def _fit_reward_direction(design: np.ndarray, block_states: np.ndarray) -> np.ndarray | None:
    """Unit ``v_RPE`` (the reward slope) from an OLS fit; ``None`` if the slope degenerates."""

    coef = np.linalg.lstsq(design, block_states, rcond=None)[0]
    vector = coef[REWARD_INDEX]
    if float(np.linalg.norm(vector)) == 0.0:
        return None
    return unit_vector(vector)


def grouped_half_indices(
    groups: dict[str, np.ndarray], rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    """Deal whole DRAWS into two count-balanced halves so no draw spans the split (leakage-safe).

    ``groups`` maps a draw id (``ev_cell`` = ``draw_id``) to its estimation row indices, so all of a
    draw's ``renderings_per_reveal × 2`` reveals stay together. A row-level shuffle would drop the
    same draw's renderings / opposite outcomes into both halves and inflate the split-half cosine —
    the same per-reveal-partition leakage the matched contests avoid (see ``matched_sign_contest``).
    """

    order = list(groups.keys())
    rng.shuffle(order)
    cut = (len(order) + 1) // 2
    left = [groups[draw] for draw in order[:cut]]
    right = [groups[draw] for draw in order[cut:]]
    empty = np.empty(0, dtype=int)
    return (
        np.concatenate(left) if left else empty,
        np.concatenate(right) if right else empty,
    )


def _rowsplit_cosine(design: np.ndarray, block_states: np.ndarray, *, seed: int) -> float:
    """The pre-fix single row-level split-half cosine (continuity with the 647a8f3 capture).

    This is the leakage-inflated metric the first R-A capture reported (0.776) — the same draw's
    renderings / outcomes can land in opposite halves. Recorded alongside the honest grouped metric
    so the pure-N effect is separable from the honesty-fix effect at the widened N.
    """

    n = block_states.shape[0]
    rng = random.Random(f"{seed}|reveal-rpe-stability")
    order = list(range(n))
    rng.shuffle(order)
    cut = (n + 1) // 2
    directions: list[np.ndarray] = []
    for rows in (np.asarray(order[:cut]), np.asarray(order[cut:])):
        direction = _fit_reward_direction(design[rows], block_states[rows])
        if direction is None:
            return 0.0
        directions.append(direction)
    return float(np.dot(directions[0], directions[1]))


def split_half_stability(
    estimation_states: np.ndarray,
    design: np.ndarray,
    draw_ids: np.ndarray,
    *,
    block: int,
    seed: int = EXTRACTION_SEED,
    n_splits: int = STABILITY_SPLITS,
) -> StabilityResult:
    """Draw-grouped, K-averaged split-half cosine of the ``v_RPE`` direction at ``block``.

    Each of ``n_splits`` seeded splits deals whole draws into two disjoint halves (so a draw never
    trains both directions), fits the reward-slope ``v_RPE`` on each half's rows, and takes their
    cosine; the mean over splits is a low-variance, leakage-safe estimate of split-half reliability
    (the gated metric). The legacy single row-level split is recorded too for continuity.
    """

    block_states = estimation_states[:, block, :]
    rowsplit = _rowsplit_cosine(design, block_states, seed=seed)

    groups = rows_by_group(draw_ids)
    rng = random.Random(f"{seed}|reveal-rpe-stability-grouped")
    cosines: list[float] = []
    for _ in range(n_splits):
        left, right = grouped_half_indices(groups, rng)
        if left.size == 0 or right.size == 0:
            continue
        d_left = _fit_reward_direction(design[left], block_states[left])
        d_right = _fit_reward_direction(design[right], block_states[right])
        if d_left is None or d_right is None:
            continue
        cosines.append(float(np.dot(d_left, d_right)))

    if not cosines:
        return StabilityResult(cosine=0.0, cosine_std=0.0, n_splits=0, rowsplit_legacy=rowsplit)
    scores = np.asarray(cosines, dtype=np.float64)
    return StabilityResult(
        cosine=float(scores.mean()),
        cosine_std=float(scores.std()),
        n_splits=len(cosines),
        rowsplit_legacy=rowsplit,
    )


def random_direction_floor(
    score: Callable[[np.ndarray], float],
    *,
    rng: np.random.Generator,
    hidden_size: int,
    n_directions: int,
    quantile: float,
    method: str,
) -> float:
    """Upper-quantile of ``score`` over ``n_directions`` gaussian random directions.

    The shared mechanism behind every random-direction null floor: one
    ``rng.standard_normal(hidden_size)`` draw per replicate, ``score`` per draw, one quantile over
    the collected values. The kernel NEVER seeds — adapters construct their own generator, so
    adopting a site leaves its draw stream byte-identical. Normalization, sign-flips and ``abs``
    belong to the caller's ``score`` closure, never here: they are what distinguishes the sites.
    ``method`` is mandatory because the interpolation-vs-nearest-rank choice flips pass/fail at
    K≈20 draws, so every adopter states it explicitly.
    """

    values = [score(rng.standard_normal(hidden_size)) for _ in range(n_directions)]
    # ``method`` is a mandatory plain ``str`` on this interface (D1); numpy's stub narrows it to a
    # literal union, so the widening lives here rather than leaking a numpy-private type outward.
    return float(np.quantile(values, quantile, method=cast(Any, method)))


def held_out_auroc_floor(
    estimation_states: np.ndarray,
    estimation_signs: np.ndarray,
    selection_states: np.ndarray,
    selection_signs: np.ndarray,
    *,
    block: int,
    n_directions: int = DEFAULT_RANDOM_DIRECTIONS,
    seed: int = EXTRACTION_SEED,
) -> float:
    """Upper-quantile held-out sign AUROC of random directions at ``block`` (the external floor).

    Each random direction is oriented on the estimation partition (flipped so the positive sign
    projects high there) and scored on selection — the same held-out discipline the fitted ``v_RPE``
    obeys. Orienting on estimation rather than folding ``max(auroc, 1-auroc)`` avoids peeking at the
    selection labels to pick the direction's sign.
    """

    est_block = estimation_states[:, block, :]
    sel_block = selection_states[:, block, :]

    def score(direction: np.ndarray) -> float:
        direction = direction / np.linalg.norm(direction)
        if sign_auroc(est_block @ direction, estimation_signs) < 0.5:
            direction = -direction
        return sign_auroc(sel_block @ direction, selection_signs)

    return random_direction_floor(
        score,
        rng=np.random.default_rng(seed_int(seed, "reveal-rpe-random-floor", block)),
        hidden_size=selection_states.shape[2],
        n_directions=n_directions,
        quantile=FLOOR_QUANTILE,
        method="linear",  # numpy's default, stated explicitly (D1)
    )


def _abs_rpe_magnitude_labels(abs_rpe: np.ndarray) -> np.ndarray:
    """High- vs low-|RPE| labels (+1/-1) via a median split — the unsigned-surprise label."""

    median = float(np.median(abs_rpe))
    return np.where(abs_rpe > median, 1, -1).astype(np.int64)


def abs_rpe_magnitude_signal(
    v_absrpe_directions: np.ndarray,
    estimation_states: np.ndarray,
    estimation_abs: np.ndarray,
    selection_states: np.ndarray,
    selection_abs: np.ndarray,
    *,
    block: int,
    n_directions: int = DEFAULT_RANDOM_DIRECTIONS,
    seed: int = EXTRACTION_SEED,
) -> tuple[float, float]:
    """(held-out |RPE|-magnitude AUROC of ``v_absrpe``, random-direction floor) at ``block``.

    Positive evidence that the reveal-token axis carries UNSIGNED surprise: the |RPE| coefficient
    direction separates high- from low-|RPE| reveals on the held-out selection partition above the
    random-direction floor. ``v_absrpe`` is already estimation-oriented by the OLS fit (it points
    toward larger |RPE|), so the floor's estimation-orient / selection-score discipline is matched.
    Without this evidence a run where nothing separates must NOT be reported as an unsigned-surprise
    collapse (it is ``indeterminate``).
    """

    est_labels = _abs_rpe_magnitude_labels(estimation_abs)
    sel_labels = _abs_rpe_magnitude_labels(selection_abs)
    observed = sign_auroc(
        project_blocks(selection_states, v_absrpe_directions)[:, block], sel_labels
    )
    floor = held_out_auroc_floor(
        estimation_states,
        est_labels,
        selection_states,
        sel_labels,
        block=block,
        n_directions=n_directions,
        seed=seed_int(seed, "abs-rpe-magnitude-floor"),
    )
    return observed, floor
