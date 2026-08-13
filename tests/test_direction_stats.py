"""Kernel behaviour for the shared direction-fitting and held-out statistics.

The parent's tests/test_direction_stats.py @ 10c4662 is a governance file: a cross-surface
byte-parity fixture plus AST guards on private imports and a frozen ``__all__`` ledger. None of
that machinery is ported, so this file tests what the kernels actually compute — the design
matrix's construction-guaranteed orthogonalities, the OLS slope recovery, the AUROC conventions,
the cross-fit's held-out discipline, and the random-direction floor's chance behaviour.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from appraisal_emotions.analysis.direction_stats import (
    ABS_RPE_INDEX,
    DESIGN_COLUMNS,
    EV_INDEX,
    FLOOR_QUANTILE,
    REWARD_INDEX,
    both_sign_cells,
    build_reveal_arrays,
    coefficient_directions,
    crossfit_held_out,
    fit_block_coefficients,
    held_out_auroc_floor,
    paired_cell_auroc,
    project_blocks,
    random_direction_floor,
    rows_by_group,
    seed_int,
    sign_auroc,
    sign_auroc_curve,
    split_half_stability,
)
from appraisal_emotions.stimuli.reveal_probes import DESIGN_REGRESSORS
from conftest import HIDDEN, N_BLOCKS, cosine, orthonormal, planted_states


def test_design_columns_mirror_the_battery_regressors():
    # signed_rpe (= reward − ev) must NEVER be a column: [1, reward, ev, signed_rpe] is rank
    # deficient, which is exactly why a diff-in-means cannot identify RPE.
    assert ("intercept", *DESIGN_REGRESSORS) == DESIGN_COLUMNS
    assert "signed_rpe" not in DESIGN_COLUMNS


def test_reveal_arrays_are_centred_full_rank_and_row_aligned(reveals):
    arrays = build_reveal_arrays(reveals)
    n = len(reveals)
    assert arrays.design.shape == (n, len(DESIGN_COLUMNS))
    assert np.allclose(arrays.design[:, 0], 1.0)  # intercept
    assert np.allclose(arrays.design[:, 1:].mean(axis=0), 0.0)  # slopes centred
    assert np.linalg.matrix_rank(arrays.design) == len(DESIGN_COLUMNS)
    # Row order is preserved: labels line up with the battery's own metadata.
    assert arrays.signs.tolist() == [int(c.metadata["rpe_sign"]) for c in reveals]
    assert arrays.ev_cell.tolist() == [str(c.metadata["ev_cell_id"]) for c in reveals]
    assert set(arrays.partition.tolist()) == {"estimation", "selection"}
    # Balanced both-outcome enumeration makes the derived signed RPE exactly orthogonal to the
    # centred EV and |RPE| columns — the identification the battery buys by construction.
    signed_rpe = np.asarray([float(c.metadata["signed_rpe"]) for c in reveals])
    assert float(signed_rpe @ arrays.design[:, EV_INDEX]) == pytest.approx(0.0, abs=1e-6)
    assert float(signed_rpe @ arrays.design[:, ABS_RPE_INDEX]) == pytest.approx(0.0, abs=1e-6)


def test_fit_block_coefficients_recovers_a_planted_slope(reveals):
    (u,) = orthonormal(seed=11, count=1)
    states = planted_states(reveals, lambda meta: 2.0 * meta["reward"] * u, seed=12, noise=0.01)
    arrays = build_reveal_arrays(reveals)
    est = arrays.partition == "estimation"
    coef = fit_block_coefficients(states[est], arrays.design[est])
    assert coef.shape == (N_BLOCKS, len(DESIGN_COLUMNS), HIDDEN)
    reward_slope = coef[0, REWARD_INDEX, :]  # H = 2 * reward * u  =>  slope ~ 2 u
    assert abs(cosine(reward_slope, u)) > 0.99
    assert np.linalg.norm(reward_slope) == pytest.approx(2.0, rel=0.1)


def test_coefficient_directions_are_unit_norm_or_exactly_zero():
    coef = np.zeros((3, len(DESIGN_COLUMNS), 4))
    coef[0, REWARD_INDEX] = [3.0, 4.0, 0.0, 0.0]
    coef[1, REWARD_INDEX] = [0.0, 0.0, 0.0, 0.0]  # a vanishing rival direction
    coef[2, REWARD_INDEX] = [0.0, -2.0, 0.0, 0.0]
    directions = coefficient_directions(coef, REWARD_INDEX)
    assert directions[0].tolist() == pytest.approx([0.6, 0.8, 0.0, 0.0])
    assert np.linalg.norm(directions[1]) == 0.0
    assert np.linalg.norm(directions[2]) == pytest.approx(1.0)


def test_project_blocks_matches_a_manual_per_block_dot():
    states = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    directions = np.eye(4)[:3]
    projections = project_blocks(states, directions)
    assert projections.shape == (2, 3)
    for row in range(2):
        for block in range(3):
            assert projections[row, block] == pytest.approx(
                float(states[row, block] @ directions[block])
            )


def test_sign_auroc_orientation_ties_and_degenerate_classes():
    signs = np.array([1, 1, -1, -1])
    assert sign_auroc(np.array([2.0, 3.0, 0.0, 1.0]), signs) == 1.0
    assert sign_auroc(np.array([0.0, 1.0, 2.0, 3.0]), signs) == 0.0
    assert sign_auroc(np.array([1.0, 1.0, 1.0, 1.0]), signs) == 0.5  # all ties count one half
    # A single-class sample cannot separate; the kernel reports chance instead of raising.
    assert sign_auroc(np.array([1.0, 2.0]), np.array([1, 1])) == 0.5


def test_sign_auroc_curve_is_the_per_block_spelling():
    projections = np.array([[2.0, 0.0], [3.0, 1.0], [0.0, 2.0], [1.0, 3.0]])
    signs = np.array([1, 1, -1, -1])
    assert sign_auroc_curve(projections, signs) == [1.0, 0.0]


def test_rows_by_group_keeps_first_appearance_order():
    cells = np.array(["b", "a", "b", "c"])
    grouped = rows_by_group(cells)
    assert list(grouped) == ["b", "a", "c"]  # load-bearing for the cross-fit fold assignment
    assert grouped["b"].tolist() == [0, 2]


def test_both_sign_cells_keeps_only_two_sided_cells_in_stimulus_order():
    cells = np.array(["z", "z", "a", "a", "m"])
    signs = np.array([1, -1, 1, 1, -1])
    assert both_sign_cells(cells, signs) == ["z"]


def test_paired_cell_auroc_cancels_between_cell_offsets():
    # Each cell carries a large offset; only the within-cell (positive − negative) difference
    # should matter, so a uniformly positive difference scores 1.0 despite the offsets.
    grouped = {"a": np.array([0, 1]), "b": np.array([2, 3])}
    signs = np.array([1, -1, 1, -1])
    projections = np.array([101.0, 100.0, -49.0, -50.0])
    assert paired_cell_auroc(projections, signs, grouped, ["a", "b"]) == 1.0
    # A cell with only one sign contributes nothing; no scorable cell returns None.
    assert paired_cell_auroc(projections, np.array([1, 1, 1, 1]), grouped, ["a", "b"]) is None


def test_crossfit_held_out_never_scores_a_cell_on_its_own_fit():
    # Two cells, opposite within-cell directions. With one cell per fold the direction scoring a
    # cell comes only from the OTHER cell, so the held-out projections must invert relative to a
    # self-fit. A leaky implementation would report a positive difference for both.
    block_states = np.array([[1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
    signs = np.array([1, -1, 1, -1])
    grouped = {"a": np.array([0, 1]), "b": np.array([2, 3])}
    held_out = crossfit_held_out(block_states, signs, grouped, ["a", "b"], 2)
    assert not np.isnan(held_out).any()
    # Cell "a" is scored by cell "b"'s direction (which points the other way), and vice versa.
    assert held_out[0] < held_out[1]
    assert held_out[2] < held_out[3]


def test_crossfit_marks_degenerate_folds_nan_instead_of_guessing():
    # A single cell means the train side of its fold is empty: the row must come back NaN so the
    # caller drops it from the scored set rather than scoring it on its own contrast.
    block_states = np.array([[1.0, 0.0], [-1.0, 0.0]])
    signs = np.array([1, -1])
    grouped = {"a": np.array([0, 1])}
    held_out = crossfit_held_out(block_states, signs, grouped, ["a"], 1)
    assert np.isnan(held_out).all()


def test_split_half_stability_separates_signal_from_noise(reveals):
    axes = orthonormal(seed=41, count=1)
    arrays = build_reveal_arrays(reveals)
    est = arrays.partition == "estimation"
    signal_states = planted_states(
        reveals, lambda meta: 8.0 * meta["signed_rpe"] * axes[0], seed=42
    )
    noise_states = planted_states(reveals, lambda _meta: np.zeros(HIDDEN), seed=43, noise=1.0)
    signal = split_half_stability(
        signal_states[est], arrays.design[est], arrays.ev_cell[est], block=0, n_splits=20
    )
    noise = split_half_stability(
        noise_states[est], arrays.design[est], arrays.ev_cell[est], block=0, n_splits=20
    )
    assert signal.cosine > 0.8
    assert noise.cosine < 0.3
    assert signal.n_splits == 20
    assert -1.0 <= signal.rowsplit_legacy <= 1.0


def test_random_direction_floor_is_the_requested_quantile():
    # The kernel draws one gaussian direction per replicate and takes the quantile of the scores;
    # with a score that ignores the draw the floor is exactly that constant.
    floor = random_direction_floor(
        lambda direction: float(direction[0] > -np.inf),
        rng=np.random.default_rng(0),
        hidden_size=4,
        n_directions=10,
        quantile=FLOOR_QUANTILE,
        method="linear",
    )
    assert floor == 1.0


def test_held_out_auroc_floor_sits_near_chance_on_unstructured_states(reveals):
    # Random directions oriented on estimation and scored on selection must not separate sign in
    # noise; a floor far above chance would mean the orientation step is peeking at the labels.
    arrays = build_reveal_arrays(reveals)
    est = arrays.partition == "estimation"
    sel = arrays.partition == "selection"
    states = planted_states(reveals, lambda _meta: np.zeros(HIDDEN), seed=51, noise=1.0)
    floor = held_out_auroc_floor(
        states[est],
        arrays.signs[est],
        states[sel],
        arrays.signs[sel],
        block=0,
        n_directions=50,
    )
    assert 0.4 < floor < 0.75


def test_seed_int_is_stable_and_parameter_sensitive():
    # Not Python's per-process-salted hash(): the same parts must give the same 64-bit seed in
    # every process, else a permutation null is not reproducible.
    assert seed_int("reveal-rpe", 7, "null") == seed_int("reveal-rpe", 7, "null")
    assert seed_int("reveal-rpe", 7, "null") != seed_int("reveal-rpe", 8, "null")
    assert 0 <= seed_int("x") < 2**64


def test_grouped_half_indices_are_reproducible_under_a_seeded_rng():
    from appraisal_emotions.analysis.direction_stats import grouped_half_indices

    groups = {name: np.array([index]) for index, name in enumerate("abcdefgh")}
    first = grouped_half_indices(groups, random.Random("fixed"))
    second = grouped_half_indices(groups, random.Random("fixed"))
    assert first[0].tolist() == second[0].tolist()
    assert first[1].tolist() == second[1].tolist()
    assert len(first[0]) + len(first[1]) == len(groups)
