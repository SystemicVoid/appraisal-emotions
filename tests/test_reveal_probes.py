"""Stimulus invariants for the R-A reveal-probe battery (rq8_rpe_valence_spec §4.4).

Adapted (imports and the shared ``battery`` fixture only) from functional-valence-validity
tests/test_reveal_probes.py @ 10c4662. The whole file survives: these are exactly the design
invariants — exact orthogonality, balanced both-outcome enumeration, matched-cell coverage,
byte-pinned reveal slot, surface uniqueness, affect neutrality — that the reveal-RPE
identification argument rests on.
"""

from __future__ import annotations

from collections import defaultdict
from typing import cast

import numpy as np
import pytest

from appraisal_emotions.stimuli.emotion_lexicon import emotion_words_in
from appraisal_emotions.stimuli.gambles import GRID_MAX, GRID_STEP, GambleGridConfig
from appraisal_emotions.stimuli.reveal_probes import REVEAL_PREFIX, build_reveal_battery
from conftest import GRID_CONFIG


def _column(reveals, key):
    return np.asarray([float(c.metadata[key]) for c in reveals])


def test_reward_and_signed_rpe_algebra(battery):
    for c in battery.reveals:
        meta = c.metadata
        expected_reward = meta["gamble_high"] if meta["realised_high"] else meta["gamble_low"]
        assert meta["reward"] == pytest.approx(expected_reward)
        assert meta["signed_rpe"] == pytest.approx(meta["reward"] - meta["ev"])
        assert meta["abs_rpe"] == pytest.approx(abs(meta["signed_rpe"]))
        assert meta["abs_rpe"] == pytest.approx(meta["spread"] / 2.0)
        assert meta["rpe_sign"] == (1 if meta["realised_high"] else -1)
        assert meta["reward_x_abs_rpe"] == pytest.approx(meta["reward"] * meta["abs_rpe"])


def test_every_draw_emits_both_outcomes_balanced(battery):
    signs_by_draw = defaultdict(set)
    counts_by_draw = defaultdict(lambda: {1: 0, -1: 0})
    for c in battery.reveals:
        draw = c.metadata["draw_id"]
        sign = c.metadata["rpe_sign"]
        signs_by_draw[draw].add(sign)
        counts_by_draw[draw][sign] += 1
    for draw, signs in signs_by_draw.items():
        assert signs == {1, -1}, f"draw {draw} is not sign-balanced"
        assert counts_by_draw[draw][1] == counts_by_draw[draw][-1]


def test_signed_rpe_exactly_orthogonal_to_ev_and_abs_rpe(battery):
    # Balanced both-outcome enumeration forces Σ srpe·ev = Σ srpe·|rpe| = 0 per draw.
    signed_rpe = _column(battery.reveals, "signed_rpe")
    ev = _column(battery.reveals, "ev")
    abs_rpe = _column(battery.reveals, "abs_rpe")
    reward = _column(battery.reveals, "reward")
    assert np.corrcoef(signed_rpe, ev)[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert np.corrcoef(signed_rpe, abs_rpe)[0, 1] == pytest.approx(0.0, abs=1e-9)
    # The one rival that stays entangled with signed RPE is realised reward.
    assert np.corrcoef(signed_rpe, reward)[0, 1] > 0.1


def test_design_is_full_rank_and_well_conditioned(battery):
    manifest = battery.manifest
    assert manifest["design_rank"] == len(manifest["design_regressors"]) + 1  # + intercept
    assert manifest["design_full_rank"] is True
    assert manifest["design_condition_number"] < 30.0
    assert manifest["vif_signed_rpe_given_ev_absrpe"] < 1.01
    assert manifest["abs_rpe_sign_balanced"] is True


def test_prompt_terminates_at_byte_pinned_reveal_slot(battery):
    for c in battery.reveals:
        assert c.prompt.endswith(REVEAL_PREFIX)
        assert not c.prompt.endswith(REVEAL_PREFIX + " ")
        read_prefix = c.metadata["read_prefix"]
        assert read_prefix.startswith(" ")
        assert read_prefix == f" {c.metadata['realised_symbol']}"
        # The revealed symbol is symbol_high iff the good outcome realised.
        expected = (
            c.metadata["symbol_high"] if c.metadata["realised_high"] else c.metadata["symbol_low"]
        )
        assert c.metadata["realised_symbol"] == expected
        # The outcome amounts are shown so EV is represented in-context.
        assert f"{c.metadata['gamble_high']} points" in c.prompt
        assert f"{c.metadata['gamble_low']} points" in c.prompt


def test_augmentation_widens_reward_matched_coverage():
    augmented = build_reveal_battery(GRID_CONFIG, reward_matched_augment=True)
    base = build_reveal_battery(GRID_CONFIG, reward_matched_augment=False)
    # Every draw is an EV-matched pair carrying both signs, by construction.
    assert augmented.manifest["n_ev_matched_pairs_both_signs"] == augmented.manifest["n_draws"]
    assert base.manifest["n_ev_matched_pairs_both_signs"] == base.manifest["n_draws"]
    # The augmentation manufactures reward-matched cells beyond the base grid's handful.
    augmented_cells = cast(int, augmented.manifest["n_reward_matched_cells_both_signs"])
    base_cells = cast(int, base.manifest["n_reward_matched_cells_both_signs"])
    assert augmented_cells > base_cells
    assert augmented_cells >= 20


def test_partitions_are_estimation_or_selection_only(battery):
    partitions = {c.metadata["partition"] for c in battery.reveals}
    assert partitions <= {"estimation", "selection"}
    assert "confirmation" not in partitions
    assert battery.manifest["reveals_by_partition"]["estimation"] > 0
    assert battery.manifest["reveals_by_partition"]["selection"] > 0


def test_comparison_ids_unique(battery):
    ids = [c.comparison_id for c in battery.reveals]
    assert len(ids) == len(set(ids))


def test_reveal_surfaces_are_unique_across_the_battery(battery):
    # On a deterministic (temperature-0) capture two identical (prompt, read_prefix) rows are the
    # SAME activation vector. If a draw's rendering_index copies collided and the partition split
    # scattered them across estimation/selection, the held-out AUROC/null would leak and effective
    # n inflate. Distinct-surface sampling makes every reveal surface unique, so no identical
    # capture can straddle the split (a fortiori none can cross partitions).
    surfaces = [(c.prompt, c.metadata["read_prefix"]) for c in battery.reveals]
    assert len(surfaces) == len(set(surfaces))
    # And no surface group spans partitions — the exact leak, stated directly.
    partitions_by_surface = defaultdict(set)
    for c in battery.reveals:
        partitions_by_surface[(c.prompt, c.metadata["read_prefix"])].add(c.metadata["partition"])
    assert all(len(parts) == 1 for parts in partitions_by_surface.values())


def test_renderings_beyond_surface_capacity_fail_closed():
    # A 2-symbol shared + 2-symbol held-out pool × 1 template family = 8 distinct surfaces; asking
    # for more distinct renderings than exist must raise, not silently duplicate or loop forever.
    narrow = GambleGridConfig(
        seed=7,
        shared_symbols=("AA", "BB"),
        held_out_symbols=("CC", "DD"),
        template_families=("ledger",),
    )
    with pytest.raises(ValueError, match="distinct reveal surfaces"):
        build_reveal_battery(narrow, renderings_per_reveal=9)


def test_overlapping_symbol_pools_fail_closed():
    # The reveal surface ignores stratum, so a symbol in BOTH pools would render a byte-identical
    # capture under each stratum with a different comparison_id / partition — re-opening the
    # estimation/selection leak distinct-surface sampling closes. Overlap must raise.
    with pytest.raises(ValueError, match="disjoint"):
        build_reveal_battery(
            GambleGridConfig(seed=7, shared_symbols=("AA", "BB"), held_out_symbols=("BB", "CC"))
        )


def test_build_is_deterministic():
    first = build_reveal_battery(GRID_CONFIG)
    second = build_reveal_battery(GRID_CONFIG)
    assert [c.comparison_id for c in first.reveals] == [c.comparison_id for c in second.reveals]
    assert [c.metadata["read_prefix"] for c in first.reveals] == [
        c.metadata["read_prefix"] for c in second.reveals
    ]


def test_surface_is_affect_neutral(battery):
    assert battery.affect_audit_warnings == ()
    for c in battery.reveals:
        assert not emotion_words_in(c.prompt)


def test_grid_legal_outcomes(battery):
    for c in battery.reveals:
        for key in ("gamble_high", "gamble_low"):
            assert c.metadata[key] % GRID_STEP == 0
            assert abs(c.metadata[key]) <= GRID_MAX
        assert c.metadata["gamble_high"] > c.metadata["gamble_low"]
