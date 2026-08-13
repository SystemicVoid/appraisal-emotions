"""Stimulus invariants for the described-gamble algebra and rendering.

Distilled from functional-valence-validity tests/test_described_gambles.py @ 10c4662 (984 lines)
down to the invariants the reveal path depends on: the round-number grid, within-cell RPE
symmetry, EV/spread decorrelation, trial-type sign rules, the byte-pinned reveal slot and the
exactly-one-leading-space symbol rendering. Tests of the dropped builders (choice prompts,
sessions, proposition / choicepointer batteries, the indifference sweep, the 3-way partition
router) went with their subjects.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from appraisal_emotions.stimuli.gambles import (
    _REVEAL_PREFIX,
    CR_GRID_MAX,
    EV_LEVELS,
    GRID_MAX,
    GRID_STEP,
    TRIAL_TYPES,
    GambleGridConfig,
    GambleSpec,
    SymbolAssignment,
    _audit_affect_neutral,
    _candidate_draws,
    _ev_level_bins,
    _outcomes_match_trial_type,
    _reveal_line,
    _select_decorrelated,
    _template_header,
    render_options_block,
)


def test_within_cell_rpe_symmetry_for_every_candidate():
    # §4.3 decorrelation 3: 50/50 gambles give RPE_good = +d/2, RPE_bad = -d/2. RPE is
    # CR-independent, so the draw enumerator (one spec per distinct high/low) suffices.
    for trial_type in TRIAL_TYPES:
        for spec in _candidate_draws(trial_type):
            assert spec.rpe_good == pytest.approx(-spec.rpe_bad)
            assert spec.rpe_good == pytest.approx(spec.abs_rpe)
            assert spec.abs_rpe == pytest.approx(spec.spread / 2.0)


def test_candidate_draws_are_grid_legal_and_sign_valid():
    # Round-number grid (F13): OUTCOMES are multiples of 10 with |.| <= 80, EV a multiple of 5.
    for trial_type in TRIAL_TYPES:
        draws = _candidate_draws(trial_type)
        assert draws
        for spec in draws:
            assert spec.high > spec.low
            for value in (spec.high, spec.low):
                assert value % GRID_STEP == 0
                assert abs(value) <= GRID_MAX
            assert (spec.ev * 2) % 10 == 0  # p = 0.5 ⇒ EV is a multiple of 5
            assert _outcomes_match_trial_type(trial_type, spec.high, spec.low)
        # Draws are distinct (high, low) pairs — the enumerator emits one spec each.
        pairs = [(spec.high, spec.low) for spec in draws]
        assert len(pairs) == len(set(pairs))


def test_trial_type_sign_rules_partition_the_grid():
    for trial_type in TRIAL_TYPES:
        for spec in _candidate_draws(trial_type):
            if trial_type == "gain":
                assert spec.low >= 0 and spec.high > 0
            elif trial_type == "loss":
                assert spec.high <= 0 and spec.low < 0
            else:
                assert spec.high > 0 and spec.low < 0


def test_ev_level_bins_are_terciles_and_cover_every_ev():
    draws = _candidate_draws("gain")
    bins = _ev_level_bins(draws)
    assert set(bins.values()) == set(EV_LEVELS)
    assert all(spec.ev in bins for spec in draws)


def test_selection_decorrelates_ev_from_spread():
    # F7: within one EV level the selection must span several spreads, else v_EV aliases a
    # stake/|RPE| axis. Round-robin over spread buckets is what guarantees it.
    import random

    for trial_type in TRIAL_TYPES:
        draws = _candidate_draws(trial_type)
        bins = _ev_level_bins(draws)
        for ev_level in EV_LEVELS:
            selected = _select_decorrelated(
                draws,
                ev_level=ev_level,
                ev_bins=bins,
                count=8,
                rng=random.Random(f"test|{trial_type}|{ev_level}"),
            )
            assert selected
            assert {bins[spec.ev] for spec in selected} == {ev_level}
            assert len({spec.spread for spec in selected}) >= 2
        # Pooled over the levels, EV and spread are near-uncorrelated across distinct draws.
        pooled = [
            spec
            for ev_level in EV_LEVELS
            for spec in _select_decorrelated(
                draws,
                ev_level=ev_level,
                ev_bins=bins,
                count=8,
                rng=random.Random(f"test-pooled|{trial_type}|{ev_level}"),
            )
        ]
        ev = np.array([spec.ev for spec in pooled])
        spread = np.array([spec.spread for spec in pooled])
        assert abs(np.corrcoef(ev, spread)[0, 1]) < 0.4


def test_selection_is_seed_deterministic():
    import random

    draws = _candidate_draws("mixed")
    bins = _ev_level_bins(draws)
    picks = [
        [
            (spec.high, spec.low)
            for spec in _select_decorrelated(
                draws, ev_level="mid", ev_bins=bins, count=6, rng=random.Random("fixed")
            )
        ]
        for _ in range(2)
    ]
    assert picks[0] == picks[1]


def test_gamble_spec_rejects_off_grid():
    with pytest.raises(ValueError):
        GambleSpec(trial_type="gain", cr=5, high=10, low=0)  # cr not a multiple of 10
    with pytest.raises(ValueError):
        GambleSpec(trial_type="gain", cr=10, high=10, low=20)  # high <= low
    with pytest.raises(ValueError, match="outcome"):
        GambleSpec(trial_type="gain", cr=10, high=90, low=0)  # outcome exceeds GRID_MAX
    with pytest.raises(ValueError, match="cr"):
        GambleSpec(trial_type="gain", cr=CR_GRID_MAX + 10, high=80, low=0)  # cr exceeds ceiling


def test_gamble_spec_allows_wide_cr():
    # CR decoupling: a certain option above the ±80 outcome grid (but <= CR_GRID_MAX) is legal —
    # the certain option is read literally, so F13's outcome bound does not apply to it.
    spec = GambleSpec(trial_type="gain", cr=160, high=80, low=0)
    assert spec.cr == 160
    assert spec.ev == 40.0  # EV derives from outcomes only, unaffected by the wide CR
    assert spec.advantage == -120.0
    assert spec.magnitude == 40.0


def test_symbol_stratum_requires_two_distinct_symbols():
    # Count alone is not enough: a duplicate stratum could let a symbol assignment draw the same
    # symbol for both outcomes, so the validator requires DISTINCT symbols.
    with pytest.raises(ValueError, match="two distinct candidate symbols"):
        GambleGridConfig(shared_symbols=("Q", "Q"))
    with pytest.raises(ValueError, match="two distinct candidate symbols"):
        GambleGridConfig(held_out_symbols=("Q", "Q"))
    with pytest.raises(ValueError, match="template family"):
        GambleGridConfig(template_families=())


def test_options_block_counterbalances_the_symbol_order():
    spec = GambleSpec(trial_type="mixed", cr=0, high=40, low=-20)
    forward = SymbolAssignment(
        stratum="shared", symbol_high="QWXV", symbol_low="ZPKM", order="forward"
    )
    reverse = SymbolAssignment(
        stratum="shared", symbol_high="QWXV", symbol_low="ZPKM", order="reverse"
    )
    assert render_options_block(spec, forward).splitlines() == [
        " QWXV = 40 points",
        " ZPKM = -20 points",
    ]
    # The reverse order lists the SAME two lines, swapped: only presentation order changes.
    assert render_options_block(spec, reverse).splitlines() == list(
        reversed(render_options_block(spec, forward).splitlines())
    )


def test_symbols_render_after_exactly_one_space():
    # The preflight single-token gate validates the single-leading-space form ``" {symbol}"``.
    # That gate is only representative of the symbol's in-prompt tokenization if EVERY
    # model-facing rendering puts the symbol after exactly one U+0020 space: a bare/newline
    # rendering, OR a multi-space indent, can tokenize the symbol as a DIFFERENT token than the
    # gated ``" symbol"`` on some tokenizers and silently defeat the gate.
    spec = GambleSpec(trial_type="gain", cr=10, high=60, low=20)
    symbols = SymbolAssignment(
        stratum="held_out", symbol_high="QWXV", symbol_low="ZPKM", order="forward"
    )
    rendered = [
        render_options_block(spec, symbols),
        _reveal_line(spec, symbols, realised_high=True),
        _reveal_line(spec, symbols, realised_high=False),
    ]
    for text in rendered:
        for symbol in (symbols.symbol_high, symbols.symbol_low):
            start = 0
            while (idx := text.find(symbol, start)) != -1:
                context = text[max(0, idx - 24) : idx + len(symbol) + 4]
                assert idx > 0 and text[idx - 1] == " " and (idx < 2 or text[idx - 2] != " "), (
                    f"symbol {symbol!r} not preceded by exactly one space: ...{context!r}"
                )
                start = idx + len(symbol)


def test_reveal_line_varies_only_the_symbol_token():
    # §4.2 R3 byte-identical outcome slot: the two realisations differ in the symbol alone.
    spec = GambleSpec(trial_type="gain", cr=10, high=60, low=20)
    symbols = SymbolAssignment(
        stratum="shared", symbol_high="QWXV", symbol_low="ZPKM", order="forward"
    )
    high = _reveal_line(spec, symbols, realised_high=True)
    low = _reveal_line(spec, symbols, realised_high=False)
    assert high.startswith(f"{_REVEAL_PREFIX} ")
    assert low.startswith(f"{_REVEAL_PREFIX} ")
    assert high.replace(symbols.symbol_high, symbols.symbol_low) == low


def test_every_template_family_has_a_distinct_header():
    headers = {family: _template_header(family) for family in GambleGridConfig().template_families}
    assert len(set(headers.values())) == len(headers)
    assert all(header and not header.endswith("\n") for header in headers.values())
    with pytest.raises(ValueError, match="unsupported gamble template family"):
        _template_header("no_such_family")


def test_affect_audit_flags_emotion_words(reveals):
    # The audited surface is clean; an injected emotion word must be caught, else the audit is
    # vacuous and the "no lexical leakage from the stimulus side" inheritance is unsupported.
    assert _audit_affect_neutral(list(reveals)) == []
    tainted = reveals[0].model_copy(update={"prompt": f"{reveals[0].prompt} You feel delighted."})
    warnings = _audit_affect_neutral([tainted])
    assert len(warnings) == 1
    assert "delighted" in warnings[0]


def test_reveal_battery_covers_every_trial_type_and_ev_level(reveals):
    by_type: dict[str, set[str]] = defaultdict(set)
    for comparison in reveals:
        by_type[comparison.metadata["trial_type"]].add(comparison.metadata["ev_level"])
    assert set(by_type) == set(TRIAL_TYPES)
    # The augmentation draws are all tagged "mid", so only the base draws span the terciles.
    base_levels = {c.metadata["ev_level"] for c in reveals if c.metadata["augmented"] is False}
    assert base_levels == set(EV_LEVELS)
