"""Estimator behaviour for the R-A reveal-token signed-RPE analyzer.

Adapted from functional-valence-validity tests/test_reveal_rpe.py @ 10c4662 (the planted-code
2x2 discrimination, the verdict-map logic, the capture slot pinning and the artifact round-trip
/ binding checks). Dropped with their subjects: the legacy-contract backfill cases (this package
writes one hashless schema) and the metadata-impostor cases against the parent's
type-identity re-authentication loop.

Synthetic fixtures plant a known hidden-state code over the real reveal battery and assert the
estimator's discrimination: signed RPE is recovered and licensed; unsigned surprise fails the
signed conjunction; pure realised reward is excluded by the reward-matched contest (the one
rival entangled with signed RPE) while the EV-matched contest (which cannot exclude reward)
still passes.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict

import numpy as np
import pytest

from appraisal_emotions.analysis.direction_stats import (
    REWARD_INDEX,
    build_reveal_arrays,
    fit_block_coefficients,
    grouped_half_indices,
    rows_by_group,
)
from appraisal_emotions.analysis.reveal_rpe import (
    RevealRpeStates,
    _verdict,
    analyze_reveal_rpe,
    build_reveal_rpe_directions,
    build_reveal_rpe_states,
    capture_reveal_states,
    read_reveal_rpe_directions,
    read_reveal_rpe_states,
    reveal_rpe_directions_path,
    reveal_rpe_states_path,
    write_reveal_rpe_directions,
    write_reveal_rpe_states,
)
from appraisal_emotions.backends.fake import FakeBackend
from appraisal_emotions.stimuli.gambles import GambleGridConfig
from appraisal_emotions.stimuli.reveal_probes import build_reveal_battery
from conftest import (
    ANALYZE_KWARGS,
    GRID_CONFIG,
    HIDDEN,
    MODEL_SPEC,
    N_BLOCKS,
    cosine,
    orthonormal,
    planted_states,
    signed_rpe_target,
)


def test_planted_signed_rpe_is_present_and_separable(reveals):
    axes = orthonormal(seed=1, count=3)
    states = planted_states(reveals, signed_rpe_target(reveals, axes), seed=2)
    report = analyze_reveal_rpe(states, reveals, **ANALYZE_KWARGS)

    assert report.verdict == "separable-signed-rpe"
    assert report.signed_conjunction_passed
    assert report.reward_matched_passed and report.ev_matched_passed
    assert report.orientation_passed
    assert report.orientation_cos_reward_ev < -0.5  # gradient oriented along reward - ev
    assert report.stability_passed
    assert report.external_passed
    assert report.signed_rpe_null_p < 0.05
    assert report.signed_rpe_null_passed  # the separable gate enforces the selection-aware null
    # both matched arms score held-out cells via cross-fitting (the underpower guard is satisfied)
    assert report.reward_matched_n_scored_cells >= 4
    assert report.ev_matched_n_scored_cells >= 4

    # v_RPE (reward slope holding EV) recovers the planted signed-RPE axis at the selected block.
    arrays = build_reveal_arrays(reveals)
    est = arrays.partition == "estimation"
    coef = fit_block_coefficients(states[est], arrays.design[est])
    from appraisal_emotions.analysis.direction_stats import coefficient_directions

    v_rpe = coefficient_directions(coef, REWARD_INDEX)[report.selected_block]
    assert abs(cosine(v_rpe, axes[0])) > 0.95


def test_planted_unsigned_surprise_only_fails_the_signed_conjunction(reveals):
    (u_surprise,) = orthonormal(seed=3, count=1)

    def target(meta):
        return 8.0 * meta["abs_rpe"] * u_surprise  # magnitude only, no sign, no value

    report = analyze_reveal_rpe(planted_states(reveals, target, seed=4), reveals, **ANALYZE_KWARGS)

    # |RPE| is constant within every matched cell, so no direction separates sign there; the sign
    # is absent from reward/EV gradients too. (Each matched contest carries a nominal 5% FPR by
    # design, so the robust discriminators are orientation and the not-separable verdict, not a
    # single contest's boolean.)
    assert not report.orientation_passed
    assert report.verdict != "separable-signed-rpe"
    # The |RPE|-magnitude axis IS robustly present here (the planted signal), so any
    # unsigned-surprise collapse rests on positive evidence, not the mere absence of a signed axis.
    assert report.abs_rpe_present


def test_planted_pure_reward_is_excluded_by_the_reward_matched_contest(reveals):
    u_reward, u_surprise = orthonormal(seed=5, count=2)

    def target(meta):
        # Realised outcome value only (+ orthogonal surprise); NO expectation reference.
        return 4.0 * meta["reward"] * u_reward + 5.0 * meta["abs_rpe"] * u_surprise

    report = analyze_reveal_rpe(planted_states(reveals, target, seed=6), reveals, **ANALYZE_KWARGS)

    # EV-matched cannot exclude reward: it separates sign via the within-draw reward flip (power
    # ~1), while orientation fails because the EV gradient b_ev is ~0 (no expectation reference).
    assert report.ev_matched_passed
    assert not report.orientation_passed
    assert report.verdict != "separable-signed-rpe"


def test_pure_noise_is_not_separable(reveals):
    states = planted_states(reveals, lambda _meta: np.zeros(HIDDEN), seed=8, noise=1.0)
    report = analyze_reveal_rpe(states, reveals, **ANALYZE_KWARGS)

    assert report.verdict != "separable-signed-rpe"
    assert not report.signed_conjunction_passed
    assert not report.orientation_passed


def test_stability_split_is_draw_grouped_and_leakage_safe(reveals):
    # The honest stability metric deals whole DRAWS (ev_cell = draw_id) into two halves so no draw
    # trains both directions. A row-level shuffle would drop the same draw's renderings / opposite
    # outcomes into both halves and inflate the split-half cosine; assert that never happens.
    arrays = build_reveal_arrays(reveals)
    est = arrays.partition == "estimation"
    draw_ids = arrays.ev_cell[est]
    groups = rows_by_group(draw_ids)
    all_draws = set(draw_ids.tolist())
    rng = random.Random("leakage-safety")
    for _ in range(64):
        left, right = grouped_half_indices(groups, rng)
        left_draws = set(draw_ids[left].tolist())
        right_draws = set(draw_ids[right].tolist())
        # No draw spans the split (the leakage a row-level shuffle would introduce).
        assert left_draws.isdisjoint(right_draws)
        # Every draw lands in exactly one half — no rows dropped, none duplicated.
        assert left_draws | right_draws == all_draws
        assert set(left.tolist()).isdisjoint(set(right.tolist()))
        assert len(left) + len(right) == len(draw_ids)


def test_grouped_stability_certifies_planted_signal_and_refuses_noise(reveals):
    # The draw-grouped, K-averaged stability must clear 0.80 on a strong planted signed-RPE code
    # and sit far below it on pure noise (two independent noise fits are ~orthogonal) — the honest
    # replacement for the leakage-inflated row-split.
    axes = orthonormal(seed=21, count=3)
    signal = analyze_reveal_rpe(
        planted_states(reveals, signed_rpe_target(reveals, axes), seed=22),
        reveals,
        **ANALYZE_KWARGS,
    )
    noise = analyze_reveal_rpe(
        planted_states(reveals, lambda _meta: np.zeros(HIDDEN), seed=23, noise=1.0),
        reveals,
        **ANALYZE_KWARGS,
    )

    assert signal.stability_passed and signal.stability_cosine >= 0.80
    assert not noise.stability_passed and noise.stability_cosine < 0.80
    # Provenance fields populated: many splits scored, a dispersion, a bounded legacy row-split.
    assert signal.stability_n_splits > 1
    assert signal.stability_cosine_std >= 0.0
    assert -1.0 <= signal.stability_cosine_rowsplit_legacy <= 1.0


def test_verdict_vocabulary_is_capped_to_present_and_separable(reveals):
    # R-A licenses at most present-and-separable: the verdict never crosses into use / welfare.
    allowed = {
        "separable-signed-rpe",
        "collapses-to-value-or-reward",
        "collapses-to-unsigned-surprise",
        "indeterminate",
    }
    states = planted_states(reveals, lambda _meta: np.zeros(HIDDEN), seed=9, noise=1.0)
    report = analyze_reveal_rpe(states, reveals, **ANALYZE_KWARGS)
    assert report.verdict in allowed
    dumped = report.model_dump_json().lower()
    for forbidden in ("welfare", "sentience", "experience", "functionally-used", "suffering"):
        assert forbidden not in dumped


def _all_gates(**overrides: bool) -> dict[str, bool]:
    gates = {
        "conjunction_passed": True,
        "orientation_passed": True,
        "stability_passed": True,
        "external_passed": True,
        "selection_null_passed": True,
        "abs_rpe_present": True,
        "reward_matched_passed": True,
        "ev_matched_passed": True,
    }
    gates.update(overrides)
    return gates


def test_separable_verdict_requires_the_selection_aware_null():
    # All other gates pass, but a non-significant held-out sign AUROC must block the separable
    # claim — the selection-aware null is a load-bearing part of the gate, not a reported aside.
    assert _verdict(**_all_gates()) == "separable-signed-rpe"
    assert _verdict(**_all_gates(selection_null_passed=False)) != "separable-signed-rpe"


def test_unsigned_surprise_collapse_requires_positive_magnitude_evidence():
    # Sign carried by neither matched contest and orientation failed. Without evidence that the
    # |RPE|-magnitude axis actually separates, the report must stay indeterminate rather than
    # positively asserting an unsigned-surprise collapse the data never showed.
    base = _all_gates(
        conjunction_passed=False,
        orientation_passed=False,
        reward_matched_passed=False,
        ev_matched_passed=False,
    )
    assert _verdict(**{**base, "abs_rpe_present": False}) == "indeterminate"
    assert _verdict(**{**base, "abs_rpe_present": True}) == "collapses-to-unsigned-surprise"


def test_single_matched_axis_is_a_value_or_reward_collapse():
    # Exactly one matched contest carries sign ⇒ a value/reward collapse (the conjunction needs
    # both). BOTH asymmetries must land there — the pure-EV analogue (reward-matched only) must
    # not silently fall through to indeterminate, symmetric with the pure-reward case.
    one_sided = _all_gates(conjunction_passed=False, orientation_passed=False)
    pure_reward = {**one_sided, "reward_matched_passed": False, "ev_matched_passed": True}
    pure_ev = {**one_sided, "reward_matched_passed": True, "ev_matched_passed": False}
    assert _verdict(**pure_reward) == "collapses-to-value-or-reward"
    assert _verdict(**pure_ev) == "collapses-to-value-or-reward"


def test_analyze_rejects_states_misaligned_to_reveals(reveals):
    states = np.zeros((len(reveals) - 1, N_BLOCKS, HIDDEN))
    with pytest.raises(ValueError, match="aligned to reveals"):
        analyze_reveal_rpe(states, reveals, **ANALYZE_KWARGS)


def test_capture_lands_on_the_read_prefix_and_is_deterministic():
    reveals = build_reveal_battery(GambleGridConfig(seed=7, single_shot_per_cell=2)).reveals
    backend = FakeBackend(MODEL_SPEC, decoder_block_count=4)
    states = capture_reveal_states(backend, reveals)
    assert states.shape == (len(reveals), 4, 8)
    assert np.array_equal(states, capture_reveal_states(backend, reveals))

    # Both outcomes of a draw share the prompt but differ in read_prefix; capture appends it, so
    # the two reveals must land on distinct byte-pinned reveal slots (distinct states).
    by_prompt: dict[str, list[int]] = defaultdict(list)
    for index, comparison in enumerate(reveals):
        by_prompt[comparison.prompt].append(index)
    pair = next(
        rows
        for rows in by_prompt.values()
        if len({reveals[r].metadata["read_prefix"] for r in rows}) > 1
    )
    left, right = pair[0], pair[1]
    assert reveals[left].prompt == reveals[right].prompt
    assert reveals[left].metadata["read_prefix"] != reveals[right].metadata["read_prefix"]
    assert not np.array_equal(states[left], states[right])


def test_capture_fails_closed_when_read_prefix_is_missing():
    # The byte-pinned reveal-token capture must refuse a reveal without metadata['read_prefix']
    # rather than silently reading position='last' on the unpinned prompt tail.
    reveals = build_reveal_battery(GambleGridConfig(seed=7, single_shot_per_cell=2)).reveals
    backend = FakeBackend(MODEL_SPEC, decoder_block_count=4)
    stripped = reveals[0].model_copy(
        update={"metadata": {k: v for k, v in reveals[0].metadata.items() if k != "read_prefix"}}
    )
    with pytest.raises(ValueError, match="read_prefix"):
        capture_reveal_states(backend, (stripped,))


def _directions_fixture(reveals):
    """A planted signed-RPE run reduced to (states artifact, report, directions, planted axis).

    Byte-identical to the parent test helper's recipe (axes seed 31, states seed 32), which is
    what produced the committed golden artifacts.
    """

    axes = orthonormal(seed=31, count=3)
    states = planted_states(reveals, signed_rpe_target(reveals, axes), seed=32)
    report = analyze_reveal_rpe(states, reveals, **ANALYZE_KWARGS)
    artifact = build_reveal_rpe_states(
        states,
        reveals,
        spec=MODEL_SPEC,
        seed=7,
        battery_contract_version="reveal_probes/v1",
    )
    directions = build_reveal_rpe_directions(
        artifact, reveals, spec=MODEL_SPEC, report=report, battery_sha256="0" * 64
    )
    return artifact, report, directions, axes[0]


def test_directions_artifact_binds_its_sources_and_recovers_the_planted_axis(reveals):
    artifact, report, directions, u_rpe = _directions_fixture(reveals)
    meta = directions.metadata
    assert meta.states_sha256 == artifact.metadata.states_sha256
    assert meta.selected_block == report.selected_block
    assert meta.source_verdict == report.verdict
    assert meta.direction_families == ("v_rpe", "v_ev", "v_absrpe")
    assert directions.directions.shape == (3, meta.n_blocks, meta.hidden_size)
    assert directions.directions.flags.writeable is False
    # Every persisted row is unit-norm (or exactly zero for a vanishing rival direction).
    norms = np.linalg.norm(directions.directions, axis=2)
    assert np.all(np.isclose(norms, 1.0, atol=1e-6) | (norms == 0.0))
    # The persisted v_rpe IS the analyze-path estimation fit: it recovers the planted axis at
    # the report's selected block, so downstream consumers read the gated direction, not a refit.
    v_rpe = directions.directions[0, meta.selected_block]
    assert abs(cosine(v_rpe, u_rpe)) > 0.95


def test_states_artifact_round_trips_and_fails_closed_on_tamper(reveals, tmp_path):
    artifact, _report, _directions, _u = _directions_fixture(reveals)
    metadata_path = tmp_path / "reveal_states.json"
    write_reveal_rpe_states(artifact, metadata_path)
    loaded = read_reveal_rpe_states(metadata_path)
    assert loaded.metadata == artifact.metadata
    assert np.array_equal(loaded.states, artifact.states)
    assert loaded.states.flags.writeable is False

    tampered = np.array(artifact.states, copy=True)
    tampered[0, 0, 0] += 1.0
    np.savez(reveal_rpe_states_path(metadata_path), reveal_states=tampered)
    with pytest.raises(ValueError, match="hash"):
        read_reveal_rpe_states(metadata_path)


def test_states_container_refuses_a_row_count_mismatch(reveals):
    artifact, _report, _directions, _u = _directions_fixture(reveals)
    with pytest.raises(ValueError, match="row count"):
        RevealRpeStates(metadata=artifact.metadata, states=np.asarray(artifact.states)[:-1])


def test_directions_artifact_round_trips_and_fails_closed_on_tamper(reveals, tmp_path):
    _artifact, _report, directions, _u = _directions_fixture(reveals)
    metadata_path = tmp_path / "reveal_directions.json"
    _meta_path, npz_path = write_reveal_rpe_directions(directions, metadata_path)
    loaded = read_reveal_rpe_directions(metadata_path)
    assert loaded.metadata == directions.metadata
    assert np.array_equal(loaded.directions, directions.directions)

    # Tamper with the npz payload: a rolled row keeps the unit-norm invariant but breaks the hash
    # binding, so the read must fail closed on the sha256, not on a shape/norm accident.
    tampered = np.array(directions.directions, copy=True)
    tampered[0, 0, :] = np.roll(tampered[0, 0, :], 1)
    np.savez(npz_path, reveal_directions=tampered)
    with pytest.raises(ValueError, match="hash"):
        read_reveal_rpe_directions(metadata_path)


def test_directions_builder_fails_closed_on_source_mismatches(reveals):
    artifact, report, _directions, _u = _directions_fixture(reveals)

    # A battery missing a captured reveal id violates the capture's canonical battery order.
    with pytest.raises(ValueError, match="canonical"):
        build_reveal_rpe_directions(
            artifact, reveals[1:], spec=MODEL_SPEC, report=report, battery_sha256="0" * 64
        )

    # A stale report from ANOTHER capture must be refused once stamped: states_sha256 is the
    # capture identity and battery_sha256 pins the design source (a same-ids battery with drifted
    # reward/EV labels must never feed the refit).
    stamped = report.model_copy(
        update={"states_sha256": artifact.metadata.states_sha256, "battery_sha256": "0" * 64}
    )
    bound = build_reveal_rpe_directions(
        artifact, reveals, spec=MODEL_SPEC, report=stamped, battery_sha256="0" * 64
    )
    assert bound.metadata.states_sha256 == artifact.metadata.states_sha256
    foreign = report.model_copy(update={"states_sha256": "f" * 64})
    with pytest.raises(ValueError, match="states_sha256"):
        build_reveal_rpe_directions(
            artifact, reveals, spec=MODEL_SPEC, report=foreign, battery_sha256="0" * 64
        )
    drifted_battery = report.model_copy(update={"battery_sha256": "b" * 64})
    with pytest.raises(ValueError, match="battery_sha256"):
        build_reveal_rpe_directions(
            artifact, reveals, spec=MODEL_SPEC, report=drifted_battery, battery_sha256="0" * 64
        )


def test_grid_config_fixture_is_the_parity_recipe():
    # The golden artifacts were produced from single_shot_per_cell=4 at seed 7; a drift here would
    # silently weaken test_golden_parity to a self-consistency check.
    assert (GRID_CONFIG.seed, GRID_CONFIG.single_shot_per_cell) == (7, 4)


def test_readers_accept_the_parents_provenance_hashed_metadata(reveals, tmp_path):
    # The drop-in route in results/ra_prime_certification.md supplies the certified R-A' artifact
    # in the parent's provenance-hashed versions (states/v2, directions/v3, one extra
    # ``provenance_hash`` field). The readers must parse those, and the payload binding must
    # still be enforced on them — the concession is metadata-shape only, never hash-skipping.
    artifact, _report, directions, _u = _directions_fixture(reveals)
    for source, version, path_of, reader, array_name in (
        (
            artifact,
            "reveal_rpe_states/v2",
            reveal_rpe_states_path,
            read_reveal_rpe_states,
            "reveal_states",
        ),
        (
            directions,
            "reveal_rpe_directions/v3",
            reveal_rpe_directions_path,
            read_reveal_rpe_directions,
            "reveal_directions",
        ),
    ):
        payload = source.metadata.model_dump(mode="json")
        payload["artifact_contract_version"] = version
        payload["provenance_hash"] = "a" * 64
        metadata_path = tmp_path / f"{array_name}.json"
        metadata_path.write_text(json.dumps(payload), encoding="utf-8")
        arrays = getattr(source, "states", None)
        arrays = source.directions if arrays is None else arrays
        np.savez(path_of(metadata_path), **{array_name: np.asarray(arrays)})

        loaded = reader(metadata_path)
        assert loaded.metadata.artifact_contract_version == version

        np.savez(path_of(metadata_path), **{array_name: np.asarray(arrays) + 1.0})
        with pytest.raises(ValueError, match="hash"):
            reader(metadata_path)
