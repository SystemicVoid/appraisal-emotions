"""E1 machinery on SYNTHETIC data with planted structure — the harness's own positive control.

This file is the sensitivity evidence for ``map_geometry``. Per
``docs/agents/experiment-gating.md``, "the harness runs" is not sensitivity: a null from E1 may
only be read as information if the harness has been shown able to recover an effect of the
claimed shape. So each emotion-concept vector is constructed as

    ``e_j = valence_j * u + appraisal_j * w + noise``

with ``u ⊥ w``, and ``v_rpe`` planted along ``w`` — or, in the harder variant, along
``normalize(w + valence_load·u)``. The valence term is the nuisance the residual regression
partials out; the ``appraisal_j`` term is exactly the residual structure the recorded expectations
claim to detect.

The planting is at FAMILY level, in the §5 recorded directions: ``outcome_pos`` and
``outcome_neg`` carry the appraisal excess on their own poles of the signed axis, and every other
family — including the valence-matched ``nonoutcome_*`` controls, ``outcome_confirm`` and ``sad``
— carries none. That is the structure the design actually predicts, so one planting exercises all
of it at once: the family contrasts, the three-level ordering, the named pairs (``disappointed``
is in ``outcome_neg`` and ``sad`` in ``nonoutcome_neg``, so the pair inherits the family split),
and P5a's expectation of ABSENCE on ``sad``.

Cases, all required:

- **recovery** — the family contrasts come out positive, clear both anisotropy floors, the
  ordering holds, and every named pair comes out in its recorded direction;
- **recovery under a valence-loaded ``v_rpe``** — the same, with ``v_rpe`` carrying a substantial
  component along the valence axis. This is the case that caught the label-shuffle floor being
  computed on the raw-cosine scale: with the floor mis-scaled it rejected every true effect here
  while passing the orthogonal-planting case, so the orthogonal case alone was vacuous;
- **rejection** — ``appraisal_j = 0`` everywhere (valence and noise only), run under the SAME
  valence-loaded geometry; the family contrast must not clear the floors;
- **P5a both ways** — ``sad`` planted at no excess must sit inside the word residuals' spread, and
  ``sad`` planted WITH an excess must not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from appraisal_emotions.analysis.emotion_mapping import format_map_geometry_summary, map_geometry
from appraisal_emotions.analysis.emotion_vectors import write_emotion_vectors
from appraisal_emotions.analysis.reveal_rpe import write_reveal_rpe_directions
from appraisal_emotions.stimuli.emotion_stories import read_emotion_words
from conftest import synthetic_directions_artifact, synthetic_emotion_artifact

WORDS_PATH = Path(__file__).resolve().parents[1] / "data" / "emotion_words.json"
HIDDEN = 24
BLOCKS = 3
PERMUTATIONS = 2000
NULL_DRAWS = 200


def _axes(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Four orthonormal axes: valence, appraisal, an unplanted third, and the style axis."""

    basis, _r = np.linalg.qr(np.random.default_rng(seed).standard_normal((HIDDEN, 4)))
    return basis[:, 0], basis[:, 1], basis[:, 2], basis[:, 3]


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _planted(
    appraisal_scale: float,
    *,
    seed: int = 5,
    valence_load: float = 0.0,
    sad_excess: float = 0.0,
):
    """Artifacts for the planted design; ``appraisal_scale=0`` is the rejection case.

    ``valence_load`` tilts ``v_rpe`` toward the valence axis — the realistic case, since any
    positive-value direction has a positive-valence component. ``sad_excess`` plants an appraisal
    excess on the P5a discriminant word, which P5a must then refuse to pass.
    """

    words = read_emotion_words(WORDS_PATH)
    labels = words.labels
    u_valence, w_appraisal, z_other, s_style = _axes(31)
    rng = np.random.default_rng(seed)

    # The §5 expectations, planted at family level: ONLY the outcome-disconfirmation families
    # carry the appraisal excess, each on its own pole of the signed axis. The valence-matched
    # non-outcome controls, the confirmation family and every other family carry none — which is
    # both what the design predicts and what makes P5a's expectation of absence on `sad` testable.
    #
    # The amplitudes are BALANCED so they sum to zero over words. A flat ±1 did that only while
    # the two families were the same size, which is how the fixture was written; the widened
    # families are 11 vs 15, so ±1 leaves the planted appraisal axis with a nonzero grand mean
    # (−0.036), and every row is then centred by an offset that carries appraisal — including the
    # style_control row, which is supposed to carry none. That leak alone drove cos_style to 0.371
    # and failed P5c: an artifact of the planting convention meeting unequal n, never of
    # `map_geometry`, which is why the fix belongs here (docs/design/e1-widening.md §7). The
    # balance below reduces to the original ±1 at equal family sizes and keeps every per-word
    # magnitude ~1.
    n_by_family = {
        family: len(words.words_in_category(family)) for family in ("outcome_pos", "outcome_neg")
    }
    balance = sum(n_by_family.values()) / 2.0
    amplitude = {
        "outcome_pos": balance / n_by_family["outcome_pos"],
        "outcome_neg": -balance / n_by_family["outcome_neg"],
    }
    appraisal = {
        word.word: amplitude.get(word.category, 0.0) * appraisal_scale for word in words.words
    }
    assert abs(sum(appraisal.values())) < 1e-12, "the planted appraisal must not shift the origin"
    appraisal[str(words.expectations["p5a"])] += sad_excess

    rows = [
        words.valence_by_word[label] * u_valence + appraisal[label] * w_appraisal
        for label in labels
    ]
    # The style_control row: valence-free and appraisal-free, so P5c's residual should sit inside
    # the word residuals' own spread — but planted on its OWN axis at the scale a word carries,
    # not at the origin. An origin-planted row has no content of its own, so after grand-mean
    # subtraction it is nothing but MINUS the grand mean: a vector ~8x shorter than every word,
    # whose cosine on any direction is that residue divided by its own small norm. P5c then
    # compares an inflated style cosine against uninflated word residuals and fails on the
    # unplanted axis by construction. The real style_control row is an ordinary vector with an
    # ordinary norm (docs/design/e1-widening.md §7), and that is the regime this control is meant
    # to test.
    rows.append(s_style)
    base = np.stack(rows, axis=0)
    vectors = np.stack(
        [base + 0.01 * rng.standard_normal(base.shape) for _ in range(BLOCKS)], axis=1
    )

    # v_rpe carries the planted appraisal axis plus, optionally, a valence component — the
    # realistic geometry, and the one that exposes a floor computed on the wrong scale.
    v_rpe = _unit(w_appraisal + valence_load * u_valence)
    directions = np.stack(
        [
            np.stack([v_rpe] * BLOCKS),  # v_rpe   — the planted appraisal axis
            np.stack([u_valence] * BLOCKS),  # v_ev    — the planted valence axis
            np.stack([z_other] * BLOCKS),  # v_absrpe — an unplanted third axis
        ],
        axis=0,
    )
    emotion = synthetic_emotion_artifact(
        vectors, labels, tuple(words.valence_by_word[label] for label in labels)
    )
    return synthetic_directions_artifact(directions), emotion


def _report(
    tmp_path: Path,
    appraisal_scale: float,
    *,
    noise_seed: int = 5,
    permutations: int = PERMUTATIONS,
    null_draws: int = NULL_DRAWS,
    valence_load: float = 0.0,
    sad_excess: float = 0.0,
):
    directions, emotion = _planted(
        appraisal_scale, seed=noise_seed, valence_load=valence_load, sad_excess=sad_excess
    )
    tag = f"{noise_seed}_{valence_load}_{sad_excess}"
    directions_path = tmp_path / f"reveal_directions_{tag}.json"
    emotion_path = tmp_path / f"emotion_vectors_{tag}.json"
    write_reveal_rpe_directions(directions, directions_path)
    write_emotion_vectors(emotion, emotion_path)
    return map_geometry(
        directions_path,
        emotion_path,
        WORDS_PATH,
        None,
        seed=7,
        n_permutations=permutations,
        n_null_draws=null_draws,
    )


@pytest.mark.parametrize("valence_load", [0.0, 0.8])
def test_recovers_the_planted_family_contrasts(tmp_path, valence_load):
    """Recovery must survive a v_rpe that is substantially valence-loaded.

    At ``valence_load=0.8`` the raw cosines are dominated by valence while the residuals are not;
    a floor computed by permuting the RAW cosines therefore sits ~4x too high and rejects a true
    effect. Both floors here resample the same post-residual scale as the statistic.
    """

    words = read_emotion_words(WORDS_PATH)
    report = _report(tmp_path, appraisal_scale=1.0, valence_load=valence_load)
    block = report.blocks[0]
    assert block.valence_source == "binary_project_labels"
    assert {contrast.pole for contrast in block.family_contrasts} == {"positive", "negative"}
    for contrast in block.family_contrasts:
        assert contrast.statistic > 0.0, f"{contrast.pole} pole lost its planted direction"
        assert contrast.p_value < 0.05, f"{contrast.pole} p={contrast.p_value}"
        outcome = len(words.words_in_category(contrast.outcome_family))
        control = len(words.words_in_category(contrast.control_family))
        assert (contrast.n_outcome, contrast.n_control) == (outcome, control)
    # And the effect clears both anisotropy floors, which is what makes it readable at all.
    assert block.clears_both_floors
    assert block.observed_family_contrast_max > block.label_shuffled_p95
    assert block.observed_family_contrast_max > block.random_direction_p95


def test_recovers_the_three_level_outcome_ordering(tmp_path):
    ordering = _report(tmp_path, appraisal_scale=1.0).blocks[0].outcome_ordering
    assert ordering.families == ("outcome_pos", "outcome_confirm", "outcome_neg")
    assert ordering.ordering_holds, ordering.mean_residuals
    assert ordering.trend_statistic > 0.0
    assert ordering.p_value < 0.05


def test_recovers_the_named_pairs_in_their_recorded_directions(tmp_path):
    block = _report(tmp_path, appraisal_scale=1.0).blocks[0]
    by_outcome = {pair.outcome: pair for pair in block.expected_pairs}
    assert set(by_outcome) == {"disappointed", "relieved", "elated"}
    for pair in block.expected_pairs:
        assert pair.statistic > 0.0, f"{pair.outcome}/{pair.control} lost its recorded direction"


def test_named_pairs_permute_inside_the_strict_pool_not_the_whole_valence_pole(tmp_path):
    """§5's resolution table: the pool is the same-pole outcome / non-outcome / confirmation
    families, not every word sharing a binary valence label. The expected sizes are derived from
    the word file here rather than restated, so the assertion follows the data."""

    words = read_emotion_words(WORDS_PATH)
    pool = [
        word for family in words.pair_pool_families() for word in words.words_in_category(family)
    ]
    expected = {
        pole: sum(1 for word in pool if words.valence_by_word[word] == pole) for pole in (1, -1)
    }
    # Derived, not pinned: the pool is exactly the same-pole words of the pair-pool families, and
    # both poles must be big enough for a within-pool permutation to have a null at all.
    assert set(expected) == {1, -1} and all(size >= 3 for size in expected.values()), expected
    block = _report(tmp_path, appraisal_scale=1.0, permutations=200, null_draws=50).blocks[0]
    for pair in block.expected_pairs:
        assert pair.n_pool == expected[words.valence_by_word[pair.outcome]]
        # A pool strictly smaller than the whole valence pole is the point of the fix.
        assert pair.n_pool < sum(
            1 for word in words.labels if words.valence_by_word[word] == pair.expected_sign
        )


def test_p5a_holds_when_sad_carries_no_excess(tmp_path):
    """P5a is an expectation of ABSENCE, and it must be able to FAIL — see the paired test."""

    block = _report(tmp_path, appraisal_scale=1.0).blocks[0]
    assert block.p5a.word == "sad"
    assert block.p5a.passed, block.p5a.model_dump()
    assert abs(block.p5a.residual) <= block.p5a.word_residual_p95
    assert block.p5a.n_words == len(read_emotion_words(WORDS_PATH).labels)
    # And the contrast it is a discriminant AGAINST is positive, which is what gives it meaning.
    assert all(contrast.statistic > 0.0 for contrast in block.family_contrasts)


def test_p5a_fails_when_sad_is_planted_with_an_rpe_excess(tmp_path):
    block = _report(
        tmp_path, appraisal_scale=1.0, sad_excess=-3.0, permutations=200, null_draws=50
    ).blocks[0]
    assert not block.p5a.passed, block.p5a.model_dump()
    assert abs(block.p5a.residual) > block.p5a.word_residual_p95
    assert block.p5a.abs_rank == 1


def test_every_word_is_tabled_with_its_family_and_valence(tmp_path):
    report = _report(tmp_path, appraisal_scale=1.0, permutations=200, null_draws=50)
    words = read_emotion_words(WORDS_PATH)
    for block in report.blocks:
        assert len(block.word_residuals) == len(words.labels)
        assert {row.word for row in block.word_residuals} == set(words.labels)
        assert all(row.family == words.category_by_word[row.word] for row in block.word_residuals)
        assert all(row.valence == words.valence_by_word[row.word] for row in block.word_residuals)
    # The summary shows them all, sorted, rather than hiding them behind a summary statistic.
    summary = format_map_geometry_summary(report)
    assert f"all {len(words.labels)} words by valence residual" in summary
    for label in words.labels:
        assert label in summary


def test_rejects_when_only_valence_is_planted(tmp_path):
    """Rejection is a FALSE-POSITIVE RATE check, not a per-seed determinism claim.

    Run under the SAME valence-loaded geometry as the recovery case, so the two are a matched
    pair: with v_rpe carrying a large valence component and no appraisal structure planted, the
    floors must still refuse. A 95th-percentile floor is allowed to be cleared on ~1 realisation
    in 20; the defect this guards against is a residual that leaks valence structure into the
    family contrast, which would fire on most of them.
    """

    fired = 0
    for noise_seed in range(20, 26):
        block = _report(
            tmp_path,
            appraisal_scale=0.0,
            noise_seed=noise_seed,
            valence_load=0.8,
            permutations=400,
            null_draws=60,
        ).blocks[0]
        fired += int(block.clears_both_floors)
    assert fired <= 1, f"the family contrast cleared both floors on {fired}/6 valence-only draws"


def test_p5c_style_control_is_flat_when_the_style_row_is_valence_free(tmp_path):
    block = _report(tmp_path, appraisal_scale=1.0).blocks[0]
    assert block.p5c_passed, [entry.model_dump() for entry in block.p5c]
    assert {entry.family for entry in block.p5c} == {"v_rpe", "v_ev", "v_absrpe"}


def test_p1_and_the_subspace_fractions_are_reported(tmp_path):
    block = _report(
        tmp_path, appraisal_scale=1.0, valence_load=0.8, permutations=200, null_draws=50
    ).blocks[0]
    # With v_rpe valence-loaded, P1's sanity correlation is positive as the design expects.
    assert block.p1_positive and block.p1_spearman_rho > 0.0
    assert 0.0 <= block.subspace_fraction_pc12_plane <= 1.0001
    assert 0.0 <= block.subspace_fraction_emotion_span <= 1.0001


def test_report_carries_its_input_digests_and_gate_cap(tmp_path):
    directions, emotion = _planted(1.0)
    directions_path = tmp_path / "reveal_directions.json"
    emotion_path = tmp_path / "emotion_vectors.json"
    write_reveal_rpe_directions(directions, directions_path)
    write_emotion_vectors(emotion, emotion_path)
    report = map_geometry(
        directions_path,
        emotion_path,
        WORDS_PATH,
        None,
        seed=7,
        n_permutations=200,
        n_null_draws=50,
    )
    assert report.directions_sha256 == directions.metadata.directions_sha256
    assert report.emotion_vectors_sha256 == emotion.metadata.vectors_sha256
    assert report.seed == 7 and report.n_permutations == 200
    assert report.sensitivity_gate == "G0=pass"
    assert "present-and-separable" in report.verdict_cap
    assert len(report.block_sweep) == BLOCKS
    summary = format_map_geometry_summary(report)
    assert "recorded expectations" in summary
    # No confirmatory/exploratory caste survives anywhere in the report.
    assert "confirmatory" not in report.model_dump_json().lower()


def test_gate_failure_caps_the_verdict(tmp_path):
    directions, emotion = _planted(1.0)
    failed = synthetic_emotion_artifact(
        np.asarray(emotion.vectors),
        emotion.word_labels,
        tuple(int(value) for value in emotion.word_valence),
        gate_verdict="harness_inadequate",
    )
    directions_path = tmp_path / "reveal_directions.json"
    emotion_path = tmp_path / "emotion_vectors.json"
    write_reveal_rpe_directions(directions, directions_path)
    write_emotion_vectors(failed, emotion_path)
    report = map_geometry(
        directions_path, emotion_path, WORDS_PATH, None, seed=7, n_permutations=100, n_null_draws=20
    )
    assert report.sensitivity_gate == "G0=harness_inadequate"
    assert report.verdict_cap.startswith("harness_inadequate")


def test_map_geometry_refuses_mismatched_artifacts(tmp_path):
    directions, emotion = _planted(1.0)
    truncated = np.asarray(directions.directions)[:, :, : HIDDEN - 1]
    truncated = truncated / np.linalg.norm(truncated, axis=2, keepdims=True)
    narrower = synthetic_directions_artifact(truncated)
    directions_path = tmp_path / "reveal_directions.json"
    emotion_path = tmp_path / "emotion_vectors.json"
    write_reveal_rpe_directions(narrower, directions_path)
    write_emotion_vectors(emotion, emotion_path)
    with pytest.raises(ValueError, match="hidden size"):
        map_geometry(directions_path, emotion_path, WORDS_PATH, None, seed=7, n_permutations=10)


def _norms_csv(
    path: Path, words, *, skip: tuple[str, ...] = (), blank_arousal: tuple[str, ...] = ()
) -> Path:
    """A fetched-subset CSV in scripts/fetch_norms.py's output schema.

    Arousal VARIES across words rather than sitting at a constant: a constant column is collinear
    with the intercept, so a design built on it would be singular and the arousal option would
    appear to work while partialling out nothing. ``blank_arousal`` leaves the arousal cell empty
    for named words, which is how the all-or-nothing rule gets tested on the second column.
    """

    scale = {1: 7.4, -1: 2.6, 0: 5.1}
    rows = ["word,valence,arousal,source"]
    for index, label in enumerate(words.labels):
        if label in skip:
            continue
        arousal = "" if label in blank_arousal else f"{3.0 + (index % 7) * 0.5:.2f}"
        rows.append(f"{label},{scale[words.valence_by_word[label]]},{arousal},synthetic")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _norms_report(tmp_path: Path, norms_kwargs: dict, **map_kwargs):
    directions, emotion = _planted(1.0)
    words = read_emotion_words(WORDS_PATH)
    directions_path = tmp_path / "reveal_directions.json"
    emotion_path = tmp_path / "emotion_vectors.json"
    write_reveal_rpe_directions(directions, directions_path)
    write_emotion_vectors(emotion, emotion_path)
    norms = _norms_csv(tmp_path / "norms.csv", words, **norms_kwargs)
    return words, map_geometry(
        directions_path,
        emotion_path,
        WORDS_PATH,
        norms,
        seed=7,
        n_permutations=200,
        n_null_draws=50,
        **map_kwargs,
    )


def test_the_arousal_column_is_partialled_out_when_asked_and_never_by_default(tmp_path):
    """The widened run's pre-registered primary readout, and the default that must not move.

    E1's published contrast is the valence-only one, so the default has to stay reproducible from
    this code (docs/design/e1-widening.md §7). Adding arousal must therefore be a request, visible
    in the report, and it must actually change the residuals — a design that silently dropped the
    column would look identical and mean something else.
    """

    _words, valence_only = _norms_report(tmp_path / "a", {})
    _words2, partialled = _norms_report(tmp_path / "b", {}, residualize_on=("valence", "arousal"))

    assert valence_only.residualize_on == ("valence",)
    assert valence_only.valence_source == "numeric_norms"
    assert partialled.residualize_on == ("valence", "arousal")
    # The source string SAYS which nuisances went, so the two readouts are never confusable.
    assert partialled.valence_source == "numeric_norms[valence+arousal]"

    before = {row.word: row.residual for row in valence_only.blocks[0].word_residuals}
    after = {row.word: row.residual for row in partialled.blocks[0].word_residuals}
    assert set(before) == set(after)
    assert any(abs(before[word] - after[word]) > 1e-9 for word in before), (
        "partialling arousal changed nothing; the column is not reaching the design"
    )
    # And the readout still works: both family contrasts survive the extra covariate.
    assert all(c.statistic > 0.0 for c in partialled.blocks[0].family_contrasts)


def test_a_missing_arousal_rating_blocks_the_upgrade_for_the_whole_set(tmp_path):
    """All-or-nothing spans EVERY requested column, not just the first one.

    A word with a valence rating and no arousal rating would otherwise silently produce a design
    on a subset nobody chose — the same incommensurability the valence rule already refuses.
    """

    words, report = _norms_report(
        tmp_path, {"blank_arousal": ("wistful",)}, residualize_on=("valence", "arousal")
    )
    assert report.valence_source == "binary_project_labels"
    assert report.norms_covered_words == len(words.labels) - 1
    assert report.norms_missing_words == ("wistful",)


def test_full_norm_coverage_upgrades_the_valence_scale(tmp_path):
    directions, emotion = _planted(1.0)
    words = read_emotion_words(WORDS_PATH)
    directions_path = tmp_path / "reveal_directions.json"
    emotion_path = tmp_path / "emotion_vectors.json"
    write_reveal_rpe_directions(directions, directions_path)
    write_emotion_vectors(emotion, emotion_path)
    norms = _norms_csv(tmp_path / "vad_subset.csv", words)
    report = map_geometry(
        directions_path,
        emotion_path,
        WORDS_PATH,
        norms,
        seed=7,
        n_permutations=200,
        n_null_draws=50,
    )
    assert report.valence_source == "numeric_norms"
    assert report.norms_covered_words == len(words.labels)
    assert report.norms_file == str(norms)
    # The named pairs stay valence-MATCHED on the binary labels even when the regression upgrades
    # to a numeric scale: the matched set is what the null permutes over.
    assert all(pair.n_pool > 2 for pair in report.blocks[0].expected_pairs)


def test_partial_norm_coverage_falls_back_to_the_binary_labels(tmp_path):
    directions, emotion = _planted(1.0)
    words = read_emotion_words(WORDS_PATH)
    directions_path = tmp_path / "reveal_directions.json"
    emotion_path = tmp_path / "emotion_vectors.json"
    write_reveal_rpe_directions(directions, directions_path)
    write_emotion_vectors(emotion, emotion_path)
    norms = _norms_csv(tmp_path / "partial.csv", words, skip=("wistful", "crestfallen"))
    report = map_geometry(
        directions_path,
        emotion_path,
        WORDS_PATH,
        norms,
        seed=7,
        n_permutations=200,
        n_null_draws=50,
    )
    # Silent partial coverage is the failure to avoid: a mixed scale would make the residuals
    # incommensurable, so the report says which scale it actually used and how many it found.
    assert report.valence_source == "binary_project_labels"
    assert report.norms_covered_words == len(words.labels) - 2
