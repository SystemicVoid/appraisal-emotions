"""The paper's confound-removal step: the corpus it is fit on, the fit, and the projection.

Split from ``test_sofroniew_stories.py`` because the object under test is different: that file
checks the *stimulus* arm, this one checks a linear-algebra step and the stimulus that feeds it.

Nothing here restates the paper's prompt or its speaker rename. Both are read from
``data/sofroniew2026/``, which ``scripts/fetch_sofroniew_recipe.py`` extracts from the arXiv
source (``docs/agents/rails.md`` clause 1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from appraisal_emotions.analysis.emotion_vectors import (
    FirstContactFailure,
    NeutralCorpusTooThin,
    extract_emotion_vectors,
    fit_neutral_projection,
)
from appraisal_emotions.analysis.neutral_projection import (
    fit_neutral_basis,
    project_out,
)
from appraisal_emotions.backends.fake import FakeBackend
from appraisal_emotions.core.schema import ModelSpec
from appraisal_emotions.stimuli.emotion_stories import read_emotion_words
from appraisal_emotions.stimuli.sofroniew_stories import (
    DIALOGUE_SEPARATOR,
    NEUTRAL_DIALOGUE_LABEL,
    apply_speaker_substitutions,
    build_neutral_dialogue_grid,
    build_sofroniew_story_grid,
    neutral_completion_splitter,
    read_sofroniew_recipe,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sofroniew2026"
WORDS_PATH = ROOT / "data" / "emotion_words.json"
SPEC = ModelSpec(key="fake", backend="fake", model_id="fake")


@pytest.fixture(scope="module")
def recipe():
    return read_sofroniew_recipe(DATA_DIR)


# --------------------------------------------------------------------------------------
# The corpus: the paper's prompt, and the paper's post-hoc speaker rename
# --------------------------------------------------------------------------------------


def test_the_speaker_rename_is_loaded_and_agrees_with_the_prompt_it_applies_to(recipe):
    """The rename was parsed from prose; the prompt from a listing. They must agree.

    Both halves are loads, so this is a cross-check between two independently extracted parts of
    the source — not a literal compared against a restatement of itself.
    """

    assert recipe.speaker_substitutions, "no speaker rename was extracted"
    for source, target in recipe.speaker_substitutions:
        assert source and target
        assert source != target
        # The prompt asks the model to emit the source tag...
        assert source in recipe.neutral_dialogue_prompt_template
        # ...and the target is what the paper says the transcript ends up carrying.
        assert target not in recipe.neutral_dialogue_prompt_template


def test_apply_speaker_substitutions_rewrites_a_transcript(recipe):
    tags = [source for source, _ in recipe.speaker_substitutions]
    transcript = "\n\n".join(f"{tag} something" for tag in tags)
    rewritten = apply_speaker_substitutions(transcript, recipe.speaker_substitutions)
    for source, target in recipe.speaker_substitutions:
        assert source not in rewritten
        assert target in rewritten


def test_apply_speaker_substitutions_refuses_a_chaining_rewrite():
    """A pair whose replacement contains an earlier source would rewrite twice, silently."""

    with pytest.raises(ValueError, match="chain"):
        apply_speaker_substitutions("A: hi", (("A:", "B:"), ("B:", "C:")))


def test_the_neutral_splitter_cuts_on_the_separator_and_renames(recipe):
    split = neutral_completion_splitter(recipe.speaker_substitutions)
    source, target = recipe.speaker_substitutions[0]
    completion = f"{DIALOGUE_SEPARATOR}\n{source} one\n{DIALOGUE_SEPARATOR}\n{source} two\n"
    pieces = split(completion)
    assert len(pieces) == 2
    assert all(piece.startswith(target) for piece in pieces)
    # Deliberately literal, exactly like the story splitter: no separator means one piece, and
    # nothing at all means nothing (the caller turns that into one empty, audited row).
    assert len(split("a single undivided transcript")) == 1
    assert split("   ") == ()


@pytest.mark.parametrize("config_path", sorted(ROOT.glob("configs/*_projected*.yaml")))
def test_every_shipped_projected_config_keeps_the_neutral_corpus_inside_the_story_topics(
    recipe, config_path
):
    """The invariant, checked against the numbers the SHIPPED configs actually carry.

    Reproduced at hand-matched numbers this test previously used (6 story topics / 12 transcripts
    / 2 per call) the invariant held; both shipped configs violated it. The real arm asked for 10
    neutral calls against 6 story topics and the smoke for 6 against 1, so the confound corpus
    covered topics no story ever saw — several of them affect-laden, which is the specific hazard
    the module docstring's point 4 names. Parametrizing over the config glob is what makes a new
    arm inherit the check instead of a reviewer having to remember it.
    """

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))["emotion_vectors"]
    assert payload["story_recipe"] == "sofroniew", "the neutral arm only exists under this recipe"
    topics_per_label = payload["stories_per_emotion"] // payload["sofroniew_stories_per_call"]
    stories = build_sofroniew_story_grid(
        ("elated",),
        recipe=recipe,
        topics_per_label=topics_per_label,
        stories_per_call=payload["sofroniew_stories_per_call"],
        seed=payload["seed"],
    )
    neutral = build_neutral_dialogue_grid(
        recipe=recipe,
        n_transcripts=payload["neutral_dialogues"],
        dialogues_per_call=payload["neutral_dialogues_per_call"],
        story_topics_per_label=topics_per_label,
        seed=payload["seed"],
    )
    story_topics = [request.topic for request in stories if request.emotion == "elated"]
    assert [request.topic for request in neutral] == story_topics[: len(neutral)]
    assert {request.emotion for request in neutral} == {NEUTRAL_DIALOGUE_LABEL}
    # Every prompt is the paper's neutral prompt with its two slots bound.
    for request in neutral:
        assert "{topic}" not in request.prompt
        assert "{n_stories}" not in request.prompt
        assert request.topic in request.prompt


def test_the_neutral_grid_refuses_more_calls_than_the_stories_have_topics(recipe):
    """The topics past the story prefix are premises no emotion vector was ever built from."""

    with pytest.raises(ValueError, match="no story ever saw"):
        build_neutral_dialogue_grid(
            recipe=recipe,
            n_transcripts=12,
            dialogues_per_call=2,
            story_topics_per_label=5,
            seed=7,
        )


def test_the_neutral_grid_refuses_an_uneven_batch(recipe):
    with pytest.raises(ValueError, match="multiple"):
        build_neutral_dialogue_grid(
            recipe=recipe,
            n_transcripts=7,
            dialogues_per_call=2,
            story_topics_per_label=6,
            seed=7,
        )


def _fit_neutral(**overrides):
    """`fit_neutral_projection` at the fake-backend smoke shape, with one knob moved per test."""

    settings = {
        "layers": (0, 1, 2, 3),
        "n_transcripts": 4,
        "dialogues_per_call": 1,
        "variance_target": 0.5,
        "seed": 7,
        "min_token": 20,
        "max_tokens": 160,
        "temperature": 1.0,
        "min_kept": 2,
        "max_drop_rate": 0.5,
        "first_contact_n": 4,
        "first_contact_max_drop_rate": 0.5,
        "story_topics_per_label": 4,
        "data_dir": DATA_DIR,
    }
    return fit_neutral_projection(FakeBackend(SPEC), **(settings | overrides))


def test_the_neutral_corpus_has_a_floor_above_the_two_the_algebra_needs():
    """What this floor catches: a rank-1 basis fit on the survivors of a mostly-dropped corpus.

    `fit_neutral_basis` refusing below 2 was the whole adequacy gate, and 2 is only the point
    where PCA stops being undefined — it says nothing about whether the subspace is the confound
    subspace. The basis is subtracted from every emotion vector BEFORE G0, so a bad one is not
    something a downstream gate can notice.
    """

    with pytest.raises(NeutralCorpusTooThin, match="below the floor of 8"):
        _fit_neutral(min_kept=8)


def test_the_neutral_corpus_has_a_drop_ceiling():
    """min_token above every transcript's length drops the lot — the corpus is not the design.

    The first-contact checkpoint is disabled here so the ceiling is what fires: the two overlap
    on a corpus that over-drops from the start, and the ceiling's own job is the corpus that
    passes its first N and then falls apart.
    """

    with pytest.raises(NeutralCorpusTooThin, match="the filter dropped 1.00"):
        _fit_neutral(min_token=100_000, max_drop_rate=0.5, first_contact_max_drop_rate=1.0)


def test_the_neutral_corpus_gets_its_own_first_contact_checkpoint():
    """The <NEW DIALOGUE> splitter is a BLIND freeze, and rails.md licenses one only with this.

    The story-side checkpoint cannot cover it: it reads `stories`, and no neutral transcript is
    ever in that list. The failure names its surface so the written sample lands in the right file.
    """

    with pytest.raises(FirstContactFailure, match="neutral transcripts") as excinfo:
        _fit_neutral(min_token=100_000, first_contact_max_drop_rate=0.0)
    assert excinfo.value.surface == "neutral transcripts"
    assert excinfo.value.sample, "the checkpoint must hand back the sample a human has to read"


def test_the_record_separates_configured_obtained_and_kept():
    """Three numbers, because a corpus falls short in two different places.

    Reproduced: with `neutral_dialogues: 12, per_call: 2` the artifact read
    `n_requested: 6, n_kept: 6, drop_rate: 0.0` — the 12 the config asked for appeared nowhere,
    and a corpus at half its designed size read as a clean run.
    """

    _basis, record, transcripts = _fit_neutral(n_transcripts=4, dialogues_per_call=2)
    # 4 asked for over 2 calls; the fake backend emits no separator, so 2 come back.
    assert record.n_transcripts_configured == 4
    assert record.n_calls_requested == 2
    assert record.n_pieces_obtained == len(transcripts) == 2
    assert record.n_kept == 2
    # The filter's drop rate is 0 — and that is exactly why it cannot be the only number here.
    assert record.drop_rate == 0.0
    assert record.n_pieces_obtained < record.n_transcripts_configured


# --------------------------------------------------------------------------------------
# The fit: "the top principal components ... enough to explain 50% of the variance"
# --------------------------------------------------------------------------------------


def _planted(n: int, hidden: int, *, dominant: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """``n`` samples whose variance is overwhelmingly along ``dominant``."""

    rng = np.random.default_rng(11)
    strong = rng.normal(size=n) * 100.0
    faint = rng.normal(size=n) * 0.01
    return np.outer(strong, dominant) + np.outer(faint, weak)


def test_a_single_dominant_direction_needs_exactly_one_component():
    hidden = 8
    dominant = np.eye(hidden)[0]
    weak = np.eye(hidden)[1]
    states = _planted(20, hidden, dominant=dominant, weak=weak)[:, None, :]

    basis = fit_neutral_basis(states, variance_target=0.5)
    assert basis.n_blocks == 1
    assert basis.rows[0].n_components == 1
    assert basis.rows[0].cumulative_explained_variance >= 0.5
    # ...and it is the planted direction, up to sign.
    assert abs(float(basis.components[0][0] @ dominant)) == pytest.approx(1.0, abs=1e-6)


def test_a_higher_variance_target_never_takes_fewer_components():
    rng = np.random.default_rng(3)
    states = rng.normal(size=(30, 4, 12))
    counts = [
        sum(row.n_components for row in fit_neutral_basis(states, variance_target=t).rows)
        for t in (0.25, 0.5, 0.75, 0.95)
    ]
    assert counts == sorted(counts)
    assert counts[0] >= 1


def test_the_component_count_is_capped_by_the_corpus_rank():
    """With N transcript means the covariance has rank <= N-1; nothing beyond that is structure."""

    rng = np.random.default_rng(5)
    states = rng.normal(size=(4, 2, 64))
    basis = fit_neutral_basis(states, variance_target=1.0)
    assert all(row.n_components <= 3 for row in basis.rows)
    assert all(row.rank <= 3 for row in basis.rows)


def test_a_degenerate_corpus_removes_nothing():
    """Identical neutral transcripts carry no variance, so no direction is confounded."""

    states = np.tile(np.arange(6.0), (5, 2, 1))
    basis = fit_neutral_basis(states, variance_target=0.5)
    assert all(row.n_components == 0 for row in basis.rows)
    vectors = np.random.default_rng(1).normal(size=(3, 2, 6))
    assert np.array_equal(project_out(vectors, basis), vectors)


def test_one_transcript_is_refused():
    with pytest.raises(ValueError, match="at least 2"):
        fit_neutral_basis(np.zeros((1, 2, 4)), variance_target=0.5)


# --------------------------------------------------------------------------------------
# The projection itself
# --------------------------------------------------------------------------------------


def test_projection_removes_the_confound_direction_and_keeps_the_rest():
    hidden = 8
    confound = np.eye(hidden)[0]
    signal = np.eye(hidden)[3]
    states = _planted(20, hidden, dominant=confound, weak=np.eye(hidden)[1])[:, None, :]
    basis = fit_neutral_basis(states, variance_target=0.5)

    vectors = np.stack([confound + signal, 2.0 * confound, signal])[:, None, :]
    projected = project_out(vectors, basis)

    # The confound component is gone from every row...
    assert np.allclose(projected[:, 0, :] @ confound, 0.0, atol=1e-9)
    # ...and the orthogonal signal is untouched.
    assert np.allclose(projected[:, 0, :] @ signal, vectors[:, 0, :] @ signal, atol=1e-9)


def test_projection_preserves_grand_mean_centring():
    """Projection is linear, so rows that summed to zero still do — G0's frame survives it."""

    rng = np.random.default_rng(17)
    states = rng.normal(size=(24, 3, 10))
    basis = fit_neutral_basis(states, variance_target=0.5)
    raw = rng.normal(size=(9, 3, 10))
    centred = raw - raw.mean(axis=0)

    projected = project_out(centred, basis)
    assert np.allclose(projected.sum(axis=0), 0.0, atol=1e-9)


def test_projection_is_idempotent():
    rng = np.random.default_rng(23)
    basis = fit_neutral_basis(rng.normal(size=(24, 3, 10)), variance_target=0.5)
    vectors = rng.normal(size=(5, 3, 10))
    once = project_out(vectors, basis)
    assert np.allclose(project_out(once, basis), once, atol=1e-9)


def test_projection_refuses_a_block_count_mismatch():
    basis = fit_neutral_basis(np.random.default_rng(2).normal(size=(6, 3, 5)), variance_target=0.5)
    with pytest.raises(ValueError, match="blocks"):
        project_out(np.zeros((4, 2, 5)), basis)


# --------------------------------------------------------------------------------------
# End to end on the fake backend (contract only — every number here is meaningless)
# --------------------------------------------------------------------------------------


def test_fit_neutral_projection_generates_captures_and_records():
    backend = FakeBackend(SPEC)
    basis, record, transcripts = fit_neutral_projection(
        backend,
        layers=(0, 1, 2, 3),
        n_transcripts=8,
        dialogues_per_call=2,
        variance_target=0.5,
        seed=7,
        min_token=20,
        max_tokens=160,
        temperature=1.0,
        min_kept=2,
        max_drop_rate=0.5,
        first_contact_n=4,
        first_contact_max_drop_rate=0.5,
        story_topics_per_label=4,
        data_dir=DATA_DIR,
    )
    # 8 transcripts asked for, 2 per call = 4 calls. The fake backend emits no <NEW DIALOGUE>,
    # so each call yields one piece — the same known smoke behaviour the story arm has, and the
    # reason n_transcripts is a request target rather than a guarantee. All three numbers are
    # recorded, because a corpus that came back at half its request must not read as a clean run.
    assert len(transcripts) == 4
    assert {t.emotion for t in transcripts} == {NEUTRAL_DIALOGUE_LABEL}
    assert record.n_transcripts_configured == 8
    assert record.n_calls_requested == 4
    assert record.n_pieces_obtained == 4
    # No transcript may be dropped for naming a target: there is no target to name.
    assert "names_target" not in record.drop_counts_by_reason
    assert record.n_kept == sum(1 for t in transcripts if t.kept)
    assert basis.n_blocks == 4
    assert len(record.blocks) == 4
    assert record.total_components == sum(row.n_components for row in record.blocks)
    assert record.variance_target == 0.5


def test_the_projection_arm_records_g0_on_both_sides_of_the_projection():
    """The artifact carries the gate before AND after — never only the flattering half."""

    words = read_emotion_words(WORDS_PATH)
    common = {
        "spec": SPEC,
        "stories_per_emotion": 2,
        "story_recipe": "sofroniew",
        # 1 per call: the fake backend emits no <NEW STORY>, so anything higher would fall short
        # of stories_per_emotion and the run would refuse before it got to the projection.
        "sofroniew_stories_per_call": 1,
        "sofroniew_data_dir": DATA_DIR,
        "seed": 7,
        "min_token": 20,
        "max_tokens": 160,
        "fallback_blocks": 4,
        "first_contact_max_drop_rate": 0.6,
    }
    plain, _stories, no_neutral = extract_emotion_vectors(FakeBackend(SPEC), words, **common)
    projected, _stories2, neutral = extract_emotion_vectors(
        FakeBackend(SPEC),
        words,
        neutral_projection=True,
        neutral_dialogues=2,
        neutral_dialogues_per_call=1,
        neutral_variance_target=0.5,
        # The structural minimum: this grid can issue 2 calls and the fake backend returns one
        # transcript each, so 2 is all a fake-backend corpus can be. The real arm's floor is 8.
        neutral_min_kept=2,
        **common,
    )

    assert no_neutral == ()
    assert plain.metadata.neutral_projection is None
    assert plain.metadata.g0_table_before_projection == ()

    assert len(neutral) == 2  # 2 calls x 1 unsplit piece each, under the fake backend
    record = projected.metadata.neutral_projection
    assert record is not None
    assert record.n_pieces_obtained == len(neutral)
    assert record.blocks[0].block == 0
    # The pre-projection table is the unprojected run's table: same stories, same centring, and
    # the projection is the only thing that happens between them.
    assert projected.metadata.g0_table_before_projection == plain.metadata.g0_table
    # ...and the vectors really did move.
    assert projected.metadata.vectors_sha256 != plain.metadata.vectors_sha256
