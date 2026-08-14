"""The paper-prompt E0 arm: loading, grid construction, and the A/B's one-variable property.

Every assertion about the paper's text reads ``data/sofroniew2026/``, which
``scripts/fetch_sofroniew_recipe.py`` extracts from the arXiv source. Nothing here restates the
prompt, the topics or the word list — a test comparing our copy against a restatement of our copy
is no check at all (``docs/agents/rails.md`` clause 1).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from appraisal_emotions.analysis.emotion_vectors import build_stimulus_plan, generate_stories
from appraisal_emotions.backends.fake import FakeBackend
from appraisal_emotions.core.schema import ModelSpec
from appraisal_emotions.stimuli.emotion_stories import STYLE_CONTROL_LABEL, read_emotion_words
from appraisal_emotions.stimuli.sofroniew_stories import (
    STORY_SEPARATOR,
    build_sofroniew_story_grid,
    build_style_control_prompt,
    read_sofroniew_recipe,
    split_completion,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sofroniew2026"
WORDS_PATH = ROOT / "data" / "emotion_words.json"
SPEC = ModelSpec(key="fake", backend="fake", model_id="fake")


@pytest.fixture(scope="module")
def recipe():
    return read_sofroniew_recipe(DATA_DIR)


# --------------------------------------------------------------------------------------
# The extracted data is what the paper says it is
# --------------------------------------------------------------------------------------


def test_extracted_counts_match_the_paper(recipe):
    """171 emotion words and 100 topics — the counts the paper states in prose."""

    assert len(recipe.paper_emotion_words) == 171
    assert len(recipe.topics) == 100
    assert len(set(recipe.topics)) == len(recipe.topics)


def test_prompt_carries_its_slots_and_separator(recipe):
    template = recipe.story_prompt_template
    for slot in ("{n_stories}", "{topic}", "{emotion}"):
        assert slot in template
    assert STORY_SEPARATOR in template


def test_normalization_is_recorded_and_reversible():
    """The one source-side substitution is declared, evidenced, and shown alongside the raw."""

    payload = json.loads((DATA_DIR / "prompts.json").read_text(encoding="utf-8"))
    story = payload["prompts"]["emotional_stories"]
    evidence = payload["normalization_evidence"]
    # Zero apostrophes across every promptblock in the paper is what licenses reading the
    # backticks as apostrophes; if a re-extraction ever shows apostrophes, the rule must be
    # revisited rather than silently kept.
    assert evidence["ascii_apostrophes"] == 0
    assert evidence["backticks"] > 0
    assert story["raw"].replace("`", "'") == story["text"]


def test_provenance_pins_the_source(recipe):
    assert recipe.arxiv_id == "2604.07729"
    assert len(recipe.main_tex_sha256) == 64
    assert set(recipe.source_sha256) == {
        "prompts.json",
        "topics_100.json",
        "emotion_words_171.json",
        "provenance.json",
    }


# --------------------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------------------


def test_topics_are_crossed_not_sampled_per_label(recipe):
    """Every label sees the SAME topics, so topic cancels in the grand-mean centring.

    This is the property that makes affect-laden paper topics safe. A per-label topic draw would
    let a topic effect masquerade as an emotion effect.
    """

    grid = build_sofroniew_story_grid(
        ("elated", "disappointed", "calm"),
        recipe=recipe,
        topics_per_label=6,
        stories_per_call=2,
        seed=7,
    )
    by_label = {}
    for request in grid:
        by_label.setdefault(request.emotion, []).append(request.topic)
    topic_sets = {label: sorted(topics) for label, topics in by_label.items()}
    assert len(set(map(tuple, topic_sets.values()))) == 1
    assert all(len(set(topics)) == 6 for topics in by_label.values())


def test_grid_covers_every_label_and_appends_the_style_control(recipe):
    labels = ("elated", "disappointed")
    grid = build_sofroniew_story_grid(
        labels, recipe=recipe, topics_per_label=3, stories_per_call=4, seed=7
    )
    assert len(grid) == (len(labels) + 1) * 3
    assert {request.emotion for request in grid} == {*labels, STYLE_CONTROL_LABEL}
    # story_index steps by stories_per_call so a completion's pieces get consecutive indices
    # without colliding with the next call's.
    indices = sorted({request.story_index for request in grid})
    assert indices == [0, 4, 8]


def test_rendered_prompt_names_the_emotion_and_the_batch_size(recipe):
    grid = build_sofroniew_story_grid(
        ("disappointed",), recipe=recipe, topics_per_label=1, stories_per_call=3, seed=7
    )
    story = next(r for r in grid if r.emotion == "disappointed")
    assert "disappointed" in story.prompt
    assert "Write 3 different stories" in story.prompt
    assert story.topic in story.prompt
    assert "{" not in story.prompt  # every slot bound


def test_style_control_prompt_drops_every_emotion_slot(recipe):
    control = build_style_control_prompt(recipe.story_prompt_template)
    assert "{emotion}" not in control
    assert "{n_stories}" in control and "{topic}" in control
    # It keeps the paper's scaffolding — the format block and the separator — so the control and
    # the stories differ only where they must.
    assert STORY_SEPARATOR in control


def test_style_control_substitution_fails_loudly_when_the_prompt_changes():
    with pytest.raises(ValueError, match="matched 0 times"):
        build_style_control_prompt("Write {n_stories} stories about {topic}.")


# --------------------------------------------------------------------------------------
# The splitter
# --------------------------------------------------------------------------------------


def test_split_recovers_stories_around_the_separator():
    completion = f"{STORY_SEPARATOR}\nFirst.\n{STORY_SEPARATOR}\nSecond.\n"
    assert split_completion(completion) == ("First.", "Second.")


def test_split_of_an_unseparated_completion_yields_one_piece():
    """BLIND-freeze behaviour: no repair, one over-long piece the first-contact check will show."""

    assert split_completion("A single unbroken paragraph.") == ("A single unbroken paragraph.",)


def test_split_of_empty_text_yields_nothing():
    assert split_completion("   ") == ()


# --------------------------------------------------------------------------------------
# The plan: what makes this an A/B rather than two experiments
# --------------------------------------------------------------------------------------


def test_both_arms_produce_the_same_story_count_per_label():
    labels = read_emotion_words(WORDS_PATH).labels[:4]
    project = build_stimulus_plan(labels, story_recipe="project", stories_per_emotion=12, seed=7)
    sofroniew = build_stimulus_plan(
        labels,
        story_recipe="sofroniew",
        stories_per_emotion=12,
        seed=7,
        sofroniew_stories_per_call=2,
        sofroniew_data_dir=DATA_DIR,
    )
    assert project.stories_per_emotion == sofroniew.stories_per_emotion == 12
    # Same labels, different prompts: the manipulation is the stimulus and nothing else.
    assert {r.emotion for r in project.requests} == {r.emotion for r in sofroniew.requests}
    assert project.stimulus_hash != sofroniew.stimulus_hash
    assert sofroniew.recipe_manifest["arxiv_id"] == "2604.07729"


def test_uneven_batch_size_is_refused():
    with pytest.raises(ValueError, match="not a multiple"):
        build_stimulus_plan(
            ("elated",),
            story_recipe="sofroniew",
            stories_per_emotion=12,
            seed=7,
            sofroniew_stories_per_call=5,
            sofroniew_data_dir=DATA_DIR,
        )


def test_unknown_recipe_is_refused():
    with pytest.raises(ValueError, match="unknown story_recipe"):
        build_stimulus_plan(("elated",), story_recipe="paper", stories_per_emotion=12, seed=7)


def test_batched_generation_emits_one_story_per_split_piece():
    """The split feeds the existing filter, so batching does not lose the drop audit."""

    backend = FakeBackend(SPEC)
    plan = build_stimulus_plan(
        ("elated", "disappointed"),
        story_recipe="sofroniew",
        stories_per_emotion=4,
        seed=7,
        sofroniew_stories_per_call=2,
        sofroniew_data_dir=DATA_DIR,
    )
    stories = generate_stories(
        backend,
        plan.requests,
        min_token=20,
        max_tokens=200,
        temperature=1.0,
        split=lambda text: (text, text),
    )
    assert len(stories) == 2 * len(plan.requests)
    for label in ("elated", "disappointed"):
        indices = [s.story_index for s in stories if s.emotion == label]
        assert len(indices) == len(set(indices)), "split pieces collided on story_index"
