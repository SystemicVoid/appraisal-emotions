"""Invariants of the E0 story stimuli.

The headline invariant is the stimulus-side affect audit, run with the SAME committed lexicon
the gamble surface passes (``stimuli.emotion_lexicon``): the topics and the prompt scaffolding
must contribute zero emotion words, or the "emotion" vector would partly be a topic vector. The
scaffolding is audited with the ``{emotion}`` slot removed, since that slot is the one place an
emotion word is supposed to appear.

No test here retypes the prompt template or the topic list — both are imported from the single
module that defines them (``docs/agents/rails.md``, clause 2).
"""

from __future__ import annotations

import os

import pytest

from appraisal_emotions.stimuli.emotion_lexicon import emotion_words_in
from appraisal_emotions.stimuli.emotion_stories import (
    STORY_PROMPT_TEMPLATE,
    STORY_TOPICS,
    STYLE_CONTROL_LABEL,
    STYLE_CONTROL_PROMPT_TEMPLATE,
    build_story_grid,
)

EMOTIONS = ("elated", "disappointed", "calm")


def test_topics_are_valence_neutral_under_the_committed_lexicon():
    assert len(STORY_TOPICS) >= 20
    assert len(set(STORY_TOPICS)) == len(STORY_TOPICS)
    for topic in STORY_TOPICS:
        assert emotion_words_in(topic) == (), f"topic {topic!r} carries affect words"


@pytest.mark.parametrize("template", [STORY_PROMPT_TEMPLATE, STYLE_CONTROL_PROMPT_TEMPLATE])
def test_prompt_scaffolding_is_affect_free_outside_the_emotion_slot(template):
    scaffolding = template.replace("{emotion}", "").replace("{topic}", "")
    assert emotion_words_in(scaffolding) == ()


def test_story_prompt_names_its_target_exactly_once_and_nothing_else():
    """The 2606.26987 lexical control, tested FUNCTIONALLY rather than by retyping the clause."""

    rendered = STORY_PROMPT_TEMPLATE.format(topic=STORY_TOPICS[0], emotion="elated")
    # The emotion is the target, named once; the committed lexicon sees it and nothing else.
    assert rendered.count("elated") == 1
    assert emotion_words_in(rendered) == ("elated",)
    # Substituting a different emotion changes exactly that one occurrence — the slot is the
    # ONLY place the template mentions the target.
    other = STORY_PROMPT_TEMPLATE.format(topic=STORY_TOPICS[0], emotion="disappointed")
    assert rendered.replace("elated", "disappointed") == other


def test_style_control_renders_without_an_emotion_argument_at_all():
    rendered = STYLE_CONTROL_PROMPT_TEMPLATE.format(topic=STORY_TOPICS[0])
    assert emotion_words_in(rendered) == ()
    assert "{" not in rendered


def test_the_two_prompts_share_their_shape():
    """Design §9 holds the two surfaces to one shape — checked BY COMPARING THEM TO EACH OTHER.

    No line of either template is retyped here (docs/agents/rails.md): the story prompt is the
    reference and the style control is measured against it, so the test cannot drift from the
    constants and cannot pass by agreeing with a stale copy of them.
    """

    story = STORY_PROMPT_TEMPLATE.format(topic=STORY_TOPICS[0], emotion="elated").splitlines()
    style = STYLE_CONTROL_PROMPT_TEMPLATE.format(topic=STORY_TOPICS[0]).splitlines()
    assert len(story) == len(style) == 3
    # Closer: byte-identical. Opening and middle: a shared prefix, then each surface's own tail.
    assert story[2] == style[2]
    for line_index, minimum in ((0, 40), (1, 60)):
        prefix = os.path.commonprefix([story[line_index], style[line_index]])
        assert len(prefix) >= minimum, f"line {line_index} shares only {len(prefix)} chars"
    # The style control keeps a character (or P5c measures a no-character vector, not a register
    # vector); only the story prompt has a target word to forbid synonyms of, so only it is longer
    # in the middle sentence.
    assert "character" in style[0]
    assert len(story[1]) > len(style[1])


def test_style_control_prompt_names_no_emotion_at_all():
    rendered = STYLE_CONTROL_PROMPT_TEMPLATE.format(
        topic=STORY_TOPICS[1], emotion=STYLE_CONTROL_LABEL
    )
    assert emotion_words_in(rendered) == ()
    assert "{emotion}" not in rendered


def test_grid_shape_and_style_control_arm():
    grid = build_story_grid(EMOTIONS, stories_per_emotion=4, seed=7)
    assert len(grid) == 4 * (len(EMOTIONS) + 1)
    assert {request.emotion for request in grid} == {*EMOTIONS, STYLE_CONTROL_LABEL}
    assert all(request.topic in STORY_TOPICS for request in grid)
    style = [request for request in grid if request.is_style_control]
    assert len(style) == 4
    assert all(emotion_words_in(request.prompt) == () for request in style)


def test_grid_is_seeded_and_per_emotion_independent():
    same = build_story_grid(EMOTIONS, stories_per_emotion=3, seed=7)
    again = build_story_grid(EMOTIONS, stories_per_emotion=3, seed=7)
    other_seed = build_story_grid(EMOTIONS, stories_per_emotion=3, seed=8)
    assert [r.prompt for r in same] == [r.prompt for r in again]
    assert [r.prompt for r in same] != [r.prompt for r in other_seed]
    # Adding a word must not re-roll another word's stimuli (per-emotion seeding).
    widened = build_story_grid((*EMOTIONS, "hopeful"), stories_per_emotion=3, seed=7)
    for emotion in EMOTIONS:
        assert [r.topic for r in same if r.emotion == emotion] == [
            r.topic for r in widened if r.emotion == emotion
        ]


def test_grid_topics_do_not_repeat_below_the_topic_ceiling():
    grid = build_story_grid(("elated",), stories_per_emotion=len(STORY_TOPICS), seed=7)
    topics = [request.topic for request in grid if request.emotion == "elated"]
    assert sorted(topics) == sorted(STORY_TOPICS)


def test_grid_refuses_degenerate_requests():
    with pytest.raises(ValueError, match="stories_per_emotion"):
        build_story_grid(EMOTIONS, stories_per_emotion=0, seed=7)
    with pytest.raises(ValueError, match="at least one emotion"):
        build_story_grid((), stories_per_emotion=2, seed=7)
