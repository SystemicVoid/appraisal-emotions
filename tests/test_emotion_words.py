"""Integrity of the §5 word set and its recorded expectations (``data/emotion_words.json``).

The word list, its minted binary valence labels and the recorded directional expectations are
DATA: the design doc fixes them before any emotion vector is extracted, and
``docs/agents/rails.md`` forbids restating them in code. So these tests never spell a word list of
their own — every expected name is read out of the file's own ``expectations`` block — and what is
checked is internal consistency: the expectation readouts must reference words and families the
list actually carries, every §5 family must be populated, and the named pairs must really be
valence-matched (an unmatched pair would make the within-valence permutation meaningless).

No width is pinned anywhere either: the set is widened by editing the file (see
``docs/design/e1-widening.md``), and a test that had to be re-edited alongside it would be a
second authority for the same data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from appraisal_emotions.stimuli.emotion_stories import (
    EMOTION_CATEGORIES,
    STYLE_CONTROL_LABEL,
    read_emotion_words,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WORDS_PATH = REPO_ROOT / "data" / "emotion_words.json"


@pytest.fixture(scope="module")
def words():
    return read_emotion_words(WORDS_PATH)


def test_every_design_family_is_populated(words):
    assert words.words, "the word set is empty"
    for category in EMOTION_CATEGORIES:
        assert words.words_in_category(category), f"§5 family {category!r} has no words"


def test_the_word_set_is_internally_consistent_at_whatever_width_it_carries(words):
    """The file is the sole authority for the §5 set (CLAUDE.md), so nothing here pins a width.

    This test used to parse ``**84 words**`` out of ``experiment.md`` and assert the file agreed.
    The E1 widening (``docs/design/e1-widening.md``) moved the file to a new width and its §5 count
    line is stale until ``experiment.md``'s owner updates the per-family tables — an edit the
    widening branch deliberately left to that document. Meanwhile the invariants that actually
    protect the run are internal: every family the loader knows is populated, the per-family counts
    account for every word exactly once, and no word appears twice.

    A width assertion here would have to be re-edited on every widening, which is what made this
    a wiring blocker rather than a finding.
    """

    counts = {family: len(words.words_in_category(family)) for family in EMOTION_CATEGORIES}
    assert set(counts) == set(EMOTION_CATEGORIES)
    assert all(count > 0 for count in counts.values()), counts
    # Every word lands in exactly one family the loader knows about, so the family counts are a
    # partition of the set rather than an overlapping cover.
    assert sum(counts.values()) == len(words.words)
    assert len(set(words.labels)) == len(words.words)
    assert {word.category for word in words.words} == set(EMOTION_CATEGORIES)


def test_valence_labels_are_the_committed_ternary_vocabulary(words):
    assert {word.valence for word in words.words} <= {-1, 0, 1}
    # The surprise family is deliberately valence 0: P4 claims v_absrpe picks it out WITHOUT
    # loading a valence pole, so giving it a pole would beg that question.
    assert all(word.valence == 0 for word in words.words if word.category == "surprise")


def test_expectation_readouts_reference_words_and_families_that_exist(words):
    labels = set(words.labels)
    families = set(EMOTION_CATEGORIES)
    for pair in words.expected_pairs():
        assert pair.outcome in labels, f"pair references unknown word {pair.outcome!r}"
        assert pair.control in labels, f"pair references unknown word {pair.control!r}"
    for contrast in words.expected_family_contrasts():
        assert {contrast.outcome_family, contrast.control_family} <= families
    assert set(words.outcome_ordering()) <= families
    for key in ("surprise", "arousal_matched"):
        assert set(words.p4_words(key)) <= labels, f"P4 {key} references an unknown word"
    assert str(words.expectations["p5a"]) in labels
    # P5c names the pseudo-emotion, which must NOT be an emotion word.
    assert str(words.expectations["p5c"]) == STYLE_CONTROL_LABEL
    assert STYLE_CONTROL_LABEL not in labels


def test_named_pairs_are_valence_matched(words):
    valence = words.valence_by_word
    for pair in words.expected_pairs():
        assert valence[pair.outcome] == valence[pair.control], (
            f"pair {pair.outcome}/{pair.control} is not valence-matched; the readout residualizes "
            "on valence and permutes within the valence-matched set, so an unmatched pair has no "
            "null"
        )


def test_expected_signs_follow_the_signed_rpe_axis(words):
    """The recorded direction of each pair is the pole of the signed v_RPE axis its
    outcome-disconfirmation concept occupies under OCC — which, on this word set, is exactly the
    outcome word's minted valence label."""

    valence = words.valence_by_word
    for pair in words.expected_pairs():
        assert pair.expected_sign == valence[pair.outcome], (
            f"pair {pair.outcome}/{pair.control} records expected_sign={pair.expected_sign} but "
            f"{pair.outcome!r} has valence {valence[pair.outcome]}; the OCC expectation ties the "
            "pole to the outcome word's disconfirmation sign"
        )


def test_family_contrasts_are_valence_matched_and_signed_by_pole(words):
    valence = words.valence_by_word
    for contrast in words.expected_family_contrasts():
        poles = {
            valence[word]
            for family in (contrast.outcome_family, contrast.control_family)
            for word in words.words_in_category(family)
        }
        assert len(poles) == 1, f"{contrast.pole} contrast is not valence-matched: {poles}"
        assert contrast.expected_sign == poles.pop()


def test_outcome_ordering_runs_positive_to_negative(words):
    """The OCC three-level structure: disconfirmed-positive > confirmed > disconfirmed-negative."""

    ordering = words.outcome_ordering()
    assert len(ordering) == 3
    assert ordering[1] == "outcome_confirm"
    valence = words.valence_by_word
    first = {valence[word] for word in words.words_in_category(ordering[0])}
    last = {valence[word] for word in words.words_in_category(ordering[-1])}
    assert first == {1} and last == {-1}


def test_p4_contrast_sets_are_disjoint(words):
    surprise = set(words.p4_words("surprise"))
    controls = set(words.p4_words("arousal_matched"))
    assert surprise and controls
    assert not surprise & controls


def test_word_set_loader_rejects_a_duplicate_word(tmp_path):
    payload = WORDS_PATH.read_text(encoding="utf-8")
    duplicated = payload.replace(
        '{"word": "happy", "category": "anchor", "valence": 1}',
        '{"word": "happy", "category": "anchor", "valence": 1},\n'
        '    {"word": "happy", "category": "anchor", "valence": 1}',
        1,
    )
    path = tmp_path / "duplicated.json"
    path.write_text(duplicated, encoding="utf-8")
    with pytest.raises(ValueError, match="repeats a word"):
        read_emotion_words(path)
