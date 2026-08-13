"""Integrity of the pre-registered §5 word set (``data/emotion_words.json``).

The word list and its minted binary valence labels are DATA: the design doc fixes them before
any emotion vector is extracted, and ``docs/agents/rails.md`` forbids restating them in code. So
these tests never spell a word list of their own — every expected name is read out of the file's
own ``confirmatory`` block, and what is checked is internal consistency: the confirmatory
readouts must reference words the list actually carries, every §5 family must be populated, and
the P2 pairs must really be valence-matched (an unmatched pair would make P2's within-valence
permutation meaningless).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from appraisal_emotions.stimuli.emotion_stories import (
    EMOTION_CATEGORIES,
    STYLE_CONTROL_LABEL,
    read_emotion_words,
)

WORDS_PATH = Path(__file__).resolve().parents[1] / "data" / "emotion_words.json"


@pytest.fixture(scope="module")
def words():
    return read_emotion_words(WORDS_PATH)


def test_every_pre_registered_category_is_populated(words):
    assert words.words, "the word set is empty"
    for category in EMOTION_CATEGORIES:
        assert words.words_in_category(category), f"§5 family {category!r} has no words"


def test_valence_labels_are_the_committed_ternary_vocabulary(words):
    assert {word.valence for word in words.words} <= {-1, 0, 1}
    # The surprise family is deliberately valence 0: P4 claims v_absrpe picks it out WITHOUT
    # loading a valence pole, so giving it a pole would beg that question.
    assert all(word.valence == 0 for word in words.words if word.category == "surprise")


def test_confirmatory_words_all_appear_in_the_word_list(words):
    labels = set(words.labels)
    for high, low in words.confirmatory_pairs():
        assert high in labels, f"P2 pair references unknown word {high!r}"
        assert low in labels, f"P2 pair references unknown word {low!r}"
    for key in ("surprise", "arousal_matched"):
        for word in words.confirmatory_set(key):
            assert word in labels, f"P4 {key} references unknown word {word!r}"
    assert str(words.confirmatory["p5a"]) in labels
    # P5c names the pseudo-emotion, which must NOT be an emotion word.
    assert str(words.confirmatory["p5c"]) == STYLE_CONTROL_LABEL
    assert STYLE_CONTROL_LABEL not in labels


def test_p2_pairs_are_valence_matched(words):
    valence = words.valence_by_word
    for high, low in words.confirmatory_pairs():
        assert valence[high] == valence[low], (
            f"pair {high}/{low} is not valence-matched; P2 residualizes on valence and permutes "
            "within the valence-matched set, so an unmatched pair has no null"
        )


def test_p4_contrast_sets_are_disjoint(words):
    surprise = set(words.confirmatory_set("surprise"))
    controls = set(words.confirmatory_set("arousal_matched"))
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
