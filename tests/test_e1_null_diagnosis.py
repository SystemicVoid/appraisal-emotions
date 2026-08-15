"""The one thing in ``scripts/e1_null_diagnosis.py`` that a word-set change can silently break.

The script is a post-run analysis over artifacts that are not all in the repo (the NRC-VAD subset
CSV is research-use-only and gitignored), so it cannot be run end to end in CI. What CAN be
covered, and what actually broke, is its design matrix: it built ``[1, valence]`` at a hard-coded
width of 84 in two places. A word file that is not 84 words wide makes that either raise on a
shape mismatch or — with the wrong kind of luck — analyse a truncated set. The E1 widening moved
the file to 111 words, so the pin was already wrong when this test was written.

Nothing here asserts a width. The point is that the design FOLLOWS the word file, whatever it
says, so the next widening needs no edit to this file either.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scripts.e1_null_diagnosis as diagnosis

from appraisal_emotions.stimuli.emotion_stories import read_emotion_words

WORDS_PATH = Path(__file__).resolve().parents[1] / "data" / "emotion_words.json"


def test_the_valence_design_follows_the_shipped_word_file():
    """One row per word the file carries, at whatever width it carries them."""

    labels = read_emotion_words(WORDS_PATH).labels
    design = diagnosis._valence_design(np.linspace(-1.0, 1.0, len(labels)))
    assert design.shape == (len(labels), 2)
    assert np.array_equal(design[:, 0], np.ones(len(labels)))


@pytest.mark.parametrize("n_words", [5, 84, 111, 200])
def test_the_valence_design_width_is_never_pinned(n_words):
    """A literal would pass at exactly one of these and fail at the rest."""

    assert diagnosis._valence_design(np.zeros(n_words)).shape == (n_words, 2)


def test_the_valence_design_carries_extra_norm_columns():
    """Arousal rides in as another column, so the widened readout needs no second code path."""

    assert diagnosis._valence_design(np.zeros((30, 2))).shape == (30, 3)


def test_the_valence_design_refuses_an_empty_norm_set():
    with pytest.raises(ValueError, match="one row per word"):
        diagnosis._valence_design(np.zeros(0))
