"""Committed emotion-word lexicon for the affect-neutrality audit.

Extracted VERBATIM (imports only) from functional-valence-validity
src/fv_validity/stimuli/emotion_lexicon.py @ 10c4662. Nothing dropped.

The calibration-slice spec (notes/briefs/polarity_asymmetry_calibration_slice_spec.md
§4.2) requires `emotion_word_present` to be set by a committed lexicon check, not
hand-judgment, so the flag is auditable and testable. The lexicon covers explicit
emotion words plus strong evaluative anchor-class words (terrible/awful/wonderful
class) whose presence in a stimulus would entangle the prompt with the §4.3 anchor
sets. Matching is exact-form, case-insensitive, and word-boundary delimited; the
lexicon lists surface forms explicitly rather than stemming.

This is the audit the R-A′ gamble surface passes with zero hits, which is what lets the
emotion-vector layer treat any alignment between ``v_RPE`` and an emotion-concept vector as
something other than lexical leakage from the stimulus side.
"""

from __future__ import annotations

import re

EMOTION_LEXICON_VERSION = "emotion_lexicon/v1"

EMOTION_WORDS: tuple[str, ...] = (
    # Explicit positive emotion words.
    "happy",
    "happiness",
    "glad",
    "joy",
    "joyful",
    "delight",
    "delighted",
    "cheerful",
    "excited",
    "excitement",
    "thrilled",
    "elated",
    "euphoric",
    "ecstatic",
    "overjoyed",
    "pleased",
    "pleasure",
    "proud",
    "pride",
    "grateful",
    "gratitude",
    "relieved",
    "relief",
    "enjoy",
    "enjoyed",
    "enjoyment",
    "love",
    "loved",
    # Explicit negative emotion words.
    "sad",
    "sadness",
    "unhappy",
    "miserable",
    "depressed",
    "depressing",
    "gloomy",
    "grief",
    "sorrow",
    "mournful",
    "anger",
    "angry",
    "furious",
    "rage",
    "irritated",
    "annoyed",
    "frustrated",
    "frustration",
    "fear",
    "afraid",
    "scared",
    "terrified",
    "anxious",
    "anxiety",
    "dread",
    "worried",
    "panic",
    "distress",
    "distressed",
    "upset",
    "ashamed",
    "shame",
    "embarrassed",
    "guilt",
    "guilty",
    "hopeless",
    "despair",
    "heartbroken",
    "devastated",
    "devastating",
    "resentful",
    "resentment",
    "disappointed",
    "disappointment",
    "stressed",
    "stressful",
    "nervous",
    "jealous",
    "envious",
    "lonely",
    "bored",
    "boredom",
    # Strong evaluative anchor-class words (kept out of stimuli so the §4.3
    # anchor readout never sees its own anchors echoed in the prompt).
    "terrible",
    "awful",
    "horrible",
    "dreadful",
    "wonderful",
    "fantastic",
    "amazing",
    "delightful",
)

_EMOTION_WORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in EMOTION_WORDS) + r")\b",
    re.IGNORECASE,
)


def emotion_words_in(text: str) -> tuple[str, ...]:
    """Return the sorted unique lexicon words present in ``text``."""

    return tuple(sorted({match.lower() for match in _EMOTION_WORD_PATTERN.findall(text)}))
