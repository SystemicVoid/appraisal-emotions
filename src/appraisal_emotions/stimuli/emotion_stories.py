"""E0 story stimuli: the pre-registered word set plus the generation-prompt grid.

NEW module (no parent counterpart — the parent program has no emotion-concept surface).

Implements the stimulus half of the Sofroniew et al. 2026 emotion-basis recipe as design
``docs/design/experiment.md`` §3 pins it: the model writes short stories in which a character
experiences an emotion, and the residual stream averaged over those stories (token 50 onward,
grand mean subtracted) is the emotion-concept vector. Two clauses are load-bearing and both are
carried by :data:`STORY_PROMPT_TEMPLATE`:

- **"without naming the emotion"** — the lexical control the arXiv:2606.26987 open-model
  replication adds to the recipe (``docs/literature.md`` records that the attribution between
  that replication and the original Sofroniew prompt is *provisional*; we adopt the clause either
  way because it is the vector-side lexical-leakage control the design's §6 threats table names).
  The clause is only a request — compliance is measured, not assumed, and
  ``analysis.emotion_vectors`` drops the stories that name the target anyway.
- **valence-neutral topics** — the story setting must not itself carry affect, or the "emotion"
  vector would be a topic vector. :data:`STORY_TOPICS` is audited against the committed
  ``stimuli.emotion_lexicon`` in the test suite, the same fail-closed audit the gamble surface
  passes.

The word set itself is DATA (``data/emotion_words.json``), loaded here and never retyped — the
§5 list and its minted binary valence labels exist in exactly one place
(``docs/agents/rails.md``, "load source text; never transcribe it").
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from appraisal_emotions.core.util import file_sha256

__all__ = [
    "DEFAULT_EMOTION_WORDS_PATH",
    "EMOTION_CATEGORIES",
    "STORY_PROMPT_TEMPLATE",
    "STORY_TOPICS",
    "STYLE_CONTROL_LABEL",
    "STYLE_CONTROL_PROMPT_TEMPLATE",
    "EmotionWord",
    "EmotionWordSet",
    "StoryRequest",
    "build_story_grid",
    "read_emotion_words",
]

EMOTION_STORIES_CONTRACT_VERSION = "emotion_stories/v1"

DEFAULT_EMOTION_WORDS_PATH = Path("data/emotion_words.json")

EmotionCategory = Literal[
    "outcome_pos",
    "outcome_neg",
    "nonoutcome_pos",
    "nonoutcome_neg",
    "prospect",
    "surprise",
    "arousal_control",
    "agency_ext",
    "anchor",
]
EMOTION_CATEGORIES: tuple[str, ...] = (
    "outcome_pos",
    "outcome_neg",
    "nonoutcome_pos",
    "nonoutcome_neg",
    "prospect",
    "surprise",
    "arousal_control",
    "agency_ext",
    "anchor",
)

# The P5c style control (design §4 E1.5c): a valence-free "formal register" pseudo-emotion run
# through the identical recipe. Not an emotion word and never scored as one — it is the control
# that makes the cosine scale interpretable.
STYLE_CONTROL_LABEL = "style_control"

# ~25 generic, valence-neutral settings. Zero emotion-lexicon hits (tested), no outcome framing,
# no evaluative adjectives: the topic must contribute nothing the emotion axis could read.
STORY_TOPICS: tuple[str, ...] = (
    "a morning commute on a city bus",
    "a kitchen at the end of the day",
    "a garden after a long rain",
    "an open-plan office on a Tuesday",
    "a laundromat late at night",
    "a public library reading room",
    "a train platform between departures",
    "a hardware store aisle",
    "a clinic waiting room",
    "a beach in the off-season",
    "a mountain trail at first light",
    "a hotel lobby during a conference",
    "a school gymnasium after hours",
    "a corner grocery at closing time",
    "a car repair shop waiting area",
    "a rooftop with a water tank",
    "a river ferry crossing",
    "a woodworking shed",
    "a university lecture hall",
    "a bakery before opening",
    "a bus depot maintenance bay",
    "a small-town post office",
    "an apartment stairwell",
    "a community swimming pool",
    "a roadside diner at midnight",
)

# ONE spelling of the story prompt. Extraction and every test that mentions it import THIS
# constant; nothing retypes it (rails.md clause 2). The emotion goes in double quotes and is the
# only affect-bearing text in the rendered prompt.
STORY_PROMPT_TEMPLATE = (
    "Write one paragraph of about 120 words: a short story set in {topic}.\n"
    'The main character is experiencing the emotion "{emotion}".\n'
    "Convey it only through what happens, what the character notices, and what they do.\n"
    'Do not use the word "{emotion}" anywhere, and do not use an obvious synonym for it.\n'
    "Do not name any emotion at all.\n"
    "Write the paragraph and nothing else."
)

# The P5c control prompt: same length, same topic grid, same "one paragraph" shape, register
# instead of emotion as the manipulated variable.
STYLE_CONTROL_PROMPT_TEMPLATE = (
    "Write one paragraph of about 120 words: a short account set in {topic}.\n"
    "Write it in a strictly formal register: full sentences, no contractions, no slang,\n"
    "the impersonal third person throughout.\n"
    "Describe only what happens and what is observable.\n"
    "Do not name any emotion at all.\n"
    "Write the paragraph and nothing else."
)


@dataclass(frozen=True)
class EmotionWord:
    """One pre-registered word: its §5 family and its MINTED binary valence label."""

    word: str
    category: str
    valence: int


@dataclass(frozen=True)
class EmotionWordSet:
    """The loaded ``data/emotion_words.json`` contents plus its file digest."""

    version: int
    words: tuple[EmotionWord, ...]
    confirmatory: dict[str, object]
    source_path: Path
    source_sha256: str

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(word.word for word in self.words)

    @property
    def valence_by_word(self) -> dict[str, int]:
        return {word.word: word.valence for word in self.words}

    def words_in_category(self, category: str) -> tuple[str, ...]:
        return tuple(word.word for word in self.words if word.category == category)

    def confirmatory_pairs(self) -> tuple[tuple[str, str, int], ...]:
        """The P2 pairs as ``(outcome, control, predicted_sign)`` triples.

        ``predicted_sign`` is the pre-registered direction of the outcome word's valence
        residual relative to its control on ``cos(v_RPE, e_j)``: +1 predicts outcome > control,
        -1 predicts outcome < control (§5 amendment record, 2026-08-13, pre-data).
        """

        pairs = self.confirmatory["p2_pairs"]
        if not isinstance(pairs, list):
            raise ValueError("confirmatory.p2_pairs must be a list of pair mappings")
        triples: list[tuple[str, str, int]] = []
        for pair in pairs:
            if not isinstance(pair, dict):
                raise ValueError(
                    "confirmatory.p2_pairs entries must be {outcome, control, predicted_sign}"
                )
            sign = int(pair["predicted_sign"])
            if sign not in (-1, 1):
                raise ValueError(f"predicted_sign must be -1 or +1, got {sign}")
            triples.append((str(pair["outcome"]), str(pair["control"]), sign))
        return tuple(triples)

    def confirmatory_set(self, key: str) -> tuple[str, ...]:
        block = self.confirmatory["p4"]
        if not isinstance(block, dict):
            raise ValueError("confirmatory.p4 must be a mapping")
        return tuple(str(word) for word in block[key])


def read_emotion_words(path: Path = DEFAULT_EMOTION_WORDS_PATH) -> EmotionWordSet:
    """Load the pre-registered §5 word set; the file is the sole authority for words and labels."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_words = payload["words"]
    if not raw_words:
        raise ValueError(f"{path} lists no words")
    words = tuple(
        EmotionWord(
            word=str(entry["word"]), category=str(entry["category"]), valence=int(entry["valence"])
        )
        for entry in raw_words
    )
    labels = [word.word for word in words]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{path} repeats a word; each word must be listed once")
    unknown = sorted({word.category for word in words} - set(EMOTION_CATEGORIES))
    if unknown:
        raise ValueError(f"{path} uses unknown categories {unknown}")
    if any(word.valence not in (-1, 0, 1) for word in words):
        raise ValueError(f"{path} carries a valence label outside -1/0/+1")
    if STYLE_CONTROL_LABEL in labels:
        raise ValueError(
            f"{path} lists {STYLE_CONTROL_LABEL!r} as an emotion word; it is the P5c pseudo-emotion"
        )
    return EmotionWordSet(
        version=int(payload["version"]),
        words=words,
        confirmatory=dict(payload["confirmatory"]),
        source_path=Path(path),
        source_sha256=file_sha256(Path(path)),
    )


@dataclass(frozen=True)
class StoryRequest:
    """One (emotion, topic) generation request and its rendered prompt."""

    emotion: str
    topic: str
    prompt: str
    story_index: int

    @property
    def is_style_control(self) -> bool:
        return self.emotion == STYLE_CONTROL_LABEL


def _topics_for(emotion: str, *, stories_per_emotion: int, seed: int) -> list[str]:
    """A seeded topic assignment for one emotion: shuffled, cycled if more stories than topics.

    Per-emotion seeding (not one global stream) keeps a word's topic list independent of how many
    other words the run extracts, so adding a word never re-rolls another word's stimuli.
    """

    order = list(STORY_TOPICS)
    random.Random(f"{seed}|emotion-stories|{emotion}").shuffle(order)
    return [order[index % len(order)] for index in range(stories_per_emotion)]


def build_story_grid(
    emotions: tuple[str, ...],
    *,
    stories_per_emotion: int,
    seed: int,
    include_style_control: bool = True,
) -> tuple[StoryRequest, ...]:
    """The (emotion × topic) generation grid, deterministic in ``seed``.

    ``include_style_control`` appends the P5c ``style_control`` pseudo-emotion, generated from
    :data:`STYLE_CONTROL_PROMPT_TEMPLATE` over the same topic grid at the same count — same
    recipe, no emotion slot, so its residual alignment measures the cosine scale rather than any
    emotion.
    """

    if stories_per_emotion < 1:
        raise ValueError("stories_per_emotion must be >= 1")
    if not emotions:
        raise ValueError("build_story_grid needs at least one emotion")
    labels = list(emotions) + ([STYLE_CONTROL_LABEL] if include_style_control else [])
    requests: list[StoryRequest] = []
    for emotion in labels:
        template = (
            STYLE_CONTROL_PROMPT_TEMPLATE
            if emotion == STYLE_CONTROL_LABEL
            else STORY_PROMPT_TEMPLATE
        )
        for story_index, topic in enumerate(
            _topics_for(emotion, stories_per_emotion=stories_per_emotion, seed=seed)
        ):
            requests.append(
                StoryRequest(
                    emotion=emotion,
                    topic=topic,
                    prompt=template.format(topic=topic, emotion=emotion),
                    story_index=story_index,
                )
            )
    return tuple(requests)
