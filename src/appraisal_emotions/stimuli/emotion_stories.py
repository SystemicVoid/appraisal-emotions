"""E0 story stimuli: the §5 word set plus the generation-prompt grid.

NEW module (no parent counterpart — the parent program has no emotion-concept surface).

Implements the stimulus half of the Sofroniew et al. 2026 emotion-basis recipe as design
``docs/design/experiment.md`` §3 pins it: the model writes short stories in which a character
experiences an emotion, and the residual stream averaged over those stories (token 50 onward,
grand mean subtracted) is the emotion-concept vector. Two clauses are load-bearing and both are
carried by :data:`STORY_PROMPT_TEMPLATE`:

- **"without naming the emotion"** — the vector-side lexical-leakage control the design's §6
  threats table names. It comes from the **original** Sofroniew prompt, whose own wording is
  "IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in the
  stories" (primary-verified 2026-08-14; the earlier provisional attribution to the
  arXiv:2606.26987 replication is withdrawn — ``docs/design/sofroniew-recipe.md``).
  The clause is only a request — compliance is measured, not assumed, and
  ``analysis.emotion_vectors`` drops the stories that name the target anyway.
- **valence-neutral topics** — the story setting must not itself carry affect, or the "emotion"
  vector would be a topic vector. :data:`STORY_TOPICS` is audited against the committed
  ``stimuli.emotion_lexicon`` in the test suite, the same fail-closed audit the gamble surface
  passes.

The word set itself is DATA (``data/emotion_words.json``), loaded here and never retyped — the
§5 list, its minted binary valence labels and the recorded directional expectations exist in
exactly one place (``docs/agents/rails.md``, "load source text; never transcribe it").
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

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
    "ExpectedFamilyContrast",
    "ExpectedPair",
    "StoryRequest",
    "build_story_grid",
    "read_emotion_words",
]

EMOTION_STORIES_CONTRACT_VERSION = "emotion_stories/v1"

DEFAULT_EMOTION_WORDS_PATH = Path("data/emotion_words.json")

EmotionCategory = Literal[
    "outcome_pos",
    "outcome_neg",
    "outcome_confirm",
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
    "outcome_confirm",
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

# ONE spelling of the story prompt (design §9, which now points HERE rather than quoting a second
# copy). Extraction and every test that mentions it import THIS constant; nothing retypes it
# (rails.md clause 2). Three sentences: opening, middle, closer. The emotion appears exactly ONCE,
# in double quotes, and is the only affect-bearing text in the rendered prompt.
STORY_PROMPT_TEMPLATE = (
    "Write one paragraph of about 120 words: a short story set in {topic}, about a character "
    'who is experiencing "{emotion}".\n'
    "Show it only through what happens, what the character notices, and what they do — do not "
    "name any emotion, and do not use that word or an obvious synonym for it.\n"
    "Write the paragraph and nothing else."
)

# The P5c control prompt: same three-sentence shape, same opening/middle/closer, same topic grid;
# register instead of emotion is the manipulated variable, and the character stays (a prompt with
# no character would make P5c a no-character vector rather than a register vector). The story
# prompt's extra "or an obvious synonym" clause has no counterpart here — there is no target word.
STYLE_CONTROL_PROMPT_TEMPLATE = (
    "Write one paragraph of about 120 words: a short story set in {topic}, about a character "
    "going about their day, written throughout in a formal register — full sentences, no "
    "contractions, no slang, third person.\n"
    "Show it only through what happens, what the character notices, and what they do — do not "
    "name any emotion.\n"
    "Write the paragraph and nothing else."
)


@dataclass(frozen=True)
class EmotionWord:
    """One §5 word: its family and its MINTED binary valence label."""

    word: str
    category: str
    valence: int


@dataclass(frozen=True)
class ExpectedPair:
    """One recorded named-pair expectation: outcome word vs its valence-matched control.

    ``expected_sign`` is the direction the §5 record puts the outcome word's valence residual in
    relative to its control on ``cos(v_RPE, e_j)``: +1 expects outcome > control, -1 expects
    outcome < control. It is a record of what we thought before we looked, not a contract.
    """

    outcome: str
    control: str
    expected_sign: int


@dataclass(frozen=True)
class ExpectedFamilyContrast:
    """One recorded family-contrast expectation: outcome family vs valence-matched control family."""

    pole: str
    outcome_family: str
    control_family: str
    expected_sign: int


@dataclass(frozen=True)
class EmotionWordSet:
    """The loaded ``data/emotion_words.json`` contents plus its file digest."""

    version: int
    words: tuple[EmotionWord, ...]
    expectations: dict[str, object]
    source_path: Path
    source_sha256: str

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(word.word for word in self.words)

    @property
    def valence_by_word(self) -> dict[str, int]:
        return {word.word: word.valence for word in self.words}

    @property
    def category_by_word(self) -> dict[str, str]:
        return {word.word: word.category for word in self.words}

    def words_in_category(self, category: str) -> tuple[str, ...]:
        return tuple(word.word for word in self.words if word.category == category)

    def _block(self, key: str) -> object:
        return self.expectations[key]

    def expected_pairs(self) -> tuple[ExpectedPair, ...]:
        """The §5 named-pair expectations."""

        return tuple(
            ExpectedPair(
                outcome=str(entry["outcome"]),
                control=str(entry["control"]),
                expected_sign=_sign(entry["expected_sign"]),
            )
            for entry in _mappings(self._block("pairs"), "expectations.pairs")
        )

    def expected_family_contrasts(self) -> tuple[ExpectedFamilyContrast, ...]:
        """The §5 family-contrast expectations, one per valence pole."""

        return tuple(
            ExpectedFamilyContrast(
                pole=str(entry["pole"]),
                outcome_family=str(entry["outcome"]),
                control_family=str(entry["control"]),
                expected_sign=_sign(entry["expected_sign"]),
            )
            for entry in _mappings(self._block("family_contrasts"), "expectations.family_contrasts")
        )

    def pair_pool_families(self) -> tuple[str, ...]:
        """The families a named pair's permutation draws its null from (§5 resolution table).

        The union of the families the recorded expectations already name — the two family
        contrasts plus the three-level ordering — so the pool cannot drift from the contrasts.
        On the shipped word set that is outcome_pos / nonoutcome_pos / outcome_neg /
        nonoutcome_neg / outcome_confirm.
        """

        families: list[str] = []
        for contrast in self.expected_family_contrasts():
            families += [contrast.outcome_family, contrast.control_family]
        families += list(self.outcome_ordering())
        return tuple(dict.fromkeys(families))

    def outcome_ordering(self) -> tuple[str, ...]:
        """The OCC three-level family ordering, highest expected signed residual first."""

        return tuple(str(family) for family in cast(list, self._block("outcome_ordering")))

    def p4_words(self, key: str) -> tuple[str, ...]:
        """The P4 ``surprise`` / ``arousal_matched`` sets, resolved from the named category.

        The expectations block names a CATEGORY, never a word list: membership lives once, in
        ``words``, so the two can never drift apart.
        """

        block = self._block("p4")
        if not isinstance(block, dict):
            raise ValueError("expectations.p4 must be a mapping of role -> category")
        return self.words_in_category(str(block[key]))


def _sign(value: object) -> int:
    sign = int(cast(int, value))
    if sign not in (-1, 1):
        raise ValueError(f"expected_sign must be -1 or +1, got {sign}")
    return sign


def _mappings(block: object, name: str) -> list[dict]:
    if not isinstance(block, list) or not all(isinstance(entry, dict) for entry in block):
        raise ValueError(f"{name} must be a list of mappings")
    return cast(list[dict], block)


def read_emotion_words(path: Path = DEFAULT_EMOTION_WORDS_PATH) -> EmotionWordSet:
    """Load the §5 word set; the file is the sole authority for words, labels and expectations."""

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
        expectations=dict(payload["expectations"]),
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
