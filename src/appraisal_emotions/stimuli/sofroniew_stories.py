"""The paper-faithful E0 stimulus arm: Sofroniew et al. 2026's own generation prompt.

Sibling of :mod:`appraisal_emotions.stimuli.emotion_stories`, which carries the *project* recipe
(a three-sentence prompt, one paragraph per call, 25 valence-neutral settings). This module
carries the recipe as the paper actually ran it, so the two can be run as an A/B on the same
model, the same word set and the same capture window, with the stimulus recipe as the only
manipulated variable.

Why the arm exists: a reviewer can ask whether our E0 result is a property of emotion-concept
geometry or of *our* prompt. That question is answerable, cheaply, by re-extracting under the
original prompt — so it should be answered with data rather than argued.

**Nothing here is transcribed.** The prompt, the 100 topics and the 171-word emotion list are
loaded from ``data/sofroniew2026/``, which ``scripts/fetch_sofroniew_recipe.py`` extracts from
the arXiv:2604.07729 LaTeX source (``docs/agents/rails.md`` clause 1, case 3). The word list is
loaded for comparison and provenance; the arm's *extraction* runs on the project's §5 word set,
because the paper's 171 words do not contain the vocabulary E1/E2 are about — see
``docs/design/sofroniew-recipe.md``.

It also carries the paper's *neutral dialogue* corpus — the emotionally-neutral Human/Assistant
transcripts whose principal components the paper projects out of the emotion vectors. Those are
not stimuli for any emotion; :mod:`appraisal_emotions.analysis.neutral_projection` says what is
done with them.

Four faithfulness points that are decisions, not accidents, and are recorded as such:

1. **Batched generation.** The paper's prompt asks for ``{n_stories}`` stories in ONE completion
   separated by ``<NEW STORY>``, and instructs cross-story diversity ("not use the same turns of
   phrase", "a mix of third-person and first-person narration"). That instruction only has
   meaning across stories the model writes together, so the arm batches, and
   :func:`split_completion` recovers the individual stories.
2. **Topics are crossed, not sampled per emotion.** The paper's 100 topics are affect-laden
   premises ("A student learns their scholarship application was denied"), the opposite of the
   project recipe's deliberately neutral settings. That is safe in the paper because every topic
   is used for every emotion, so topic contributes equally to every emotion mean and cancels in
   the grand-mean subtraction. It is NOT safe under per-emotion topic sampling, which would turn
   a topic effect into an emotion effect. This arm therefore draws ONE seeded topic subset and
   uses it for every label.
3. **The P5c style control is ours, derived here.** The paper has no style control; P5c is what
   makes our cosine scale readable, so it must run through whatever recipe the arm uses. The
   control prompt is built from the loaded paper prompt by replacing its three emotion-bearing
   lines, each replacement asserted to have hit (:func:`build_style_control_prompt`), so the
   control and the stories differ only where they must.
4. **The neutral corpus reuses the story topics.** The paper's appendix says the same 100 topics
   "seed the generation of our stories and dialogues datasets", so the confound subspace it
   removes is fit on the same topic material the stories were written from. At our scale that has
   to be arranged rather than inherited: :func:`build_neutral_dialogue_grid` draws from the SAME
   seeded shuffle
   :func:`build_sofroniew_story_grid` draws from, so the neutral corpus spans topics the stories
   saw instead of a disjoint sample of the 100.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from appraisal_emotions.core.util import file_sha256
from appraisal_emotions.stimuli.emotion_stories import STYLE_CONTROL_LABEL, StoryRequest

__all__ = [
    "DEFAULT_SOFRONIEW_DATA_DIR",
    "DIALOGUE_SEPARATOR",
    "NEUTRAL_DIALOGUE_LABEL",
    "SOFRONIEW_STORIES_CONTRACT_VERSION",
    "STORY_SEPARATOR",
    "SofroniewRecipe",
    "apply_speaker_substitutions",
    "build_neutral_dialogue_grid",
    "build_sofroniew_story_grid",
    "build_style_control_prompt",
    "neutral_completion_splitter",
    "read_sofroniew_recipe",
    "split_completion",
]

SOFRONIEW_STORIES_CONTRACT_VERSION = "sofroniew_stories/v1"

DEFAULT_SOFRONIEW_DATA_DIR = Path("data/sofroniew2026")

# The separator the paper's prompt instructs the model to emit between stories.
STORY_SEPARATOR = "<NEW STORY>"
# ...and between neutral dialogues.
DIALOGUE_SEPARATOR = "<NEW DIALOGUE>"

# The label carried by neutral-dialogue transcripts through the shared generation path. It is NOT
# an emotion and never becomes a basis row: the transcripts exist only to fit the confound
# subspace that gets projected out (``analysis.neutral_projection``).
NEUTRAL_DIALOGUE_LABEL = "neutral_dialogue"

# The three lines of the paper prompt that name the emotion, and what the P5c control puts in
# their place. These replacements are OUR text — they reproduce nothing held in a file, so no
# load applies. Each is asserted to match exactly once; a silent miss would make the control a
# different experiment.
_STYLE_CONTROL_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    (
        "The story should follow a character who is feeling {emotion}.",
        "The story should follow a character going about their day, written throughout in a "
        "formal register: full sentences, no contractions, no slang, third person.",
    ),
    (
        "IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in the "
        "stories. Instead, convey the emotion ONLY through:",
        "IMPORTANT: You must NEVER name an emotion in the stories. Instead, convey the "
        "character's situation ONLY through:",
    ),
    (
        "The emotion should be clearly conveyed to the reader through these indirect means, but "
        "never explicitly named.",
        "The formal register should be maintained throughout, and no emotion should ever be named.",
    ),
)


@dataclass(frozen=True)
class SofroniewRecipe:
    """The loaded ``data/sofroniew2026/`` contents plus the digests of the files it came from.

    ``paper_emotion_words`` is carried for provenance and for the word-set comparison in
    ``docs/design/sofroniew-recipe.md``; the arm does not extract on it by default.
    """

    story_prompt_template: str
    neutral_dialogue_prompt_template: str
    # The paper's post-hoc speaker rename, extracted from its prose (Person: -> Human:,
    # AI: -> Assistant:). Loaded, never typed here: it decides what tokens the neutral
    # transcripts actually carry, and the model's Human/Assistant formatting is exactly the
    # confound direction the projection is meant to remove.
    speaker_substitutions: tuple[tuple[str, str], ...]
    topics: tuple[str, ...]
    paper_emotion_words: tuple[str, ...]
    arxiv_id: str
    arxiv_version: str
    main_tex_sha256: str
    source_dir: Path
    source_sha256: dict[str, str]

    @property
    def style_control_prompt_template(self) -> str:
        return build_style_control_prompt(self.story_prompt_template)

    def manifest(self) -> dict[str, object]:
        """The provenance rows a run manifest records for this arm."""

        return {
            "contract_version": SOFRONIEW_STORIES_CONTRACT_VERSION,
            "arxiv_id": self.arxiv_id,
            "arxiv_version": self.arxiv_version,
            "main_tex_sha256": self.main_tex_sha256,
            "source_dir": str(self.source_dir),
            "source_sha256": dict(self.source_sha256),
            "n_topics": len(self.topics),
            "n_paper_emotion_words": len(self.paper_emotion_words),
            "speaker_substitutions": [
                {"from": src, "to": dst} for src, dst in self.speaker_substitutions
            ],
        }


def read_sofroniew_recipe(data_dir: Path = DEFAULT_SOFRONIEW_DATA_DIR) -> SofroniewRecipe:
    """Load the extracted paper recipe; these files are the sole authority for its text."""

    data_dir = Path(data_dir)
    names = ("prompts.json", "topics_100.json", "emotion_words_171.json", "provenance.json")
    missing = [name for name in names if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{data_dir} is missing {missing}. Run scripts/fetch_sofroniew_recipe.py to extract "
            "the recipe from the arXiv source."
        )
    payloads = {name: json.loads((data_dir / name).read_text(encoding="utf-8")) for name in names}

    story = payloads["prompts.json"]["prompts"]["emotional_stories"]
    template = str(story["text"])
    for slot in ("{n_stories}", "{topic}", "{emotion}"):
        if slot not in template:
            raise ValueError(f"the loaded story prompt has no {slot} slot; re-run the fetch script")
    if STORY_SEPARATOR not in template:
        raise ValueError(
            f"the loaded story prompt never mentions {STORY_SEPARATOR!r}; split_completion would "
            "have nothing to split on"
        )

    neutral = payloads["prompts.json"]["prompts"]["neutral_dialogues"]
    neutral_template = str(neutral["text"])
    for slot in ("{n_stories}", "{topic}"):
        if slot not in neutral_template:
            raise ValueError(
                f"the loaded neutral-dialogue prompt has no {slot} slot; re-run the fetch script"
            )
    if DIALOGUE_SEPARATOR not in neutral_template:
        raise ValueError(
            f"the loaded neutral-dialogue prompt never mentions {DIALOGUE_SEPARATOR!r}; the "
            "transcript splitter would have nothing to split on"
        )
    substitutions = tuple(
        (str(pair["from"]), str(pair["to"])) for pair in neutral["post_hoc_speaker_substitutions"]
    )
    if not substitutions:
        raise ValueError("the loaded neutral-dialogue prompt carries no speaker substitutions")
    for source, _target in substitutions:
        # Load-vs-load consistency, not literal-vs-literal: the rename was parsed from one place
        # in the paper and the prompt from another, so a drifted parse shows up here.
        if source not in neutral_template:
            raise ValueError(
                f"speaker substitution {source!r} does not occur in the neutral-dialogue prompt"
            )

    topics = tuple(str(topic) for topic in payloads["topics_100.json"]["topics"])
    words = tuple(str(word) for word in payloads["emotion_words_171.json"]["words"])
    if not topics or not words:
        raise ValueError(f"{data_dir} carries an empty topic or word list")

    source = payloads["provenance.json"]["source"]
    return SofroniewRecipe(
        story_prompt_template=template,
        neutral_dialogue_prompt_template=neutral_template,
        speaker_substitutions=substitutions,
        topics=topics,
        paper_emotion_words=words,
        arxiv_id=str(source["arxiv_id"]),
        arxiv_version=str(source["arxiv_version"]),
        main_tex_sha256=str(source["main_tex_sha256"]),
        source_dir=data_dir,
        source_sha256={name: file_sha256(data_dir / name) for name in names},
    )


def build_style_control_prompt(story_prompt_template: str) -> str:
    """The P5c control prompt: the paper prompt with its emotion-bearing lines replaced.

    Every substitution must hit exactly once. A missed substitution would leave ``{emotion}``
    unbound (a crash) or leave the control asking for an emotion (a silent wrong control), so
    this fails loudly instead of best-effort patching.
    """

    prompt = story_prompt_template
    for target, replacement in _STYLE_CONTROL_SUBSTITUTIONS:
        occurrences = prompt.count(target)
        if occurrences != 1:
            raise ValueError(
                f"style-control substitution matched {occurrences} times, expected 1: {target!r}. "
                "The paper prompt changed; update _STYLE_CONTROL_SUBSTITUTIONS deliberately."
            )
        prompt = prompt.replace(target, replacement)
    if "{emotion}" in prompt:
        raise ValueError("the style-control prompt still carries an {emotion} slot")
    return prompt


def split_completion(text: str) -> tuple[str, ...]:
    """Recover the individual stories from one batched completion.

    Frozen BLIND: no story from the run's model has been read, because generation needs the GPU
    the arm is being prepared for (``docs/agents/rails.md`` clause "Observation before
    imagination" licenses a blind freeze only against an early-N checkpoint, and the E0 config
    carries one). The checkpoint that catches THIS function failing is the yield check in
    :func:`~appraisal_emotions.analysis.emotion_vectors.extract_emotion_vectors`, not the length
    or lexical filter: a completion this function cannot split yields one piece that is a
    perfectly well-formed story of ordinary length, so no per-piece filter can see it. What is
    observable is that the label came back with fewer stories than were asked for.

    Deliberately literal: split on the separator the prompt asked for, drop empties, keep
    everything else as written. The model may or may not emit a leading separator, and may append
    trailing commentary; neither is repaired here, because a repair frozen against unread output
    is a guess, and the checkpoint is where guesses get corrected.
    """

    pieces = [piece.strip() for piece in text.split(STORY_SEPARATOR)]
    return tuple(piece for piece in pieces if piece)


def apply_speaker_substitutions(text: str, substitutions: tuple[tuple[str, str], ...]) -> str:
    """The paper's post-hoc rename of the generated transcript's speaker tags.

    Order matters only if one substitution's output is another's input; the paper's two pairs
    (``Person:`` -> ``Human:``, ``AI:`` -> ``Assistant:``) do not overlap, and this asserts that
    rather than assuming it, because a chained rewrite would silently produce a third string.
    """

    for index, (source, target) in enumerate(substitutions):
        for later_source, _ in substitutions[index + 1 :]:
            if later_source in target:
                raise ValueError(
                    f"speaker substitution {source!r}->{target!r} writes text that a later "
                    f"substitution ({later_source!r}) would rewrite again; the rewrite would chain"
                )
        text = text.replace(source, target)
    return text


def neutral_completion_splitter(
    substitutions: tuple[tuple[str, str], ...],
) -> Callable[[str], tuple[str, ...]]:
    """Split one batched neutral completion into transcripts, then rename its speakers.

    Same BLIND-freeze status and same deliberate literalness as :func:`split_completion` — this is
    the dialogue-side twin of it, with the paper's post-hoc rename folded in so no transcript can
    reach the capture with the prompt's ``Person:``/``AI:`` tags still on it.
    """

    def split(text: str) -> tuple[str, ...]:
        pieces = [piece.strip() for piece in text.split(DIALOGUE_SEPARATOR)]
        return tuple(apply_speaker_substitutions(piece, substitutions) for piece in pieces if piece)

    return split


def _shared_topics(*, topics: tuple[str, ...], topics_per_label: int, seed: int) -> list[str]:
    """ONE seeded topic subset, used for every label so topic cancels in the grand mean.

    Sampled without replacement when the subset fits, so no label sees a topic twice.
    """

    if topics_per_label < 1:
        raise ValueError("topics_per_label must be >= 1")
    if topics_per_label > len(topics):
        raise ValueError(
            f"topics_per_label={topics_per_label} exceeds the {len(topics)} topics the paper lists"
        )
    order = list(topics)
    random.Random(f"{seed}|sofroniew-topics").shuffle(order)
    return order[:topics_per_label]


def build_sofroniew_story_grid(
    emotions: tuple[str, ...],
    *,
    recipe: SofroniewRecipe,
    topics_per_label: int,
    stories_per_call: int,
    seed: int,
    include_style_control: bool = True,
) -> tuple[StoryRequest, ...]:
    """The paper-recipe generation grid: ``topics_per_label`` calls per label, batched.

    Each request asks for ``stories_per_call`` stories, so a label ends up with
    ``topics_per_label * stories_per_call`` stories — set the two so that product matches the
    project arm's ``stories_per_emotion``, or the A/B compares sample sizes as well as prompts.

    ``story_index`` steps by ``stories_per_call`` so the stories recovered from one completion
    can take consecutive indices without colliding with the next call's.
    """

    if not emotions:
        raise ValueError("build_sofroniew_story_grid needs at least one emotion")
    if stories_per_call < 1:
        raise ValueError("stories_per_call must be >= 1")

    topics = _shared_topics(topics=recipe.topics, topics_per_label=topics_per_label, seed=seed)
    labels = list(emotions) + ([STYLE_CONTROL_LABEL] if include_style_control else [])
    style_template = recipe.style_control_prompt_template

    requests: list[StoryRequest] = []
    for label in labels:
        is_control = label == STYLE_CONTROL_LABEL
        template = style_template if is_control else recipe.story_prompt_template
        for call_index, topic in enumerate(topics):
            rendered = (
                template.format(n_stories=stories_per_call, topic=topic)
                if is_control
                else template.format(n_stories=stories_per_call, topic=topic, emotion=label)
            )
            requests.append(
                StoryRequest(
                    emotion=label,
                    topic=topic,
                    prompt=rendered,
                    story_index=call_index * stories_per_call,
                )
            )
    return tuple(requests)


def build_neutral_dialogue_grid(
    *,
    recipe: SofroniewRecipe,
    n_transcripts: int,
    dialogues_per_call: int,
    seed: int,
) -> tuple[StoryRequest, ...]:
    """The confound corpus: ``n_transcripts`` emotionally neutral Human/Assistant transcripts.

    These are NOT stimuli for any emotion. Their activations fit the subspace the paper projects
    out of the emotion vectors before using them ("model activations on a set of emotionally
    neutral transcripts ... enough to explain 50% of the variance"), so what matters about them is
    coverage of the ordinary-conversation directions, not balance across emotions.

    Topics come from the SAME seeded shuffle of the paper's 100 that the story grid draws from, so
    when the two counts coincide the neutral corpus spans exactly the topics the stories did. The
    paper gets this for free -- its appendix seeds both datasets from the same 100 topics -- while
    at our scale it has to be arranged, or the projection would remove directions belonging to
    topics no story ever saw.

    ``n_transcripts`` is a request target, not a guarantee: it fixes how many CALLS are issued
    (``n_transcripts // dialogues_per_call``), and how many transcripts come back depends on
    whether the model emits the separators it was asked for. The realised count is what the
    artifact records.
    """

    if n_transcripts < 1:
        raise ValueError("n_transcripts must be >= 1")
    if dialogues_per_call < 1:
        raise ValueError("dialogues_per_call must be >= 1")
    if n_transcripts % dialogues_per_call:
        raise ValueError(
            f"n_transcripts={n_transcripts} is not a multiple of "
            f"dialogues_per_call={dialogues_per_call}"
        )

    n_calls = n_transcripts // dialogues_per_call
    topics = _shared_topics(topics=recipe.topics, topics_per_label=n_calls, seed=seed)
    return tuple(
        StoryRequest(
            emotion=NEUTRAL_DIALOGUE_LABEL,
            topic=topic,
            prompt=recipe.neutral_dialogue_prompt_template.format(
                n_stories=dialogues_per_call, topic=topic
            ),
            story_index=call_index * dialogues_per_call,
        )
        for call_index, topic in enumerate(topics)
    )
