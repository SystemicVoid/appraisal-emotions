"""E0 — emotion-concept basis extraction and the G0 sensitivity gate (design §4 E0).

NEW module (no parent counterpart). Downscaled Sofroniew et al. 2026 recipe on the model that
carries the certified appraisal directions:

1. generate ``stories_per_emotion`` stories per §5 word, each asking for a character who
   experiences the emotion **without naming it** (``stimuli.emotion_stories``);
2. FILTER the stories that name the target anyway (or its naive morphological variants) and the
   ones too short to have a token in the capture window — both drop rates are recorded, because a
   high drop rate is itself a harness finding;
3. re-feed ONLY the story text (instruction stripped — otherwise every vector would share the
   template's activation) and average the residual stream over tokens from ``min_token`` (default
   50, the recipe's window) onward, at every block;
4. emotion-concept vector ``e_j`` = that emotion's per-block story mean minus the grand mean over
   the emotion words, at every block;
5. OPTIONALLY (``neutral_projection``, default off) the paper's confound-removal step: project
   out the top PCs of activations on an emotionally neutral dialogue corpus, enough to explain
   50% of their variance (``analysis.neutral_projection``). Off by default because the paper's own
   footnote 3 reports it as a denoiser whose qualitative findings hold without it — so it is an
   arm, and a run that uses it records the G0 table on both sides of it.

**Gate G0** (the manipulation check that licenses every later null): PC1 of the per-block
mean-centred ``{e_j}`` correlates with the §5 minted binary valence labels at
``|Spearman rho| >= g0_threshold``. The block is SELECTED by that criterion — no LLM judge, no
peeking at any appraisal direction. G0 failing means the emotion basis, not the hypothesis,
failed: the artifact records ``gate_verdict = "harness_inadequate"`` and every downstream null
inherits that cap (design §4 E0 diagnosticity clause).

**Freeze status of the story filter: BLIND** (``docs/agents/rails.md``, "observation before
imagination"). No story from this model has been read — GPU runs gate behind human approval and
none has been authorized — so the drop rule is frozen against outputs it has not seen. The
licensed compensation is TWO early-N checks, both before the capture pass, because they catch
different failures. The first-contact checkpoint reads the run's own first ``first_contact_n``
generations against the frozen rule and routes to ``harness_inadequate`` if the drop rate exceeds
``first_contact_max_drop_rate``; the sample is written out for a human to read. The yield check
(:class:`StoryYieldShortfall`) compares the stories each label actually came back with against
``stories_per_emotion`` — the batched Sofroniew arm's completion splitter can return fewer pieces
than the prompt asked for, and every piece it does return passes the length and lexical filters,
so the drop audit reports 0.00 on a sample half its designed size. Replace this paragraph with a
real reality-sample frequency table once one exists.

Licence: the emotion basis is a measurement instrument. Nothing here licenses any claim about
what the model feels; ``e_disappointed`` is the *concept vector for the word* "disappointed".
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.stats import spearmanr

from appraisal_emotions.activation.capture import (
    StoryCaptureBackend,
    capture_token_mean_states,
    decoder_layers,
)
from appraisal_emotions.analysis.neutral_projection import (
    DEFAULT_NEUTRAL_VARIANCE_TARGET,
    NeutralBasis,
    NeutralProjectionRecord,
    basis_arrays,
    basis_from_arrays,
    basis_sha256,
    fit_neutral_basis,
    project_out,
)
from appraisal_emotions.core.schema import ModelSpec, StrictModel
from appraisal_emotions.core.util import (
    EXTRACTION_SEED,
    ensure_parent,
    read_json,
    stable_hash,
    states_sha256,
    write_json,
)
from appraisal_emotions.stimuli.emotion_stories import (
    STORY_PROMPT_TEMPLATE,
    STORY_TOPICS,
    STYLE_CONTROL_LABEL,
    STYLE_CONTROL_PROMPT_TEMPLATE,
    EmotionWordSet,
    StoryRequest,
    build_story_grid,
)
from appraisal_emotions.stimuli.sofroniew_stories import (
    DEFAULT_SOFRONIEW_DATA_DIR,
    NEUTRAL_DIALOGUE_LABEL,
    build_neutral_dialogue_grid,
    build_sofroniew_story_grid,
    neutral_completion_splitter,
    read_sofroniew_recipe,
    split_completion,
)

__all__ = [
    "DEFAULT_G0_THRESHOLD",
    "DEFAULT_MIN_TOKEN",
    "EMOTION_VECTORS_CONTRACT_VERSION",
    "EmotionVectors",
    "EmotionVectorsMetadata",
    "FirstContactFailure",
    "G0BlockRow",
    "GeneratedStory",
    "StoryYieldShortfall",
    "emotion_vectors_path",
    "extract_emotion_vectors",
    "fit_neutral_projection",
    "generate_stories",
    "naive_variants",
    "read_emotion_vectors",
    "valence_oriented_pc_axes",
    "write_emotion_vectors",
]

EMOTION_VECTORS_CONTRACT_VERSION = "emotion_vectors/v1"
_VECTORS_SUFFIX = ".vectors.npz"
_VECTORS_ARRAY_NAME = "emotion_vectors"

# Sofroniew recipe: the residual stream is averaged from token 50 onward, past the story's
# opening scaffolding. Configurable because a downscaled run generates shorter stories.
DEFAULT_MIN_TOKEN = 50
# Design §4 E0: |Spearman rho| >= 0.6 between PC1 scores and the valence labels.
DEFAULT_G0_THRESHOLD = 0.6
DEFAULT_FALLBACK_BLOCKS = 4
GateVerdict = Literal["pass", "harness_inadequate"]

_STEM_SUFFIXES = ("ed", "ing", "ment", "ness", "ly", "es", "s", "d")
_GROWTH_SUFFIXES = ("s", "d", "ed", "ing", "ment", "ness", "ly")


# --------------------------------------------------------------------------------------
# Lexical filter (the vector-side leakage control; §6 threats table)
# --------------------------------------------------------------------------------------


def naive_variants(word: str) -> tuple[str, ...]:
    """The target word plus its naive morphological neighbourhood.

    Deliberately over-inclusive and deliberately dumb: strip one known suffix to get a stem, then
    grow the stem back with the same suffix inventory. It generates non-words
    (``sadd``, ``calmment``) which simply never match, and it will miss irregular relatives
    (``angry`` -> ``anger``). Over-dropping costs stories; under-dropping costs the control, so
    the asymmetry is intentional. The realised drop rate is recorded in the artifact.
    """

    base = word.strip().lower()
    if not base:
        raise ValueError("cannot build variants of an empty word")
    stems = {base}
    for suffix in _STEM_SUFFIXES:
        if base.endswith(suffix) and len(base) > len(suffix) + 2:
            stems.add(base[: -len(suffix)])
    variants: set[str] = set(stems)
    for stem in stems:
        variants.update(stem + suffix for suffix in _GROWTH_SUFFIXES)
        if stem.endswith("e"):
            variants.update({stem[:-1] + "ing", stem[:-1] + "ed"})
        if stem.endswith("y"):
            variants.update({stem[:-1] + "ily", stem[:-1] + "iness"})
    return tuple(sorted(variants))


def _target_pattern(word: str) -> re.Pattern[str]:
    joined = "|".join(re.escape(variant) for variant in naive_variants(word))
    return re.compile(rf"\b(?:{joined})\b", re.IGNORECASE)


def names_target(text: str, word: str) -> bool:
    """True when the story names the emotion it was told not to name."""

    return _target_pattern(word).search(text) is not None


# --------------------------------------------------------------------------------------
# Generation + first-contact checkpoint
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratedStory:
    """One generated story with its filter verdict (``drop_reason is None`` ⇒ kept)."""

    emotion: str
    topic: str
    story_index: int
    text: str
    token_count: int
    drop_reason: str | None

    @property
    def kept(self) -> bool:
        return self.drop_reason is None


def story_log_records(stories: tuple[GeneratedStory, ...]) -> list[dict[str, object]]:
    """The on-disk shape of ``stories.json`` — the ONLY definition of it."""

    return [
        {
            "emotion": story.emotion,
            "topic": story.topic,
            "story_index": story.story_index,
            "token_count": story.token_count,
            "drop_reason": story.drop_reason,
            "text": story.text,
        }
        for story in stories
    ]


def read_story_log(path: Path) -> tuple[GeneratedStory, ...]:
    """Read ``stories.json`` back, verdicts and order intact.

    A re-capture must see exactly the rows E0 captured — same texts, same order, same drop
    reasons — or the centring frame it rebuilds is not E0's. Round-tripping through the writer
    above is what makes that a property of one function rather than of two hand-matched dicts.
    """

    return tuple(
        GeneratedStory(
            emotion=str(row["emotion"]),
            topic=str(row["topic"]),
            story_index=int(row["story_index"]),
            text=str(row["text"]),
            token_count=int(row["token_count"]),
            drop_reason=None if row["drop_reason"] is None else str(row["drop_reason"]),
        )
        for row in read_json(path)
    )


class FirstContactFailure(RuntimeError):
    """The BLIND story filter dropped too much of its first-contact sample.

    Raised before full spend so the run records ``harness_inadequate`` rather than producing an
    emotion basis built from whatever survived a mis-frozen rule.
    """

    def __init__(self, *, drop_rate: float, threshold: float, sample: tuple[GeneratedStory, ...]):
        super().__init__(
            f"first-contact checkpoint failed: {drop_rate:.2f} of the first {len(sample)} stories "
            f"were dropped by the BLIND story filter (threshold {threshold:.2f}). Read the "
            "written sample, fix the prompt or the filter, then re-run — do NOT score this basis."
        )
        self.drop_rate = drop_rate
        self.threshold = threshold
        self.sample = sample


class StoryYieldShortfall(RuntimeError):
    """A label came back with fewer stories than the recipe asked for.

    This is the failure the BLIND ``<NEW STORY>`` freeze is actually exposed to, and it is not a
    filter drop: a completion the splitter cannot cut yields ONE piece where ``stories_per_call``
    were requested, so the run proceeds on a fraction of the configured sample with a drop rate of
    0.00 and nothing in the artifact contradicting ``stories_per_emotion``. Reproduced on the
    shipped projected smoke config: 2 stories per label configured, 1 realised, "85/85 stories
    kept ... drop rate 0.000" printed.

    Raised before the capture pass, so the refusal costs the generations already spent and not the
    forwards. The counts ride along for the operator's report.
    """

    def __init__(self, *, requested: int, realized: dict[str, int]):
        short = {label: count for label, count in sorted(realized.items()) if count < requested}
        super().__init__(
            f"{len(short)} of {len(realized)} labels came back with fewer than the configured "
            f"stories_per_emotion={requested} stories: {short}. The splitter returned fewer "
            "pieces than the prompt asked for, so this run would score a basis at an unrecorded "
            "fraction of its designed sample size while reporting a clean drop rate. Read the "
            "first-contact sample, fix the prompt or the splitter (or lower "
            "sofroniew_stories_per_call to what the model actually emits), then re-run."
        )
        self.requested = requested
        self.realized = realized
        self.short = short


def _yield_check(stories: tuple[GeneratedStory, ...], *, requested: int) -> dict[str, int]:
    """Realised stories per label vs the configured count — the one the drop audit cannot see."""

    realized = dict(Counter(story.emotion for story in stories))
    if any(count < requested for count in realized.values()):
        raise StoryYieldShortfall(requested=requested, realized=realized)
    return realized


# Labels that are not emotion words, so the "did it name the target?" check does not apply: the
# P5c style control was never told an emotion, and a neutral dialogue is a confound sample rather
# than a stimulus for anything.
_NON_EMOTION_LABELS = frozenset({STYLE_CONTROL_LABEL, NEUTRAL_DIALOGUE_LABEL})


def _classify(text: str, request: StoryRequest, *, min_token: int, token_count: int) -> str | None:
    if not text.strip():
        return "empty"
    if request.emotion not in _NON_EMOTION_LABELS and names_target(text, request.emotion):
        return "names_target"
    if token_count <= min_token:
        return "too_short"
    return None


def _single_story(text: str) -> tuple[str, ...]:
    """The project recipe's split: one completion is one story."""

    return (text,)


def generate_stories(
    backend: StoryCaptureBackend,
    requests: tuple[StoryRequest, ...],
    *,
    min_token: int,
    max_tokens: int,
    temperature: float,
    split: Callable[[str], tuple[str, ...]] = _single_story,
) -> tuple[GeneratedStory, ...]:
    """Generate and filter the stories each request asks for, in request order.

    ``split`` turns one completion into the stories it carries. The project recipe asks for one
    story per call and leaves this at :func:`_single_story`; the Sofroniew arm asks for several
    per call and passes
    :func:`~appraisal_emotions.stimuli.sofroniew_stories.split_completion`. A completion that
    yields nothing becomes one empty story, so the request still appears in the drop audit rather
    than vanishing from it.
    """

    stories: list[GeneratedStory] = []
    for request in requests:
        rendered = backend.render_prompt(request.prompt).rendered_prompt
        result = backend.generate_with_metadata(
            rendered, max_tokens=max_tokens, temperature=temperature
        )
        pieces = split(result.text.strip()) or ("",)
        for offset, piece in enumerate(pieces):
            text = piece.strip()
            token_count = len(backend.token_ids(text)) if text else 0
            stories.append(
                GeneratedStory(
                    emotion=request.emotion,
                    topic=request.topic,
                    story_index=request.story_index + offset,
                    text=text,
                    token_count=token_count,
                    drop_reason=_classify(
                        text, request, min_token=min_token, token_count=token_count
                    ),
                )
            )
    return tuple(stories)


def _first_contact_check(
    stories: tuple[GeneratedStory, ...], *, n: int, max_drop_rate: float
) -> tuple[int, float]:
    """Read the run's own first ``n`` stories against the frozen filter; refuse if it over-drops."""

    sample = stories[:n]
    if not sample:
        return 0, 0.0
    drop_rate = sum(1 for story in sample if not story.kept) / len(sample)
    if drop_rate > max_drop_rate:
        raise FirstContactFailure(
            drop_rate=drop_rate, threshold=max_drop_rate, sample=tuple(sample)
        )
    return len(sample), drop_rate


# --------------------------------------------------------------------------------------
# Vector assembly + PCA / G0
# --------------------------------------------------------------------------------------


def _emotion_means(
    states: np.ndarray, stories: tuple[GeneratedStory, ...], labels: tuple[str, ...]
) -> np.ndarray:
    """Per-label mean over that label's KEPT stories: ``(n_labels, n_blocks, hidden)``."""

    rows = [index for index, story in enumerate(stories) if story.kept]
    kept = [stories[index] for index in rows]
    means = np.zeros((len(labels), states.shape[1], states.shape[2]), dtype=np.float64)
    for label_index, label in enumerate(labels):
        selected = [i for i, story in enumerate(kept) if story.emotion == label]
        if not selected:
            raise ValueError(
                f"emotion {label!r} kept zero stories after the lexical/length filter; the basis "
                "is undefined. Raise stories_per_emotion or loosen the generation cap."
            )
        means[label_index] = states[np.asarray(selected)].mean(axis=0)
    return means


def valence_oriented_pc_axes(
    word_vectors: np.ndarray, valence: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """PCA of the emotion-vector set at one block, with PC1 oriented toward positive valence.

    ``word_vectors`` is ``(n_words, hidden)``. Returns ``(components (k, hidden), scores
    (n_words, k))``. PC1's sign is arbitrary out of an SVD, so it is flipped to correlate
    non-negatively with the valence labels — that fixes the sign of the "affect-concept valence"
    axis once, here, for G0 and for every downstream projection (E2 reads this same helper). PC2
    is NOT oriented: no arousal labels are committed, so its sign is reported as-is.
    """

    centered = word_vectors - word_vectors.mean(axis=0)
    components = np.linalg.svd(centered, full_matrices=False)[2]
    scores = centered @ components.T
    if float(np.dot(scores[:, 0], valence - valence.mean())) < 0.0:
        components = components.copy()
        components[0] = -components[0]
        scores = centered @ components.T
    return components, scores


class G0BlockRow(StrictModel):
    """One block's PC1-vs-valence correlation — the G0 selection criterion, per block."""

    block: int
    spearman_rho: float
    spearman_p: float
    n_words: int


def _g0_table(word_vectors: np.ndarray, valence: np.ndarray) -> tuple[G0BlockRow, ...]:
    rows: list[G0BlockRow] = []
    for block in range(word_vectors.shape[1]):
        _components, scores = valence_oriented_pc_axes(word_vectors[:, block, :], valence)
        result = spearmanr(scores[:, 0], valence)
        rho = float(np.nan_to_num(result.statistic, nan=0.0))
        p_value = float(np.nan_to_num(result.pvalue, nan=1.0))
        rows.append(
            G0BlockRow(block=block, spearman_rho=rho, spearman_p=p_value, n_words=int(len(valence)))
        )
    return tuple(rows)


# --------------------------------------------------------------------------------------
# Hash-bound artifact
# --------------------------------------------------------------------------------------


class EmotionVectorsMetadata(StrictModel):
    """Identity, recipe, drop accounting and the G0 gate record for one emotion basis."""

    artifact_contract_version: Literal["emotion_vectors/v1"] = EMOTION_VECTORS_CONTRACT_VERSION
    model_key: str
    backend: str
    model_id: str
    revision: str | None
    tokenizer_id: str
    dtype: str | None
    seed: int
    # The word list and the two prompt templates are the stimulus identity of this basis.
    words_file: str
    words_file_sha256: str
    stimulus_hash: str
    # Which E0 stimulus arm generated the stories: the §5 project recipe, or the Sofroniew et al.
    # 2026 appendix prompt (docs/design/sofroniew-recipe.md). Everything else is held identical,
    # so two bases differing only here are the prompt A/B.
    story_recipe: str = "project"
    recipe_manifest: dict[str, object] = {}
    # The paper's confound-removal step (analysis.neutral_projection). ``None`` means it was not
    # run, which is the default: footnote 3 of the paper calls the projection a denoiser and says
    # its qualitative findings hold without it, so enabling it is a deliberate arm, not a silent
    # change to the instrument E1-E3 already read.
    neutral_projection: NeutralProjectionRecord | None = None
    stories_per_emotion: int
    min_token: int
    max_tokens: int
    temperature: float
    # Row order of the vectors array: the §5 words, then the P5c style_control pseudo-emotion.
    vector_labels: tuple[str, ...]
    categories: tuple[str, ...]
    valence_labels: tuple[int, ...]
    n_word_rows: int  # rows [0:n_word_rows) are emotion words; the rest are pseudo-emotions
    n_requested: int
    n_kept: int
    drop_rate: float
    drop_counts_by_reason: dict[str, int]
    # Realised stories per label BEFORE the filter, against the configured `stories_per_emotion`
    # above. The two are equal on a run whose splitter returned what the prompt asked for, and the
    # run refuses when they are not (:class:`StoryYieldShortfall`) — so this pair is the record
    # that the A/B compared at matched sample size, not just at matched configuration.
    generated_by_label: dict[str, int]
    kept_by_label: dict[str, int]
    first_contact_n: int
    first_contact_drop_rate: float
    first_contact_max_drop_rate: float
    filter_freeze_status: Literal["BLIND", "reality-sampled"]
    g0_threshold: float
    g0_table: tuple[G0BlockRow, ...]
    # The same table computed on the UNPROJECTED vectors, present only when the neutral projection
    # ran. Recorded unconditionally in that case, in both directions: whether the projection
    # helped the gate or hurt it is a fact about the run, not something to look up only when it
    # flatters the arm (docs/agents/rails.md, symmetric-amendment).
    g0_table_before_projection: tuple[G0BlockRow, ...] = ()
    selected_block: int
    g0_abs_rho: float
    g0_spearman_rho: float
    g0_spearman_p: float
    gate_verdict: GateVerdict
    gate_note: str
    n_blocks: int
    hidden_size: int
    vectors_dtype: str
    vectors_sha256: str


@dataclass(frozen=True)
class EmotionVectors:
    """Validated per-block emotion-concept vectors plus binding metadata (frozen read-only).

    ``neutral_basis`` is the confound subspace this run subtracted, present exactly when
    ``metadata.neutral_projection`` is. It rides with the artifact rather than being recoverable
    only from a re-fit because two readers need it: a reviewer auditing WHICH directions were
    removed, and P1, whose re-capture has to apply the same subtraction to its per-story states or
    its faithfulness identity against these vectors cannot hold.
    """

    metadata: EmotionVectorsMetadata
    vectors: np.ndarray  # (n_labels, n_blocks, hidden)
    neutral_basis: NeutralBasis | None = None

    def __post_init__(self) -> None:
        array = np.array(self.vectors, copy=True)
        meta = self.metadata
        self._validate_neutral_basis()
        if array.ndim != 3:
            raise ValueError("emotion vectors must be (n_labels, n_blocks, hidden)")
        labels, blocks, hidden = (int(dim) for dim in array.shape)
        if labels != len(meta.vector_labels):
            raise ValueError("vector row count does not match metadata.vector_labels")
        if len(meta.categories) != labels or len(meta.valence_labels) != labels:
            raise ValueError("metadata categories/valence_labels must align with vector_labels")
        if not 0 < meta.n_word_rows <= labels:
            raise ValueError("metadata.n_word_rows out of range")
        if blocks != meta.n_blocks or hidden != meta.hidden_size:
            raise ValueError("emotion vectors block/hidden dims do not match metadata")
        if str(array.dtype) != meta.vectors_dtype:
            raise ValueError("emotion vectors dtype does not match metadata")
        if states_sha256(array) != meta.vectors_sha256:
            raise ValueError("emotion vectors hash does not match metadata")
        if not np.all(np.isfinite(array)):
            raise ValueError("emotion vectors must be finite")
        if not 0 <= meta.selected_block < meta.n_blocks:
            raise ValueError("metadata.selected_block out of range for n_blocks")
        array.flags.writeable = False
        object.__setattr__(self, "vectors", array)

    def _validate_neutral_basis(self) -> None:
        """Present iff the run projected, shaped as its record says, and digest-bound to it."""

        record = self.metadata.neutral_projection
        if record is None:
            if self.neutral_basis is not None:
                raise ValueError(
                    "a confound basis is attached but metadata.neutral_projection is None; the "
                    "artifact would claim an unprojected basis while carrying a projection"
                )
            return
        if self.neutral_basis is None:
            raise ValueError(
                "metadata.neutral_projection records a projection but no confound basis is "
                "attached; the removed directions would be unauditable and P1 could not "
                "reproduce them"
            )
        if self.neutral_basis.rows != record.blocks:
            raise ValueError("the confound basis rows disagree with metadata.neutral_projection")
        if basis_sha256(self.neutral_basis) != record.components_sha256:
            raise ValueError("confound basis hash does not match metadata")

    @property
    def word_labels(self) -> tuple[str, ...]:
        return tuple(self.metadata.vector_labels[: self.metadata.n_word_rows])

    @property
    def word_vectors(self) -> np.ndarray:
        """The emotion-word rows only — the PCA basis (pseudo-emotions are never basis members)."""

        return self.vectors[: self.metadata.n_word_rows]

    @property
    def word_valence(self) -> np.ndarray:
        return np.asarray(self.metadata.valence_labels[: self.metadata.n_word_rows], dtype=float)

    def row(self, label: str) -> np.ndarray:
        return self.vectors[self.metadata.vector_labels.index(label)]


def emotion_vectors_path(metadata_path: Path) -> Path:
    if metadata_path.suffix != ".json":
        raise ValueError("emotion-vectors metadata path must end in .json")
    return metadata_path.with_name(f"{metadata_path.stem}{_VECTORS_SUFFIX}")


def write_emotion_vectors(artifact: EmotionVectors, metadata_path: Path) -> tuple[Path, Path]:
    """Write the metadata JSON and the sibling vectors npz; return both paths.

    The npz carries the confound basis beside the vectors when the run projected, so the removed
    directions travel with the thing they were removed from.
    """

    vectors_path = emotion_vectors_path(metadata_path)
    ensure_parent(metadata_path)
    payload = {_VECTORS_ARRAY_NAME: np.asarray(artifact.vectors)}
    if artifact.neutral_basis is not None:
        payload |= basis_arrays(artifact.neutral_basis)
    np.savez(vectors_path, **payload)
    write_json(metadata_path, artifact.metadata)
    return metadata_path, vectors_path


def read_emotion_vectors(metadata_path: Path) -> EmotionVectors:
    """Read and revalidate an emotion-vectors artifact (metadata + payload hash binding)."""

    metadata = EmotionVectorsMetadata.model_validate(read_json(metadata_path))
    with np.load(emotion_vectors_path(metadata_path), allow_pickle=False) as bundle:
        array = np.array(bundle[_VECTORS_ARRAY_NAME])
        basis = (
            None
            if metadata.neutral_projection is None
            else basis_from_arrays(
                metadata.neutral_projection.blocks,
                {name: np.array(bundle[name]) for name in bundle.files},
            )
        )
    return EmotionVectors(metadata=metadata, vectors=array, neutral_basis=basis)


# --------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------


def _gate(table: tuple[G0BlockRow, ...], threshold: float) -> tuple[int, float, GateVerdict, str]:
    best = max(table, key=lambda row: abs(row.spearman_rho))
    abs_rho = abs(best.spearman_rho)
    if abs_rho >= threshold:
        return (
            best.block,
            abs_rho,
            "pass",
            (
                f"G0 PASS: |rho|={abs_rho:.3f} >= {threshold} at block {best.block}; the basis "
                "detects valence structure, which is what licenses E1's nulls to mean anything."
            ),
        )
    return (
        best.block,
        abs_rho,
        "harness_inadequate",
        (
            f"G0 FAIL: best |rho|={abs_rho:.3f} < {threshold} (block {best.block}). The emotion "
            "basis, not the inheritance hypothesis, failed — every downstream null records "
            "harness_inadequate and the claim stays OPEN (design §4 E0)."
        ),
    )


StoryRecipeName = Literal["project", "sofroniew"]


@dataclass(frozen=True)
class StimulusPlan:
    """One recipe's generation grid, its completion splitter, and its provenance rows."""

    recipe: str
    requests: tuple[StoryRequest, ...]
    split: Callable[[str], tuple[str, ...]]
    stimulus_hash: str
    stories_per_emotion: int
    recipe_manifest: dict[str, object]


def build_stimulus_plan(
    labels: tuple[str, ...],
    *,
    story_recipe: StoryRecipeName,
    stories_per_emotion: int,
    seed: int,
    sofroniew_stories_per_call: int = 2,
    sofroniew_data_dir: Path = DEFAULT_SOFRONIEW_DATA_DIR,
) -> StimulusPlan:
    """Resolve the E0 stimulus arm.

    ``project`` is the §5 recipe (three-sentence prompt, one paragraph per call, 25 neutral
    settings). ``sofroniew`` is the paper's own appendix prompt over the paper's 100 topics,
    batched ``sofroniew_stories_per_call`` stories per completion — the A/B whose whole point is
    that everything downstream (word set, capture window, centring, G0) is held identical, so a
    difference in the result is a difference the prompt made.

    Under ``sofroniew``, ``stories_per_emotion`` still names the target stories per label; the
    number of CALLS per label is ``stories_per_emotion // sofroniew_stories_per_call``, and it
    must divide evenly or the two arms would not compare at matched sample size.
    """

    if story_recipe == "project":
        return StimulusPlan(
            recipe="project",
            requests=build_story_grid(labels, stories_per_emotion=stories_per_emotion, seed=seed),
            split=_single_story,
            stimulus_hash=stable_hash(
                [STORY_PROMPT_TEMPLATE, STYLE_CONTROL_PROMPT_TEMPLATE, list(STORY_TOPICS)]
            ),
            stories_per_emotion=stories_per_emotion,
            recipe_manifest={"recipe": "project", "n_topics": len(STORY_TOPICS)},
        )
    if story_recipe != "sofroniew":
        raise ValueError(f"unknown story_recipe {story_recipe!r}; expected project or sofroniew")

    if sofroniew_stories_per_call < 1:
        raise ValueError("sofroniew_stories_per_call must be >= 1")
    if stories_per_emotion % sofroniew_stories_per_call:
        raise ValueError(
            f"stories_per_emotion={stories_per_emotion} is not a multiple of "
            f"sofroniew_stories_per_call={sofroniew_stories_per_call}; the arms would compare at "
            "different sample sizes"
        )
    recipe = read_sofroniew_recipe(sofroniew_data_dir)
    topics_per_label = stories_per_emotion // sofroniew_stories_per_call
    manifest = recipe.manifest()
    manifest |= {
        "recipe": "sofroniew",
        "topics_per_label": topics_per_label,
        "stories_per_call": sofroniew_stories_per_call,
    }
    return StimulusPlan(
        recipe="sofroniew",
        requests=build_sofroniew_story_grid(
            labels,
            recipe=recipe,
            topics_per_label=topics_per_label,
            stories_per_call=sofroniew_stories_per_call,
            seed=seed,
        ),
        split=split_completion,
        # Hashes the loaded paper text, not a copy of it: the arm's stimulus identity is the
        # arXiv source it was extracted from plus how far it was downscaled.
        stimulus_hash=stable_hash(
            [
                recipe.story_prompt_template,
                recipe.style_control_prompt_template,
                list(recipe.topics),
                topics_per_label,
                sofroniew_stories_per_call,
            ]
        ),
        stories_per_emotion=stories_per_emotion,
        recipe_manifest=manifest,
    )


def fit_neutral_projection(
    backend: StoryCaptureBackend,
    *,
    layers: tuple[int, ...],
    n_transcripts: int,
    dialogues_per_call: int,
    variance_target: float,
    seed: int,
    min_token: int,
    max_tokens: int,
    temperature: float,
    data_dir: Path = DEFAULT_SOFRONIEW_DATA_DIR,
) -> tuple[NeutralBasis, NeutralProjectionRecord, tuple[GeneratedStory, ...]]:
    """Generate the neutral corpus, capture it, and fit the confound basis the paper projects out.

    Returns the basis, the record for the artifact, and the transcripts themselves so the caller
    can write them into the run's story log — an unread confound corpus is exactly the kind of
    thing a first-contact checkpoint exists to catch, and it costs nothing to keep.

    The lexical/length filter is the same one the stories go through, minus the "did it name the
    target" clause (there is no target). The length clause matters here for the same reason it
    does there: a transcript with no token past ``min_token`` contributes nothing but noise to the
    covariance.
    """

    recipe = read_sofroniew_recipe(data_dir)
    requests = build_neutral_dialogue_grid(
        recipe=recipe,
        n_transcripts=n_transcripts,
        dialogues_per_call=dialogues_per_call,
        seed=seed,
    )
    transcripts = generate_stories(
        backend,
        requests,
        min_token=min_token,
        max_tokens=max_tokens,
        temperature=temperature,
        split=neutral_completion_splitter(recipe.speaker_substitutions),
    )
    kept = tuple(transcript for transcript in transcripts if transcript.kept)
    if len(kept) < 2:
        raise ValueError(
            f"only {len(kept)} of {len(transcripts)} neutral transcripts survived the filter; a "
            "confound basis needs at least 2. Raise neutral_dialogues or max_tokens, or lower "
            "min_token."
        )
    states = capture_token_mean_states(
        backend, [transcript.text for transcript in kept], layers=layers, min_token=min_token
    )
    basis = fit_neutral_basis(states, variance_target=variance_target)
    drop_counts = Counter(t.drop_reason for t in transcripts if t.drop_reason is not None)
    record = NeutralProjectionRecord(
        variance_target=variance_target,
        pooling=f"mean over transcript tokens from token {min_token} onward, per block",
        n_requested=len(transcripts),
        n_kept=len(kept),
        drop_rate=1.0 - len(kept) / len(transcripts),
        drop_counts_by_reason=dict(sorted(drop_counts.items())),
        dialogues_per_call=dialogues_per_call,
        stimulus_hash=stable_hash(
            [
                recipe.neutral_dialogue_prompt_template,
                [list(pair) for pair in recipe.speaker_substitutions],
                [request.topic for request in requests],
                dialogues_per_call,
            ]
        ),
        blocks=basis.rows,
        total_components=basis.total_components,
        max_components_in_a_block=max(row.n_components for row in basis.rows),
        components_sha256=basis_sha256(basis),
    )
    return basis, record, transcripts


def extract_emotion_vectors(
    backend: StoryCaptureBackend,
    word_set: EmotionWordSet,
    *,
    spec: ModelSpec,
    stories_per_emotion: int,
    story_recipe: StoryRecipeName = "project",
    sofroniew_stories_per_call: int = 2,
    sofroniew_data_dir: Path = DEFAULT_SOFRONIEW_DATA_DIR,
    neutral_projection: bool = False,
    neutral_dialogues: int = 40,
    neutral_dialogues_per_call: int = 4,
    neutral_variance_target: float = DEFAULT_NEUTRAL_VARIANCE_TARGET,
    seed: int = EXTRACTION_SEED,
    min_token: int = DEFAULT_MIN_TOKEN,
    max_tokens: int = 320,
    temperature: float = 1.0,
    g0_threshold: float = DEFAULT_G0_THRESHOLD,
    fallback_blocks: int = DEFAULT_FALLBACK_BLOCKS,
    first_contact_n: int = 10,
    first_contact_max_drop_rate: float = 0.5,
) -> tuple[EmotionVectors, tuple[GeneratedStory, ...], tuple[GeneratedStory, ...]]:
    """Run E0 end to end: generate → filter → capture → centre → [project] → PCA → G0 gate.

    Returns the hash-bound artifact and every generated story (kept and dropped) so the caller
    can write the first-contact sample and the drop audit. Raises :class:`FirstContactFailure`
    before the capture pass when the BLIND filter over-drops its first-contact sample.

    ``neutral_projection`` adds the paper's confound-removal step: generate ``neutral_dialogues``
    emotionally neutral Human/Assistant transcripts from the paper's own prompt, fit the top PCs
    explaining ``neutral_variance_target`` of their activation variance per block, and project
    those out of the centred vectors before G0
    (:mod:`appraisal_emotions.analysis.neutral_projection`). Default OFF: the paper's footnote 3
    reports the projection as a denoiser whose qualitative findings hold without it, and turning
    it on silently would change the instrument E1-E3 have already been read against.

    The third return value is the neutral corpus (empty when the step is off). It is returned
    SEPARATELY from the story log rather than appended to it, because the story log is P1's input
    and a neutral transcript is not a story of any emotion — but it still gets written out, so the
    corpus behind a projection is readable rather than a number in a metadata block.
    """

    labels = word_set.labels
    plan = build_stimulus_plan(
        labels,
        story_recipe=story_recipe,
        stories_per_emotion=stories_per_emotion,
        seed=seed,
        sofroniew_stories_per_call=sofroniew_stories_per_call,
        sofroniew_data_dir=sofroniew_data_dir,
    )
    stories = generate_stories(
        backend,
        plan.requests,
        min_token=min_token,
        max_tokens=max_tokens,
        temperature=temperature,
        split=plan.split,
    )
    contact_n, contact_rate = _first_contact_check(
        stories, n=first_contact_n, max_drop_rate=first_contact_max_drop_rate
    )
    # Before the capture pass, and separate from the drop audit: the filter can only report on
    # pieces that exist, so a splitter that returned half of them looks like a clean run to it.
    generated_by_label = _yield_check(stories, requested=stories_per_emotion)

    kept = tuple(story for story in stories if story.kept)
    if not kept:
        raise ValueError("every generated story was dropped; no emotion basis can be built")
    layers = decoder_layers(backend, fallback_blocks=fallback_blocks)
    states = capture_token_mean_states(
        backend, [story.text for story in kept], layers=layers, min_token=min_token
    )

    row_labels = (*labels, STYLE_CONTROL_LABEL)
    means = _emotion_means(states, stories, row_labels)
    # Grand mean over the EMOTION WORDS only (the recipe's centring set); the style control is
    # centred by the same vector so it lives in the same frame without shifting it.
    grand_mean = means[: len(labels)].mean(axis=0)
    vectors = np.ascontiguousarray(means - grand_mean, dtype=np.float64)

    valence = np.asarray([word_set.valence_by_word[label] for label in labels], dtype=float)
    neutral_transcripts: tuple[GeneratedStory, ...] = ()
    projection_record: NeutralProjectionRecord | None = None
    neutral_basis: NeutralBasis | None = None
    table_before: tuple[G0BlockRow, ...] = ()
    if neutral_projection:
        # After centring, before the gate: projection is linear, so word rows that summed to zero
        # still do, and G0 has to score the vectors the run keeps.
        table_before = _g0_table(vectors[: len(labels)], valence)
        neutral_basis, projection_record, neutral_transcripts = fit_neutral_projection(
            backend,
            layers=layers,
            n_transcripts=neutral_dialogues,
            dialogues_per_call=neutral_dialogues_per_call,
            variance_target=neutral_variance_target,
            seed=seed,
            min_token=min_token,
            max_tokens=max_tokens,
            temperature=temperature,
            data_dir=sofroniew_data_dir,
        )
        vectors = project_out(vectors, neutral_basis)

    table = _g0_table(vectors[: len(labels)], valence)
    selected_block, abs_rho, verdict, note = _gate(table, g0_threshold)
    best_row = table[selected_block]

    drop_counts = Counter(story.drop_reason for story in stories if story.drop_reason is not None)
    metadata = EmotionVectorsMetadata(
        model_key=spec.key,
        backend=spec.backend,
        model_id=spec.model_id,
        revision=spec.revision,
        tokenizer_id=spec.tokenizer_id or spec.model_id,
        dtype=spec.dtype,
        seed=seed,
        words_file=str(word_set.source_path),
        words_file_sha256=word_set.source_sha256,
        stimulus_hash=plan.stimulus_hash,
        story_recipe=plan.recipe,
        recipe_manifest=plan.recipe_manifest,
        neutral_projection=projection_record,
        stories_per_emotion=stories_per_emotion,
        min_token=min_token,
        max_tokens=max_tokens,
        temperature=temperature,
        vector_labels=row_labels,
        categories=(*(word.category for word in word_set.words), STYLE_CONTROL_LABEL),
        valence_labels=(*(word.valence for word in word_set.words), 0),
        n_word_rows=len(labels),
        n_requested=len(stories),
        n_kept=len(kept),
        drop_rate=1.0 - len(kept) / len(stories),
        drop_counts_by_reason=dict(sorted(drop_counts.items())),
        generated_by_label=dict(sorted(generated_by_label.items())),
        kept_by_label=dict(Counter(story.emotion for story in kept)),
        first_contact_n=contact_n,
        first_contact_drop_rate=contact_rate,
        first_contact_max_drop_rate=first_contact_max_drop_rate,
        filter_freeze_status="BLIND",
        g0_threshold=g0_threshold,
        g0_table=table,
        g0_table_before_projection=table_before,
        selected_block=selected_block,
        g0_abs_rho=abs_rho,
        g0_spearman_rho=best_row.spearman_rho,
        g0_spearman_p=best_row.spearman_p,
        gate_verdict=verdict,
        gate_note=note,
        n_blocks=int(vectors.shape[1]),
        hidden_size=int(vectors.shape[2]),
        vectors_dtype=str(vectors.dtype),
        vectors_sha256=states_sha256(vectors),
    )
    artifact = EmotionVectors(metadata=metadata, vectors=vectors, neutral_basis=neutral_basis)
    return artifact, stories, neutral_transcripts
