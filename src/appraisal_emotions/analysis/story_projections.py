"""Per-story projections onto the fixed direction set — E0's missing within-word error term.

E0 keeps only the MEAN of a word's ~12 stories, so every E1 readout is one number per word with
no within-word variance, no reliability, and therefore no way to tell a real absence from an
instrument too coarse to resolve the effect. That is the gap the E1 null diagnosis measured
(`scripts/e1_null_diagnosis.py`): the family contrast came in at roughly half the smallest effect
the test could detect, and nothing in the artifact said which half of the noise was fixable.

This re-captures the SAME stories E0 already generated and stored — no generation, no new
stimulus, no new seed — and keeps, per story and per block, the projection onto each direction
the analysis actually reads, plus the story's norm. Everything E1 computes is a function of those
two numbers, so ~2 MB replaces the 5 GB of residual states that were never retainable. That is
the second thing this fixes: projections sync back from a rented instance where states cannot.

**The faithfulness gate.** Centring is by the same grand mean E0 used, so the per-story rows
average to E0's own word vectors exactly:

    mean_i (x_ij - g) = mean_i x_ij - g = e_j

and therefore ``mean_i (y_ij . d) == e_j . d`` for every direction, word and block — an exact
linear identity. It is checked at write time against the shipped emotion-vector artifact and
recorded in the metadata; a re-capture that read the wrong window, ordered the stories
differently, or landed on a different model breaks it by orders of magnitude, so the artifact
cannot be built from a capture that is not the same capture.

**Why the within-word Gram is stored too.** Projections and norms span every *story-level* linear
readout, but not the *word-level* one: a leave-one-story-out or split-half word cosine needs
``e_j^(half) . x_ij``, an inner product between a subset mean and a single story, which no
per-story scalar can reconstruct. Storing the same-word Gram of the centred states (k x k per
label per block, ~8 MB) makes those exact instead of delta-method approximations — and reliability
is the entire point of P1, so approximating it would defeat the capture. Cross-word inner products
are NOT stored: they are 500x larger and nothing reads them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from appraisal_emotions.activation.capture import (
    StoryCaptureBackend,
    capture_token_mean_states,
    decoder_layers,
)
from appraisal_emotions.analysis.emotion_vectors import (
    DEFAULT_FALLBACK_BLOCKS,
    DEFAULT_MIN_TOKEN,
    EmotionVectors,
    GeneratedStory,
    valence_oriented_pc_axes,
)
from appraisal_emotions.analysis.reveal_rpe import DIRECTION_FAMILIES, RevealRpeDirections
from appraisal_emotions.core.schema import ModelSpec, StrictModel
from appraisal_emotions.core.util import ensure_parent, read_json, states_sha256, write_json

STORY_PROJECTIONS_CONTRACT_VERSION = "story_projections/v1"

# The two PC axes E1 reads (`block_geometry` uses `components[:2]`) appended to the three certified
# reveal directions. Order is the column order of the projections array and never changes without
# a contract-version bump.
PROJECTION_DIRECTIONS: tuple[str, ...] = (*DIRECTION_FAMILIES, "pc1", "pc2")

_PROJECTIONS_SUFFIX = ".projections.npz"
_PROJECTIONS_ARRAY_NAME = "projections"
_NORMS_ARRAY_NAME = "norms"
_GRAMS_ARRAY_NAME = "word_grams"

# Padding for the ragged story counts (11 or 12 per label). NaN, not zero: reading past a label's
# count is a bug, and a NaN propagates into whatever consumed it instead of quietly biasing a mean.
_GRAM_PAD = np.nan

# Sited on measured separation, not asserted (`tests/test_story_projections.py`): on the fake
# backend an aligned re-capture lands at 2.9e-15, while the SUBTLEST misalignment worth catching —
# a one-token shift of the capture window — reads 0.045, a two-story label swap 0.68 and a
# re-generated story set 3.5. 1e-2 sits ~4.5x below the smallest defect and twelve orders above
# exact arithmetic. What that measurement cannot cover is bf16 kernel drift from capturing on
# different silicon than E0 used; hence the quarantine below rather than a looser threshold.
GATE_MAX_RELATIVE_DEVIATION = 1e-2


class StoryProjectionGateFailure(RuntimeError):
    """The re-capture does not reproduce E0's word vectors — carries the artifact for quarantine.

    Raised instead of returning, so no caller can use a capture that is not a decomposition of the
    published basis. The artifact rides along because the alternative is discarding a GPU capture
    over a threshold that may itself be wrong: the operator writes it to a quarantine path, reads
    the recorded deviation, and decides. Mirrors E0's :class:`FirstContactFailure`.
    """

    def __init__(self, artifact: StoryProjections, message: str):
        super().__init__(message)
        self.artifact = artifact


class StoryProjectionsMetadata(StrictModel):
    """Identity of the re-capture, the row order, and the faithfulness gate against E0."""

    artifact_contract_version: Literal["story_projections/v1"] = STORY_PROJECTIONS_CONTRACT_VERSION
    model_key: str
    backend: str
    model_id: str
    revision: str | None
    tokenizer_id: str
    dtype: str | None
    min_token: int
    # What this capture is bound to: it is only interpretable against the artifacts whose
    # directions and word vectors define the projection targets and the centring.
    emotion_vectors_sha256: str
    emotion_vectors_selected_block: int
    directions_sha256: str
    directions_selected_block: int
    words_file_sha256: str
    # Row order of the projections array.
    story_labels: tuple[str, ...]
    story_topics: tuple[str, ...]
    story_indices: tuple[int, ...]
    n_stories: int
    n_blocks: int
    hidden_size: int
    direction_names: tuple[str, ...]
    stories_per_label: dict[str, int]
    # Row order of the Gram array's first axis, and the padded extent of its last two.
    gram_labels: tuple[str, ...]
    gram_max_stories_per_label: int
    # The gate: max |mean_i(y_ij . d) - e_j . d| over words, blocks and directions, divided by the
    # across-word spread of e_j . d at that block and direction.
    gate_max_relative_deviation: float
    gate_max_relative_deviation_threshold: float
    gate_word_mean_correlation: float
    # `harness_inadequate` marks a quarantined capture: written so the GPU spend is not lost, and
    # refused by every reader of the analysis path.
    gate_verdict: Literal["pass", "harness_inadequate"]
    projections_dtype: str
    projections_sha256: str


@dataclass(frozen=True)
class StoryProjections:
    """Validated per-story projections plus binding metadata (frozen read-only)."""

    metadata: StoryProjectionsMetadata
    projections: np.ndarray  # (n_stories, n_blocks, n_directions)
    norms: np.ndarray  # (n_stories, n_blocks) — ||y_ij||, so cosine = projection / norm
    word_grams: np.ndarray  # (n_labels, n_blocks, k_max, k_max) — y_ij . y_mj, NaN-padded

    def __post_init__(self) -> None:
        meta = self.metadata
        projections = np.array(self.projections, copy=True)
        norms = np.array(self.norms, copy=True)
        if projections.ndim != 3:
            raise ValueError("projections must be (n_stories, n_blocks, n_directions)")
        stories, blocks, directions = (int(dim) for dim in projections.shape)
        if stories != meta.n_stories or blocks != meta.n_blocks:
            raise ValueError("projections shape does not match metadata")
        if directions != len(meta.direction_names):
            raise ValueError("projection columns do not match metadata.direction_names")
        if len(meta.story_labels) != stories or len(meta.story_topics) != stories:
            raise ValueError("story metadata rows must align with the projections array")
        if len(meta.story_indices) != stories:
            raise ValueError("story metadata rows must align with the projections array")
        if norms.shape != (stories, blocks):
            raise ValueError("norms must be (n_stories, n_blocks)")
        if str(projections.dtype) != meta.projections_dtype:
            raise ValueError("projections dtype does not match metadata")
        if states_sha256(projections) != meta.projections_sha256:
            raise ValueError("projections hash does not match metadata")
        if not np.all(np.isfinite(projections)) or not np.all(np.isfinite(norms)):
            raise ValueError("projections and norms must be finite")
        if not np.all(norms > 0.0):
            raise ValueError("a zero-norm story state makes its cosine undefined")
        grams = np.array(self.word_grams, copy=True)
        k_max = meta.gram_max_stories_per_label
        if grams.shape != (len(meta.gram_labels), blocks, k_max, k_max):
            raise ValueError("word_grams must be (n_labels, n_blocks, k_max, k_max)")
        self._validate_grams(grams, norms)
        projections.flags.writeable = False
        norms.flags.writeable = False
        grams.flags.writeable = False
        object.__setattr__(self, "projections", projections)
        object.__setattr__(self, "norms", norms)
        object.__setattr__(self, "word_grams", grams)

    def _validate_grams(self, grams: np.ndarray, norms: np.ndarray) -> None:
        """Every in-range cell finite and symmetric, every out-of-range cell padding.

        The diagonal is checked against ``norms`` because the two arrays store the same quantity
        by two routes; a disagreement means one of them was written from a different capture, and
        an artifact whose reliability estimates and whose cosines disagree is worse than none.
        """

        meta = self.metadata
        for label_index, label in enumerate(meta.gram_labels):
            rows = self.rows_for(label)
            count = rows.size
            if count != meta.stories_per_label[label]:
                raise ValueError(f"{label}: gram count disagrees with stories_per_label")
            block = grams[label_index, :, :count, :count]
            if not np.all(np.isfinite(block)):
                raise ValueError(f"{label}: within-word Gram has a non-finite entry")
            if not np.allclose(block, np.swapaxes(block, -1, -2), rtol=0.0, atol=1e-9):
                raise ValueError(f"{label}: within-word Gram is not symmetric")
            padding = grams[label_index].copy()
            padding[:, :count, :count] = _GRAM_PAD
            if not np.all(np.isnan(padding)):
                raise ValueError(f"{label}: Gram cells past the story count must be NaN padding")
            diagonal = np.einsum("bii->bi", grams[label_index, :, :count, :count])
            if not np.allclose(diagonal, norms[rows].T ** 2, rtol=1e-9, atol=0.0):
                raise ValueError(f"{label}: Gram diagonal disagrees with the stored norms")

    def cosines(self, direction: str) -> np.ndarray:
        """``(n_stories, n_blocks)`` cosine of every story with ``direction`` — E1's readout."""

        return self.projections[:, :, self.metadata.direction_names.index(direction)] / self.norms

    def rows_for(self, label: str) -> np.ndarray:
        """Row indices of ``label``'s stories, in capture order — also the Gram's k-axis order."""

        return np.asarray(
            [i for i, story in enumerate(self.metadata.story_labels) if story == label], dtype=int
        )

    def gram_for(self, label: str) -> np.ndarray:
        """``(n_blocks, k, k)`` inner products of ``label``'s centred story states, unpadded."""

        count = self.metadata.stories_per_label[label]
        return self.word_grams[self.metadata.gram_labels.index(label), :, :count, :count]


def story_projections_path(metadata_path: Path) -> Path:
    if metadata_path.suffix != ".json":
        raise ValueError("story-projections metadata path must end in .json")
    return metadata_path.with_name(f"{metadata_path.stem}{_PROJECTIONS_SUFFIX}")


def write_story_projections(artifact: StoryProjections, metadata_path: Path) -> tuple[Path, Path]:
    payload_path = story_projections_path(metadata_path)
    ensure_parent(metadata_path)
    np.savez(
        payload_path,
        **{
            _PROJECTIONS_ARRAY_NAME: np.asarray(artifact.projections),
            _NORMS_ARRAY_NAME: np.asarray(artifact.norms),
            _GRAMS_ARRAY_NAME: np.asarray(artifact.word_grams),
        },
    )
    write_json(metadata_path, artifact.metadata)
    return metadata_path, payload_path


def read_story_projections(metadata_path: Path) -> StoryProjections:
    metadata = StoryProjectionsMetadata.model_validate(read_json(metadata_path))
    with np.load(story_projections_path(metadata_path), allow_pickle=False) as bundle:
        projections = np.array(bundle[_PROJECTIONS_ARRAY_NAME])
        norms = np.array(bundle[_NORMS_ARRAY_NAME])
        grams = np.array(bundle[_GRAMS_ARRAY_NAME])
    return StoryProjections(
        metadata=metadata, projections=projections, norms=norms, word_grams=grams
    )


def direction_matrix(
    emotion: EmotionVectors, directions: RevealRpeDirections, block: int
) -> np.ndarray:
    """``(n_directions, hidden)`` UNIT rows for one block, in ``PROJECTION_DIRECTIONS`` order.

    The reveal directions come from the certified artifact; the PC axes are recomputed from the
    emotion basis by the same helper `block_geometry` calls, so a projection here and a cosine in
    `map_geometry` are the same number by construction rather than by coincidence.
    """

    reveal = directions.directions[:, block, :]
    components, _scores = valence_oriented_pc_axes(
        emotion.word_vectors[:, block, :], emotion.word_valence
    )
    stacked = np.vstack([reveal, components[:2]])
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    if not np.all(norms > 0.0):
        raise ValueError(f"block {block} has a zero-norm projection direction")
    return stacked / norms


def _gate_against_emotion_basis(
    projections: np.ndarray,
    story_labels: tuple[str, ...],
    emotion: EmotionVectors,
    unit_directions: np.ndarray,
) -> tuple[float, float]:
    """Max relative deviation of the re-capture's word means from E0's, and their correlation."""

    observed, expected = [], []
    for label_index, label in enumerate(emotion.metadata.vector_labels):
        rows = np.asarray([i for i, story in enumerate(story_labels) if story == label])
        if rows.size == 0:
            continue
        observed.append(projections[rows].mean(axis=0))
        expected.append(np.einsum("bh,bdh->bd", emotion.vectors[label_index], unit_directions))
    observed_array, expected_array = np.asarray(observed), np.asarray(expected)
    # Scale each (block, direction) cell by the across-word spread of the expected value there:
    # an absolute tolerance would be meaningless across blocks whose cosine scales differ 10x.
    spread = expected_array.std(axis=0, ddof=1)
    spread[spread == 0.0] = np.nan
    relative = np.abs(observed_array - expected_array) / spread
    correlation = float(np.corrcoef(observed_array.reshape(-1), expected_array.reshape(-1))[0, 1])
    return float(np.nanmax(relative)), correlation


def extract_story_projections(
    backend: StoryCaptureBackend,
    stories: tuple[GeneratedStory, ...],
    emotion: EmotionVectors,
    directions: RevealRpeDirections,
    *,
    spec: ModelSpec,
    words_file_sha256: str,
    min_token: int = DEFAULT_MIN_TOKEN,
    fallback_blocks: int = DEFAULT_FALLBACK_BLOCKS,
) -> StoryProjections:
    """Re-capture the KEPT stories and project each onto the block's direction set.

    ``stories`` is E0's own story log, read back from ``stories.json``: same text, same order,
    same filter verdicts. Only the kept rows are captured, exactly as E0 did, so the grand mean
    recomputed here is E0's grand mean and the gate below is an identity rather than an estimate.
    """

    kept = tuple(story for story in stories if story.kept)
    if not kept:
        raise ValueError("the story log contains no kept stories; nothing to re-capture")
    if emotion.metadata.n_blocks != directions.metadata.n_blocks:
        raise ValueError("emotion and directions artifacts have different block counts")

    layers = decoder_layers(backend, fallback_blocks=fallback_blocks)
    states = capture_token_mean_states(
        backend, [story.text for story in kept], layers=layers, min_token=min_token
    )
    story_labels = tuple(story.emotion for story in kept)

    # E0's centring, reconstructed from the same rows: per-label mean first, then the grand mean
    # over the EMOTION WORDS only. Averaging the stories directly would weight labels by their
    # surviving story count and would not reproduce E0's frame.
    word_labels = emotion.word_labels
    label_means = np.stack(
        [
            states[np.asarray([i for i, story in enumerate(story_labels) if story == label])].mean(
                axis=0
            )
            for label in word_labels
        ]
    )
    grand_mean = label_means.mean(axis=0)
    centred = states - grand_mean

    unit_directions = np.stack(
        [direction_matrix(emotion, directions, block) for block in range(emotion.metadata.n_blocks)]
    )
    projections = np.ascontiguousarray(
        np.einsum("nbh,bdh->nbd", centred, unit_directions), dtype=np.float64
    )
    norms = np.linalg.norm(centred, axis=2)

    gram_labels = emotion.metadata.vector_labels
    rows_by_label = {
        label: np.asarray([i for i, story in enumerate(story_labels) if story == label], dtype=int)
        for label in gram_labels
    }
    if any(rows.size == 0 for rows in rows_by_label.values()):
        raise ValueError("every emotion-vector row must have at least one surviving story")
    k_max = max(rows.size for rows in rows_by_label.values())
    word_grams = np.full(
        (len(gram_labels), int(centred.shape[1]), k_max, k_max), _GRAM_PAD, dtype=np.float64
    )
    for label_index, label in enumerate(gram_labels):
        block = centred[rows_by_label[label]]
        count = block.shape[0]
        word_grams[label_index, :, :count, :count] = np.einsum("ibh,mbh->bim", block, block)

    deviation, correlation = _gate_against_emotion_basis(
        projections, story_labels, emotion, unit_directions
    )
    passed = deviation <= GATE_MAX_RELATIVE_DEVIATION
    metadata = StoryProjectionsMetadata(
        model_key=spec.key,
        backend=spec.backend,
        model_id=spec.model_id,
        revision=spec.revision,
        tokenizer_id=spec.tokenizer_id or spec.model_id,
        dtype=spec.dtype,
        min_token=min_token,
        emotion_vectors_sha256=emotion.metadata.vectors_sha256,
        emotion_vectors_selected_block=emotion.metadata.selected_block,
        directions_sha256=directions.metadata.directions_sha256,
        directions_selected_block=directions.metadata.selected_block,
        words_file_sha256=words_file_sha256,
        story_labels=story_labels,
        story_topics=tuple(story.topic for story in kept),
        story_indices=tuple(story.story_index for story in kept),
        n_stories=len(kept),
        n_blocks=int(projections.shape[1]),
        hidden_size=int(states.shape[2]),
        direction_names=PROJECTION_DIRECTIONS,
        stories_per_label={label: int(rows.size) for label, rows in rows_by_label.items()},
        gram_labels=gram_labels,
        gram_max_stories_per_label=k_max,
        gate_max_relative_deviation=deviation,
        gate_max_relative_deviation_threshold=GATE_MAX_RELATIVE_DEVIATION,
        gate_word_mean_correlation=correlation,
        gate_verdict="pass" if passed else "harness_inadequate",
        projections_dtype=str(projections.dtype),
        projections_sha256=states_sha256(projections),
    )
    artifact = StoryProjections(
        metadata=metadata, projections=projections, norms=norms, word_grams=word_grams
    )
    if not passed:
        raise StoryProjectionGateFailure(
            artifact,
            f"re-capture does not reproduce the E0 word vectors: max relative deviation "
            f"{deviation:.4g} > {GATE_MAX_RELATIVE_DEVIATION} (word-mean correlation "
            f"{correlation:.6f}). The projections would not be a story-level decomposition of the "
            "published basis, so the artifact is quarantined rather than written to the analysis "
            "path.",
        )
    return artifact
