"""P1 capture: the artifact must be a decomposition of E0's OWN basis, and it must span P1's math.

Two properties are worth a test here, and they are the two the analysis silently assumes.

*Faithfulness.* Re-capturing E0's stories and centring by the recomputed grand mean reproduces
E0's word vectors exactly (`mean_i (x_ij - g) = e_j`), so every per-story row is a decomposition of
a published number rather than a parallel measurement of the same thing. The tests below check the
identity holds on an aligned re-capture and that the gate REFUSES the three misalignments that
would otherwise pass silently: a permuted story log, a different capture window, and a story set
that is not the one the basis was built from.

*Sufficiency.* P1's headline quantities are split-half and leave-one-out word-level cosines, whose
denominators need inner products between a subset mean and a single story. Per-story scalars do
not span those; the within-word Gram does. `test_gram_reproduces_split_half_cosines_exactly` is
the proof that the stored artifact is enough — without it the analysis would be a delta-method
approximation of the reliability it exists to measure.

The fake backend's states are a deterministic hash of the text, so nothing here is evidence about
any model. The structure under test is arithmetic, and arithmetic is what these fixtures check.
"""

from __future__ import annotations

import numpy as np
import pytest

from appraisal_emotions.activation.capture import capture_token_mean_states, decoder_layers
from appraisal_emotions.analysis.emotion_vectors import EmotionVectors, GeneratedStory
from appraisal_emotions.analysis.neutral_projection import (
    basis_sha256,
    fit_neutral_basis,
    project_out,
)
from appraisal_emotions.analysis.story_projections import (
    GATE_MAX_RELATIVE_DEVIATION,
    PROJECTION_DIRECTIONS,
    StoryProjectionGateFailure,
    direction_matrix,
    extract_story_projections,
    read_story_projections,
    write_story_projections,
)
from appraisal_emotions.backends.fake import FakeBackend
from appraisal_emotions.core.schema import ModelSpec
from appraisal_emotions.stimuli.emotion_stories import STYLE_CONTROL_LABEL
from conftest import synthetic_directions_artifact, synthetic_emotion_artifact

SPEC = ModelSpec(key="fake", backend="fake", model_id="fake")
WORDS = ("elated", "content", "calm", "uneasy", "bitter", "crushed")
VALENCE = (1, 1, 1, -1, -1, -1)
STORIES_PER_LABEL = 5
MIN_TOKEN = 5
N_BLOCKS = 4


def _stories(seed: int = 0) -> tuple[GeneratedStory, ...]:
    """A story log in E0's shape: every label represented, one dropped row to exercise the filter."""

    rng = np.random.default_rng(seed)
    rows: list[GeneratedStory] = []
    for label in (*WORDS, STYLE_CONTROL_LABEL):
        for index in range(STORIES_PER_LABEL):
            # Distinct text per (label, index) so the fake backend's hash gives distinct states.
            filler = " ".join(f"w{int(value)}" for value in rng.integers(0, 999, size=12))
            rows.append(
                GeneratedStory(
                    emotion=label,
                    topic=f"topic{index % 3}",
                    story_index=index,
                    text=f"{label} story {index}: {filler}.",
                    token_count=40,
                    drop_reason=None,
                )
            )
    rows.append(
        GeneratedStory(
            emotion=WORDS[0],
            topic="topic0",
            story_index=STORIES_PER_LABEL,
            text="a dropped story that must never be captured or centred over",
            token_count=40,
            drop_reason="names_target",
        )
    )
    return tuple(rows)


def _capture(backend: FakeBackend, stories: tuple[GeneratedStory, ...], min_token: int = MIN_TOKEN):
    kept = tuple(story for story in stories if story.kept)
    layers = decoder_layers(backend, fallback_blocks=N_BLOCKS)
    return kept, capture_token_mean_states(
        backend, [story.text for story in kept], layers=layers, min_token=min_token
    )


def _synthetic_e0(backend: FakeBackend, stories: tuple[GeneratedStory, ...], **kwargs):
    """Build the E0 artifact these stories WOULD have produced, by E0's own centring recipe."""

    kept, states = _capture(backend, stories, **kwargs)
    labels = (*WORDS, STYLE_CONTROL_LABEL)
    means = np.stack(
        [
            states[[i for i, story in enumerate(kept) if story.emotion == label]].mean(axis=0)
            for label in labels
        ]
    )
    grand_mean = means[: len(WORDS)].mean(axis=0)
    return synthetic_emotion_artifact(means - grand_mean, WORDS, VALENCE), states, kept


@pytest.fixture
def fixture():
    backend = FakeBackend(SPEC, decoder_block_count=N_BLOCKS)
    stories = _stories()
    emotion, states, kept = _synthetic_e0(backend, stories)
    rng = np.random.default_rng(3)
    raw = rng.standard_normal((3, N_BLOCKS, emotion.metadata.hidden_size))
    directions = synthetic_directions_artifact(raw / np.linalg.norm(raw, axis=2, keepdims=True))
    return backend, stories, emotion, directions, states, kept


@pytest.fixture
def projected_fixture():
    """The same fixture on the PAPER-FAITHFUL arm: E0's word vectors carry a confound removal.

    Built exactly as ``extract_emotion_vectors`` builds it — centre, then project — so the artifact
    under test is the one the projected config actually writes, basis included.
    """

    backend = FakeBackend(SPEC, decoder_block_count=N_BLOCKS)
    stories = _stories()
    kept, states = _capture(backend, stories)
    labels = (*WORDS, STYLE_CONTROL_LABEL)
    means = np.stack(
        [
            states[[i for i, story in enumerate(kept) if story.emotion == label]].mean(axis=0)
            for label in labels
        ]
    )
    centred = means - means[: len(WORDS)].mean(axis=0)
    # A neutral corpus in the same space the stories live in, so the removed directions are ones
    # the emotion vectors actually have a component along — a basis orthogonal to them would make
    # the projection a no-op and the test vacuous.
    neutral = states[:: max(1, len(states) // 8)]
    basis = fit_neutral_basis(neutral, variance_target=0.5)
    emotion = synthetic_emotion_artifact(
        project_out(centred, basis), WORDS, VALENCE, neutral_basis=basis
    )
    rng = np.random.default_rng(3)
    raw = rng.standard_normal((3, N_BLOCKS, emotion.metadata.hidden_size))
    directions = synthetic_directions_artifact(raw / np.linalg.norm(raw, axis=2, keepdims=True))
    return backend, stories, emotion, directions, states, kept


def _extract(fixture, **kwargs):
    backend, stories, emotion, directions, _states, _kept = fixture
    return extract_story_projections(
        backend,
        kwargs.pop("stories", stories),
        emotion,
        directions,
        spec=SPEC,
        words_file_sha256="0" * 64,
        min_token=kwargs.pop("min_token", MIN_TOKEN),
        fallback_blocks=N_BLOCKS,
        **kwargs,
    )


def test_word_means_reproduce_the_emotion_basis_exactly(fixture):
    """The identity the whole artifact rests on: mean_i (y_ij . d) == e_j . d, to float precision."""

    _backend, _stories, emotion, directions, _states, _kept = fixture
    artifact = _extract(fixture)

    unit = np.stack([direction_matrix(emotion, directions, b) for b in range(N_BLOCKS)])
    for label_index, label in enumerate(emotion.metadata.vector_labels):
        observed = artifact.projections[artifact.rows_for(label)].mean(axis=0)
        expected = np.einsum("bh,bdh->bd", emotion.vectors[label_index], unit)
        assert np.allclose(observed, expected, rtol=0.0, atol=1e-12)

    # The gate agrees, and it agrees with room to spare rather than by a hair.
    assert artifact.metadata.gate_max_relative_deviation < GATE_MAX_RELATIVE_DEVIATION / 100
    assert artifact.metadata.gate_word_mean_correlation > 1 - 1e-12
    assert artifact.metadata.direction_names == PROJECTION_DIRECTIONS
    assert artifact.metadata.n_stories == len(WORDS + (STYLE_CONTROL_LABEL,)) * STORIES_PER_LABEL
    assert artifact.metadata.gate_verdict == "pass"


def test_word_means_reproduce_a_projected_emotion_basis(projected_fixture):
    """The same identity on the paper-faithful arm, whose basis had a confound subspace removed.

    Reproduced before the fix: E0 subtracted the neutral basis from its published word vectors
    while P1 re-derived unprojected story means, so the gate compared two different spaces and
    read a max relative deviation of 3.469 (word-mean correlation 0.718) on the shipped projected
    smoke config. The subtraction is linear, so applying it per story restores the identity
    exactly rather than merely loosening the threshold.
    """

    _backend, _stories, emotion, directions, _states, _kept = projected_fixture
    assert emotion.neutral_basis is not None
    assert emotion.neutral_basis.total_components > 0, "a basis that removes nothing proves nothing"
    artifact = _extract(projected_fixture)

    unit = np.stack([direction_matrix(emotion, directions, b) for b in range(N_BLOCKS)])
    for label_index, label in enumerate(emotion.metadata.vector_labels):
        observed = artifact.projections[artifact.rows_for(label)].mean(axis=0)
        expected = np.einsum("bh,bdh->bd", emotion.vectors[label_index], unit)
        assert np.allclose(observed, expected, rtol=0.0, atol=1e-12), label
    assert artifact.metadata.gate_verdict == "pass"
    assert artifact.metadata.gate_max_relative_deviation < GATE_MAX_RELATIVE_DEVIATION / 100
    # The capture records which space it lives in, so a reader never has to consult the config.
    assert artifact.metadata.neutral_basis_sha256 == basis_sha256(emotion.neutral_basis)

    # ...and the per-story states really were projected: their component in the removed subspace
    # is gone. Without this the test above would pass on a basis that removed nothing.
    _kept_rows, raw = _capture(_backend := projected_fixture[0], projected_fixture[1])
    assert np.linalg.norm(raw @ emotion.neutral_basis.components[0].T) > 1e-6


def test_a_projected_basis_without_its_components_is_unreadable():
    """The compounding half of the same defect: counts recorded, directions not.

    An artifact that says a projection ran but does not carry the subspace it removed cannot be
    audited by a reviewer and cannot be re-applied by P1, so it is refused at construction rather
    than producing a confusing gate failure two commands later.
    """

    vectors = np.random.default_rng(4).standard_normal((len(WORDS) + 1, N_BLOCKS, 16))
    basis = fit_neutral_basis(
        np.random.default_rng(5).standard_normal((8, N_BLOCKS, 16)), variance_target=0.5
    )
    artifact = synthetic_emotion_artifact(vectors, WORDS, VALENCE, neutral_basis=basis)
    with pytest.raises(ValueError, match="no confound basis is attached"):
        EmotionVectors(metadata=artifact.metadata, vectors=artifact.vectors)


def test_a_failed_gate_still_yields_a_quarantinable_artifact(fixture):
    """The capture is a rented-GPU cost: a gate failure must not throw the numbers away."""

    with pytest.raises(StoryProjectionGateFailure) as caught:
        _extract(fixture, min_token=MIN_TOKEN + 20)
    artifact = caught.value.artifact
    assert artifact.metadata.gate_verdict == "harness_inadequate"
    assert artifact.metadata.gate_max_relative_deviation > GATE_MAX_RELATIVE_DEVIATION
    assert artifact.projections.shape[0] == artifact.metadata.n_stories


def test_dropped_stories_are_not_captured(fixture):
    """The drop_reason row must be absent: E0 centred over the survivors, so P1 must too."""

    artifact = _extract(fixture)
    assert artifact.metadata.stories_per_label[WORDS[0]] == STORIES_PER_LABEL
    assert all(index < STORIES_PER_LABEL for index in artifact.metadata.story_indices)


def test_gate_refuses_a_permuted_story_log(fixture):
    """Same texts, same count, labels rotated by one: the identity breaks and the write is refused."""

    _backend, stories, _emotion, _directions, _states, _kept = fixture
    kept = [story for story in stories if story.kept]
    rotated = tuple(
        GeneratedStory(
            emotion=kept[(i + STORIES_PER_LABEL) % len(kept)].emotion,
            topic=story.topic,
            story_index=story.story_index,
            text=story.text,
            token_count=story.token_count,
            drop_reason=None,
        )
        for i, story in enumerate(kept)
    )
    with pytest.raises(StoryProjectionGateFailure, match="does not reproduce"):
        _extract(fixture, stories=rotated)


def test_gate_refuses_a_different_capture_window(fixture):
    """The most plausible silent misalignment: right stories, wrong min_token."""

    with pytest.raises(StoryProjectionGateFailure, match="does not reproduce"):
        _extract(fixture, min_token=MIN_TOKEN + 20)


def test_gate_refuses_stories_the_basis_was_not_built_from(fixture):
    """A re-generated story set with the same shape is a different measurement, not a re-capture."""

    with pytest.raises(StoryProjectionGateFailure, match="does not reproduce"):
        _extract(fixture, stories=_stories(seed=99))


def test_gram_reproduces_split_half_cosines_exactly(fixture):
    """Half-mean word cosines from (projections, Gram) match the ones from the full states.

    This is the sufficiency proof. The numerator is linear so the projections carry it; the
    DENOMINATOR ``||mean_A y||`` is not, and it is exactly what the stored Gram supplies. If this
    passes, P1's split-half reliability is computed on the real statistic rather than on a
    first-order approximation of it.
    """

    _backend, _stories, emotion, directions, states, kept = fixture
    artifact = _extract(fixture)
    unit = np.stack([direction_matrix(emotion, directions, b) for b in range(N_BLOCKS)])
    grand_mean = np.stack(
        [
            states[[i for i, story in enumerate(kept) if story.emotion == label]].mean(axis=0)
            for label in WORDS
        ]
    ).mean(axis=0)
    centred = states - grand_mean

    for label in emotion.metadata.vector_labels:
        rows = artifact.rows_for(label)
        gram = artifact.gram_for(label)
        half = np.arange(rows.size) < rows.size // 2
        for subset in (half, ~half):
            picks = np.flatnonzero(subset)
            # From the artifact alone: projections give the numerator, the Gram the norm.
            numerator = artifact.projections[rows[picks]].mean(axis=0)
            norm = np.sqrt(gram[:, picks][:, :, picks].mean(axis=(1, 2)))
            from_artifact = numerator / norm[:, None]
            # From the full states, which the artifact is meant to replace.
            mean_state = centred[rows[picks]].mean(axis=0)
            truth = np.einsum("bh,bdh->bd", mean_state, unit) / np.linalg.norm(
                mean_state, axis=1, keepdims=True
            )
            assert np.allclose(from_artifact, truth, rtol=1e-12, atol=1e-12)


def test_gram_diagonal_is_gated_against_the_stored_norms(fixture):
    """Two routes to the same number, so a mismatch means one array came from another capture."""

    artifact = _extract(fixture)
    corrupted = np.array(artifact.norms)
    corrupted[0, 0] *= 1.5
    with pytest.raises(ValueError, match="Gram diagonal disagrees"):
        type(artifact)(
            metadata=artifact.metadata,
            projections=artifact.projections,
            norms=corrupted,
            word_grams=artifact.word_grams,
        )


def test_artifact_round_trips_through_disk(tmp_path, fixture):
    artifact = _extract(fixture)
    metadata_path, payload_path = write_story_projections(artifact, tmp_path / "p1.json")
    assert payload_path.exists()
    restored = read_story_projections(metadata_path)
    assert restored.metadata == artifact.metadata
    assert np.array_equal(restored.projections, artifact.projections)
    assert np.array_equal(restored.norms, artifact.norms)
    assert np.array_equal(
        np.nan_to_num(restored.word_grams, nan=-1.0), np.nan_to_num(artifact.word_grams, nan=-1.0)
    )
    assert restored.cosines("v_rpe").shape == (artifact.metadata.n_stories, N_BLOCKS)
