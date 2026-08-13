"""E0 extraction on the deterministic fake backend, plus the filter's own unit invariants.

The end-to-end case asserts the CONTRACT — artifact exists, shapes are consistent, the G0 table
is well-formed, the drop accounting adds up — and never the science: the fake backend's stories
are seeded templates and its hidden states are a hash of the text, so the G0 number is
meaningless by construction (which is exactly why the smoke run reports ``harness_inadequate``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from appraisal_emotions.analysis.emotion_vectors import (
    FirstContactFailure,
    extract_emotion_vectors,
    generate_stories,
    naive_variants,
    names_target,
    read_emotion_vectors,
    valence_oriented_pc_axes,
    write_emotion_vectors,
)
from appraisal_emotions.backends.fake import FakeBackend
from appraisal_emotions.core.schema import ModelSpec
from appraisal_emotions.stimuli.emotion_stories import (
    STYLE_CONTROL_LABEL,
    build_story_grid,
    read_emotion_words,
)

WORDS_PATH = Path(__file__).resolve().parents[1] / "data" / "emotion_words.json"
SPEC = ModelSpec(key="fake", backend="fake", model_id="fake")
# The shipped smoke recipe (configs/emotion_vectors_smoke.yaml), so the deterministic fake
# generations here are the same ones the smoke run reads.
SMOKE = {"stories_per_emotion": 3, "min_token": 20, "max_tokens": 160, "seed": 7}


@pytest.fixture(scope="module")
def words():
    return read_emotion_words(WORDS_PATH)


@pytest.fixture(scope="module")
def extracted(words):
    backend = FakeBackend(SPEC)
    return extract_emotion_vectors(
        backend,
        words,
        spec=SPEC,
        fallback_blocks=4,
        first_contact_max_drop_rate=0.6,
        **SMOKE,
    )


def test_naive_variants_cover_the_obvious_morphology():
    variants = set(naive_variants("disappointed"))
    assert {"disappoint", "disappointed", "disappointing", "disappointment"} <= variants
    assert "happiness" in set(naive_variants("happy"))
    with pytest.raises(ValueError):
        naive_variants("  ")


def test_names_target_is_word_boundary_delimited_and_case_insensitive():
    assert names_target("She was Elated by the news.", "elated")
    assert names_target("a small disappointment", "disappointed")
    assert not names_target("the sadiron on the shelf", "sad")
    assert not names_target("nothing of the sort", "calm")


def test_generate_stories_records_a_drop_reason_for_instruction_violations():
    backend = FakeBackend(SPEC)
    grid = build_story_grid(("elated", "calm"), stories_per_emotion=6, seed=7)
    stories = generate_stories(backend, grid, min_token=20, max_tokens=160, temperature=1.0)
    assert len(stories) == len(grid)
    assert {story.drop_reason for story in stories} <= {None, "names_target", "too_short", "empty"}
    # Every dropped-for-naming story really does contain its target; every kept one does not.
    for story in stories:
        if story.drop_reason == "names_target":
            assert names_target(story.text, story.emotion)
        if story.kept and story.emotion != STYLE_CONTROL_LABEL:
            assert not names_target(story.text, story.emotion)


def test_first_contact_checkpoint_refuses_an_over_dropping_filter(words):
    backend = FakeBackend(SPEC)
    with pytest.raises(FirstContactFailure) as excinfo:
        extract_emotion_vectors(
            backend,
            words,
            spec=SPEC,
            fallback_blocks=4,
            # An impossible bar: any drop at all in the sample trips the checkpoint, which is
            # the BLIND freeze's compensation path (docs/agents/rails.md).
            first_contact_max_drop_rate=0.0,
            **SMOKE,
        )
    assert excinfo.value.sample, "the checkpoint must hand back the sample a human has to read"


def test_end_to_end_artifact_is_shape_consistent_and_round_trips(extracted, words, tmp_path):
    artifact, stories = extracted
    meta = artifact.metadata
    n_labels = len(words.labels) + 1  # + style_control

    assert artifact.vectors.shape == (n_labels, meta.n_blocks, meta.hidden_size)
    assert meta.vector_labels[-1] == STYLE_CONTROL_LABEL
    assert meta.n_word_rows == len(words.labels)
    assert artifact.word_vectors.shape[0] == len(words.labels)
    assert len(meta.categories) == n_labels and len(meta.valence_labels) == n_labels

    # Drop accounting adds up and matches the returned stories.
    assert meta.n_requested == len(stories) == n_labels * SMOKE["stories_per_emotion"]
    assert meta.n_kept == sum(1 for story in stories if story.kept)
    assert sum(meta.drop_counts_by_reason.values()) == meta.n_requested - meta.n_kept
    assert meta.drop_rate == pytest.approx(1.0 - meta.n_kept / meta.n_requested)
    assert set(meta.kept_by_label) == set(meta.vector_labels)

    # G0 table: one row per block, in block order, and the gate reads the max-|rho| row.
    assert [row.block for row in meta.g0_table] == list(range(meta.n_blocks))
    assert all(-1.0 <= row.spearman_rho <= 1.0 for row in meta.g0_table)
    assert all(row.n_words == len(words.labels) for row in meta.g0_table)
    best = max(meta.g0_table, key=lambda row: abs(row.spearman_rho))
    assert meta.selected_block == best.block
    assert meta.g0_abs_rho == pytest.approx(abs(best.spearman_rho))
    assert meta.gate_verdict == (
        "pass" if meta.g0_abs_rho >= meta.g0_threshold else "harness_inadequate"
    )

    # The grand mean is subtracted over the emotion-word rows, so they sum to ~0 per block.
    assert np.allclose(artifact.word_vectors.mean(axis=0), 0.0, atol=1e-9)

    path = tmp_path / "emotion_vectors.json"
    write_emotion_vectors(artifact, path)
    reloaded = read_emotion_vectors(path)
    assert reloaded.metadata.vectors_sha256 == meta.vectors_sha256
    assert np.array_equal(reloaded.vectors, artifact.vectors)


def test_artifact_read_refuses_a_tampered_payload(extracted, tmp_path):
    artifact, _stories = extracted
    path = tmp_path / "tampered.json"
    _metadata_path, vectors_path = write_emotion_vectors(artifact, path)
    tampered = np.array(artifact.vectors)
    tampered[0, 0, 0] += 1.0
    np.savez(vectors_path, emotion_vectors=tampered)
    with pytest.raises(ValueError, match="hash does not match"):
        read_emotion_vectors(path)


def test_pc1_orientation_follows_the_valence_labels():
    rng = np.random.default_rng(11)
    axis = np.zeros(6)
    axis[0] = 1.0
    valence = np.asarray([1.0, 1.0, -1.0, -1.0, 0.0])
    vectors = np.outer(valence, axis) + 0.01 * rng.standard_normal((5, 6))
    components, scores = valence_oriented_pc_axes(vectors, valence)
    assert float(np.dot(scores[:, 0], valence)) > 0.0
    # Flipping the label sign flips the axis, which is what makes the axis sign well-defined.
    flipped, _scores = valence_oriented_pc_axes(vectors, -valence)
    assert np.allclose(flipped[0], -components[0])
