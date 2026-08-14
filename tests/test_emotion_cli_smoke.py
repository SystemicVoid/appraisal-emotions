"""End-to-end contract smoke for the emotion layer: the E0 -> P1 -> E1 -> E2 -> E3 chain on the
fake backend.

Runs the six shipped commands in order through the real CLI, into a tmp run root, using the
shipped smoke configs with only their output/registry/word paths redirected. Everything asserted
here is CONTRACT — artifacts exist, revalidate, and bind to each other by digest; the reports
carry every word's residual, the recorded-expectation readouts, the patching arms, and the
inherited gate cap. The fake backend's stories are seeded templates and its hidden states are a
hash of the text, so every NUMBER in this chain is meaningless and none of it is evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from appraisal_emotions.analysis.activation_patching import ARMS
from appraisal_emotions.analysis.emotion_vectors import read_emotion_vectors
from appraisal_emotions.analysis.reveal_rpe import read_reveal_rpe_directions
from appraisal_emotions.analysis.story_projections import (
    direction_matrix,
    read_story_projections,
)
from appraisal_emotions.cli import app
from appraisal_emotions.stimuli.emotion_stories import read_emotion_words

REPO_ROOT = Path(__file__).resolve().parents[1]
RPE_SMOKE = REPO_ROOT / "configs" / "reveal_rpe_smoke.yaml"
EMOTION_SMOKE = REPO_ROOT / "configs" / "emotion_vectors_smoke.yaml"
WORDS = REPO_ROOT / "data" / "emotion_words.json"


def _config_into(source: Path, tmp_path: Path, name: str) -> Path:
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["run"]["output_root"] = str(tmp_path / "runs")
    payload["model"]["registry_path"] = str(REPO_ROOT / "configs" / "model_registry.yaml")
    if "emotion_vectors" in payload:
        payload["emotion_vectors"]["words_file"] = str(WORDS)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


@dataclass(frozen=True)
class _Chain:
    """The E0/R-A artifacts every downstream smoke reads, captured once per module."""

    rpe_dir: Path
    emotion_dir: Path
    emotion_config: Path


@pytest.fixture(scope="module")
def chain(tmp_path_factory) -> _Chain:
    """Run extract-rpe + extract-emotions once; the E1/E2/E3 commands are read-only over them."""

    tmp_path = tmp_path_factory.mktemp("emotion_chain")
    runner = CliRunner()
    rpe_config = _config_into(RPE_SMOKE, tmp_path, "rpe.yaml")
    emotion_config = _config_into(EMOTION_SMOKE, tmp_path, "emotions.yaml")
    result = runner.invoke(app, ["extract-rpe", "--config", str(rpe_config)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["extract-emotions", "--config", str(emotion_config)])
    assert result.exit_code == 0, result.output
    return _Chain(
        rpe_dir=tmp_path / "runs" / "reveal_rpe_smoke" / "reveal_rpe",
        emotion_dir=tmp_path / "runs" / "emotion_vectors_smoke" / "emotions",
        emotion_config=emotion_config,
    )


def test_e0_writes_its_artifacts_and_reports_the_gate_cap(chain):
    emotion_dir = chain.emotion_dir
    for name in (
        "emotion_vectors.json",
        "emotion_vectors.vectors.npz",
        "stories.json",
        "first_contact_sample.json",
    ):
        assert (emotion_dir / name).is_file(), f"missing E0 artifact {name}"
    emotion = read_emotion_vectors(emotion_dir / "emotion_vectors.json")
    # The BLIND freeze's compensation is a written sample a human can read, not a log line.
    sample = json.loads((emotion_dir / "first_contact_sample.json").read_text(encoding="utf-8"))
    assert 0 < len(sample) <= emotion.metadata.first_contact_n
    assert all("text" in record and "drop_reason" in record for record in sample)
    # Hash-derived hidden states cannot carry valence structure, so the smoke MUST report the
    # cap rather than a pass — a "pass" here would mean the gate is not doing anything.
    assert emotion.metadata.gate_verdict == "harness_inadequate"


def test_p1_projections_bind_to_the_artifacts_they_decompose(chain):
    runner = CliRunner()
    rpe_dir, emotion_dir = chain.rpe_dir, chain.emotion_dir
    result = runner.invoke(
        app,
        [
            "extract-story-projections",
            "--config",
            str(chain.emotion_config),
            "--directions",
            str(rpe_dir / "reveal_directions.json"),
        ],
    )
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert not (emotion_dir / "story_projections_gate_failed.json").exists()

    emotion = read_emotion_vectors(emotion_dir / "emotion_vectors.json")
    directions = read_reveal_rpe_directions(rpe_dir / "reveal_directions.json")
    projections = read_story_projections(emotion_dir / "story_projections.json")
    meta = projections.metadata
    assert meta.gate_verdict == "pass"
    # The digests are the whole point of the capture: these projections are interpretable only
    # against the basis they centre on and the directions they project onto, so the binding is
    # checked against those two artifacts rather than against the projection metadata itself.
    assert meta.emotion_vectors_sha256 == emotion.metadata.vectors_sha256
    assert meta.directions_sha256 == directions.metadata.directions_sha256
    assert meta.n_stories == emotion.metadata.n_kept
    assert set(emotion.metadata.vector_labels) <= set(meta.stories_per_label)

    # The identity the artifact exists for, exercised through the CLI's config -> path wiring:
    # a re-capture that read a different window or a different run dir breaks it immediately.
    unit = np.stack(
        [direction_matrix(emotion, directions, block) for block in range(emotion.metadata.n_blocks)]
    )
    for index, label in enumerate(emotion.metadata.vector_labels):
        observed = projections.projections[projections.rows_for(label)].mean(axis=0)
        expected = np.einsum("bh,bdh->bd", emotion.vectors[index], unit)
        assert np.allclose(observed, expected, rtol=0.0, atol=1e-12), label


def test_e1_map_geometry_shows_every_word_and_the_recorded_expectations(chain):
    runner = CliRunner()
    rpe_dir, emotion_dir = chain.rpe_dir, chain.emotion_dir
    emotion = read_emotion_vectors(emotion_dir / "emotion_vectors.json")
    e1_out = emotion_dir / "map_geometry_report.json"
    result = runner.invoke(
        app,
        [
            "map-geometry",
            "--directions",
            str(rpe_dir / "reveal_directions.json"),
            "--emotions",
            str(emotion_dir / "emotion_vectors.json"),
            "--words",
            str(WORDS),
            "--out",
            str(e1_out),
            "--seed",
            "7",
            "--permutations",
            "200",
            "--null-draws",
            "50",
        ],
    )
    assert result.exit_code == 0, result.output
    e1 = json.loads(e1_out.read_text(encoding="utf-8"))
    assert e1["emotion_vectors_sha256"] == emotion.metadata.vectors_sha256
    assert e1["sensitivity_gate"] == "G0=harness_inadequate"
    assert e1["verdict_cap"].startswith("harness_inadequate")
    assert len(e1["block_sweep"]) == emotion.metadata.n_blocks
    words = read_emotion_words(WORDS)
    for block in e1["blocks"]:
        # Every word is shown — no readout hides behind a summary statistic.
        assert {row["word"] for row in block["word_residuals"]} == set(words.labels)
        assert {c["pole"] for c in block["family_contrasts"]} == {"positive", "negative"}
        assert block["outcome_ordering"]["families"] == list(words.outcome_ordering())
        assert {pair["outcome"] for pair in block["expected_pairs"]} == {
            "disappointed",
            "relieved",
            "elated",
        }
    # The confirmatory/exploratory caste is gone from the artifact, not merely from the prose.
    assert "confirmatory" not in e1_out.read_text(encoding="utf-8").lower()
    assert "all 84 words by valence residual" in result.output
    assert "recorded expectations" in result.output


def test_e2_and_e3_read_the_same_artifacts_and_cap_their_claims(chain):
    runner = CliRunner()
    rpe_dir, emotion_dir = chain.rpe_dir, chain.emotion_dir
    emotion = read_emotion_vectors(emotion_dir / "emotion_vectors.json")
    e2_out = emotion_dir / "expectation_control_report.json"
    result = runner.invoke(
        app,
        [
            "expectation-control",
            "--states",
            str(rpe_dir / "reveal_states.json"),
            "--battery",
            str(rpe_dir / "battery.json"),
            "--emotions",
            str(emotion_dir / "emotion_vectors.json"),
            "--out",
            str(e2_out),
            "--seed",
            "7",
            "--permutations",
            "200",
        ],
    )
    assert result.exit_code == 0, result.output
    e2 = json.loads(e2_out.read_text(encoding="utf-8"))
    assert e2["states_sha256"] and e2["battery_sha256"]
    assert e2["emotion_vectors_sha256"] == emotion.metadata.vectors_sha256
    assert e2["sensitivity_gate"] == "G0=harness_inadequate"
    assert len(e2["axes"]) == 2
    assert all(axis["n_cells"] >= 2 for axis in e2["axes"])

    e3_out = emotion_dir / "activation_patching_report.json"
    result = runner.invoke(
        app,
        [
            "patch-reveals",
            "--states",
            str(rpe_dir / "reveal_states.json"),
            "--battery",
            str(rpe_dir / "battery.json"),
            "--directions",
            str(rpe_dir / "reveal_directions.json"),
            "--emotions",
            str(emotion_dir / "emotion_vectors.json"),
            "--out",
            str(e3_out),
            "--seed",
            "7",
        ],
    )
    assert result.exit_code == 0, result.output
    e3 = json.loads(e3_out.read_text(encoding="utf-8"))
    assert e3["emotion_vectors_sha256"] == emotion.metadata.vectors_sha256
    assert e3["sensitivity_gate"] == "G0=harness_inadequate"
    assert e3["n_pairs"] > 0 and e3["n_reward_cells"] > 0
    assert {arm["arm"] for arm in e3["arms"]} == set(ARMS)
    assert len(e3["arms"]) == 2 * len(ARMS)
    assert e3["mode"] == "state" and e3["continuations"] == []
    # The mode note is printed with every run, not buried in a docstring.
    assert "STATE-LEVEL PREVIEW" in result.output

    e3_forward = emotion_dir / "activation_patching_forward.json"
    result = runner.invoke(
        app,
        [
            "patch-reveals",
            "--states",
            str(rpe_dir / "reveal_states.json"),
            "--battery",
            str(rpe_dir / "battery.json"),
            "--directions",
            str(rpe_dir / "reveal_directions.json"),
            "--emotions",
            str(emotion_dir / "emotion_vectors.json"),
            "--out",
            str(e3_forward),
            "--mode",
            "forward",
            "--config",
            str(chain.emotion_config),
            "--max-pairs",
            "2",
            "--random-draws",
            "2",
            "--seed",
            "7",
        ],
    )
    assert result.exit_code == 0, result.output
    forward = json.loads(e3_forward.read_text(encoding="utf-8"))
    assert forward["mode"] == "forward"
    assert forward["model_key"] == "fake-functional"
    # The causal tier reads downstream of the patch, and stores raw continuations for reading.
    assert max(forward["readout_blocks"]) > forward["block"]
    assert forward["continuations"]
    assert "FORWARD-PATCHED" in result.output
