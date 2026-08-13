"""End-to-end contract smoke for the emotion layer: the E0 -> E1 -> E2 chain on the fake backend.

Runs the four shipped commands in order through the real CLI, into a tmp run root, using the
shipped smoke configs with only their output/registry/word paths redirected. Everything asserted
here is CONTRACT — artifacts exist, revalidate, and bind to each other by digest; reports carry
the confirmatory / exploratory split and the inherited gate cap. The fake backend's stories are
seeded templates and its hidden states are a hash of the text, so every NUMBER in this chain is
meaningless and none of it is evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from appraisal_emotions.analysis.emotion_vectors import read_emotion_vectors
from appraisal_emotions.cli import app

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


def test_emotion_chain_smoke(tmp_path):
    runner = CliRunner()
    rpe_config = _config_into(RPE_SMOKE, tmp_path, "rpe.yaml")
    emotion_config = _config_into(EMOTION_SMOKE, tmp_path, "emotions.yaml")

    result = runner.invoke(app, ["extract-rpe", "--config", str(rpe_config)])
    assert result.exit_code == 0, result.output
    rpe_dir = tmp_path / "runs" / "reveal_rpe_smoke" / "reveal_rpe"

    result = runner.invoke(app, ["extract-emotions", "--config", str(emotion_config)])
    assert result.exit_code == 0, result.output
    emotion_dir = tmp_path / "runs" / "emotion_vectors_smoke" / "emotions"
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
    assert e1["confirmatory"] and e1["exploratory"]
    assert len(e1["block_sweep"]) == emotion.metadata.n_blocks
    assert {pair["high"] for block in e1["confirmatory"] for pair in block["p2_pairs"]} == {
        "disappointed",
        "relieved",
        "elated",
    }
    assert "CONFIRMATORY" in result.output and "EXPLORATORY" in result.output

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
