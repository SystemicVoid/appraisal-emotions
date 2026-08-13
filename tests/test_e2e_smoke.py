"""End-to-end contract smoke: ``extract-rpe`` on the fake backend writes valid artifacts.

Exercises the shipped smoke config through the real CLI (preflight → battery → capture →
estimator → hash-bound artifacts) into a tmp run root. The fake backend's hidden states are a
deterministic hash of the prompt, so the GATE NUMBERS ARE MEANINGLESS — this asserts the
contract (artifacts exist, revalidate, and bind to each other), never the science.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from typer.testing import CliRunner

from appraisal_emotions.analysis.reveal_rpe import (
    read_reveal_rpe_directions,
    read_reveal_rpe_states,
)
from appraisal_emotions.cli import app
from appraisal_emotions.core.util import file_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPO_ROOT / "configs" / "reveal_rpe_smoke.yaml"


def _config_into(tmp_path: Path) -> Path:
    """The shipped smoke config, redirected to a tmp run root (values otherwise untouched)."""

    payload = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    payload["run"]["output_root"] = str(tmp_path / "runs")
    payload["model"]["registry_path"] = str(REPO_ROOT / "configs" / "model_registry.yaml")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_extract_rpe_smoke_writes_bound_artifacts(tmp_path):
    result = CliRunner().invoke(app, ["extract-rpe", "--config", str(_config_into(tmp_path))])
    assert result.exit_code == 0, result.output

    run_dir = tmp_path / "runs" / "reveal_rpe_smoke" / "reveal_rpe"
    for name in (
        "battery.json",
        "symbol_preflight.json",
        "reveal_rpe_report.json",
        "reveal_states.json",
        "reveal_states.states.npz",
        "reveal_directions.json",
        "reveal_directions.directions.npz",
    ):
        assert (run_dir / name).is_file(), f"missing artifact {name}"

    # Both payload-bearing artifacts revalidate their sha256 bindings on read.
    states = read_reveal_rpe_states(run_dir / "reveal_states.json")
    directions = read_reveal_rpe_directions(run_dir / "reveal_directions.json")

    report = json.loads((run_dir / "reveal_rpe_report.json").read_text(encoding="utf-8"))
    battery = json.loads((run_dir / "battery.json").read_text(encoding="utf-8"))
    preflight = json.loads((run_dir / "symbol_preflight.json").read_text(encoding="utf-8"))

    # The report is bound to the capture it scored and the battery that supplied the design.
    assert report["states_sha256"] == states.metadata.states_sha256
    assert report["battery_sha256"] == file_sha256(run_dir / "battery.json")
    assert directions.metadata.states_sha256 == states.metadata.states_sha256
    assert directions.metadata.battery_sha256 == report["battery_sha256"]
    assert directions.metadata.selected_block == report["selected_block"]
    assert directions.metadata.source_verdict == report["verdict"]

    # Row order and shapes line up with the battery actually written.
    analysed_ids = [
        c["comparison_id"]
        for c in battery["reveals"]
        if c["metadata"]["partition"] in ("estimation", "selection")
    ]
    assert list(states.metadata.reveal_ids) == analysed_ids
    assert states.states.shape == (
        len(analysed_ids),
        states.metadata.n_blocks,
        states.metadata.hidden_size,
    )
    assert directions.directions.shape == (3, states.metadata.n_blocks, states.metadata.hidden_size)
    norms = np.linalg.norm(directions.directions, axis=2)
    assert np.all(np.isclose(norms, 1.0, atol=1e-6) | (norms == 0.0))

    # The smoke's single-letter symbols pass the preflight, and the audited surface stays clean.
    assert preflight["passed"] is True
    assert battery["affect_audit_warnings"] == []
    # Licence discipline: the verdict vocabulary never crosses into use / welfare.
    assert report["verdict"] in {
        "separable-signed-rpe",
        "collapses-to-value-or-reward",
        "collapses-to-unsigned-surprise",
        "indeterminate",
    }


def test_extract_rpe_refuses_a_one_sided_partition_set(tmp_path):
    payload = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    payload["run"]["output_root"] = str(tmp_path / "runs")
    payload["model"]["registry_path"] = str(REPO_ROOT / "configs" / "model_registry.yaml")
    payload["reveal_rpe"]["partitions"] = ["estimation"]
    path = tmp_path / "one_sided.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(app, ["extract-rpe", "--config", str(path)])
    assert result.exit_code != 0
    assert "estimation and selection" in result.output


def test_config_rejects_the_confirmation_partition(tmp_path):
    # §4.3 F14 is structural here: the pilot partition vocabulary makes a confirmation-touching
    # run unrepresentable rather than refusing it in a CLI guard.
    payload = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    payload["run"]["output_root"] = str(tmp_path / "runs")
    payload["reveal_rpe"]["partitions"] = ["estimation", "selection", "confirmation"]
    path = tmp_path / "confirmation.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(app, ["extract-rpe", "--config", str(path)])
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "config validation failed" in str(result.exception)
