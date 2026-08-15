"""End-to-end contract smoke for the emotion layer: the E0 -> P1 -> E1 -> E2 -> E3 chain on the
fake backend.

Runs the six shipped commands in order through the real CLI, into a tmp run root, using the
shipped smoke configs with only their output/registry/word paths redirected. Every E0 smoke
config gets the E0 -> P1 chain run over it, not just the base one: the two Sofroniew arms differ
from the base arm in the parts that broke (batched generation, and a confound subspace subtracted
from the published vectors), and a config no test loads is a config nothing checks. Everything asserted
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
from appraisal_emotions.config import load_config
from appraisal_emotions.stimuli.emotion_stories import read_emotion_words

REPO_ROOT = Path(__file__).resolve().parents[1]
RPE_SMOKE = REPO_ROOT / "configs" / "reveal_rpe_smoke.yaml"
EMOTION_SMOKE = REPO_ROOT / "configs" / "emotion_vectors_smoke.yaml"
# Globbed, not listed: a new E0 smoke arm inherits the E0 -> P1 chain by existing, which is what
# stops the next arm from shipping with no test that ever loaded it. The glob was `sofroniew*`
# until the widened arm shipped and was, of course, not covered by it; every arm config is named
# `emotion_vectors_<arm>_smoke.yaml`, and the un-armed `emotion_vectors_smoke.yaml` is excluded by
# the pattern because the module fixture already runs it.
ARM_SMOKE_CONFIGS = sorted(
    path.name for path in (REPO_ROOT / "configs").glob("emotion_vectors_*_smoke.yaml")
)
WORDS = REPO_ROOT / "data" / "emotion_words.json"


def _config_into(source: Path, tmp_path: Path, name: str) -> Path:
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["run"]["output_root"] = str(tmp_path / "runs")
    payload["model"]["registry_path"] = str(REPO_ROOT / "configs" / "model_registry.yaml")
    if "emotion_vectors" in payload:
        payload["emotion_vectors"]["words_file"] = str(WORDS)
        if "sofroniew_data_dir" in payload["emotion_vectors"]:
            # Absolutized like the other inputs: the config's relative path resolves against the
            # CWD, which is not the repo root for every way of invoking pytest.
            payload["emotion_vectors"]["sofroniew_data_dir"] = str(
                REPO_ROOT / payload["emotion_vectors"]["sofroniew_data_dir"]
            )
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


@pytest.mark.parametrize("config_name", ARM_SMOKE_CONFIGS)
def test_the_shipped_arms_run_e0_then_p1_through_the_cli(tmp_path, chain, config_name):
    """E0 -> P1 on every shipped arm smoke config, through the real command surface.

    No test loaded or ran ANY of these configs before, and the gap was not theoretical: the
    projected arm's E0 wrote an artifact its own P1 structurally rejected (max relative deviation
    3.469 against a 0.01 threshold, exit 1, capture quarantined), and both configs asked the
    neutral grid for more topics than their stories were written over. Both are one CLI invocation
    away from visible. Parametrizing over the config names is what keeps a third arm from shipping
    untested.

    The identity assertion at the end is the one that catches the P1 defect specifically: a
    projected basis whose per-story rows do not average back to it fails here by orders of
    magnitude, whatever the gate threshold happens to be.
    """

    runner = CliRunner()
    config = _config_into(REPO_ROOT / "configs" / config_name, tmp_path, config_name)
    run_id = yaml.safe_load(config.read_text(encoding="utf-8"))["run"]["id"]
    emotion_dir = tmp_path / "runs" / run_id / "emotions"

    result = runner.invoke(app, ["extract-emotions", "--config", str(config)])
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    emotion = read_emotion_vectors(emotion_dir / "emotion_vectors.json")
    # Realised sample size equals the configured one — the check that reads 0.00 drop rate on a
    # half-size sample cannot be satisfied by a clean drop audit alone.
    assert all(
        count == emotion.metadata.stories_per_emotion
        for count in emotion.metadata.generated_by_label.values()
    )

    projected = emotion.metadata.neutral_projection is not None
    assert projected == ("projected" in config_name)
    if projected:
        # The removed subspace is IN the artifact, so a reviewer can see which directions went.
        assert emotion.neutral_basis is not None
        assert (emotion_dir / "neutral_dialogues.json").is_file()
        assert (emotion_dir / "neutral_first_contact_sample.json").is_file()

    result = runner.invoke(
        app,
        [
            "extract-story-projections",
            "--config",
            str(config),
            "--directions",
            str(chain.rpe_dir / "reveal_directions.json"),
        ],
    )
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert not (emotion_dir / "story_projections_gate_failed.json").exists()

    directions = read_reveal_rpe_directions(chain.rpe_dir / "reveal_directions.json")
    projections = read_story_projections(emotion_dir / "story_projections.json")
    assert projections.metadata.gate_verdict == "pass"
    assert projections.metadata.emotion_vectors_sha256 == emotion.metadata.vectors_sha256
    unit = np.stack(
        [direction_matrix(emotion, directions, block) for block in range(emotion.metadata.n_blocks)]
    )
    for index, label in enumerate(emotion.metadata.vector_labels):
        observed = projections.projections[projections.rows_for(label)].mean(axis=0)
        expected = np.einsum("bh,bdh->bd", emotion.vectors[index], unit)
        assert np.allclose(observed, expected, rtol=0.0, atol=1e-12), label


def test_a_refused_e0_quarantines_the_corpora_it_already_paid_for(tmp_path):
    """A refusal must not throw away the generations that produced it.

    All three E0 refusals fire after generation and before the artifact is written. They used to
    exit(1) before any write, discarding the whole story corpus — and the yield check's own
    message told the operator to go read a first-contact sample that path never wrote. Here the
    projected smoke config is pushed into a shortfall (2 stories per call from a backend that
    emits one undivided completion), and the files it names must be on disk afterwards.
    """

    runner = CliRunner()
    name = "emotion_vectors_sofroniew_projected_smoke.yaml"
    config = _config_into(REPO_ROOT / "configs" / name, tmp_path, name)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["emotion_vectors"]["sofroniew_stories_per_call"] = 2
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    emotion_dir = tmp_path / "runs" / payload["run"]["id"] / "emotions"

    result = runner.invoke(app, ["extract-emotions", "--config", str(config)])
    assert result.exit_code == 1, result.output
    assert "harness_inadequate" in result.output
    # The refusal's own advice, honoured.
    assert (emotion_dir / "stories.json").is_file()
    assert (emotion_dir / "first_contact_sample.json").is_file()
    # ...and it stopped before scoring anything, so no basis was published.
    assert not (emotion_dir / "emotion_vectors.json").exists()
    log = json.loads((emotion_dir / "stories.json").read_text(encoding="utf-8"))
    assert log, "the quarantined log must hold the generations, not be an empty placeholder"


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
    assert f"all {len(words.labels)} words by valence residual" in result.output
    assert "recorded expectations" in result.output


def test_the_readout_columns_come_from_the_config_and_default_to_valence_alone(chain, tmp_path):
    """Which nuisances get partialled out is the RUN's decision, recorded in its config.

    Two things are under test and only one of them is the new feature. With no ``--config`` the
    command must still produce E1's certified valence-only estimand — the published 0.018572 has
    to stay reproducible from this code — and with the widened arm's config it must carry arousal
    into the design. The chosen columns are written into the report, so a reader of the artifact
    never has to reconstruct which estimand it is.
    """

    runner = CliRunner()
    common = [
        "--directions",
        str(chain.rpe_dir / "reveal_directions.json"),
        "--emotions",
        str(chain.emotion_dir / "emotion_vectors.json"),
        "--words",
        str(WORDS),
        "--seed",
        "7",
        "--permutations",
        "50",
        "--null-draws",
        "20",
    ]
    default_out = tmp_path / "default.json"
    result = runner.invoke(app, ["map-geometry", *common, "--out", str(default_out)])
    assert result.exit_code == 0, result.output
    assert json.loads(default_out.read_text(encoding="utf-8"))["residualize_on"] == ["valence"]

    wide_out = tmp_path / "wide.json"
    result = runner.invoke(
        app,
        [
            "map-geometry",
            *common,
            "--out",
            str(wide_out),
            "--config",
            str(REPO_ROOT / "configs" / "emotion_vectors_wide_smoke.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(wide_out.read_text(encoding="utf-8"))["residualize_on"] == [
        "valence",
        "arousal",
    ]


def test_only_the_widened_arm_partials_out_arousal():
    """The certified path's default must not move when a new arm ships.

    Every other shipped E0 config feeds analyses whose published numbers were computed on
    ``[1, valence]``; a config that quietly gained a second nuisance column would change what those
    numbers ESTIMATE without changing anything a reader can see. So the split is asserted over the
    shipped set rather than over a list this test keeps, and a new arm has to declare itself here.
    """

    residualize = {
        path.name: load_config(path).emotion_vectors.residualize_on
        for path in sorted((REPO_ROOT / "configs").glob("emotion_vectors_*.yaml"))
    }
    assert residualize, "no shipped emotion_vectors config was loaded"
    widened = {name for name, columns in residualize.items() if "arousal" in columns}
    assert widened == {"emotion_vectors_wide.yaml", "emotion_vectors_wide_smoke.yaml"}, residualize
    assert all(columns[0] == "valence" for columns in residualize.values())


def test_a_readout_that_does_not_partial_valence_is_refused(tmp_path):
    """Valence first and always: every readout here is a valence-RESIDUAL readout, and the column
    order is the design matrix's, so arousal alone is not a variant of this analysis at all."""

    source = REPO_ROOT / "configs" / "emotion_vectors_wide_smoke.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["emotion_vectors"]["residualize_on"] = ["arousal"]
    path = tmp_path / "arousal_only.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="config validation failed") as failure:
        load_config(path)
    # `load_config` re-raises with the config's path; the reason is on the cause it chains to.
    assert "must start with 'valence'" in str(failure.value.__cause__)


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
    assert [arm["cell_family"] for arm in e2["arms"]] == ["reward_matched", "ev_matched"]
    for arm in e2["arms"]:
        assert len(arm["axes"]) == 2
        assert all(axis["n_cells"] >= 2 for axis in arm["axes"])
    assert len(e2["comparison_signature"]) == 2

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
