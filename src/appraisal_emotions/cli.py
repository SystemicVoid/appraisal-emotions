"""``appraisal-emotions`` command surface.

Extracted from functional-valence-validity src/fv_validity/cli_described_gambles.py
(``run_reveal_rpe``, L445-608) @ 10c4662, retargeted onto this package's config loader.

Kept: the preflight → battery → partition filter → capture → analyze → bind → write chain,
including the pre-capture ``>= 2 reveals per partition`` check (a degenerate config otherwise
wastes the whole base-model capture pass) and the states/battery digest stamping that binds the
report to what it scored.

Dropped: the provenance sidecar writes (two ``write_provenance`` calls per run), the
``adapter_dir`` refusal and the ``confirmation``-partition refusal from
``_guard_reveal_rpe_runnable`` — both are structural here (see ``config.RevealRpeConfig``), so
there is nothing left to guard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console

from appraisal_emotions.activation.capture import RenderedCaptureBackend
from appraisal_emotions.analysis.reveal_rpe import (
    analyze_reveal_rpe,
    build_reveal_rpe_directions,
    build_reveal_rpe_states,
    capture_reveal_states,
    write_reveal_rpe_directions,
    write_reveal_rpe_states,
)
from appraisal_emotions.analysis.symbol_preflight import preflight_symbols
from appraisal_emotions.backends.factory import create_backend, free_backend
from appraisal_emotions.cli_emotions import register as register_emotion_commands
from appraisal_emotions.config import StudyConfig, load_config, resolve_model_spec
from appraisal_emotions.core.util import file_sha256, write_json
from appraisal_emotions.stimuli.gambles import GambleGridConfig
from appraisal_emotions.stimuli.reveal_probes import (
    REVEAL_PROBES_CONTRACT_VERSION,
    build_reveal_battery,
)

app = typer.Typer(add_completion=False, help="Appraisal-direction extraction and analysis.")
console = Console()

ConfigOption = Annotated[Path, typer.Option("--config", help="Path to the run YAML.")]


@app.callback()
def main() -> None:
    """Group callback for the ``appraisal-emotions <command>`` surface.

    Originally here to stop Typer collapsing a one-command app into a bare command; the app now
    also carries the E0-E2 emotion commands, registered from ``cli_emotions`` at import time so
    both surfaces stay in their own module while sharing one entry point.
    """


def _reveal_rpe_paths(cfg: StudyConfig) -> dict[str, Path]:
    root = cfg.run_dir / "reveal_rpe"
    # "states"/"directions" are the metadata JSONs; the writers drop sibling ``.npz`` payloads.
    return {
        "battery": root / "battery.json",
        "states": root / "reveal_states.json",
        "preflight": root / "symbol_preflight.json",
        "report": root / "reveal_rpe_report.json",
        "directions": root / "reveal_directions.json",
    }


@app.command("extract-rpe")
def extract_rpe(config: ConfigOption = Path("configs/reveal_rpe_smoke.yaml")) -> None:
    """Rung R-A: BASE reveal-token signed-RPE measurement (rq8_rpe_valence_spec §4.4).

    Captures base residual-stream states at the byte-pinned outcome-reveal token of the balanced
    reveal battery (read-only; no steering) and tests whether a SIGNED reward-prediction-error
    direction is present and separable from realised reward, EV, and unsigned surprise |RPE|.
    The licence is capped to present-and-separable — R-A has no causal arm and licenses no
    functional / welfare / experience claim, and a re-extraction here does NOT inherit the
    parent program's R-A′ certification. Writes the battery, the hash-bound states artifact, the
    symbol preflight, the gate report, and the fitted per-block directions.
    """

    cfg = load_config(config)
    spec = resolve_model_spec(cfg)
    rr = cfg.reveal_rpe
    partitions = tuple(rr.partitions)
    missing = {"estimation", "selection"} - set(partitions)
    if missing:
        # Fail fast BEFORE backend creation + capture: analyze_reveal_rpe fits on estimation and
        # scores held out on selection, so a one-sided partition set always errors after the
        # (expensive) base-model forwards, discarding the whole capture with no artifact.
        raise typer.BadParameter(
            "extract-rpe needs BOTH the estimation and selection partitions (the estimator fits "
            f"on estimation and scores held-out on selection); missing {sorted(missing)}."
        )
    backend = create_backend(spec)
    try:
        preflight = preflight_symbols(
            backend,
            candidate_shared=rr.shared_symbols,
            candidate_held_out=rr.held_out_symbols,
        )
        if not preflight.passed:
            raise typer.BadParameter(preflight.note)
        grid = GambleGridConfig(
            seed=rr.seed,
            single_shot_per_cell=rr.single_shot_per_cell,
            shared_symbols=preflight.valid_shared,
            held_out_symbols=preflight.valid_held_out,
        )
        battery = build_reveal_battery(
            grid,
            renderings_per_reveal=rr.renderings_per_reveal,
            reward_matched_augment=rr.reward_matched_augment,
        )
        reveals = tuple(
            comparison
            for comparison in battery.reveals
            if str(comparison.metadata.get("partition")) in partitions
        )
        # Mirror analyze_reveal_rpe's >= 2-per-partition requirement HERE, before the expensive
        # base-model capture: a degenerate small config can leave one hash-split side
        # under-filled, and failing only inside the analyzer would waste the whole capture pass.
        n_est = sum(1 for c in reveals if str(c.metadata.get("partition")) == "estimation")
        n_sel = sum(1 for c in reveals if str(c.metadata.get("partition")) == "selection")
        if n_est < 2 or n_sel < 2:
            raise typer.BadParameter(
                f"reveal battery yields estimation={n_est}, selection={n_sel} reveals; the "
                "held-out estimator needs >= 2 in each. Widen the grid / augmentation or change "
                "the seed (checked here to fail before the expensive base-model capture)."
            )
        states = capture_reveal_states(cast(RenderedCaptureBackend, backend), reveals)
    finally:
        free_backend(backend)

    report = analyze_reveal_rpe(
        states,
        reveals,
        seed=rr.seed,
        n_permutations=rr.n_permutations,
        n_random_directions=rr.n_random_directions,
        alpha=rr.alpha,
    )
    artifact = build_reveal_rpe_states(
        states,
        reveals,
        spec=spec,
        seed=rr.seed,
        battery_contract_version=REVEAL_PROBES_CONTRACT_VERSION,
    )
    paths = _reveal_rpe_paths(cfg)
    write_json(paths["battery"], battery)
    battery_sha256 = file_sha256(paths["battery"])
    # Hash-bind the report to the states it scored AND the battery that supplied the design: a
    # stale same-shape report can never lend its selected_block/verdict to a different capture,
    # and a same-ids battery with drifted reward/EV labels can never feed a backfilled refit.
    report = report.model_copy(
        update={
            "states_sha256": artifact.metadata.states_sha256,
            "battery_sha256": battery_sha256,
        }
    )
    write_reveal_rpe_states(artifact, paths["states"])
    write_json(paths["preflight"], preflight)
    write_json(paths["report"], report)
    directions = build_reveal_rpe_directions(
        artifact, reveals, spec=spec, report=report, battery_sha256=battery_sha256
    )
    write_reveal_rpe_directions(directions, paths["directions"])
    console.print(
        f"reveal-RPE R-A: {len(reveals)} reveals ({', '.join(partitions)}) "
        f"-> [cyan]{report.verdict}[/cyan] -> {paths['report']}"
    )
    console.print(
        f"  signed-conjunction={report.signed_conjunction_passed} "
        f"(reward-matched p={report.reward_matched_p}, ev-matched p={report.ev_matched_p}); "
        f"orientation={report.orientation_passed} (cos={report.orientation_cos_reward_ev:.3f}); "
        f"stability={report.stability_passed}; external={report.external_passed}; "
        f"signed-null={report.signed_rpe_null_passed} (p={report.signed_rpe_null_p}); "
        f"|RPE|-axis-present={report.abs_rpe_present}"
    )
    console.print(
        f"  matched cells scored: reward={report.reward_matched_n_scored_cells}/"
        f"{report.reward_matched_n_cells}, ev={report.ev_matched_n_scored_cells}/"
        f"{report.ev_matched_n_cells} — licence capped to PRESENT-AND-SEPARABLE (no use/welfare)."
    )


register_emotion_commands(app)


if __name__ == "__main__":  # pragma: no cover
    app()
