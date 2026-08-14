"""The E0-E3 command surface: ``extract-emotions`` / ``map-geometry`` / ``expectation-control``
/ ``patch-reveals``.

NEW module (no parent counterpart). A sibling of ``cli.py`` rather than four more commands in
it: ``cli.py`` is the ported reveal-RPE surface and stays the size it was extracted at, and the
emotion tiers carry their own artifact paths, options and licence language. :func:`register`
attaches the commands to the one shared Typer app, so ``appraisal-emotions --help`` lists all
five in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console

from appraisal_emotions.activation.capture import StoryCaptureBackend
from appraisal_emotions.analysis.activation_patching import (
    DEFAULT_CONTINUATION_TOKENS,
    DEFAULT_MAX_CONTINUATIONS,
    activation_patching,
    format_activation_patching_summary,
)
from appraisal_emotions.analysis.emotion_mapping import (
    format_map_geometry_summary,
    map_geometry,
)
from appraisal_emotions.analysis.emotion_vectors import (
    EmotionVectors,
    FirstContactFailure,
    GeneratedStory,
    extract_emotion_vectors,
    write_emotion_vectors,
)
from appraisal_emotions.analysis.expectation_control import (
    expectation_control,
    format_expectation_control_summary,
)
from appraisal_emotions.backends.base import PatchedForwardBackend
from appraisal_emotions.backends.factory import create_backend, free_backend
from appraisal_emotions.config import StudyConfig, load_config, resolve_model_spec
from appraisal_emotions.core.util import write_json
from appraisal_emotions.stimuli.emotion_stories import read_emotion_words

console = Console()

ConfigOption = Annotated[Path, typer.Option("--config", help="Path to the run YAML.")]
SeedOption = Annotated[int, typer.Option("--seed", help="Seed for every permutation / bootstrap.")]


def emotion_paths(cfg: StudyConfig) -> dict[str, Path]:
    root = cfg.run_dir / "emotions"
    return {
        "vectors": root / "emotion_vectors.json",
        "stories": root / "stories.json",
        "first_contact": root / "first_contact_sample.json",
    }


def _story_records(stories: tuple[GeneratedStory, ...]) -> list[dict[str, object]]:
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


def _print_e0(artifact: EmotionVectors, path: Path) -> None:
    meta = artifact.metadata
    console.print(
        f"E0 emotion basis: {meta.n_kept}/{meta.n_requested} stories kept over "
        f"{meta.n_word_rows} words + style_control -> [cyan]{meta.gate_verdict}[/cyan] -> {path}"
    )
    console.print(
        f"  drop rate {meta.drop_rate:.3f} {meta.drop_counts_by_reason}; "
        f"first-contact {meta.first_contact_drop_rate:.2f} over n={meta.first_contact_n} "
        f"(filter freeze: {meta.filter_freeze_status})"
    )
    console.print(
        f"  G0: |rho|={meta.g0_abs_rho:.3f} vs threshold {meta.g0_threshold} at block "
        f"{meta.selected_block}/{meta.n_blocks} (p={meta.g0_spearman_p:.4f})"
    )
    console.print(f"  {meta.gate_note}")


def register(app: typer.Typer) -> None:
    """Attach the four emotion-layer commands to the shared app."""

    app.command("extract-emotions")(extract_emotions)
    app.command("map-geometry")(map_geometry_command)
    app.command("expectation-control")(expectation_control_command)
    app.command("patch-reveals")(patch_reveals_command)


def extract_emotions(config: ConfigOption = Path("configs/emotion_vectors_smoke.yaml")) -> None:
    """Rung E0: emotion-concept basis + the G0 sensitivity gate (design §4 E0).

    Generates stories per §5 word ("without naming the emotion"), drops the ones
    that name the target anyway or are too short for the capture window, re-feeds the story text
    alone, and averages the residual stream from ``min_token`` onward — the downscaled Sofroniew
    recipe. G0 (PC1 of the basis vs the valence labels, |rho| >= threshold) is the manipulation
    check for ALL of E1/E2: if it fails, this artifact records ``harness_inadequate`` and every
    downstream null inherits that cap rather than counting against the hypothesis. The basis is a
    measurement instrument; it licenses no claim about what the model feels.
    """

    cfg = load_config(config)
    spec = resolve_model_spec(cfg)
    ev = cfg.emotion_vectors
    words = read_emotion_words(ev.words_file)
    paths = emotion_paths(cfg)
    backend = create_backend(spec)
    try:
        artifact, stories = extract_emotion_vectors(
            cast(StoryCaptureBackend, backend),
            words,
            spec=spec,
            stories_per_emotion=ev.stories_per_emotion,
            seed=ev.seed,
            min_token=ev.min_token,
            max_tokens=ev.max_tokens,
            temperature=ev.temperature,
            g0_threshold=ev.g0_threshold,
            fallback_blocks=ev.fallback_blocks,
            first_contact_n=ev.first_contact_n,
            first_contact_max_drop_rate=ev.first_contact_max_drop_rate,
        )
    except FirstContactFailure as failure:
        # The BLIND filter's compensation: write the sample a human must READ, record the cap,
        # and stop before the capture pass rather than scoring a basis built from the survivors.
        write_json(paths["first_contact"], _story_records(failure.sample))
        console.print(f"[red]E0 harness_inadequate[/red]: {failure}")
        console.print(f"  first-contact sample written to {paths['first_contact']}")
        raise typer.Exit(code=1) from failure
    finally:
        free_backend(backend)

    write_json(paths["stories"], _story_records(stories))
    write_json(paths["first_contact"], _story_records(stories[: ev.first_contact_n]))
    write_emotion_vectors(artifact, paths["vectors"])
    _print_e0(artifact, paths["vectors"])


def map_geometry_command(
    directions: Annotated[Path, typer.Option("--directions", help="reveal_directions.json")],
    emotions: Annotated[Path, typer.Option("--emotions", help="emotion_vectors.json")],
    out: Annotated[Path, typer.Option("--out", help="Report JSON path.")],
    words: Annotated[Path, typer.Option("--words")] = Path("data/emotion_words.json"),
    norms: Annotated[
        Path | None, typer.Option("--norms", help="Fetched VAD subset CSV (optional).")
    ] = None,
    seed: SeedOption = 7,
    permutations: Annotated[int, typer.Option("--permutations")] = 10_000,
    null_draws: Annotated[int, typer.Option("--null-draws")] = 1_000,
) -> None:
    """Rung E1: valence-residual geometry — the headline (design §4 E1).

    EVERY word's valence residual is reported. The §5 recorded expectations — the two family
    contrasts, the three-level outcome ordering, the three named pairs — additionally carry an
    effect size and a one-sided permutation p in their recorded direction; there is no
    multiple-comparison correction and no confirmatory/exploratory caste. A flat residual with G0
    or P5c failed is ``harness_inadequate``, NOT evidence against appraisal inheritance. Licence
    capped at present-and-separable.
    """

    report = map_geometry(
        directions,
        emotions,
        words,
        norms,
        seed=seed,
        n_permutations=permutations,
        n_null_draws=null_draws,
    )
    write_json(out, report)
    console.print(format_map_geometry_summary(report))
    console.print(f"\nwritten to {out}")


def expectation_control_command(
    states: Annotated[Path, typer.Option("--states", help="reveal_states.json")],
    battery: Annotated[Path, typer.Option("--battery", help="battery.json")],
    emotions: Annotated[Path, typer.Option("--emotions", help="emotion_vectors.json")],
    out: Annotated[Path, typer.Option("--out", help="Report JSON path.")],
    block: Annotated[
        int | None, typer.Option("--block", help="Defaults to the emotion artifact's block.")
    ] = None,
    seed: SeedOption = 7,
    permutations: Annotated[int, typer.Option("--permutations")] = 10_000,
) -> None:
    """Rung E2: expectation vs situation on the reward-matched cells (design §4 E2).

    Projects reveal-token states onto the emotion valence axis and onto the elated/disappointed
    contrast, then regresses on signed RPE WITHIN outcome-fixed cells with a cluster-aware
    permutation null. A sign-congruent within-cell slope favours expectation-tracking; a null
    within cells is what the situational-context rival predicts. Licence capped at
    present-and-separable — E2 has no causal arm.
    """

    report = expectation_control(
        states, battery, emotions, block=block, seed=seed, n_permutations=permutations
    )
    write_json(out, report)
    console.print(format_expectation_control_summary(report))
    console.print(f"\nwritten to {out}")


def patch_reveals_command(
    states: Annotated[Path, typer.Option("--states", help="reveal_states.json")],
    battery: Annotated[Path, typer.Option("--battery", help="battery.json")],
    directions: Annotated[Path, typer.Option("--directions", help="reveal_directions.json")],
    emotions: Annotated[Path, typer.Option("--emotions", help="emotion_vectors.json")],
    out: Annotated[Path, typer.Option("--out", help="Report JSON path.")],
    mode: Annotated[
        str, typer.Option("--mode", help="'state' (zero forwards) or 'forward' (causal tier).")
    ] = "state",
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Run YAML naming the model. REQUIRED for --mode forward."),
    ] = None,
    block: Annotated[
        int | None, typer.Option("--block", help="Defaults to the emotion artifact's block.")
    ] = None,
    seed: SeedOption = 7,
    max_pairs: Annotated[
        int | None, typer.Option("--max-pairs", help="Defaults to 480 (state) / 60 (forward).")
    ] = None,
    random_draws: Annotated[
        int | None,
        typer.Option("--random-draws", help="Random-direction floor draws; 200 / 5 by mode."),
    ] = None,
    continuation_tokens: Annotated[
        int,
        typer.Option("--continuation-tokens", help="Greedy continuation cap; 0 disables."),
    ] = DEFAULT_CONTINUATION_TOKENS,
    max_continuations: Annotated[
        int,
        typer.Option("--max-continuations", help="Pairs that get continuations stored."),
    ] = DEFAULT_MAX_CONTINUATIONS,
) -> None:
    """Rung E3: in-distribution activation patching on the reward-matched cells (design §4 E3).

    Substitutes a donor reveal-token state into a recipient from the same reward-matched cell
    (same realised reward and |RPE|, opposite signed RPE, same template family and outcome symbol)
    across four arms: the full-residual ceiling, the certified ``v_rpe`` component, a
    random-direction floor, and a same-condition donor as the no-op control.

    ``--mode state`` is the zero-forward preview: pure arithmetic over the captured states, no
    model, capped at present-and-separable. ``--mode forward`` is the causal tier — it re-runs each
    recipient's real prompt with the value substituted at the reveal token and reads the emotion
    axes downstream, where the patch has propagated; that is the only tier that can earn
    functionally-used, and it needs ``--config`` to name the model. Continuations are stored raw
    and UNSCORED: no grader exists and none may be written before a reality sample of that surface.
    """

    backend = None
    spec = None
    if mode == "forward":
        if config is None:
            raise typer.BadParameter("--mode forward needs --config to name the model")
        spec = resolve_model_spec(load_config(config))
        backend = create_backend(spec)
    try:
        report = activation_patching(
            states,
            battery,
            directions,
            emotions,
            mode=mode,
            backend=cast(PatchedForwardBackend | None, backend),
            model_key=None if spec is None else spec.key,
            block=block,
            seed=seed,
            max_pairs=max_pairs,
            n_random_draws=random_draws,
            continuation_tokens=continuation_tokens,
            max_continuations=max_continuations,
        )
    finally:
        if backend is not None:
            free_backend(backend)
    write_json(out, report)
    console.print(format_activation_patching_summary(report))
    console.print(f"\nwritten to {out}")
