"""The E0-E4 command surface: ``extract-emotions`` / ``extract-story-projections`` /
``map-geometry`` / ``expectation-control`` / ``patch-reveals`` / ``behavioral-transfer``.

NEW module (no parent counterpart). A sibling of ``cli.py`` rather than four more commands in
it: ``cli.py`` is the ported reveal-RPE surface and stays the size it was extracted at, and the
emotion tiers carry their own artifact paths, options and licence language. :func:`register`
attaches the commands to the one shared Typer app, so ``appraisal-emotions --help`` lists them
all in one place.
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
from appraisal_emotions.analysis.behavioral_transfer import (
    DEFAULT_MAX_PAIRS,
    DEFAULT_MAX_PER_CELL,
    DEFAULT_PERMUTATIONS,
    DEFAULT_RANDOM_DRAWS,
    DEFAULT_REACHABILITY_PAIRS,
    ChatPatchingBackend,
    behavioral_transfer,
    format_behavioral_transfer_summary,
)
from appraisal_emotions.analysis.emotion_mapping import (
    format_map_geometry_summary,
    map_geometry,
)
from appraisal_emotions.analysis.emotion_vectors import (
    EmotionVectors,
    FirstContactFailure,
    extract_emotion_vectors,
    read_emotion_vectors,
    read_story_log,
    story_log_records,
    write_emotion_vectors,
)
from appraisal_emotions.analysis.expectation_control import (
    expectation_control,
    format_expectation_control_summary,
)
from appraisal_emotions.analysis.reveal_rpe import read_reveal_rpe_directions
from appraisal_emotions.analysis.story_projections import (
    StoryProjectionGateFailure,
    extract_story_projections,
    story_projections_path,
    write_story_projections,
)
from appraisal_emotions.backends.base import PatchedForwardBackend
from appraisal_emotions.backends.factory import create_backend, free_backend
from appraisal_emotions.config import StudyConfig, load_config, resolve_model_spec
from appraisal_emotions.core.util import file_sha256, write_json
from appraisal_emotions.stimuli.decision_probes import ANSWER_FORMS, AnswerForm
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
        "projections": root / "story_projections.json",
        "projections_quarantine": root / "story_projections_gate_failed.json",
    }


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
    """Attach the emotion-layer commands to the shared app."""

    app.command("extract-emotions")(extract_emotions)
    app.command("extract-story-projections")(extract_story_projections_command)
    app.command("map-geometry")(map_geometry_command)
    app.command("expectation-control")(expectation_control_command)
    app.command("patch-reveals")(patch_reveals_command)
    app.command("behavioral-transfer")(behavioral_transfer_command)


def behavioral_transfer_command(
    config: ConfigOption,
    states: Annotated[Path, typer.Option("--states", help="reveal_states.json")],
    battery: Annotated[Path, typer.Option("--battery", help="battery.json")],
    directions: Annotated[Path, typer.Option("--directions", help="reveal_directions.json")],
    emotions: Annotated[Path, typer.Option("--emotions", help="emotion_vectors.json")],
    out: Annotated[Path, typer.Option("--out", help="Report JSON path.")],
    # REQUIRED, and deliberately so. The answer form is a measurement -- which of ` KER` and `KER`
    # the model actually continues with at a slot that follows the template's own newlines -- and
    # the run cannot infer it. Both forms are single-token by construction (the pool is gated on
    # both), so guessing wrong does not raise: it reads a margin between two tokens the model was
    # never going to emit, B0 nulls, and the report writes that up as a fact about the model. A
    # default here would be the whole failure, so there is none. `e4-surface-preflight` prints the
    # value to pass.
    answer_form: Annotated[
        str,
        typer.Option(
            "--answer-form",
            help="Answer-slot token form, SET BY THE REALITY SAMPLE: bare | leading_space.",
        ),
    ],
    block: Annotated[
        int | None,
        typer.Option("--block", help="Patch block; defaults to the DIRECTIONS artifact's."),
    ] = None,
    seed: SeedOption = 7,
    max_pairs: Annotated[int, typer.Option("--max-pairs")] = DEFAULT_MAX_PAIRS,
    max_per_cell: Annotated[
        int,
        typer.Option("--max-per-cell", help="Cap per reward cell BEFORE --max-pairs truncates."),
    ] = DEFAULT_MAX_PER_CELL,
    random_draws: Annotated[
        int, typer.Option("--random-draws", help="Random-direction floor draws.")
    ] = DEFAULT_RANDOM_DRAWS,
    permutations: Annotated[int, typer.Option("--permutations")] = DEFAULT_PERMUTATIONS,
    reachability_pairs: Annotated[
        int,
        typer.Option(
            "--reachability-pairs",
            help="Off-cell pairs for the positive control that makes every null readable.",
        ),
    ] = DEFAULT_REACHABILITY_PAIRS,
) -> None:
    """Rung E4: does the patched expectation state change what the model DOES (e4-prereg)?

    Extends the byte-pinned reveal prompt into a three-message chat — reveal, the model's own
    measured completion, then a decision probe — and reads a next-token logit MARGIN at the answer
    slot with zero decode steps. Two things separate it from E3's behavioural leg. The readout sits
    at a LATER token than the patch, where the residual stream's identity path cannot reach it, so
    a shift means an attention head read the patched value; and the transfer fraction's denominator
    is MEASURED first, unpatched, by the B0 gate. If B0 fails, no patched forward is spent and the
    window records harness_inadequate rather than a null nobody can read.

    Always a GPU run: there is no state-mode preview, because a state-mode preview is exactly the
    same-position arithmetic this rung exists to escape.
    """

    if answer_form not in ANSWER_FORMS:
        raise typer.BadParameter(
            f"--answer-form must be one of {list(ANSWER_FORMS)}, not {answer_form!r}. It is set by "
            "the reality sample: run `just e4-surface-preflight` without --tokenizer-only and pass "
            "the form it prints."
        )
    spec = resolve_model_spec(load_config(config))
    backend = create_backend(spec)
    try:
        report = behavioral_transfer(
            states,
            battery,
            directions,
            emotions,
            backend=cast(ChatPatchingBackend, backend),
            model_key=spec.key,
            block=block,
            seed=seed,
            max_pairs=max_pairs,
            max_per_cell=max_per_cell,
            n_random_draws=random_draws,
            n_permutations=permutations,
            n_reachability_pairs=reachability_pairs,
            answer_form=cast(AnswerForm, answer_form),
        )
    finally:
        free_backend(backend)
    write_json(out, report)
    console.print(format_behavioral_transfer_summary(report))
    console.print(f"\nwritten to {out}")


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
        write_json(paths["first_contact"], story_log_records(failure.sample))
        console.print(f"[red]E0 harness_inadequate[/red]: {failure}")
        console.print(f"  first-contact sample written to {paths['first_contact']}")
        raise typer.Exit(code=1) from failure
    finally:
        free_backend(backend)

    write_json(paths["stories"], story_log_records(stories))
    write_json(paths["first_contact"], story_log_records(stories[: ev.first_contact_n]))
    write_emotion_vectors(artifact, paths["vectors"])
    _print_e0(artifact, paths["vectors"])


def extract_story_projections_command(
    config: ConfigOption = Path("configs/emotion_vectors_smoke.yaml"),
    directions: Annotated[
        Path, typer.Option("--directions", help="reveal_directions.json at the SAME model key")
    ] = Path("runs/reveal_rpe_base/reveal_rpe/reveal_directions.json"),
) -> None:
    """Rung P1: per-story projections onto the fixed directions — E0's within-word error term.

    Re-feeds the stories E0 already generated and stored, with no generation and no new stimulus,
    and keeps one scalar per story x block x direction plus the same-word Gram. That buys the
    split-half reliability, the variance decomposition and the attenuation ceiling that E0's
    word-mean-only artifact cannot support, at ~8 MB instead of the 5 GB of residual states.

    P1 runs NO new test of the family contrast: the point estimate and its p are already fixed by
    E1's data and are not recomputed here. It measures how much of E1's word-level noise is
    within-word (fixable by more stories) rather than between-word (not), which is what decides
    whether a larger capture is worth renting a GPU for.
    """

    cfg = load_config(config)
    spec = resolve_model_spec(cfg)
    paths = emotion_paths(cfg)
    emotion = read_emotion_vectors(paths["vectors"])
    stories = read_story_log(paths["stories"])
    reveal = read_reveal_rpe_directions(directions)
    if reveal.metadata.model_key != spec.key:
        raise typer.BadParameter(
            f"directions are for model {reveal.metadata.model_key!r}, config resolves "
            f"{spec.key!r}; the geometry comparison needs matched model identity"
        )

    backend = create_backend(spec)
    try:
        artifact = extract_story_projections(
            cast(StoryCaptureBackend, backend),
            stories,
            emotion,
            reveal,
            spec=spec,
            words_file_sha256=file_sha256(cfg.emotion_vectors.words_file),
            min_token=cfg.emotion_vectors.min_token,
            fallback_blocks=cfg.emotion_vectors.fallback_blocks,
        )
    except StoryProjectionGateFailure as failure:
        # Quarantine rather than discard: the capture is a rented-GPU cost and the recorded
        # deviation is the only evidence about WHY the gate fired. Nothing downstream reads
        # this path, so a quarantined artifact cannot leak into an analysis.
        write_story_projections(failure.artifact, paths["projections_quarantine"])
        console.print(f"[red]P1 harness_inadequate[/red]: {failure}")
        console.print(f"  quarantined capture written to {paths['projections_quarantine']}")
        raise typer.Exit(code=1) from failure
    finally:
        free_backend(backend)

    write_story_projections(artifact, paths["projections"])
    meta = artifact.metadata
    console.print(
        f"P1 story projections: {meta.n_stories} stories x {meta.n_blocks} blocks x "
        f"{len(meta.direction_names)} directions -> {paths['projections']}"
    )
    console.print(
        f"  faithfulness gate: max relative deviation {meta.gate_max_relative_deviation:.3g} "
        f"<= {meta.gate_max_relative_deviation_threshold} (word-mean correlation "
        f"{meta.gate_word_mean_correlation:.6f})"
    )
    console.print(f"  payload {story_projections_path(paths['projections'])}")


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
    """Rungs E2 + E2b: expectation vs situation, and expectation vs comparison (design §4 E2;
    ``docs/design/e2b-prereg.md``).

    Projects reveal-token states onto the emotion valence axis and onto the elated/disappointed
    contrast, then regresses on signed RPE WITHIN matched cells with a cluster-aware permutation
    null — over BOTH matched families. Reward-matched cells pin the outcome and vary the stated
    EV; EV-matched cells do the reverse. One arm alone cannot tell a comparison from a code for
    the quantity that arm happens to vary, so the verdict is the conjunction reported in
    ``comparison_signature``. Licence capped at present-and-separable — neither arm has a causal
    leg.
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
