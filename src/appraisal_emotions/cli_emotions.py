"""The E0-E3 command surface: ``extract-emotions`` / ``extract-story-projections`` /
``map-geometry`` / ``expectation-control`` / ``patch-reveals``.

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
    E0Refusal,
    EmotionVectors,
    GeneratedStory,
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
from appraisal_emotions.stimuli.emotion_stories import read_emotion_words

console = Console()

ConfigOption = Annotated[Path, typer.Option("--config", help="Path to the run YAML.")]
SeedOption = Annotated[int, typer.Option("--seed", help="Seed for every permutation / bootstrap.")]


def emotion_paths(cfg: StudyConfig) -> dict[str, Path]:
    root = cfg.run_dir / "emotions"
    return {
        "vectors": root / "emotion_vectors.json",
        "stories": root / "stories.json",
        "neutral_dialogues": root / "neutral_dialogues.json",
        "first_contact": root / "first_contact_sample.json",
        "neutral_first_contact": root / "neutral_first_contact_sample.json",
        "projections": root / "story_projections.json",
        "projections_quarantine": root / "story_projections_gate_failed.json",
    }


def _write_corpora(
    paths: dict[str, Path],
    stories: tuple[GeneratedStory, ...],
    neutral: tuple[GeneratedStory, ...],
    *,
    n: int,
) -> list[Path]:
    """Write every generated corpus and its early-N sample; return what was written.

    The SAME four files on the success path and on every refusal path, because the operator's
    question is identical either way ("what did the model actually produce?") and a refusal is
    when they most need the answer. The neutral corpus keeps its own file rather than being
    appended to the story log: P1 re-feeds that log as emotion stories, and a neutral transcript
    is not a story of any emotion.
    """

    written: list[Path] = []
    if stories:
        write_json(paths["stories"], story_log_records(stories))
        write_json(paths["first_contact"], story_log_records(stories[:n]))
        written += [paths["stories"], paths["first_contact"]]
    if neutral:
        write_json(paths["neutral_dialogues"], story_log_records(neutral))
        write_json(paths["neutral_first_contact"], story_log_records(neutral[:n]))
        written += [paths["neutral_dialogues"], paths["neutral_first_contact"]]
    return written


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
    # Realised against configured, printed rather than left for a reader of the JSON: a drop rate
    # of 0.00 says nothing about whether the splitter returned the sample that was asked for.
    console.print(
        f"  stories per label: {meta.stories_per_emotion} configured, "
        f"{min(meta.generated_by_label.values())}-{max(meta.generated_by_label.values())} "
        f"generated, {min(meta.kept_by_label.values())}-{max(meta.kept_by_label.values())} kept"
    )
    if meta.neutral_projection is not None:
        proj = meta.neutral_projection
        best_before = max(
            (abs(row.spearman_rho) for row in meta.g0_table_before_projection), default=0.0
        )
        best_random = max(
            (abs(row.spearman_rho) for row in meta.g0_table_random_projection), default=0.0
        )
        console.print(
            f"  neutral projection: {proj.total_components} components over "
            f"{len(proj.blocks)} blocks (<= {proj.max_components_in_a_block} per block) at "
            f"{proj.variance_target:.0%} of variance"
        )
        # Three numbers, because a corpus can fall short at two different places: the splitter
        # returning fewer pieces than were asked for, and the filter dropping what did come back.
        console.print(
            f"    corpus: {proj.n_transcripts_configured} configured over "
            f"{proj.n_calls_requested} calls, {proj.n_pieces_obtained} obtained, "
            f"{proj.n_kept} kept (floor {proj.min_kept}); drop rate {proj.drop_rate:.3f} "
            f"(ceiling {proj.max_drop_rate:.2f}); first-contact "
            f"{proj.first_contact_drop_rate:.2f} over n={proj.first_contact_n}"
        )
        # Three numbers, not two: a random frame of the same width is what says whether the move
        # from `before` to the gate line above belongs to the neutral subspace or to removing k
        # directions at all.
        console.print(
            f"    G0 |rho| before projection {best_before:.3f}; "
            f"random {proj.max_components_in_a_block}-frame control {best_random:.3f}"
        )
    console.print(
        f"  G0: |rho|={meta.g0_abs_rho:.3f} vs threshold {meta.g0_threshold} at block "
        f"{meta.selected_block}/{meta.n_blocks} (p={meta.g0_spearman_p:.4f})"
    )
    console.print(f"  {meta.gate_note}")


def register(app: typer.Typer) -> None:
    """Attach the four emotion-layer commands to the shared app."""

    app.command("extract-emotions")(extract_emotions)
    app.command("extract-story-projections")(extract_story_projections_command)
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
        artifact, stories, neutral = extract_emotion_vectors(
            cast(StoryCaptureBackend, backend),
            words,
            spec=spec,
            stories_per_emotion=ev.stories_per_emotion,
            story_recipe=ev.story_recipe,
            sofroniew_stories_per_call=ev.sofroniew_stories_per_call,
            sofroniew_data_dir=ev.sofroniew_data_dir,
            neutral_projection=ev.neutral_projection,
            neutral_dialogues=ev.neutral_dialogues,
            neutral_dialogues_per_call=ev.neutral_dialogues_per_call,
            neutral_variance_target=ev.neutral_variance_target,
            neutral_min_kept=ev.neutral_min_kept,
            neutral_max_drop_rate=ev.neutral_max_drop_rate,
            seed=ev.seed,
            min_token=ev.min_token,
            max_tokens=ev.max_tokens,
            temperature=ev.temperature,
            g0_threshold=ev.g0_threshold,
            fallback_blocks=ev.fallback_blocks,
            first_contact_n=ev.first_contact_n,
            first_contact_max_drop_rate=ev.first_contact_max_drop_rate,
        )
    except E0Refusal as failure:
        # QUARANTINE, not discard. All three refusals fire after generation and before the
        # verdict, so the generations are already paid for and they are the only evidence for why
        # the run refused — and two of the messages send the operator to go read a sample. Write
        # the corpora first, then exit non-zero: same shape as the P1 gate handler below.
        written = _write_corpora(paths, failure.stories, failure.neutral, n=ev.first_contact_n)
        console.print(f"[red]E0 harness_inadequate[/red]: {failure}")
        for path in written:
            console.print(f"  quarantined: {path}")
        raise typer.Exit(code=1) from failure
    finally:
        free_backend(backend)

    _write_corpora(paths, stories, neutral, n=ev.first_contact_n)
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
