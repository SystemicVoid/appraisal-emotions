"""E1 — valence-residual geometry between appraisal directions and emotion concepts (design §4).

NEW module (no parent counterpart). Pure CPU over two hash-bound artifacts: the reveal-RPE
directions (``v_rpe``, ``v_ev``, ``v_absrpe``) and the E0 emotion basis. No model, no forwards.
The per-block estimands live in ``analysis.valence_residual``; this module chooses the blocks and
the valence scale, runs the nulls, and assembles the report.

The steelmanned claim (§1) is NOT "``v_RPE`` points at positive emotions" — that is near-certain
and uninformative, since any positive-value direction has a positive-valence component. It is
that after removing the valence axis the emotion space is STILL organized by appraisal
variables. So every readout is a *residual* readout, and EVERY word's residual is tabled:

- **P1 (sanity, not result):** ``rho(cos(v_RPE, e_j), valence_j) > 0`` over all words. Expected
  true; a failure indicts the extraction, not the theory.
- **The recorded §5 expectations**, each with an effect size and a cheap one-sided permutation p
  in its recorded direction: the two **family contrasts** (outcome-disconfirmation words vs
  valence-matched controls, per pole — the powered statistic), the three-level **outcome
  ordering** (positive-disconfirmation > confirmation > negative-disconfirmation), and the three
  **named pairs** (``disappointed < sad``, ``relieved > calm``, ``elated > content``), reported as
  three individual points, each permuted inside the §5 same-pole pool (the outcome / non-outcome
  / confirmation families, n = 22 positive and 20 negative — not every word of that valence). No
  Holm correction and no confirmatory caste (operator decision, 2026-08-13): the expectations
  record what we thought before we looked, and the data is analysed directly.
- **P4:** ``v_absrpe`` prefers the surprise family over arousal-matched valenced controls, and
  loads PC2 rather than PC1. *Stated every time:* ``v_RPE ⊥ v_absrpe`` holds by construction of
  the reveal design matrix; the informative half is which WORDS ``v_absrpe`` picks out.
- **P5a (expectation of absence):** ``sad`` shows no RPE-excess after valence partialling. It
  holds by the word's residual sitting inside the word residuals' own spread — the same p95 band
  P5c uses, for the same reason: one measurement per word means there is no within-word sampling
  error to interval. It only means something while the family contrasts are positive.
- **P5c (scale control):** the valence-free ``style_control`` vector shows ~0 residual alignment
  with all three directions. If it fails, the cosine scale itself is invalid and E1 records
  ``harness_inadequate`` (design §4 E1.5c) — not evidence about inheritance.

What makes a null readable is kept, all of it: the inherited G0 sensitivity gate, P5c, and two
floors against the anisotropy threat (§6) — label-shuffled emotion vectors and norm-matched
random directions, both resampling the headline family contrast. Comparisons are per-block
mean-centred, and the full block sweep is reported rather than only the two artifacts' selected
blocks (site/object mismatch is a finding, not something to hide).

Licence: present-and-separable at most. Geometry is not use, and no outcome here licenses
welfare, sentience or experience claims.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import numpy as np

from appraisal_emotions.analysis.direction_stats import seed_int
from appraisal_emotions.analysis.emotion_vectors import read_emotion_vectors
from appraisal_emotions.analysis.reveal_rpe import DIRECTION_FAMILIES, read_reveal_rpe_directions
from appraisal_emotions.analysis.valence_residual import (
    BlockGeometry,
    ExpectedPairResult,
    FamilyContrastResult,
    OutcomeOrderingResult,
    P4Result,
    P5aResult,
    P5cStyleResult,
    block_geometry,
    cosines,
    expected_pair_results,
    family_contrast_results,
    family_contrast_statistics,
    family_rows,
    label_shuffled_floor,
    norm_fraction,
    ols_residuals,
    outcome_ordering_result,
    p4_surprise_contrast,
    p5a_discriminant_null,
    p5c_style_control,
    pair_pool_rows,
    random_direction_residual_floor,
    spearman,
    two_sample_statistic,
)
from appraisal_emotions.core.schema import StrictModel
from appraisal_emotions.core.util import EXTRACTION_SEED, file_sha256
from appraisal_emotions.stimuli.emotion_stories import EmotionWordSet, read_emotion_words

__all__ = [
    "MAP_GEOMETRY_CONTRACT_VERSION",
    "BlockReport",
    "BlockSweepRow",
    "MapGeometryReport",
    "WordResidual",
    "format_map_geometry_summary",
    "map_geometry",
    "read_valence_norms",
]

MAP_GEOMETRY_CONTRACT_VERSION = "map_geometry/v2"

# --------------------------------------------------------------------------------------
# Numeric norms (fetched, never transcribed — scripts/fetch_norms.py)
# --------------------------------------------------------------------------------------


def read_valence_norms(
    path: Path, labels: tuple[str, ...]
) -> tuple[np.ndarray | None, int, tuple[str, ...]]:
    """Numeric valence norms for ``labels`` from the fetched subset CSV, or ``None``.

    Returns ``(valence array aligned to labels, covered word count, uncovered words)``. Coverage is
    ALL-OR-NOTHING by design: a numeric scale for some words and a binary label for others would
    make the residuals incommensurable across the word set, and every readout here is a comparison
    BETWEEN words. Partial coverage therefore falls the whole set back to the binary labels and
    the report names the words that blocked the upgrade, so the fix is a lookup, not a mystery.
    """

    values: dict[str, float] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            word = str(row["word"]).strip().lower()
            raw = str(row.get("valence", "")).strip()
            if word and raw:
                values[word] = float(raw)
    missing = tuple(label for label in labels if label not in values)
    covered = len(labels) - len(missing)
    if missing:
        return None, covered, missing
    return np.asarray([values[label] for label in labels], dtype=float), covered, ()


# --------------------------------------------------------------------------------------
# Report schema
# --------------------------------------------------------------------------------------


class WordResidual(StrictModel):
    """One word's raw cosine and its valence residual on ``v_rpe`` — every word is tabled."""

    word: str
    family: str
    valence: int
    cos_v_rpe: float
    residual: float


class BlockReport(StrictModel):
    """Everything E1 reads at one block. One block report — no confirmatory/exploratory caste."""

    block: int
    valence_source: str
    p1_spearman_rho: float
    p1_spearman_p: float
    p1_positive: bool
    word_residuals: tuple[WordResidual, ...]
    family_contrasts: tuple[FamilyContrastResult, ...]
    outcome_ordering: OutcomeOrderingResult
    expected_pairs: tuple[ExpectedPairResult, ...]
    p4: P4Result
    p5a: P5aResult
    p5c: tuple[P5cStyleResult, ...]
    p5c_passed: bool
    mean_cos_by_family: dict[str, float]
    subspace_fraction_emotion_span: float
    subspace_fraction_pc12_plane: float
    observed_family_contrast_max: float
    label_shuffled_p95: float
    random_direction_p95: float
    clears_both_floors: bool


class BlockSweepRow(StrictModel):
    """One block of the descriptive depth profile (no permutations)."""

    block: int
    p1_spearman_rho: float
    family_contrast_statistics: dict[str, float]
    p4_contrast: float
    cos_absrpe_pc1: float
    cos_absrpe_pc2: float


class MapGeometryReport(StrictModel):
    """E1 record: every word's residual, the §5 expectation readouts, floors, and the gate cap."""

    artifact_contract_version: Literal["map_geometry/v2"] = MAP_GEOMETRY_CONTRACT_VERSION
    seed: int
    n_permutations: int
    n_null_draws: int
    directions_sha256: str
    directions_selected_block: int
    directions_source_verdict: str
    emotion_vectors_sha256: str
    emotion_selected_block: int
    words_file_sha256: str
    norms_file: str | None
    norms_covered_words: int
    norms_missing_words: tuple[str, ...]
    valence_source: str
    headline_blocks: tuple[int, ...]
    sensitivity_gate: str
    verdict_cap: str
    blocks: tuple[BlockReport, ...]
    block_sweep: tuple[BlockSweepRow, ...]
    notes: tuple[str, ...]


_RECORDED_EXPECTATIONS_NOTE = (
    "RECORDED EXPECTATIONS, not a pre-registration contract (operator decision, 2026-08-13): the "
    "§5 directions were written down before extraction, every word's residual is reported, "
    "effect sizes lead and p-values follow, and there is no multiple-comparison correction. "
    "Reporting a result the expectations did not anticipate is fine; quietly rewriting an "
    "expectation is not — see the symmetric-amendment test in docs/agents/rails.md."
)
_READOUT_NOTE = (
    "SIGN AND POWER: v_RPE is a SIGNED axis, so OCC / decision affect puts "
    "positive-disconfirmation concepts on its positive pole and negative-disconfirmation "
    "concepts on its negative pole. Every expectation statistic is expected_sign * (outcome - "
    "control), so positive always means the recorded split — disappointed sits BELOW sad. The "
    "named pairs are three individual points, each permuted inside the §5 same-pole pool (the "
    "outcome / non-outcome / confirmation families), so their p floors at 1/(n(n-1)) for a pool "
    "of size n — n_pool is reported per pair. The powered statistic is the family contrast, "
    "which pools a whole family per side."
)


# --------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------


def _designs(
    binary_valence: np.ndarray, numeric_valence: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, str]:
    ones = np.ones(binary_valence.size)
    binary_design = np.column_stack([ones, binary_valence])
    if numeric_valence is None:
        return binary_design, binary_design, "binary_project_labels"
    return np.column_stack([ones, numeric_valence]), binary_design, "numeric_norms"


def _block_report(
    geometry: BlockGeometry,
    words: EmotionWordSet,
    index: dict[str, int],
    rows: dict[str, np.ndarray],
    pool: np.ndarray,
    designs: tuple[np.ndarray, np.ndarray, str],
    binary_valence: np.ndarray,
    valence: np.ndarray,
    *,
    seed: int,
    n_permutations: int,
    n_null_draws: int,
) -> BlockReport:
    design, binary_design, valence_source = designs
    cos = geometry.cos["v_rpe"]
    residuals = ols_residuals(cos, design)
    contrasts = words.expected_family_contrasts()
    categories = words.category_by_word

    def headline(candidate: np.ndarray) -> float:
        return max(family_contrast_statistics(candidate, contrasts, rows))

    def rng(name: str) -> np.random.Generator:
        return np.random.default_rng(seed_int(seed, f"map-geometry-{name}", geometry.block))

    rho, p_value = spearman(cos, valence)
    observed_max = headline(residuals)
    label_floor = label_shuffled_floor(
        geometry,
        design,
        headline,
        rng=rng("label-null"),
        n_draws=n_null_draws,
    )
    direction_floor = random_direction_residual_floor(
        geometry,
        design,
        headline,
        rng=rng("direction-null"),
        n_draws=n_null_draws,
    )
    return BlockReport(
        block=geometry.block,
        valence_source=valence_source,
        p1_spearman_rho=rho,
        p1_spearman_p=p_value,
        p1_positive=rho > 0.0,
        word_residuals=tuple(
            WordResidual(
                word=label,
                family=categories[label],
                valence=int(binary_valence[position]),
                cos_v_rpe=float(cos[position]),
                residual=float(residuals[position]),
            )
            for label, position in index.items()
        ),
        family_contrasts=family_contrast_results(
            residuals,
            contrasts,
            rows,
            rng=rng("family"),
            n_permutations=n_permutations,
        ),
        outcome_ordering=outcome_ordering_result(
            residuals,
            words.outcome_ordering(),
            rows,
            rng=rng("ordering"),
            n_permutations=n_permutations,
        ),
        expected_pairs=expected_pair_results(
            residuals,
            words.expected_pairs(),
            binary_valence,
            index,
            pool,
            rng=rng("pairs"),
            n_permutations=n_permutations,
        ),
        p4=p4_surprise_contrast(
            geometry,
            words,
            index,
            rng=rng("p4"),
            n_permutations=n_permutations,
        ),
        p5a=p5a_discriminant_null(geometry, str(words.expectations["p5a"]), index, design),
        p5c=(p5c := p5c_style_control(geometry, binary_design)),
        p5c_passed=all(entry.passed for entry in p5c),
        mean_cos_by_family={
            family: float(geometry.cos[family].mean()) for family in DIRECTION_FAMILIES
        },
        subspace_fraction_emotion_span=norm_fraction(
            geometry.family_vectors["v_rpe"], geometry.centered_words
        ),
        subspace_fraction_pc12_plane=norm_fraction(
            geometry.family_vectors["v_rpe"], geometry.components[:2]
        ),
        observed_family_contrast_max=observed_max,
        label_shuffled_p95=label_floor,
        random_direction_p95=direction_floor,
        clears_both_floors=bool(observed_max > label_floor and observed_max > direction_floor),
    )


def _sweep_row(
    geometry: BlockGeometry,
    words: EmotionWordSet,
    index: dict[str, int],
    rows: dict[str, np.ndarray],
    design: np.ndarray,
    binary_valence: np.ndarray,
    valence: np.ndarray,
) -> BlockSweepRow:
    residuals = ols_residuals(geometry.cos["v_rpe"], design)
    contrasts = words.expected_family_contrasts()
    statistics = family_contrast_statistics(residuals, contrasts, rows)
    cos = geometry.cos["v_absrpe"]
    left = np.asarray([index[word] for word in words.p4_words("surprise")])
    right = np.asarray([index[word] for word in words.p4_words("arousal_matched")])
    loading = cosines(geometry.components[:2], geometry.family_vectors["v_absrpe"])
    return BlockSweepRow(
        block=geometry.block,
        p1_spearman_rho=spearman(geometry.cos["v_rpe"], valence)[0],
        family_contrast_statistics={
            contrast.pole: value for contrast, value in zip(contrasts, statistics, strict=True)
        },
        p4_contrast=two_sample_statistic(cos, left, right, 1),
        cos_absrpe_pc1=float(loading[0]),
        cos_absrpe_pc2=float(loading[1]),
    )


def map_geometry(
    directions_artifact: Path,
    emotion_artifact: Path,
    words_file: Path,
    norms_csv: Path | None = None,
    *,
    seed: int = EXTRACTION_SEED,
    n_permutations: int = 10_000,
    n_null_draws: int = 1_000,
) -> MapGeometryReport:
    """Run E1 over the two artifacts; headline at BOTH selected blocks, sweep over every block."""

    directions = read_reveal_rpe_directions(Path(directions_artifact))
    emotion = read_emotion_vectors(Path(emotion_artifact))
    words = read_emotion_words(Path(words_file))
    if directions.metadata.hidden_size != emotion.metadata.hidden_size:
        raise ValueError("directions and emotion vectors have different hidden sizes")
    if directions.metadata.n_blocks != emotion.metadata.n_blocks:
        raise ValueError("directions and emotion vectors have different block counts")
    if tuple(emotion.word_labels) != words.labels:
        raise ValueError("the emotion artifact's word rows do not match the word file")

    labels = words.labels
    index = {label: position for position, label in enumerate(labels)}
    rows = family_rows(words, index)
    pool = pair_pool_rows(words, rows)
    binary_valence = np.asarray([words.valence_by_word[label] for label in labels], dtype=float)
    numeric_valence, covered, missing_norms = (
        read_valence_norms(Path(norms_csv), labels) if norms_csv is not None else (None, 0, ())
    )
    designs = _designs(binary_valence, numeric_valence)
    valence = binary_valence if numeric_valence is None else numeric_valence

    headline = tuple(sorted({directions.metadata.selected_block, emotion.metadata.selected_block}))
    geometries = [
        block_geometry(directions, emotion, block) for block in range(directions.metadata.n_blocks)
    ]
    gate = emotion.metadata.gate_verdict
    return MapGeometryReport(
        seed=seed,
        n_permutations=n_permutations,
        n_null_draws=n_null_draws,
        directions_sha256=directions.metadata.directions_sha256,
        directions_selected_block=directions.metadata.selected_block,
        directions_source_verdict=directions.metadata.source_verdict,
        emotion_vectors_sha256=emotion.metadata.vectors_sha256,
        emotion_selected_block=emotion.metadata.selected_block,
        words_file_sha256=file_sha256(Path(words_file)),
        norms_file=None if norms_csv is None else str(norms_csv),
        norms_covered_words=covered,
        norms_missing_words=missing_norms,
        valence_source=designs[2],
        headline_blocks=headline,
        sensitivity_gate=f"G0={gate}",
        verdict_cap=_verdict_cap(gate),
        blocks=tuple(
            _block_report(
                geometries[block],
                words,
                index,
                rows,
                pool,
                designs,
                binary_valence,
                valence,
                seed=seed,
                n_permutations=n_permutations,
                n_null_draws=n_null_draws,
            )
            for block in headline
        ),
        block_sweep=tuple(
            _sweep_row(geometry, words, index, rows, designs[0], binary_valence, valence)
            for geometry in geometries
        ),
        notes=(_RECORDED_EXPECTATIONS_NOTE, _READOUT_NOTE),
    )


def _verdict_cap(gate: str) -> str:
    if gate != "pass":
        return (
            "harness_inadequate — the E0 G0 sensitivity gate did NOT pass, so no null here is "
            "evidence against appraisal inheritance and no positive here is evidence for it. "
            "The claim stays OPEN (design §4 E0 diagnosticity clause)."
        )
    return (
        "present-and-separable, pilot-suggestive. G0 passed, so E1's nulls carry information on "
        "THIS model/surface/recipe only; a flat valence residual with P5c also passing licenses "
        "the discard 'stop investing in appraisal-residual geometry on story-mean emotion bases "
        "at this scale', and nothing wider. No welfare / sentience / experience claim either way."
    )


# --------------------------------------------------------------------------------------
# Human-readable summary — every word first, then the recorded-expectation readouts
# --------------------------------------------------------------------------------------


def _word_table(block: BlockReport) -> list[str]:
    ordered = sorted(block.word_residuals, key=lambda row: row.residual, reverse=True)
    lines = [f"    all {len(ordered)} words by valence residual on v_rpe (descending):"]
    for start in range(0, len(ordered), 3):
        lines.append(
            "      "
            + "  ".join(
                f"{row.residual:+.4f} {row.word:<13}{row.valence:+d} {row.family:<15}"
                for row in ordered[start : start + 3]
            ).rstrip()
        )
    return lines


def _block_lines(block: BlockReport) -> list[str]:
    lines = [
        f"  block {block.block} (valence source: {block.valence_source})",
        f"    P1  rho(cos(v_rpe, e_j), valence) = {block.p1_spearman_rho:+.3f} "
        f"(p={block.p1_spearman_p:.4f})",
    ]
    lines.extend(_word_table(block))
    lines.append("    recorded expectations (+ statistic = the recorded direction)")
    for contrast in block.family_contrasts:
        lines.append(
            f"      family {contrast.pole:<8} {contrast.outcome_family} "
            f"({contrast.mean_residual_outcome:+.4f}, n={contrast.n_outcome}) vs "
            f"{contrast.control_family} ({contrast.mean_residual_control:+.4f}, "
            f"n={contrast.n_control}) -> {contrast.statistic:+.4f} (p={contrast.p_value:.4f})"
        )
    ordering = block.outcome_ordering
    lines.append(
        "      ordering (expected high to low) "
        + " | ".join(
            f"{family} {mean:+.4f}"
            for family, mean in zip(ordering.families, ordering.mean_residuals, strict=True)
        )
        + f" -> three-level ordering holds: {'Y' if ordering.ordering_holds else 'n'}; "
        f"end-to-end trend (first minus last family, the middle family does NOT enter this p) "
        f"{ordering.trend_statistic:+.4f} (p={ordering.p_value:.4f})"
    )
    for pair in block.expected_pairs:
        direction = ">" if pair.expected_sign > 0 else "<"
        lines.append(
            f"      pair   {pair.outcome:>13} {direction} {pair.control:<11} "
            f"{pair.statistic:+.4f} (p={pair.p_value:.4f}, pool n={pair.n_pool})"
        )
    lines.append(
        f"    P4  surprise {block.p4.mean_cos_surprise:+.4f} vs arousal-matched "
        f"{block.p4.mean_cos_arousal_matched:+.4f} -> contrast {block.p4.contrast:+.4f} "
        f"(p={block.p4.p_value:.4f}); v_absrpe loads PC1 {block.p4.cos_absrpe_pc1:+.3f} / PC2 "
        f"{block.p4.cos_absrpe_pc2:+.3f} -> arousal-PC {'Y' if block.p4.loads_arousal_pc else 'n'}"
    )
    lines.append(
        f"    P5a {block.p5a.word} residual {block.p5a.residual:+.4f} "
        f"(|rank| {block.p5a.abs_rank}/{block.p5a.n_words}) vs word |resid| p95 "
        f"{block.p5a.word_residual_p95:.4f} -> inside the spread: "
        f"{'Y' if block.p5a.passed else 'n'}"
    )
    for entry in block.p5c:
        lines.append(
            f"    P5c {entry.family:<9} style residual {entry.residual:+.4f} "
            f"vs word |resid| p95 {entry.word_residual_p95:.4f} -> "
            f"{'PASS' if entry.passed else 'FAIL'}"
        )
    lines.append(
        f"    floors: observed family contrast {block.observed_family_contrast_max:+.4f} vs "
        f"label-shuffled p95 {block.label_shuffled_p95:+.4f} / random-direction p95 "
        f"{block.random_direction_p95:+.4f} -> "
        f"clears both: {'Y' if block.clears_both_floors else 'n'}"
    )
    return lines


def format_map_geometry_summary(report: MapGeometryReport) -> str:
    """A plain-text summary for stdout: every word, then the recorded-expectation readouts."""

    lines = [
        "E1 map-geometry — valence-residual appraisal geometry (present-and-separable at most)",
        f"  seed={report.seed} perms={report.n_permutations} null-draws={report.n_null_draws}",
        f"  sensitivity: {report.sensitivity_gate}",
        f"  verdict cap: {report.verdict_cap}",
        f"  headline blocks {list(report.headline_blocks)} "
        f"(directions {report.directions_selected_block}, emotion "
        f"{report.emotion_selected_block})",
        "",
    ]
    for block in report.blocks:
        lines.extend(_block_lines(block))
    lines.append("")
    lines.append("  block sweep (P1 rho | max family contrast | P4 contrast | v_absrpe·PC2):")
    for row in report.block_sweep:
        lines.append(
            f"    b{row.block:>3}  {row.p1_spearman_rho:+.3f}  "
            f"{max(row.family_contrast_statistics.values()):+.4f}  {row.p4_contrast:+.4f}  "
            f"{row.cos_absrpe_pc2:+.3f}"
        )
    lines.append("")
    lines.append("NOTES (read before quoting any number)")
    for note in report.notes:
        lines.append(f"  - {note}")
    return "\n".join(lines)
