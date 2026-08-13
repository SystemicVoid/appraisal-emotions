"""E1 — valence-residual geometry between appraisal directions and emotion concepts (design §4).

NEW module (no parent counterpart). Pure CPU over two hash-bound artifacts: the reveal-RPE
directions (``v_rpe``, ``v_ev``, ``v_absrpe``) and the E0 emotion basis. No model, no forwards.
The per-block estimands live in ``analysis.valence_residual``; this module chooses the blocks and
the valence scale, runs the nulls, and assembles the report.

The steelmanned claim (§1) is NOT "``v_RPE`` points at positive emotions" — that is near-certain
and uninformative, since any positive-value direction has a positive-valence component. It is
that after removing the valence axis the emotion space is STILL organized by appraisal
variables. So every readout is a *residual* readout:

- **P1 (exploratory sanity):** ``rho(cos(v_RPE, e_j), valence_j) > 0`` over all words. Expected
  true; a failure indicts the extraction, not the theory.
- **P2 (confirmatory):** regress ``cos(v_RPE, e_j)`` on valence (numeric norms when the fetched
  subset covers the set, else the §5 minted binary labels), then test the three pre-registered
  matched pairs on the residuals — one-sided, permutation p with residuals shuffled WITHIN the
  valence-matched set, Holm-corrected across the three. Read ``open_questions`` in the report
  before quoting a P2 number: it carries the power floor and an unresolved sign question on the
  ``disappointed > sad`` pair.
- **P4 (confirmatory):** ``v_absrpe`` prefers the surprise family over arousal-matched valenced
  controls, and loads PC2 rather than PC1. *Stated every time:* ``v_RPE ⊥ v_absrpe`` holds by
  construction of the reveal design matrix; the informative half is which WORDS ``v_absrpe``
  picks out.
- **P5a (confirmatory prediction of absence):** ``sad`` shows no RPE-excess after valence
  partialling. It passes by being null, and only means something while P2 is positive.
- **P5c (confirmatory scale control):** the valence-free ``style_control`` vector shows ~0
  residual alignment with all three directions. If it fails, the cosine scale itself is invalid
  and E1 records ``harness_inadequate`` (design §4 E1.5c) — not evidence about inheritance.

Two nulls guard the anisotropy threat (§6): label-shuffled emotion vectors and norm-matched
random directions. Comparisons are per-block mean-centred, and the full block sweep is reported
rather than only the two artifacts' selected blocks (site/object mismatch is a finding, not
something to hide).

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
    P4Result,
    P5aResult,
    P5cStyleResult,
    PairResult,
    block_geometry,
    cosines,
    label_shuffled_pair_floor,
    norm_fraction,
    ols_residuals,
    p2_matched_pairs,
    p4_contrast,
    p4_surprise_contrast,
    p5a_discriminant_null,
    p5c_style_control,
    pair_statistics,
    random_direction_pair_floor,
    spearman,
)
from appraisal_emotions.core.schema import StrictModel
from appraisal_emotions.core.util import EXTRACTION_SEED, file_sha256
from appraisal_emotions.stimuli.emotion_stories import EmotionWordSet, read_emotion_words

__all__ = [
    "MAP_GEOMETRY_CONTRACT_VERSION",
    "BlockSweepRow",
    "ConfirmatoryBlock",
    "ExploratoryBlock",
    "MapGeometryReport",
    "format_map_geometry_summary",
    "map_geometry",
    "read_valence_norms",
]

MAP_GEOMETRY_CONTRACT_VERSION = "map_geometry/v1"
DEFAULT_ALPHA = 0.05

# --------------------------------------------------------------------------------------
# Numeric norms (fetched, never transcribed — scripts/fetch_norms.py)
# --------------------------------------------------------------------------------------


def read_valence_norms(path: Path, labels: tuple[str, ...]) -> tuple[np.ndarray | None, int]:
    """Numeric valence norms for ``labels`` from the fetched subset CSV, or ``None``.

    Returns ``(valence array aligned to labels, covered word count)``. Partial coverage yields
    ``None``: mixing a numeric scale for some words with a binary label for others would make the
    regression's residuals incommensurable, and the binary labels are always available.
    """

    values: dict[str, float] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            word = str(row["word"]).strip().lower()
            raw = str(row.get("valence", "")).strip()
            if word and raw:
                values[word] = float(raw)
    covered = sum(1 for label in labels if label in values)
    if covered != len(labels):
        return None, covered
    return np.asarray([values[label] for label in labels], dtype=float), covered


# --------------------------------------------------------------------------------------
# Report schema
# --------------------------------------------------------------------------------------


class ConfirmatoryBlock(StrictModel):
    """The pre-registered §5 readouts at one block. Nothing exploratory belongs in here."""

    block: int
    valence_source: str
    p2_pairs: tuple[PairResult, ...]
    p2_any_passed: bool
    p4: P4Result
    p5a: P5aResult
    p5c: tuple[P5cStyleResult, ...]
    p5c_passed: bool
    label_shuffled_max_pair_p95: float
    random_direction_max_pair_p95: float
    observed_max_pair_statistic: float
    clears_both_nulls: bool


class ExploratoryBlock(StrictModel):
    """Everything NOT pre-registered at one block — labeled so it can never be read as a result."""

    block: int
    p1_spearman_rho: float
    p1_spearman_p: float
    p1_positive: bool
    mean_cos_by_family: dict[str, float]
    subspace_fraction_emotion_span: float
    subspace_fraction_pc12_plane: float
    note: str


class BlockSweepRow(StrictModel):
    """One block of the descriptive depth profile (no permutations — exploratory)."""

    block: int
    p1_spearman_rho: float
    p2_pair_statistics: dict[str, float]
    p4_contrast: float
    cos_absrpe_pc1: float
    cos_absrpe_pc2: float


class MapGeometryReport(StrictModel):
    """E1 record: confirmatory §5 readouts, exploratory context, and the inherited gate cap."""

    artifact_contract_version: Literal["map_geometry/v1"] = MAP_GEOMETRY_CONTRACT_VERSION
    seed: int
    n_permutations: int
    n_null_draws: int
    alpha: float
    directions_sha256: str
    directions_selected_block: int
    directions_source_verdict: str
    emotion_vectors_sha256: str
    emotion_selected_block: int
    words_file_sha256: str
    norms_file: str | None
    norms_covered_words: int
    valence_source: str
    headline_blocks: tuple[int, ...]
    sensitivity_gate: str
    verdict_cap: str
    confirmatory: tuple[ConfirmatoryBlock, ...]
    exploratory: tuple[ExploratoryBlock, ...]
    block_sweep: tuple[BlockSweepRow, ...]
    open_questions: tuple[str, ...]


_P2_POWER_NOTE = (
    "P2 POWER FLOOR: the within-valence permutation puts the pair's two residuals into a "
    "uniformly random ordered pair of the matched set, so the smallest attainable p is "
    "1/(n(n-1)) for a matched set of size n (~0.004 here) and only a pair straddling near the "
    "extremes of its matched set can survive Holm across three pairs. A large positive pair "
    "statistic that misses significance is a POWER finding, not a null."
)
_P2_SIGN_QUESTION = (
    "P2 SIGN, OPEN: design §4 E1.2 pre-registers all three pairs as high > low on the valence "
    "residual of cos(v_RPE, e_j), and that is what is computed here — unchanged. But the OCC "
    "reading that motivates the pairs makes 'disappointed' a NEGATIVE prediction error, which "
    "predicts e_disappointed ANTI-aligned with v_RPE and therefore disappointed < sad on a "
    "signed residual, while relieved > calm and elated > content are sign-consistent as "
    "written. Resolve this in the design doc BEFORE the first real run; changing it afterwards "
    "is a post-hoc amendment and must pass the symmetric-amendment test in docs/agents/rails.md."
)

_EXPLORATORY_NOTE = (
    "EXPLORATORY — not pre-registered in design §5. Includes the full-set P1 correlation and the "
    "subspace fractions. These may generate hypotheses; they may not be reported as results."
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


def _confirmatory_block(
    geometry: BlockGeometry,
    words: EmotionWordSet,
    index: dict[str, int],
    designs: tuple[np.ndarray, np.ndarray, str],
    binary_valence: np.ndarray,
    *,
    seed: int,
    n_permutations: int,
    n_null_draws: int,
    alpha: float,
) -> ConfirmatoryBlock:
    design, binary_design, valence_source = designs
    pairs = words.confirmatory_pairs()
    rng = np.random.default_rng(seed_int(seed, "map-geometry-p2", geometry.block))
    p2_pairs, residuals = p2_matched_pairs(
        geometry,
        pairs,
        index,
        design,
        binary_valence,
        rng=rng,
        n_permutations=n_permutations,
        alpha=alpha,
    )
    p4 = p4_surprise_contrast(
        geometry,
        words,
        index,
        rng=np.random.default_rng(seed_int(seed, "map-geometry-p4", geometry.block)),
        n_permutations=n_permutations,
        alpha=alpha,
    )
    p5a = p5a_discriminant_null(
        geometry,
        str(words.confirmatory["p5a"]),
        index,
        design,
        rng=np.random.default_rng(seed_int(seed, "map-geometry-p5a", geometry.block)),
        n_bootstrap=n_null_draws,
    )
    p5c = p5c_style_control(geometry, binary_design)
    label_floor = label_shuffled_pair_floor(
        geometry,
        pairs,
        index,
        design,
        binary_valence,
        rng=np.random.default_rng(seed_int(seed, "map-geometry-label-null", geometry.block)),
        n_draws=n_null_draws,
    )
    direction_floor = random_direction_pair_floor(
        geometry,
        pairs,
        index,
        design,
        binary_valence,
        rng=np.random.default_rng(seed_int(seed, "map-geometry-direction-null", geometry.block)),
        n_draws=n_null_draws,
    )
    observed_max = max(pair_statistics(residuals, pairs, binary_valence, index))
    return ConfirmatoryBlock(
        block=geometry.block,
        valence_source=valence_source,
        p2_pairs=p2_pairs,
        p2_any_passed=any(pair.passed for pair in p2_pairs),
        p4=p4,
        p5a=p5a,
        p5c=p5c,
        p5c_passed=all(entry.passed for entry in p5c),
        label_shuffled_max_pair_p95=label_floor,
        random_direction_max_pair_p95=direction_floor,
        observed_max_pair_statistic=observed_max,
        clears_both_nulls=bool(observed_max > label_floor and observed_max > direction_floor),
    )


def _exploratory_block(geometry: BlockGeometry, valence: np.ndarray) -> ExploratoryBlock:
    rho, p_value = spearman(geometry.cos["v_rpe"], valence)
    v_rpe = geometry.family_vectors["v_rpe"]
    return ExploratoryBlock(
        block=geometry.block,
        p1_spearman_rho=rho,
        p1_spearman_p=p_value,
        p1_positive=rho > 0.0,
        mean_cos_by_family={
            family: float(geometry.cos[family].mean()) for family in DIRECTION_FAMILIES
        },
        subspace_fraction_emotion_span=norm_fraction(v_rpe, geometry.centered_words),
        subspace_fraction_pc12_plane=norm_fraction(v_rpe, geometry.components[:2]),
        note=_EXPLORATORY_NOTE,
    )


def _sweep_row(
    geometry: BlockGeometry,
    words: EmotionWordSet,
    index: dict[str, int],
    design: np.ndarray,
    binary_valence: np.ndarray,
    valence: np.ndarray,
) -> BlockSweepRow:
    pairs = words.confirmatory_pairs()
    residuals = ols_residuals(geometry.cos["v_rpe"], design)
    statistics = pair_statistics(residuals, pairs, binary_valence, index)
    cos = geometry.cos["v_absrpe"]
    left = np.asarray([index[word] for word in words.confirmatory_set("surprise")])
    right = np.asarray([index[word] for word in words.confirmatory_set("arousal_matched")])
    loading = cosines(geometry.components[:2], geometry.family_vectors["v_absrpe"])
    return BlockSweepRow(
        block=geometry.block,
        p1_spearman_rho=spearman(geometry.cos["v_rpe"], valence)[0],
        p2_pair_statistics={
            f"{pair[0]}>{pair[1]}": value for pair, value in zip(pairs, statistics, strict=True)
        },
        p4_contrast=p4_contrast(cos, left, right),
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
    alpha: float = DEFAULT_ALPHA,
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
    binary_valence = np.asarray([words.valence_by_word[label] for label in labels], dtype=float)
    numeric_valence, covered = (
        read_valence_norms(Path(norms_csv), labels) if norms_csv is not None else (None, 0)
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
        alpha=alpha,
        directions_sha256=directions.metadata.directions_sha256,
        directions_selected_block=directions.metadata.selected_block,
        directions_source_verdict=directions.metadata.source_verdict,
        emotion_vectors_sha256=emotion.metadata.vectors_sha256,
        emotion_selected_block=emotion.metadata.selected_block,
        words_file_sha256=file_sha256(Path(words_file)),
        norms_file=None if norms_csv is None else str(norms_csv),
        norms_covered_words=covered,
        valence_source=designs[2],
        headline_blocks=headline,
        sensitivity_gate=f"G0={gate}",
        verdict_cap=_verdict_cap(gate),
        confirmatory=tuple(
            _confirmatory_block(
                geometries[block],
                words,
                index,
                designs,
                binary_valence,
                seed=seed,
                n_permutations=n_permutations,
                n_null_draws=n_null_draws,
                alpha=alpha,
            )
            for block in headline
        ),
        exploratory=tuple(_exploratory_block(geometries[block], valence) for block in headline),
        block_sweep=tuple(
            _sweep_row(geometry, words, index, designs[0], binary_valence, valence)
            for geometry in geometries
        ),
        open_questions=(_P2_POWER_NOTE, _P2_SIGN_QUESTION),
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
        "THIS model/surface/recipe only; a null P2 with P5c also passing licenses the discard "
        "'stop investing in appraisal-residual geometry at 4B on story-mean emotion bases', and "
        "nothing wider. No welfare / sentience / experience claim is licensed either way."
    )


# --------------------------------------------------------------------------------------
# Human-readable summary
# --------------------------------------------------------------------------------------


def _confirmatory_lines(block: ConfirmatoryBlock) -> list[str]:
    lines = [f"  block {block.block} (valence source: {block.valence_source})"]
    lines.append("    P2  pair                       stat      p     p_holm  pass")
    for pair in block.p2_pairs:
        lines.append(
            f"        {pair.high:>13} > {pair.low:<11} {pair.statistic:+7.4f} "
            f"{pair.p_value:6.4f} {pair.p_holm:6.4f}  {'Y' if pair.passed else 'n'}"
        )
    lines.append(
        f"    P4  surprise {block.p4.mean_cos_surprise:+.4f} vs arousal-matched "
        f"{block.p4.mean_cos_arousal_matched:+.4f} -> contrast {block.p4.contrast:+.4f} "
        f"(p={block.p4.p_value:.4f}, pass={'Y' if block.p4.passed else 'n'}); "
        f"v_absrpe loads PC1 {block.p4.cos_absrpe_pc1:+.3f} / PC2 {block.p4.cos_absrpe_pc2:+.3f}"
        f" -> arousal-PC {'Y' if block.p4.loads_arousal_pc else 'n'}"
    )
    lines.append(
        f"    P5a {block.p5a.word} residual {block.p5a.residual:+.4f} "
        f"CI [{block.p5a.ci_low:+.4f}, {block.p5a.ci_high:+.4f}] "
        f"covers 0: {'Y' if block.p5a.includes_zero else 'n'}"
    )
    for entry in block.p5c:
        lines.append(
            f"    P5c {entry.family:<9} style residual {entry.residual:+.4f} "
            f"vs word |resid| p95 {entry.word_residual_p95:.4f} -> "
            f"{'PASS' if entry.passed else 'FAIL'}"
        )
    lines.append(
        f"    nulls: observed max pair {block.observed_max_pair_statistic:+.4f} vs "
        f"label-shuffled p95 {block.label_shuffled_max_pair_p95:+.4f} / random-direction p95 "
        f"{block.random_direction_max_pair_p95:+.4f} -> "
        f"clears both: {'Y' if block.clears_both_nulls else 'n'}"
    )
    return lines


def format_map_geometry_summary(report: MapGeometryReport) -> str:
    """A plain-text summary table for stdout — confirmatory first, exploratory clearly fenced."""

    lines = [
        "E1 map-geometry — valence-residual appraisal geometry (present-and-separable at most)",
        f"  seed={report.seed} perms={report.n_permutations} null-draws={report.n_null_draws} "
        f"alpha={report.alpha}",
        f"  sensitivity: {report.sensitivity_gate}",
        f"  verdict cap: {report.verdict_cap}",
        f"  headline blocks {list(report.headline_blocks)} "
        f"(directions {report.directions_selected_block}, emotion "
        f"{report.emotion_selected_block})",
        "",
        "CONFIRMATORY (pre-registered, design §5)",
    ]
    for block in report.confirmatory:
        lines.extend(_confirmatory_lines(block))
    lines.extend(["", "EXPLORATORY (not pre-registered — hypothesis-generating only)"])
    for block in report.exploratory:
        lines.append(
            f"  block {block.block}: P1 rho={block.p1_spearman_rho:+.3f} "
            f"(p={block.p1_spearman_p:.4f}); ||v_rpe|| fraction in emotion span "
            f"{block.subspace_fraction_emotion_span:.3f}, in PC1-PC2 plane "
            f"{block.subspace_fraction_pc12_plane:.3f}"
        )
    lines.append("  block sweep (P1 rho | max pair stat | P4 contrast | v_absrpe·PC2):")
    for row in report.block_sweep:
        lines.append(
            f"    b{row.block:>3}  {row.p1_spearman_rho:+.3f}  "
            f"{max(row.p2_pair_statistics.values()):+.4f}  {row.p4_contrast:+.4f}  "
            f"{row.cos_absrpe_pc2:+.3f}"
        )
    lines.append("")
    lines.append("OPEN QUESTIONS (read before quoting any P2 number)")
    for question in report.open_questions:
        lines.append(f"  - {question}")
    return "\n".join(lines)
