"""P1 analysis of record: split E1's word-level noise into the half more stories fix and the half they don't. CPU only.

E1's positive-pole family contrast reads +0.0186 at block 63 against an MDE80 of 0.032 — half the
smallest effect its own test detects. That is an instrument report, and it is unreadable in either
direction until the residual spread `s = 0.0267` is decomposed:

    s^2  =  sigma^2_tau,resid  +  sigma^2_w / k

A within-word-dominated `s` means a larger capture is worth renting a GPU for, and this script says
how large. A between-word-dominated `s` means no story count rescues the design and the next move
is a wider word set. `docs/design/p1-prereg.md` fixed which of those verdicts fires at which
threshold, and it was committed before the capture that this reads.

**This script runs no new test of the family contrast.** E1's point estimate and its exact
enumeration p stand as published; nothing here recomputes, re-tests or rescales them. It reconstructs
the contrast only as a GATE — a re-capture that cannot reproduce E1's own word cosines is not a
decomposition of E1's basis and its variance components would describe some other measurement.

The exploratory directions (`v_ev`, `v_absrpe`, `pc1`, `pc2`) are captured but sit behind
`--exploratory`, off by default, because the pre-registration's do-not-look list is enforced by the
default rather than by intention.

Reads the P1 projections artifact plus the E0/E1 artifacts it binds to. No model, no forwards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from appraisal_emotions.analysis.emotion_mapping import read_valence_norms
from appraisal_emotions.analysis.emotion_vectors import read_emotion_vectors
from appraisal_emotions.analysis.story_projections import read_story_projections
from appraisal_emotions.analysis.story_reliability import (
    MDE80_COEFFICIENT,
    attenuation,
    half_sample_cosines,
    prophecy,
    split_half_reliability,
    topic_adjusted_word_means,
    variance_components,
    word_projection_stats,
)
from appraisal_emotions.analysis.valence_residual import (
    family_contrast_statistics,
    family_rows,
    ols_residuals,
)
from appraisal_emotions.stimuli.emotion_stories import read_emotion_words

REPO = Path(__file__).resolve().parents[1]
PROJECTIONS = REPO / "runs/emotion_vectors_base/emotions/story_projections.json"
EMOTION = REPO / "runs/emotion_vectors_base/emotions/emotion_vectors.json"
REPORT = REPO / "runs/emotion_vectors_base/emotions/map_geometry_report.json"
WORDS = REPO / "data/emotion_words.json"
NORMS = REPO / "data/norms/vad_subset.csv"
OUT = REPO / "runs/emotion_vectors_base/emotions/p1_reliability.json"

DECISION_BLOCK = 63
ROBUSTNESS_BLOCK = 35
PRIMARY_DIRECTION = "v_rpe"
EXPLORATORY_DIRECTIONS = ("v_ev", "v_absrpe", "pc1", "pc2")

# Pre-registered before the capture. An exact identity is a property of exact arithmetic; a bf16
# forward with different batching will not reproduce `e_j . d` bitwise. At an effect scale of 0.02,
# 1e-3 is decisive, and a miss above it means investigating padding/position/batching before
# anything below is interpreted.
COSINE_GATE_ABS = 1e-3

# The prereg's verdict thresholds, restated here as constants so the routing cannot drift from the
# document. Changing either requires the symmetric-amendment test, not an edit.
ICC_RECIPE_FLOOR = 0.5
ICC_PROCEED_FLOOR = 0.7
K_BUDGET = 48


def _verdict(icc: float, forecast: dict) -> tuple[str, str]:
    """Route on the pre-registered table. No branch here reads a number this script computed twice."""

    if icc < ICC_RECIPE_FLOOR:
        return "harness_inadequate_recipe_is_the_finding", (
            f"ICC_resid {icc:.3f} < {ICC_RECIPE_FLOOR}: a story-mean base at this k cannot carry "
            "word-level residual geometry at all. E1's null says nothing about inheritance; stop "
            "investing in story-mean emotion bases at this scale. The claim stays OPEN."
        )
    if icc < ICC_PROCEED_FLOOR:
        return "gray_zone", (
            f"ICC_resid {icc:.3f} is between {ICC_RECIPE_FLOOR} and {ICC_PROCEED_FLOOR}: no kill "
            "and no licence. The prophecy curve governs the recommendation."
        )
    if not forecast["detectable_at_any_k"]:
        return "word_set_is_the_finding", (
            "Reliability is adequate but the between-word-only MDE80 floor "
            f"({forecast['mde80_floor_k_infinite']:.4f}) still exceeds the de-attenuated effect "
            f"({forecast['projected_effect_k_infinite']:.4f}). No story count rescues this; the "
            "next design widens the word families, not k."
        )
    k_star = forecast["k_star"]
    if k_star is None or k_star > K_BUDGET:
        return "detectable_only_above_the_k_budget", (
            f"A crossing exists but at k* = {k_star} > {K_BUDGET} stories per word. Reachable in "
            "principle, outside the pre-registered budget: not a licence."
        )
    return "proceed", (
        f"ICC_resid {icc:.3f} and k* = {k_star} <= {K_BUDGET}. Licenses a pre-registered E1' "
        "capture at k*, with its block named in THAT prereg before generation. Nothing here "
        "upgrades E1's contrast: the ceiling stays pilot-suggestive, functional "
        "measurement-validity."
    )


def _analyse(artifact, labels, design, block, direction, shipped, *, rng_seed, n_splits, max_k):
    stats = word_projection_stats(artifact, labels, block=block, direction=direction)
    components, residuals = variance_components(stats, design)
    halves = half_sample_cosines(
        artifact,
        labels,
        block=block,
        direction=direction,
        rng=np.random.default_rng(rng_seed),
        n_splits=n_splits,
    )
    forecast = prophecy(stats, components, observed_effect=shipped["observed_effect"], max_k=max_k)
    return (
        stats,
        components,
        residuals,
        {
            "block": block,
            "direction": direction,
            "n_words": len(labels),
            "mean_stories_per_word": components.mean_k,
            "observed_residual_sd": components.observed_residual_sd,
            "within_word_variance_cos": components.within_word_variance_cos,
            "between_word_variance_resid": components.between_word_variance_resid,
            "icc_1k_resid": components.icc_1k_resid,
            "icc_1k_raw": components.icc_1k_raw,
            "split_half_exact": split_half_reliability(halves, design),
            "attenuation": attenuation(stats),
            "prophecy": forecast,
        },
    )


def _block_report(artifact, labels, design, rows, contrasts, by_block, block, args) -> dict:
    """Everything reported for one block: gate first, then the decomposition it licenses."""

    shipped_block = by_block[block]
    shipped_cos = np.asarray([row["cos_v_rpe"] for row in shipped_block["word_residuals"]])
    shipped_words = [row["word"] for row in shipped_block["word_residuals"]]
    if tuple(shipped_words) != labels:
        raise SystemExit(f"block {block}: report word order differs from the word file")
    positive = next(row for row in shipped_block["family_contrasts"] if row["pole"] == "positive")

    stats, components, residuals, summary = _analyse(
        artifact,
        labels,
        design,
        block,
        PRIMARY_DIRECTION,
        {"observed_effect": positive["statistic"]},
        rng_seed=args.seed + block,
        n_splits=args.n_splits,
        max_k=args.max_k,
    )

    # THE GATE. Everything above describes E1's basis only if the cosines reconstructed from
    # story-level scalars are E1's cosines.
    drift = float(np.abs(stats.cos - shipped_cos).max())
    reconstructed = family_contrast_statistics(residuals, contrasts, rows)
    contrast_drift = float(abs(reconstructed[0] - positive["statistic"]))
    summary["gate_vs_shipped_report"] = {
        "max_abs_word_cosine_drift": drift,
        "threshold": COSINE_GATE_ABS,
        "passed": drift <= COSINE_GATE_ABS,
        "shipped_positive_contrast": positive["statistic"],
        "reconstructed_positive_contrast": reconstructed[0],
        "contrast_drift": contrast_drift,
        "note": (
            "Reconstruction, not a re-test: E1's p stands as published and is not recomputed "
            "here. If this drifts, the variance components below describe some other capture."
        ),
    }
    if drift > COSINE_GATE_ABS:
        raise SystemExit(
            f"block {block}: reconstructed word cosines drift {drift:.3g} from the shipped "
            f"report (threshold {COSINE_GATE_ABS}). Investigate padding, position and batching "
            "before interpreting anything."
        )

    # Pre-committed SECONDARY: a nuisance-adjusted point estimate that routes nothing.
    adjusted = topic_adjusted_word_means(
        artifact, labels, stats, block=block, direction=PRIMARY_DIRECTION
    )
    adjusted_contrast = family_contrast_statistics(ols_residuals(adjusted, design), contrasts, rows)
    summary["topic_fixed_effect_secondary"] = {
        "n_topics": len(set(artifact.metadata.story_topics)),
        "adjusted_positive_contrast": adjusted_contrast[0],
        "adjusted_negative_contrast": adjusted_contrast[1],
        "shift_from_unadjusted": adjusted_contrast[0] - positive["statistic"],
        "note": (
            "Topic as a FIXED effect over 1,017 story rows. Inference never comes from this "
            "model — the p stays E1's word-level permutation. Reported because topic "
            "composition is a named nuisance, and it routes nothing."
        ),
    }

    if block == DECISION_BLOCK:
        verdict, note = _verdict(components.icc_1k_resid, summary["prophecy"])
        summary["verdict"] = verdict
        summary["verdict_note"] = note

    if args.exploratory:
        summary["exploratory_directions"] = {
            name: _analyse(
                artifact,
                labels,
                design,
                block,
                name,
                {"observed_effect": positive["statistic"]},
                rng_seed=args.seed + block,
                n_splits=args.n_splits,
                max_k=args.max_k,
            )[3]
            for name in EXPLORATORY_DIRECTIONS
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projections", type=Path, default=PROJECTIONS)
    # Explicit, because the pre-capture dry run runs this script on SYNTHETIC scalars and a
    # synthetic verdict landing in `runs/` alongside real artifacts is exactly the kind of file
    # that gets quoted six weeks later. Default is the analysis of record's path; the dry run
    # redirects it.
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-splits", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-k", type=int, default=400)
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="also report the non-v_rpe directions (pre-registered as exploratory, opened last)",
    )
    args = parser.parse_args()

    artifact = read_story_projections(args.projections)
    if artifact.metadata.gate_verdict != "pass":
        raise SystemExit(
            f"capture gate verdict is {artifact.metadata.gate_verdict!r}; a quarantined capture is "
            "not a decomposition of the published basis and must not be analysed"
        )
    emotion = read_emotion_vectors(EMOTION)
    if artifact.metadata.emotion_vectors_sha256 != emotion.metadata.vectors_sha256:
        raise SystemExit("projections were captured against a different emotion-vector artifact")

    words = read_emotion_words(WORDS)
    labels = tuple(words.labels)
    index = {label: position for position, label in enumerate(labels)}
    rows = family_rows(words, index)
    contrasts = words.expected_family_contrasts()
    numeric_valence, covered, missing = read_valence_norms(NORMS, labels)
    if numeric_valence is None or covered != len(labels):
        raise SystemExit(f"norms cover {covered}/{len(labels)}; missing {missing}")
    design = np.column_stack([np.ones(len(labels)), numeric_valence])

    report = json.loads(REPORT.read_text())
    by_block = {row["block"]: row for row in report["blocks"]}

    result: dict[str, object] = {
        "capture": {
            "model_id": artifact.metadata.model_id,
            "revision": artifact.metadata.revision,
            "n_stories": artifact.metadata.n_stories,
            "min_token": artifact.metadata.min_token,
            "gate_max_relative_deviation": artifact.metadata.gate_max_relative_deviation,
            "projections_sha256": artifact.metadata.projections_sha256,
        },
        "prereg": "docs/design/p1-prereg.md",
        "decision_block": DECISION_BLOCK,
        "mde80_coefficient": MDE80_COEFFICIENT,
        "blocks": {},
    }

    for block in (DECISION_BLOCK, ROBUSTNESS_BLOCK):
        summary = _block_report(artifact, labels, design, rows, contrasts, by_block, block, args)
        result["blocks"][str(block)] = summary

    decision = result["blocks"][str(DECISION_BLOCK)]
    result["verdict"] = decision["verdict"]
    result["verdict_note"] = decision["verdict_note"]
    result["claim_ceiling"] = (
        "functional measurement-validity, pilot-suggestive. P1 validates or condemns the "
        "INSTRUMENT; no outcome here upgrades E1's contrast into a result, and no outcome licenses "
        "any welfare, sentience or experience claim. Emotion words are concept labels throughout."
    )

    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
