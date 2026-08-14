"""Is the mid-depth peak in the E1 family contrast bigger than depth-shopping explains? CPU only.

`map_geometry`'s block sweep reports the family contrast at every block but floors none of them,
and the two headline blocks are chosen by criteria (G0's rho, the RPE certification) that are
blind to the contrast. So the sweep's peak — block 50, +0.0538, roughly twice either headline —
is currently the largest unevaluated number in E1, and it cannot be read off the shipped report:
the report tables word-level cosines at the headline blocks only. It CAN be read off the capture
artifacts, which cost nothing to re-analyse.

The test is a max-statistic permutation, the standard selection-aware correction. One shuffle of
the 84 word rows is drawn per null draw and applied to **every** block, so the null inherits the
real cross-block correlation (b35/b63 word residuals correlate at 0.81) rather than pretending 64
independent tests. `p_selection_aware` is then the honest p for "the best block over this depth
range", and comparing it with the per-block p at the same block shows exactly what depth-shopping
was worth.

Two block families are reported, and the choice between them is the whole result, so neither is
picked here:

* `all` (0-63) — no prior at all.
* `g0_passing` — every block whose G0 rho clears the threshold **recorded in the emotion-vector
  artifact before any of this was run** (0.6; blocks 2 and 6-63 pass, 0/1/3/4/5 fail). This is the
  repo's own sensitivity gate, not a band drawn around the peak: a block that fails G0 carries no
  detectable valence structure, so its family contrast is uninterpretable in either direction.

Report both, always, and read the gap. The early blocks G0 rejects are wildly noisy, so the two
families give very different answers, and any write-up that quotes one without the other is
choosing its p-value.

Reads only `runs/.../reveal_directions.json` + `emotion_vectors.json` and their npz payloads,
through the same loaders `map_geometry` uses. No model, no forwards, no new capture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from appraisal_emotions.analysis.emotion_mapping import read_valence_norms
from appraisal_emotions.analysis.emotion_vectors import read_emotion_vectors
from appraisal_emotions.analysis.reveal_rpe import read_reveal_rpe_directions
from appraisal_emotions.analysis.valence_residual import (
    block_geometry,
    family_contrast_statistics,
    family_rows,
    ols_residuals,
)
from appraisal_emotions.stimuli.emotion_stories import read_emotion_words

REPO = Path(__file__).resolve().parents[1]
DIRECTIONS = REPO / "runs/reveal_rpe_base/reveal_rpe/reveal_directions.json"
EMOTION = REPO / "runs/emotion_vectors_base/emotions/emotion_vectors.json"
REPORT = REPO / "runs/emotion_vectors_base/emotions/map_geometry_report.json"
WORDS = REPO / "data/emotion_words.json"
NORMS = REPO / "data/norms/vad_subset.csv"
OUT = REPO / "runs/emotion_vectors_base/emotions/e1_selection_aware_depth.json"

# The G0 threshold is read from the artifact, never written here: it was fixed before the capture
# and it is the only depth criterion in this project that predates the numbers being corrected.
G0_THRESHOLD_KEY = "g0_threshold"


def _residuals_by_block(directions, emotion, design: np.ndarray) -> np.ndarray:
    """(n_blocks, 84) valence residuals of cos(v_rpe, e_j), one regression per block."""

    return np.stack(
        [
            ols_residuals(block_geometry(directions, emotion, block).cos["v_rpe"], design)
            for block in range(emotion.metadata.n_blocks)
        ]
    )


def _statistics(residuals: np.ndarray, contrasts, rows: dict[str, np.ndarray]) -> np.ndarray:
    """(n_blocks, n_poles) family-contrast statistics."""

    return np.array([family_contrast_statistics(row, contrasts, rows) for row in residuals])


def _family_result(
    observed: np.ndarray, draws: np.ndarray, blocks: np.ndarray, poles: list[str]
) -> dict:
    """Max-statistic p over ``blocks``, per pole and over both poles jointly."""

    n = draws.shape[0]
    out: dict[str, dict] = {}
    for index, pole in enumerate(poles):
        obs = observed[blocks, index]
        peak = int(blocks[int(np.argmax(obs))])
        null_max = draws[:, blocks, index].max(axis=1)
        out[pole] = {
            "argmax_block": peak,
            "statistic_at_argmax": float(obs.max()),
            "p_selection_aware": float((np.sum(null_max >= obs.max()) + 1) / (n + 1)),
            "p_at_that_block_uncorrected": float(
                (np.sum(draws[:, peak, index] >= obs.max()) + 1) / (n + 1)
            ),
            "null_max_p95": float(np.quantile(null_max, 0.95)),
        }
    obs_both = observed[blocks].max()
    null_both = draws[:, blocks, :].max(axis=(1, 2))
    out["either_pole"] = {
        "statistic": float(obs_both),
        "p_selection_aware": float((np.sum(null_both >= obs_both) + 1) / (n + 1)),
        "null_max_p95": float(np.quantile(null_both, 0.95)),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    directions = read_reveal_rpe_directions(DIRECTIONS)
    emotion = read_emotion_vectors(EMOTION)
    words = read_emotion_words(WORDS)
    index = {label: position for position, label in enumerate(words.labels)}
    rows = family_rows(words, index)
    contrasts = words.expected_family_contrasts()
    poles = [contrast.pole for contrast in contrasts]

    numeric_valence, covered, missing = read_valence_norms(NORMS, words.labels)
    if numeric_valence is None or covered != len(words.labels):
        raise SystemExit(f"norms cover {covered}/{len(words.labels)}; missing {missing}")
    design = np.column_stack([np.ones(len(words.labels)), numeric_valence])

    residuals = _residuals_by_block(directions, emotion, design)
    observed = _statistics(residuals, contrasts, rows)

    # Gate: the recomputed sweep must equal the shipped one before any of it is interpreted.
    report = json.loads(REPORT.read_text())
    shipped = np.array(
        [
            [row["family_contrast_statistics"][pole] for pole in poles]
            for row in report["block_sweep"]
        ]
    )
    drift = float(np.abs(observed - shipped).max())
    if drift > 1e-9:
        raise SystemExit(f"recomputed sweep disagrees with the shipped report by {drift:.3g}")

    # One shuffle per draw, applied to EVERY block: the null keeps the cross-block correlation
    # that makes 64 blocks worth far fewer than 64 independent looks.
    rng = np.random.default_rng(args.seed)
    draws = np.empty((args.n_draws, residuals.shape[0], len(poles)))
    for draw in range(args.n_draws):
        order = rng.permutation(residuals.shape[1])
        draws[draw] = _statistics(residuals[:, order], contrasts, rows)

    vectors_meta = json.loads(EMOTION.read_text())
    threshold = float(vectors_meta[G0_THRESHOLD_KEY])
    g0_passing = np.array(
        [row["block"] for row in vectors_meta["g0_table"] if abs(row["spearman_rho"]) >= threshold]
    )
    families = {
        "all": np.arange(residuals.shape[0]),
        "g0_passing": g0_passing,
        # Reported because it was computed, and dropping it would be the mirror image of the
        # p-hacking this script exists to price. Blocks 16-63 is the contiguous deep band; it
        # excludes block 2 and 6-15, which G0 ADMITS, on no criterion that predates the peak. It
        # is the most favourable defensible family and it is not the headline.
        "post_hoc_deep_band_16_63": np.arange(16, residuals.shape[0]),
    }
    result = {
        "n_draws": args.n_draws,
        "seed": args.seed,
        "sweep_reproduction_max_abs_diff": drift,
        "g0_threshold": threshold,
        "g0_passing_blocks": f"{g0_passing.size}/{residuals.shape[0]}",
        "g0_failing_blocks": [
            block for block in range(residuals.shape[0]) if block not in set(g0_passing.tolist())
        ],
        "headline_blocks": report["headline_blocks"],
        "observed_at_headline": {
            str(block): dict(zip(poles, observed[block].tolist(), strict=True))
            for block in report["headline_blocks"]
        },
        "families": {
            name: _family_result(observed, draws, blocks, poles)
            for name, blocks in families.items()
        },
        "note": (
            "p_selection_aware is the p for 'the best block in this family'. Compare it with "
            "p_at_that_block_uncorrected at the SAME block: the gap is the price of depth "
            "shopping, and it is small only because neighbouring blocks are highly correlated. "
            "Nothing here is confirmatory — the sweep was looked at AFTER the headline blocks "
            "came back null, so this is a lead to pre-register on fresh data, not a finding."
        ),
    }
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
