"""E1 null diagnosis — what effect size could this pipeline have detected? CPU only.

Post-run analysis over the EXISTING artifacts (`map_geometry_report.json`, the norms CSV, the
story log). No model, no forwards, no new capture. It answers, in order:

1. **reproduce** — the report's word residuals are recomputed from its own cosines + the norms
   CSV, so every later number stands on a verified reconstruction.
2. **scale** — the observed family contrast placed against the instrument's own nuisance scale:
   the raw-cosine valence footprint, the P5c style vector's residual, the word-residual spread.
3. **power_empirical** — the missing power analysis. Noise = the OBSERVED word residuals
   (label-permuted, the same exchangeability the report's own floor uses); a known family effect
   delta is planted on the cosines, the valence regression is re-fit (so the covariate absorbs its
   share, as it would for a real effect), and detection = clearing the one-sided 5% permutation
   bar — the report's own criterion. Sweeping delta gives the minimum detectable effect (MDE) in
   the report's own units, which is what the observed +0.0254 must be judged against.
4. **power_planted** — the same question through the FULL map_geometry pipeline (regression,
   permutations, BOTH anisotropy floors), on a planted artifact whose geometry is calibrated to
   the observed instrument: per-word idiosyncratic content sized to the observed word-residual
   spread, a valence axis sized to the observed raw-cosine valence footprint, and a family
   appraisal signal swept over a ladder of amplitudes. Cross-validates 3 end to end.

   Why this is NOT the tests' ``_planted`` construction at lower noise: that construction gives a
   word vector ONLY its valence component (``e_j = valence_j·u + noise``), so every valence-0
   word is a near-zero-norm vector whose cosine is pure noise — at noise calibrated to the
   observed spread its word-residual p95 comes out ~0.56 vs the observed 0.098, and it gets WORSE
   as noise shrinks. The shipped positive control therefore only operates in its strong-signal
   regime (appraisal_scale=1.0 vs noise 0.01, SNR ~100): it demonstrates the machinery recovers a
   huge planted effect, and cannot be extended to ask about effects of the observed size. That is
   itself a sensitivity finding (docs/agents/experiment-gating.md: "the harness runs" is not
   sensitivity at the claim's scale).
5. **arousal** — the regression re-run with the arousal column the norms CSV already carries
   (both directions of every change reported; symmetric-amendment discipline).
6. **reversal** — the `disappointed`/`sad` named-pair reversal decomposed into raw cosine,
   valence-predicted, and residual parts.
7. **selection** — how much the G0 block-selection criterion actually discriminates, and where
   the family contrast sits over depth relative to the two headline blocks.
8. **topics** — per-word story-topic composition (each word draws a different random 12 of 25
   topics, which does NOT cancel in the grand-mean subtraction).

Writes ``e1_null_diagnosis.json`` next to the report and prints a summary. Nothing here is a new
headline statistic: it is the sensitivity accounting that decides whether the E1 null routes to
``diagnostic`` or ``harness_inadequate`` (docs/agents/experiment-gating.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from appraisal_emotions.analysis.emotion_mapping import read_valence_norms
from appraisal_emotions.analysis.valence_residual import ols_residuals
from appraisal_emotions.stimuli.emotion_stories import read_emotion_words

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "runs/emotion_vectors_base/emotions/map_geometry_report.json"
REPORT_BINARY = (
    REPO / "runs/emotion_vectors_base/emotions/map_geometry_report_binary_sensitivity.json"
)
STORIES = REPO / "runs/emotion_vectors_base/emotions/stories.json"
NORMS = REPO / "data/norms/vad_subset.csv"
WORDS = REPO / "data/emotion_words.json"
OUT = REPO / "runs/emotion_vectors_base/emotions/e1_null_diagnosis.json"

POLES = (
    ("positive", "outcome_pos", "nonoutcome_pos", 1),
    ("negative", "outcome_neg", "nonoutcome_neg", -1),
)


def _block_frame(report: dict, block: int) -> dict:
    entry = next(b for b in report["blocks"] if b["block"] == block)
    words = [row["word"] for row in entry["word_residuals"]]
    return {
        "entry": entry,
        "words": words,
        "family": {row["word"]: row["family"] for row in entry["word_residuals"]},
        "cos": np.array([row["cos_v_rpe"] for row in entry["word_residuals"]]),
        "resid_reported": np.array([row["residual"] for row in entry["word_residuals"]]),
        "binary": np.array([row["valence"] for row in entry["word_residuals"]], dtype=float),
    }


def _rows(frame: dict) -> dict[str, np.ndarray]:
    rows: dict[str, list[int]] = defaultdict(list)
    for i, w in enumerate(frame["words"]):
        rows[frame["family"][w]].append(i)
    return {k: np.array(v) for k, v in rows.items()}


def _pole_stats(resid: np.ndarray, rows: dict) -> dict[str, float]:
    return {
        pole: float(sign * (resid[rows[out]].mean() - resid[rows[ctl]].mean()))
        for pole, out, ctl, sign in POLES
    }


def _perm_p_and_floor(resid: np.ndarray, rows: dict, rng, n: int = 10_000) -> dict:
    """One-sided permutation p per pole plus the p95 floor of each pole and of the max."""

    stats = {pole: [] for pole, *_ in POLES}
    for _ in range(n):
        drawn = _pole_stats(rng.permutation(resid), rows)
        for pole, *_ in POLES:
            stats[pole].append(drawn[pole])
    observed = _pole_stats(resid, rows)
    out = {}
    for pole, *_ in POLES:
        draws = np.array(stats[pole])
        out[pole] = {
            "statistic": observed[pole],
            "p": float((np.sum(draws >= observed[pole]) + 1) / (n + 1)),
            "floor_p95": float(np.quantile(draws, 0.95)),
        }
    max_draws = np.maximum(np.array(stats["positive"]), np.array(stats["negative"]))
    out["max_floor_p95"] = float(np.quantile(max_draws, 0.95))
    return out


# ------------------------------------------------------------------ 3. empirical power


def empirical_power(
    frame: dict, design: np.ndarray, rows: dict, *, observed: float, n_sims: int, seed: int
) -> dict:
    """Plant delta on the cosines, re-fit the valence regression, count detections.

    Noise pool = the observed residuals, label-permuted — identical exchangeability to the
    report's own label-shuffle floor, so this measures the power of the test as fielded, against
    the noise it actually saw. The plant is +delta on outcome_pos and -delta on outcome_neg (the
    theory-expected two-pole structure); detection per pole is a one-sided 5% permutation test —
    the report's own criterion (its p95 label floor IS that test's critical value).
    """

    rng = np.random.default_rng(seed)
    resid_obs = ols_residuals(frame["cos"], design)
    plant = np.zeros(len(frame["words"]))
    plant[rows["outcome_pos"]] = 1.0
    plant[rows["outcome_neg"]] = -1.0

    # How much of a UNIT planted family effect survives the valence regression (per pole).
    resid_plant = ols_residuals(plant, design)
    absorption = _pole_stats(resid_plant, rows)

    deltas = sorted({0.0, 0.01, 0.02, round(observed, 4), 0.03, 0.04, 0.05, 0.06, 0.08, 0.10})
    n_floor = 300
    curve = []
    for delta in deltas:
        hits = {"positive": 0, "negative": 0, "both_floor_max": 0}
        for _ in range(n_sims):
            cos_sim = rng.permutation(resid_obs) + delta * plant
            resid_sim = ols_residuals(cos_sim, design)
            sim_stat = _pole_stats(resid_sim, rows)
            draws = {pole: np.empty(n_floor) for pole, *_ in POLES}
            for k in range(n_floor):
                drawn = _pole_stats(rng.permutation(resid_sim), rows)
                for pole, *_ in POLES:
                    draws[pole][k] = drawn[pole]
            for pole, *_ in POLES:
                if sim_stat[pole] > np.quantile(draws[pole], 0.95):
                    hits[pole] += 1
            max_floor = np.quantile(np.maximum(draws["positive"], draws["negative"]), 0.95)
            if max(sim_stat.values()) > max_floor:
                hits["both_floor_max"] += 1
        curve.append({"delta": delta, **{k: v / n_sims for k, v in hits.items()}})

    def mde(target: float, key: str) -> float | None:
        xs = [c["delta"] for c in curve]
        ys = [c[key] for c in curve]
        for x0, x1, y0, y1 in zip(xs, xs[1:], ys, ys[1:], strict=False):
            if y0 < target <= y1:
                return round(x0 + (x1 - x0) * (target - y0) / (y1 - y0), 4)
        return None

    sd = float(resid_obs.std(ddof=1))
    return {
        "residual_sd": sd,
        "analytic_se_contrast": sd * float(np.sqrt(1 / 9 + 1 / 10)),
        "unit_effect_surviving_valence_regression": absorption,
        "power_curve": curve,
        "mde50_positive_pole": mde(0.5, "positive"),
        "mde80_positive_pole": mde(0.8, "positive"),
        "mde80_max_criterion": mde(0.8, "both_floor_max"),
        "observed_positive_statistic": observed,
        "power_at_observed_positive": next(
            c["positive"] for c in curve if abs(c["delta"] - round(observed, 4)) < 1e-9
        ),
    }


# ------------------------------------------------------------------ 4. planted-pipeline power


def planted_pipeline_power(
    scratch: Path, *, target_resid_sd: float, target_footprint: float, seeds: int
) -> dict:
    """A calibrated-geometry planted artifact swept over amplitude through the REAL map_geometry.

    Construction (per seed): ``e_j = g_j + c_v·valence_j·u + a_j·w``, with ``g_j`` a per-word
    idiosyncratic content vector (what real emotion vectors mostly are), ``u`` the valence axis,
    ``w`` the appraisal axis, ``v_rpe = normalize(w + 0.8·u)`` (valence-loaded, as observed:
    P1 rho ~0.86). The hidden width sets the content-driven cosine spread (~1/sqrt(hidden)), so
    it is chosen to match the observed word-residual sd; ``c_v`` is chosen to match the observed
    raw-cosine valence footprint. The family plant ``a_j = ±amp`` on the outcome families is the
    §5 structure. Detection uses the report's own criteria: one-sided permutation p < 0.05 on the
    positive-pole family contrast, and ``clears_both_floors``.
    """

    sys.path.insert(0, str(REPO / "tests"))
    from appraisal_emotions.analysis.emotion_mapping import map_geometry  # noqa: PLC0415
    from appraisal_emotions.analysis.emotion_vectors import write_emotion_vectors  # noqa: PLC0415
    from appraisal_emotions.analysis.reveal_rpe import write_reveal_rpe_directions  # noqa: PLC0415
    from conftest import synthetic_directions_artifact, synthetic_emotion_artifact  # noqa: PLC0415

    words = read_emotion_words(WORDS)
    labels = words.labels
    valence = np.array([words.valence_by_word[w] for w in labels], dtype=float)
    hidden = int(round(1.0 / target_resid_sd**2))  # content cosine sd ~ 1/sqrt(hidden)
    # cos(u, v_rpe) = 0.8/sqrt(1.64); footprint = 2 * c_v * that (unit-norm content vectors).
    c_v = target_footprint / (2.0 * 0.8 / np.sqrt(1.0 + 0.8**2))

    def artifacts(amp: float, seed: int):
        rng = np.random.default_rng(seed)
        basis, _ = np.linalg.qr(rng.standard_normal((hidden, 3)))
        u, w, z = basis[:, 0], basis[:, 1], basis[:, 2]
        content = rng.standard_normal((len(labels) + 1, hidden))
        content /= np.linalg.norm(content, axis=1, keepdims=True)
        plant = np.array(
            [
                {"outcome_pos": amp, "outcome_neg": -amp}.get(words.category_by_word[w], 0.0)
                for w in labels
            ]
            + [0.0]  # style_control row: content only
        )
        vecs = content + np.outer(np.append(c_v * valence, 0.0), u) + np.outer(plant, w)
        v_rpe = (w + 0.8 * u) / np.linalg.norm(w + 0.8 * u)
        directions = np.stack([v_rpe[None, :], u[None, :], z[None, :]], axis=0)
        emotion = synthetic_emotion_artifact(
            vecs[:, None, :], labels, tuple(int(v) for v in valence)
        )
        return synthetic_directions_artifact(directions), emotion

    def run(amp: float, seed: int):
        directions, emotion = artifacts(amp, seed)
        dpath = scratch / "planted_directions.json"
        epath = scratch / "planted_emotion.json"
        write_reveal_rpe_directions(directions, dpath)
        write_emotion_vectors(emotion, epath)
        return map_geometry(
            dpath, epath, WORDS, None, seed=seed, n_permutations=1000, n_null_draws=200
        )

    # Calibration check at amp=0: realized residual sd and valence footprint vs targets.
    check = run(0.0, 999).blocks[0]
    check_sd = float(np.std([r.residual for r in check.word_residuals], ddof=1))
    cos0 = np.array([r.cos_v_rpe for r in check.word_residuals])
    check_footprint = float(cos0[valence > 0].mean() - cos0[valence < 0].mean())

    ladder = []
    for amp in [0.0, 0.013, 0.026, 0.033, 0.04, 0.052, 0.065, 0.078, 0.104]:
        realized, detected_p, detected_floors = [], 0, 0
        for seed in range(seeds):
            block = run(amp, 100 + seed).blocks[0]
            positive = next(c for c in block.family_contrasts if c.pole == "positive")
            realized.append(positive.statistic)
            detected_p += int(positive.p_value < 0.05)
            detected_floors += int(block.clears_both_floors)
        ladder.append(
            {
                "planted_amp": amp,
                "mean_realized_positive_statistic": float(np.mean(realized)),
                "power_positive_p05": detected_p / seeds,
                "power_clears_both_floors": detected_floors / seeds,
            }
        )
    return {
        "hidden": hidden,
        "c_valence": c_v,
        "target_resid_sd": target_resid_sd,
        "calibrated_resid_sd": check_sd,
        "target_footprint": target_footprint,
        "calibrated_footprint": check_footprint,
        "n_seeds_per_scale": seeds,
        "ladder": ladder,
    }


# ------------------------------------------------------------------ per-block sections 1/2/3/5/6


def _diagnose_block(
    report: dict,
    block: int,
    words,
    numeric_valence: np.ndarray,
    arousal: dict[str, float],
    *,
    n_sims: int,
) -> dict:
    labels = words.labels
    frame = _block_frame(report, block)
    assert frame["words"] == list(labels)
    rows = _rows(frame)
    design = np.column_stack([np.ones(84), numeric_valence])
    resid = ols_residuals(frame["cos"], design)

    # 1. reproduce
    reproduce = {"max_abs_residual_diff": float(np.abs(resid - frame["resid_reported"]).max())}

    # 2. scale
    raw_by_pole = float(
        frame["cos"][frame["binary"] > 0].mean() - frame["cos"][frame["binary"] < 0].mean()
    )
    p5c_rpe = next(e for e in frame["entry"]["p5c"] if e["family"] == "v_rpe")
    observed_positive = frame["entry"]["family_contrasts"][0]["statistic"]
    scale = {
        "raw_cos_valence_footprint(mean_pos_minus_mean_neg)": raw_by_pole,
        "observed_positive_pole_contrast": observed_positive,
        "style_control_residual_v_rpe": p5c_rpe["residual"],
        "style_over_observed_ratio": abs(p5c_rpe["residual"] / observed_positive),
        "word_residual_sd": float(resid.std(ddof=1)),
        "subspace_fraction_v_rpe_in_emotion_span": frame["entry"]["subspace_fraction_emotion_span"],
    }

    # 3. empirical power
    power = empirical_power(
        frame, design, rows, observed=observed_positive, n_sims=n_sims, seed=1000 + block
    )

    # 5. arousal partialled (both directions of every change reported)
    arousal_vec = np.array([arousal[w] for w in labels])
    design3 = np.column_stack([design, arousal_vec])
    resid3 = ols_residuals(frame["cos"], design3)
    arousal_out = {
        "valence_only": _perm_p_and_floor(resid, rows, np.random.default_rng(3000 + block)),
        "valence_plus_arousal": _perm_p_and_floor(
            resid3, rows, np.random.default_rng(2000 + block)
        ),
        "pairs_valence_only": {},
        "pairs_valence_plus_arousal": {},
    }
    for pair in words.expected_pairs():
        i, j = labels.index(pair.outcome), labels.index(pair.control)
        arousal_out["pairs_valence_only"][f"{pair.outcome}_vs_{pair.control}"] = float(
            pair.expected_sign * (resid[i] - resid[j])
        )
        arousal_out["pairs_valence_plus_arousal"][f"{pair.outcome}_vs_{pair.control}"] = float(
            pair.expected_sign * (resid3[i] - resid3[j])
        )

    # 6. reversal decomposition
    i, j = labels.index("disappointed"), labels.index("sad")
    fitted = frame["cos"] - resid
    reversal = {
        "raw_cos": {"disappointed": float(frame["cos"][i]), "sad": float(frame["cos"][j])},
        "valence_norms": {
            "disappointed": float(numeric_valence[i]),
            "sad": float(numeric_valence[j]),
        },
        "fitted_from_valence": {"disappointed": float(fitted[i]), "sad": float(fitted[j])},
        "residual": {"disappointed": float(resid[i]), "sad": float(resid[j])},
        "residual_arousal_partialled": {
            "disappointed": float(resid3[i]),
            "sad": float(resid3[j]),
        },
        "outcome_neg_family_mean_residual": float(resid[rows["outcome_neg"]].mean()),
        "nonoutcome_neg_family_mean_residual": float(resid[rows["nonoutcome_neg"]].mean()),
    }

    return {
        "reproduce": reproduce,
        "scale": scale,
        "power_empirical": power,
        "arousal": arousal_out,
        "reversal": reversal,
    }


# ------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sims", type=int, default=1000)
    parser.add_argument("--planted-seeds", type=int, default=24)
    parser.add_argument(
        "--scratch", type=Path, required=True, help="scratch dir for the planted sweep"
    )
    args = parser.parse_args()

    report = json.loads(REPORT.read_text())
    binary_report = json.loads(REPORT_BINARY.read_text())
    words = read_emotion_words(WORDS)
    labels = words.labels
    numeric_valence, covered, missing = read_valence_norms(NORMS, labels)
    assert numeric_valence is not None and covered == 84, (covered, missing)
    arousal = {
        row["word"]: float(row["arousal"]) for row in __import__("csv").DictReader(NORMS.open())
    }
    out: dict = {"inputs": {"report": str(REPORT), "norms": str(NORMS)}}

    for block in report["headline_blocks"]:
        out[f"block_{block}"] = _diagnose_block(
            report, block, words, numeric_valence, arousal, n_sims=args.n_sims
        )

    # 4. planted-pipeline power, geometry calibrated to the block-35 observations. The synthetic
    # run uses the binary labels (no norms), so the binary-design spread is the matching target.
    binary_b35 = next(b for b in binary_report["blocks"] if b["block"] == 35)
    b35_resid = np.array([r["residual"] for r in binary_b35["word_residuals"]])
    b35_cos = np.array([r["cos_v_rpe"] for r in binary_b35["word_residuals"]])
    b35_val = np.array([r["valence"] for r in binary_b35["word_residuals"]], dtype=float)
    args.scratch.mkdir(parents=True, exist_ok=True)
    out["power_planted_pipeline"] = planted_pipeline_power(
        args.scratch,
        target_resid_sd=float(b35_resid.std(ddof=1)),
        target_footprint=float(b35_cos[b35_val > 0].mean() - b35_cos[b35_val < 0].mean()),
        seeds=args.planted_seeds,
    )
    out["power_planted_pipeline"]["binary_report_observed_positive"] = binary_b35[
        "family_contrasts"
    ][0]["statistic"]

    # 7. selection-criterion accounting
    vectors_meta = json.loads(
        (REPO / "runs/emotion_vectors_base/emotions/emotion_vectors.json").read_text()
    )
    g0 = [(r["block"], abs(r["spearman_rho"])) for r in vectors_meta["g0_table"]]
    deep = [v for b, v in g0 if b >= 16]
    sweep = {r["block"]: r["family_contrast_statistics"]["positive"] for r in report["block_sweep"]}
    positive_signs = [b for b in range(20, 64) if sweep[b] > 0]
    argmax_block = max(sweep, key=lambda b: sweep[b] if b >= 20 else -1)
    frame35, frame63 = _block_frame(report, 35), _block_frame(report, 63)
    design_n = np.column_stack([np.ones(84), numeric_valence])
    resid_corr = float(
        np.corrcoef(
            ols_residuals(frame35["cos"], design_n), ols_residuals(frame63["cos"], design_n)
        )[0, 1]
    )
    out["selection"] = {
        "word_residual_corr_b35_b63": resid_corr,
        "g0_selected_block": vectors_meta["selected_block"],
        "g0_rho_at_selected": max(v for b, v in g0 if b == vectors_meta["selected_block"]),
        "g0_runner_up": sorted(g0, key=lambda t: -t[1])[1],
        "g0_rho_range_blocks_16_to_63": [min(deep), max(deep)],
        "positive_pole_sign_positive_blocks_20_to_63": f"{len(positive_signs)}/44",
        "positive_pole_argmax_block_from_20": [argmax_block, sweep[argmax_block]],
        "positive_pole_at_headline": {"35": sweep[35], "63": sweep[63]},
        "note": (
            "The sweep has no per-block floors and word-level cosines are only reported at the "
            "two headline blocks, so whether the mid-band peak clears its own selection-aware "
            "floor is NOT computable from the shipped report — it needs the emotion_vectors npz."
        ),
    }

    # 8. topic composition
    stories = json.loads(STORIES.read_text())
    topics_by_word: dict[str, set] = defaultdict(set)
    for story in stories:
        if story["drop_reason"] is None and story["emotion"] in labels:
            topics_by_word[story["emotion"]].add(story["topic"])
    overlaps = []
    word_list = list(labels)
    for a in range(len(word_list)):
        for b in range(a + 1, len(word_list)):
            overlaps.append(len(topics_by_word[word_list[a]] & topics_by_word[word_list[b]]))
    out["topics"] = {
        "topics_per_word_min_max": [
            min(len(v) for v in topics_by_word.values()),
            max(len(v) for v in topics_by_word.values()),
        ],
        "pairwise_shared_topics_min_mean_max": [
            int(min(overlaps)),
            float(np.mean(overlaps)),
            int(max(overlaps)),
        ],
        "disappointed_sad_shared_topics": len(
            topics_by_word["disappointed"] & topics_by_word["sad"]
        ),
        "note": (
            "Each word draws a random 12 of 25 topics (per-word seed), so topic composition is a "
            "word-level nuisance that the grand-mean subtraction does not cancel; at "
            "stories_per_emotion=25 the grid is fully crossed and it cancels exactly."
        ),
    }

    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
