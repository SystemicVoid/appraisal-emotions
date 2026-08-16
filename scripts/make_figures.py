"""Generate the four figures for submission.md from versioned run artifacts.

Every plotted value and every annotated number is loaded from a JSON artifact at
runtime (docs/agents/rails.md: never hand-copy text or numbers held in a file).

Sources
-------
Fig 1  runs/emotion_vectors_base/emotions/expectation_control_report.json
Fig 2  runs/emotion_vectors_base/emotions/behavioral_transfer_report_widened.json
Fig 3  runs/emotion_vectors_wide/emotions/map_geometry_report.json
Fig 4  runs/emotion_vectors_wide/emotions/map_geometry_report_valence_only.json
       runs/emotion_vectors_base/emotions/map_geometry_report.json
       (the base contract residualizes on valence only — config.py defaults
       residualize_on=("valence",) — so the base main report IS the matched
       valence-only sweep; the wide run's main report adds arousal, hence its
       separate *_valence_only.json file.)

Usage:  uv run --with matplotlib python scripts/make_figures.py
Writes: figures/fig{1..4}_*.png (300 dpi) and .pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "figures"

# Colorblind-safe palette (dataviz reference palette, light mode; the
# blue/orange pair passes all validator gates: worst CVD dE 24.7, normal 33.6).
BLUE = "#2a78d6"  # categorical slot 1
ORANGE = "#eb6834"  # categorical slot 2
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.labelcolor": INK,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)


def load(rel: str) -> dict:
    with open(ROOT / rel) as fh:
        return json.load(fh)


def save(fig: plt.Figure, stem: str) -> list[str]:
    FIGDIR.mkdir(exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = FIGDIR / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths.append(str(p))
    plt.close(fig)
    return paths


def perm_p_label(p: float, n_permutations: int) -> str:
    """Render the permutation-floor p-value the way the paper does (1/10001)."""
    denom = round(1.0 / p)
    if denom == n_permutations + 1:
        return f"1/{denom}"
    return f"{p:.2g}"


# ---------------------------------------------------------------- Figure 1
def fig1_matched_effects() -> list[str]:
    rep = load("runs/emotion_vectors_base/emotions/expectation_control_report.json")
    axes_order = ["pc1_affect_concept_valence", "elated_minus_disappointed"]
    axis_names = {
        "pc1_affect_concept_valence": "PC1 (emotion-concept valence)",
        "elated_minus_disappointed": "elated − disappointed",
    }
    arms = {arm["cell_family"]: arm for arm in rep["arms"]}

    def slope(arm_key: str, axis_key: str) -> dict:
        (row,) = [a for a in arms[arm_key]["axes"] if a["axis"] == axis_key]
        return row

    rm = [slope("reward_matched", ax) for ax in axes_order]
    ev = [slope("ev_matched", ax) for ax in axes_order]
    n_rm = rm[0]["n_cells"]
    n_ev = ev[0]["n_cells"]

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    x = np.arange(len(axes_order))
    width = 0.32
    b1 = ax.bar(
        x - width / 2,
        [r["pooled_within_cell_slope"] for r in rm],
        width,
        color=BLUE,
        label=f"Reward-matched ({n_rm} cells): expectation contribution",
        zorder=3,
    )
    b2 = ax.bar(
        x + width / 2,
        [r["pooled_within_cell_slope"] for r in ev],
        width,
        color=ORANGE,
        label=f"EV-matched ({n_ev} pairs): reward contribution",
        zorder=3,
    )
    for bars in (b1, b2):
        for rect in bars:
            ax.annotate(
                f"+{rect.get_height():.4f}",
                (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=INK2,
            )
    # Ratio of matched effects, computed from the loaded slopes.
    ymax = max(r["pooled_within_cell_slope"] for r in rm + ev)
    for i in range(len(axes_order)):
        ratio = ev[i]["pooled_within_cell_slope"] / rm[i]["pooled_within_cell_slope"]
        ax.annotate(
            f"ratio {ratio:.2f}",
            (x[i], ymax * 1.16),
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
        )
        ax.plot(
            [x[i] - width / 2, x[i] + width / 2],
            [ymax * 1.14, ymax * 1.14],
            color=MUTED,
            lw=0.8,
            solid_capstyle="butt",
        )
        for xx in (x[i] - width / 2, x[i] + width / 2):
            ax.plot([xx, xx], [ymax * 1.11, ymax * 1.14], color=MUTED, lw=0.8)

    p_labels = {perm_p_label(r["p_value"], rep["n_permutations"]) for r in rm + ev}
    assert len(p_labels) == 1, "expected one shared permutation p-value"
    ax.text(
        0.0,
        -0.26,
        f"All four effects: permutation p = {p_labels.pop()} "
        f"({rep['n_permutations']:,} permutations).",
        transform=ax.transAxes,
        fontsize=8,
        color=INK2,
    )
    ax.set_xticks(x, [axis_names[a] for a in axes_order])
    ax.set_ylabel("Pooled within-cell slope\n(projection per reward unit)")
    ax.set_ylim(0, ymax * 1.26)
    ax.yaxis.grid(True, zorder=0)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncols=1,
        handlelength=1.2,
        borderaxespad=0.0,
    )
    ax.axhline(0, color=BASELINE, lw=0.8)
    return save(fig, "fig1_matched_effects")


# ---------------------------------------------------------------- Figure 2
def fig2_carryover() -> list[str]:
    rep = load("runs/emotion_vectors_base/emotions/behavioral_transfer_report_widened.json")
    gaps = np.asarray(rep["gate"]["per_pair_gap"], dtype=float)
    n = len(gaps)
    mean = gaps.mean()
    share_pos = (gaps > 0).mean()

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    # Gaps are quantized on a 1/32-logit grid; align bin edges to half-steps.
    step = 4 / 32
    lo = np.floor(gaps.min() / step) * step - step / 2
    hi = np.ceil(gaps.max() / step) * step + step
    bins = np.arange(lo, hi, step)
    ax.hist(gaps, bins=bins, color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(0, color=INK2, lw=0.9, zorder=4)
    ax.axvline(mean, color=ORANGE, lw=1.4, zorder=4)
    ax.annotate(
        f"mean {mean:+.2f}",
        (mean, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(5, -2),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=9,
        color=ORANGE,
    )
    ax.annotate(
        "0",
        (0, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(-5, -2),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=9,
        color=INK2,
    )
    ax.text(
        0.97,
        0.80,
        f"{share_pos:.0%} of {n} pairs > 0",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        color=INK,
    )
    ax.set_xlabel(
        "Per-pair risk-choice logit gap\n(after positive − after negative first-round outcome)"
    )
    ax.set_ylabel("Matched pairs")
    ax.yaxis.grid(True, zorder=0)
    return save(fig, "fig2_carryover")


# ---------------------------------------------------------------- Figure 3
def fig3_family_residuals() -> list[str]:
    rep = load("runs/emotion_vectors_wide/emotions/map_geometry_report.json")
    (blk,) = [b for b in rep["blocks"] if b["block"] == rep["emotion_selected_block"]]
    rows = blk["word_residuals"]
    (pos_contrast,) = [c for c in blk["family_contrasts"] if c["pole"] == "positive"]
    (style,) = [s for s in blk["p5c"] if s["family"] == "v_rpe"]

    families = [
        ("outcome_pos", "Outcome-linked\npositive"),
        ("nonoutcome_pos", "Non-outcome\npositive"),
        ("outcome_neg", "Outcome-linked\nnegative"),
        ("nonoutcome_neg", "Non-outcome\nnegative"),
        ("outcome_confirm", "Expectation-\nconfirmation"),
    ]
    labelled = {
        "elated": (9, 0),
        "amused": (9, 0),
        "disappointed": (-9, 3),
        "underwhelmed": (9, 3),
        "sad": (9, 0),
        "vindicated": (9, 0),
    }

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    rng = np.random.default_rng(7)
    means = {}
    for i, (fam, disp) in enumerate(families):
        frows = [r for r in rows if r["family"] == fam]
        ys = np.array([r["residual"] for r in frows])
        xs = i + rng.uniform(-0.16, 0.16, size=len(frows))
        ax.scatter(
            xs,
            ys,
            s=22,
            color=BLUE,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        means[fam] = ys.mean()
        ax.plot(
            [i - 0.26, i + 0.26],
            [ys.mean()] * 2,
            color=INK,
            lw=1.6,
            zorder=4,
            solid_capstyle="butt",
        )
        for r, xx, yy in zip(frows, xs, ys):
            if r["word"] in labelled:
                dx, dy = labelled[r["word"]]
                text = r["word"]
                if r["word"] == "amused":  # residual value is discussed in the text
                    text = f"amused ({r['residual']:.3f})".replace("-", "−")
                ax.annotate(
                    text,
                    (xx, yy),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha="left" if dx > 0 else "right",
                    va="center",
                    fontsize=8,
                    style="italic",
                    color=INK2,
                    zorder=5,
                )

    ax.axhline(0, color=BASELINE, lw=0.8, zorder=1)

    # Style-control pseudo-word: scale reference (not part of any family).
    ax.axhline(style["residual"], color=ORANGE, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.annotate(
        f"style-control pseudo-word ({style['residual']:.3f})",
        (len(families) - 0.55, style["residual"]),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        color=ORANGE,
    )

    # Headline family gap, computed from the loaded contrast row.
    m_out = pos_contrast["mean_residual_outcome"]
    m_ctl = pos_contrast["mean_residual_control"]
    xb = 1.42
    ax.plot([xb, xb], [m_ctl, m_out], color=INK, lw=1.0)
    for yy in (m_ctl, m_out):
        ax.plot([xb - 0.04, xb], [yy, yy], color=INK, lw=1.0)
    ax.annotate(
        f"{pos_contrast['statistic']:+.4f}\n(p = {pos_contrast['p_value']:.4f})",
        (xb + 0.05, m_out),
        ha="left",
        va="center",
        fontsize=8,
        color=INK,
    )

    ax.set_xticks(
        range(len(families)),
        [
            f"{disp}\n(n = {sum(r['family'] == fam for r in rows)})"
            for fam, disp in families
        ],
    )
    ax.set_xlim(-0.6, len(families) - 0.4)
    ax.set_ylabel("RPE-alignment residual\n(valence + arousal removed)")
    ax.yaxis.grid(True, zorder=0)
    return save(fig, "fig3_family_residuals")


# ---------------------------------------------------------------- Figure 4
def fig4_depth_profile() -> list[str]:
    wide = load("runs/emotion_vectors_wide/emotions/map_geometry_report_valence_only.json")
    base = load("runs/emotion_vectors_base/emotions/map_geometry_report.json")
    assert base.get("residualize_on") is None  # base contract default: ("valence",)
    assert wide["residualize_on"] == ["valence"]

    def series(rep: dict) -> tuple[np.ndarray, np.ndarray]:
        blocks = np.array([b["block"] for b in rep["block_sweep"]])
        vals = np.array(
            [b["family_contrast_statistics"]["positive"] for b in rep["block_sweep"]]
        )
        order = np.argsort(blocks)
        return blocks[order], vals[order]

    bx, by = series(base)
    wx, wy = series(wide)
    r = np.corrcoef(by, wy)[0, 1]
    n_wide = wide["norms_covered_words"]
    n_base = base["norms_covered_words"]
    rpe_block = wide["directions_selected_block"]

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.axvspan(47.5, 50.5, color=GRID, alpha=0.6, zorder=1)
    ax.axvline(rpe_block, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.axhline(0, color=BASELINE, lw=0.8, zorder=1)
    ax.plot(wx, wy, color=BLUE, lw=1.8, zorder=3, label=f"Widened run ({n_wide} words)")
    ax.plot(bx, by, color=ORANGE, lw=1.8, zorder=3, label=f"Base run ({n_base} words)")

    ax.annotate(
        f"block {rpe_block}\n(RPE instrument)",
        (rpe_block, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(-5, -2),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8,
        color=INK2,
    )
    ax.annotate(
        "blocks 48–50",
        (49, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(5, -2),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        color=INK2,
    )
    ax.text(
        0.02,
        0.95,
        f"depth profiles r = {r:.2f}",
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK,
        va="top",
    )
    # Direct end labels so identity is not carried by color alone in print.
    for name, ys in (("widened", wy), ("base", by)):
        ax.annotate(
            name,
            (63, ys[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
            color=INK2,
            annotation_clip=False,
        )
    ax.set_xlabel("Transformer block")
    ax.set_ylabel("Outcome-positive family contrast\n(valence-residualized)")
    ax.set_xlim(0, 63)
    ax.yaxis.grid(True, zorder=0)
    ax.legend(loc="lower right", handlelength=1.6)
    return save(fig, "fig4_depth_profile")


def main() -> None:
    written = []
    written += fig1_matched_effects()
    written += fig2_carryover()
    written += fig3_family_residuals()
    written += fig4_depth_profile()
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
