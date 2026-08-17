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
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "figures"

# ---------------------------------------------------------------- design system
# One palette, one type family, one spine treatment across all four figures.
#
# Two categorical hues (a desaturated slate ink-blue and a warm terracotta) plus
# a neutral ink ramp. The pair passes every dataviz validator gate in light mode:
# lightness band, chroma floor, CVD separation (worst adjacent dE 16.3 protan,
# 23.6 tritan), normal-vision floor (23.1) and >= 3:1 contrast on the surface.
SLATE = "#255d94"  # categorical slot 1
RUST = "#a95c38"  # categorical slot 2
SLATE_TINT = "#cfdce8"  # slot-1 fill tint (histogram bodies)
BAND = "#f2f0ea"  # highlighted-region wash
INK = "#1a1a18"  # primary text
INK2 = "#54524c"  # secondary text / value labels
MUTED = "#8b8981"  # tertiary text, reference rules
HAIR = "#cfcdc5"  # spines, baselines
GRID = "#eceae4"  # gridlines

# Body text of the paper is Georgia; DejaVu Serif is the closest installed
# companion — same sturdy, low-contrast, large-x-height screen-serif lineage —
# and it ships with matplotlib, so the figures render identically from a clean
# checkout. (Noto Serif is a nearer match on paper but has no U+2212 MINUS
# SIGN, which every negative tick label and the "elated − disappointed" axis
# name need; Liberation/Nimbus Roman are Times clones and read colder.)
SERIF = ["DejaVu Serif", "Liberation Serif", "serif"]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": SERIF,
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": HAIR,
        "axes.linewidth": 0.6,
        "xtick.color": HAIR,
        "ytick.color": HAIR,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 0.0,
        "xtick.major.pad": 4,
        "ytick.major.pad": 3,
        "axes.labelcolor": INK,
        "axes.labelpad": 5,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "legend.handlelength": 1.0,
        "legend.handleheight": 0.9,
        "legend.handletextpad": 0.55,
        "legend.labelspacing": 0.45,
        "legend.columnspacing": 1.4,
        "legend.borderaxespad": 0.0,
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42,
    }
)

# Halos lift floating labels off dense marks without a boxy background; the
# light one is for labels that sit against a reference rule.
HALO = [pe.withStroke(linewidth=2.4, foreground="white")]
HALO_LIGHT = [pe.withStroke(linewidth=1.6, foreground="white")]

# All four figures are placed in the same PDF column (88% of the A4 text width,
# ~14.3 cm), so a wider canvas is shrunk further and its type lands smaller on
# the page. Figures 3 and 4 need a wider canvas than 1 and 2; these multipliers
# scale their type back up so every figure's smallest text is the same physical
# size in the printed paper.
TYPE3 = 1.15
TYPE4 = 1.25


def style_axes(ax: plt.Axes, *, xticks: bool = True) -> None:
    """Shared frame: left/top/right spines dropped, one hairline base rule,
    a recessive horizontal grid, and tick marks only where a scale is numeric."""
    ax.spines["bottom"].set_color(HAIR)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.yaxis.grid(True)
    if not xticks:
        ax.tick_params(axis="x", length=0)


def compact_legend(ax: plt.Axes, **kwargs) -> None:
    """Frameless legend parked above the plot area so it never competes with
    the marks; label text takes the secondary ink, not the series color."""
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.015), **kwargs)
    for text in leg.get_texts():
        text.set_color(INK2)


def load(rel: str) -> dict:
    with open(ROOT / rel) as fh:
        return json.load(fh)


def save(fig: plt.Figure, stem: str) -> list[str]:
    FIGDIR.mkdir(exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = FIGDIR / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.04)
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
    width = 0.24
    offset = width / 2 + 0.02  # 2px-equivalent surface gap between paired bars
    b1 = ax.bar(
        x - offset,
        [r["pooled_within_cell_slope"] for r in rm],
        width,
        color=SLATE,
        label=f"Reward-matched ({n_rm} cells): expectation contribution",
        zorder=3,
    )
    b2 = ax.bar(
        x + offset,
        [r["pooled_within_cell_slope"] for r in ev],
        width,
        color=RUST,
        label=f"EV-matched ({n_ev} pairs): reward contribution",
        zorder=3,
    )
    for bars in (b1, b2):
        for rect in bars:
            ax.annotate(
                f"+{rect.get_height():.4f}",
                (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK2,
            )
    # Ratio of matched effects, computed from the loaded slopes.
    ymax = max(r["pooled_within_cell_slope"] for r in rm + ev)
    for i in range(len(axes_order)):
        ratio = ev[i]["pooled_within_cell_slope"] / rm[i]["pooled_within_cell_slope"]
        ax.annotate(
            f"ratio {ratio:.2f}",
            (x[i], ymax * 1.165),
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK2,
        )
        # Hairline span bracket over the pair the ratio is taken from.
        ax.plot(
            [x[i] - offset, x[i] + offset],
            [ymax * 1.145, ymax * 1.145],
            color=MUTED,
            lw=0.55,
            solid_capstyle="butt",
            zorder=4,
        )
        for xx in (x[i] - offset, x[i] + offset):
            ax.plot(
                [xx, xx],
                [ymax * 1.115, ymax * 1.145],
                color=MUTED,
                lw=0.55,
                solid_capstyle="butt",
                zorder=4,
            )

    p_labels = {perm_p_label(r["p_value"], rep["n_permutations"]) for r in rm + ev}
    assert len(p_labels) == 1, "expected one shared permutation p-value"
    ax.text(
        0.0,
        -0.235,
        f"All four effects: permutation p = {p_labels.pop()} "
        f"({rep['n_permutations']:,} permutations).",
        transform=ax.transAxes,
        fontsize=7.5,
        color=MUTED,
    )
    ax.set_xticks(x, [axis_names[a] for a in axes_order])
    ax.set_ylabel("Pooled within-cell slope\n(projection per reward unit)")
    ax.set_xlim(-0.6, len(axes_order) - 0.4)
    ax.set_ylim(0, ymax * 1.26)
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    style_axes(ax, xticks=False)
    compact_legend(ax, ncols=1)
    return save(fig, "fig1_matched_effects")


# ---------------------------------------------------------------- Figure 2
def fig2_carryover() -> list[str]:
    rep = load("runs/emotion_vectors_base/emotions/behavioral_transfer_report_widened.json")
    gaps = np.asarray(rep["gate"]["per_pair_gap"], dtype=float)
    n = len(gaps)
    mean = gaps.mean()
    share_pos = (gaps > 0).mean()

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    # The gaps land on a dyadic lattice (multiples of 1/64 logit), so bin edges
    # are pushed a third of a bin off that lattice. Aligning them to it instead
    # puts a fifth of the sample exactly on an edge, where numpy assigns every
    # tied observation to the bin above and visibly skews the bar heights.
    step = 0.125
    lo = np.floor(gaps.min() / step) * step - step / 3
    hi = gaps.max() + step
    bins = np.arange(lo, hi, step)
    assert not np.isclose(gaps[:, None], bins[None, :]).any(), "value on a bin edge"
    ax.hist(
        gaps,
        bins=bins,
        color=SLATE_TINT,
        edgecolor=SLATE,
        linewidth=0.55,
        zorder=3,
    )
    ax.axvline(0, color=MUTED, lw=0.7, zorder=4)
    ax.axvline(mean, color=RUST, lw=1.1, zorder=5)
    ax.annotate(
        f"mean {mean:+.2f}",
        (mean, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8.5,
        color=RUST,
        zorder=6,
        path_effects=HALO,
    )
    ax.annotate(
        "0",
        (0, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(-4, 0),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8.5,
        color=MUTED,
        zorder=6,
        path_effects=HALO,
    )
    ax.text(
        0.985,
        0.86,
        f"{share_pos:.0%} of {n} pairs > 0",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=8.5,
        color=INK,
        zorder=6,
        path_effects=HALO,
    )
    ax.set_xlabel(
        "Per-pair risk-choice logit gap\n(after positive − after negative first-round outcome)"
    )
    ax.set_ylabel("Matched pairs")
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.tick_params(axis="x", which="minor", length=1.8, width=0.6, color=HAIR)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    counts, _ = np.histogram(gaps, bins=bins)
    ax.set_ylim(0, counts.max() * 1.10)  # headroom for the top-edge annotations
    style_axes(ax)
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
    # Word labels sit tight against their dot (a ~7pt offset, so the pairing is
    # unambiguous without leader lines). Sides are chosen so no label crosses
    # into a neighbouring family's dot field: `underwhelmed` and `disappointed`
    # are only ~0.003 apart in residual, so the upper one is stacked above its
    # dot and the lower one opens sideways.
    labelled = {
        "elated": (7, -3),
        "amused": (7, 0),
        "disappointed": (7, 0),
        "underwhelmed": (0, 5),
        "sad": (7, 0),
        "vindicated": (-7, 0),
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
            s=15,
            color=SLATE,
            alpha=0.62,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
        means[fam] = ys.mean()
        # Family mean: a thin rule, carried clear of the dot field by a white
        # underlay rather than by extra weight.
        for lw, colour, z in ((3.0, "white", 3.6), (1.1, INK, 4.0)):
            ax.plot(
                [i - 0.28, i + 0.28],
                [ys.mean()] * 2,
                color=colour,
                lw=lw,
                zorder=z,
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
                    ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                    va="center" if dx else "bottom",
                    fontsize=7.5 * TYPE3,
                    style="italic",
                    color=INK,
                    zorder=6,
                    path_effects=HALO_LIGHT,
                )

    ax.axhline(0, color=HAIR, lw=0.6, zorder=1)

    # Style-control pseudo-word: scale reference (not part of any family).
    ax.axhline(style["residual"], color=RUST, lw=0.9, ls=(0, (5, 3)), zorder=2)
    ax.annotate(
        f"style-control pseudo-word ({style['residual']:.3f})",
        (-0.55, style["residual"]),
        xytext=(0, 4),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=7.5 * TYPE3,
        color=RUST,
        zorder=6,
        path_effects=HALO,
    )

    # Headline family gap, computed from the loaded contrast row. The bracket
    # sits in the gutter between the two families the contrast is taken over,
    # so it cannot be read as comparing any other adjacent pair.
    m_out = pos_contrast["mean_residual_outcome"]
    m_ctl = pos_contrast["mean_residual_control"]
    fam_index = {fam: i for i, (fam, _) in enumerate(families)}
    xb = (
        fam_index[pos_contrast["outcome_family"]]
        + fam_index[pos_contrast["control_family"]]
    ) / 2
    ax.plot([xb, xb], [m_ctl, m_out], color=INK2, lw=0.7, zorder=5)
    for yy in (m_ctl, m_out):
        ax.plot([xb - 0.05, xb], [yy, yy], color=INK2, lw=0.7, zorder=5)
    ax.annotate(
        f"{pos_contrast['statistic']:+.4f}\n(p = {pos_contrast['p_value']:.4f})",
        (xb, m_out),
        xytext=(4, 1),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7.5 * TYPE3,
        color=INK,
        linespacing=1.35,
        zorder=6,
        path_effects=HALO,
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
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="both", which="major", labelsize=8 * TYPE3)
    ax.xaxis.label.set_size(8.5 * TYPE3)
    ax.yaxis.label.set_size(8.5 * TYPE3)
    style_axes(ax, xticks=False)
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
    # Highlighted band = the blocks at which the contrast peaks, one per run,
    # read off the loaded sweeps rather than written down.
    peaks = sorted({int(wx[np.argmax(wy)]), int(bx[np.argmax(by)])})
    peak_lo, peak_hi = peaks[0], peaks[-1]
    x_first, x_last = int(min(bx[0], wx[0])), int(max(bx[-1], wx[-1]))

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.axvspan(peak_lo - 0.5, peak_hi + 0.5, color=BAND, lw=0, zorder=0)
    ax.axvline(rpe_block, color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=2)
    ax.axhline(0, color=HAIR, lw=0.6, zorder=1)
    ax.plot(wx, wy, color=SLATE, lw=1.3, zorder=4, label=f"Widened run ({n_wide} words)")
    ax.plot(bx, by, color=RUST, lw=1.3, zorder=3, label=f"Base run ({n_base} words)")

    ax.annotate(
        f"block {rpe_block}\n(RPE instrument)",
        (rpe_block, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(-5, 0),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=7.5 * TYPE4,
        color=MUTED,
        linespacing=1.35,
        zorder=6,
        path_effects=HALO,
    )
    ax.annotate(
        f"blocks {peak_lo}–{peak_hi}",
        (peak_hi + 0.5, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=7.5 * TYPE4,
        color=MUTED,
        zorder=6,
        path_effects=HALO,
    )
    ax.text(
        0.015,
        0.97,
        f"depth profiles r = {r:.2f}",
        transform=ax.transAxes,
        fontsize=8 * TYPE4,
        color=INK,
        va="top",
        zorder=6,
        path_effects=HALO,
    )
    # Direct end labels so identity is not carried by color alone in print.
    for name, ys, colour in (("widened", wy, SLATE), ("base", by, RUST)):
        ax.annotate(
            name,
            (x_last, ys[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.5 * TYPE4,
            color=colour,
            annotation_clip=False,
        )
    ax.set_xlabel("Transformer block")
    ax.set_ylabel("Outcome-positive family contrast\n(valence-residualized)")
    ax.set_xlim(x_first, x_last)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(axis="x", which="minor", length=1.8, width=0.6, color=HAIR)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="both", which="major", labelsize=8 * TYPE4)
    ax.xaxis.label.set_size(8.5 * TYPE4)
    ax.yaxis.label.set_size(8.5 * TYPE4)
    style_axes(ax)
    compact_legend(ax, ncols=2, fontsize=8 * TYPE4)
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
