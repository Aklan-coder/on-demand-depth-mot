"""Presentation-ready plots (PNG, 200 dpi) of the tracking-compute trade-off."""
from __future__ import annotations

from pathlib import Path

import numpy as np

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SURFACE = "#fcfcfb"
FAMILY = {  # fixed categorical order; on-demand families get the first three slots
    "od_margin": ("#2a78d6", "On-demand (margin)"),
    "od_crowd": ("#eb6834", "On-demand (crowd)"),
    "od_hybrid": ("#1baf7a", "On-demand (hybrid)"),
}
REF = ("#52514e", "Fixed-rate (every N frames)")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _family(name: str) -> str | None:
    for k in FAMILY:
        if name == k or name.startswith(k + "_thr"):
            return k
    return None


def plot_tradeoff(df, out_dir, title_suffix: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out_dir)
    panels = [("HOTA", "HOTA (higher is better)"), ("IDF1", "IDF1 (higher is better)"),
              ("IDSW", "ID switches (lower is better)")]
    for xkey, xlabel, fname in (("depth_rate", "Frames with depth computed (%)", "tradeoff_depth_rate.png"),
                                ("latency_mean_ms", "Mean latency per frame (ms, measured on this machine)",
                                 "tradeoff_latency.png")):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
        xs = df[xkey] * (100 if xkey == "depth_rate" else 1)
        for ax, (m, ylabel) in zip(axes, panels):
            _style(ax)
            ref = df[df["type"].isin(["never", "fixed", "always"]) & (df["fallback"] == "none")]
            ref = ref.assign(_x=xs[ref.index]).sort_values("_x")
            ax.plot(ref["_x"], ref[m], color=REF[0], lw=2, ls="--", marker="o", ms=6,
                    label=REF[1], zorder=2)
            for fam, (col, lab) in FAMILY.items():
                d = df[np.array([_family(n) == fam for n in df["policy"]], bool)]
                if len(d):
                    d = d.assign(_x=xs[d.index]).sort_values("_x")
                    dotted = fam == "od_hybrid"   # often coincides with margin: draw on top
                    ax.plot(d["_x"], d[m], color=col, lw=2.2 if dotted else 2,
                            ls=":" if dotted else "-", marker="o", ms=4 if dotted else 7,
                            label=lab, zorder=4 if dotted else 3,
                            markeredgecolor=SURFACE, markeredgewidth=1.0 if dotted else 1.5)
            other = df[(df["type"] == "ondemand").to_numpy()
                       & np.array([_family(n) is None for n in df["policy"]], bool)]
            if len(other):
                ax.scatter(xs[other.index], other[m], s=55, facecolors="none",
                           edgecolors=FAMILY["od_margin"][0], linewidths=1.6, zorder=3,
                           label="On-demand variants (hold / pseudo fallback)")
            for key, lab, off, ha in (("never", "no depth", (6, -14), "left"),
                                      ("pseudo", "pseudo-depth", (6, 7), "left"),
                                      ("always", "always depth", (-6, 9), "right")):
                d = df[df["type"] == key]
                if len(d):
                    x, y = xs[d.index[0]], d[m].iloc[0]
                    ax.scatter([x], [y], s=70, color=INK, marker="D" if key == "pseudo" else "s",
                               zorder=4, edgecolors=SURFACE, linewidths=1.5)
                    ax.annotate(lab, (x, y), textcoords="offset points", xytext=off, ha=ha,
                                fontsize=9, color=INK2, zorder=5)
            ax.set_xlabel(xlabel, color=INK2, fontsize=10)
            ax.set_title(ylabel, color=INK, fontsize=11, loc="left")
            if xkey == "latency_mean_ms" and (xs > 0).all():
                ax.set_xscale("log")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=min(len(l), 5), frameon=False, fontsize=9,
                   labelcolor=INK2)
        fig.suptitle(f"Tracking accuracy vs. depth cost  {('- ' + title_suffix) if title_suffix else ''}",
                     x=0.01, ha="left", color=INK, fontsize=13)
        fig.tight_layout(rect=(0, 0.08, 1, 0.95))
        fig.savefig(out / fname, dpi=200, facecolor=SURFACE)
        plt.close(fig)


def plot_score_analysis(res: dict, out_dir, title_suffix: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rb = res["recall_at_budget"]
    b = np.array(rb["budget"]) * 100
    fig, ax = plt.subplots(figsize=(6.4, 5), facecolor=SURFACE)
    _style(ax)
    ax.plot(b, b, color=REF[0], ls="--", lw=2, label="Fixed-rate / random (expected)")
    colors = {"margin": FAMILY["od_margin"][0], "crowd": FAMILY["od_crowd"][0],
              "hybrid": FAMILY["od_hybrid"][0]}
    for s, col in colors.items():
        if s in rb:
            auc = res["auc"][s].get(rb["target"], float("nan"))
            ax.plot(b, np.array(rb[s]) * 100, color=col, lw=2.2 if s == "hybrid" else 2,
                    ls=":" if s == "hybrid" else "-", zorder=4 if s == "hybrid" else 3,
                    label=f"{s} score (AUC {auc:.2f})")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 101)
    ax.set_xlabel("Depth budget: % of frames with depth", color=INK2)
    ax.set_ylabel(f"% of '{rb['target']}' frames caught", color=INK2)
    ax.set_title(f"Does the ambiguity score find the frames where depth {rb['target']}?"
                 + (f"\n{title_suffix}" if title_suffix else ""), color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "score_recall_at_budget.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
