"""Combine cost and accuracy into one table, with a budget-matched comparison.

The research question is "at the same depth budget, does on-demand beat fixed-rate?".
Fixed-rate policies (plus no-depth at rate 0 and always-depth at rate 1) trace a curve
of accuracy vs. depth rate. For every policy we report how far it sits above or below
that curve at its own depth rate:  dHOTA_vs_fixed > 0 means "better than spending the
same depth budget on a fixed schedule".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ["HOTA", "DetA", "AssA", "IDF1", "IDSW", "MOTA", "Frag"]
COST = ["depth_rate", "latency_mean_ms", "latency_p95_ms", "fps", "tracker_ms",
        "depth_compute_s", "model_memory_mb"]
POLICY = ["policy", "type", "every", "score", "threshold", "hold", "fallback"]


def build_summary(costs: dict, metrics: dict) -> pd.DataFrame:
    rows = []
    for name, c in costs.items():
        r = {k: c.get(k) for k in POLICY + COST}
        r.update({k: metrics[name][k] for k in METRICS})
        rows.append(r)
    df = pd.DataFrame(rows)
    ref = df[df["type"].isin(["never", "fixed", "always"]) & (df["fallback"] == "none")]
    if len(ref) >= 2:
        # fixed-rate runs with the same N but different phase offsets are averaged into one
        # point of the reference curve (rate and metrics), so no single lucky/unlucky phase
        # defines "fixed-rate"
        key = ref["type"].astype(str) + "_" + ref["every"].fillna(0).astype(str)
        g = ref.groupby(key)[["depth_rate", "HOTA", "IDF1", "AssA", "IDSW"]].mean().sort_values("depth_rate")
        for m in ("HOTA", "IDF1", "AssA", "IDSW"):
            df[f"d{m}_vs_fixed"] = df[m] - np.interp(df["depth_rate"], g["depth_rate"], g[m])
    return df.sort_values(["type", "depth_rate"]).reset_index(drop=True)


def pareto_mask(rate: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """True for points not dominated (lower-or-equal depth rate AND higher-or-equal acc)."""
    keep = np.ones(len(rate), bool)
    for i in range(len(rate)):
        dom = (rate <= rate[i]) & (acc >= acc[i]) & ((rate < rate[i]) | (acc > acc[i]))
        keep[i] = not dom.any()
    return keep


def format_summary(df: pd.DataFrame) -> str:
    cols = ["policy", "depth_rate", "HOTA", "AssA", "IDF1", "IDSW", "latency_mean_ms",
            "latency_p95_ms"] + [c for c in ("dHOTA_vs_fixed", "dIDF1_vs_fixed") if c in df]
    t = df[cols].copy()
    t["depth_rate"] = (100 * t["depth_rate"]).round(1).astype(str) + "%"
    t["pareto"] = np.where(pareto_mask(df["depth_rate"].to_numpy(), df["HOTA"].to_numpy()),
                           "*", "")
    with pd.option_context("display.max_rows", 200, "display.width", 200,
                           "display.float_format", "{:.2f}".format):
        return ("\n" + t.to_string(index=False) +
                "\n\n* = Pareto-optimal (no other policy has both lower depth rate and higher "
                "HOTA).\nd*_vs_fixed = difference to the fixed-rate curve at the same depth rate.")
