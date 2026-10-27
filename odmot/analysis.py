"""Does the ambiguity score predict the frames where depth matters?

Input: frames.csv from an always-depth run in analysis mode (`odmot analyze`). On every
frame the tracker solved the first association twice - with and without the depth term -
on the same tracker state, and recorded

    changed          depth changed at least one match
    correct_with / correct_without   matches agreeing with ground-truth identity
    wrong_with / wrong_without       matches contradicting ground-truth identity

From this we label frames
    helped : depth produced more correct (or fewer wrong) matches than no depth
    hurt   : the reverse

and measure, for each candidate score, how well it ranks those frames (ROC-AUC) and how
many of them a given depth budget would catch ("recall at budget"). A fixed-rate
policy catches, in expectation, the same fraction of them as its budget - the diagonal.
This analysis is independent of the full tracking metrics and explains them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ambiguity import SCORE_TYPES


def roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUC with tie handling; NaN if a class is empty."""
    label = label.astype(bool)
    n_pos, n_neg = label.sum(), (~label).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def recall_at_budget(score: np.ndarray, label: np.ndarray, budgets: np.ndarray) -> np.ndarray:
    """Fraction of positive frames caught when depth runs on the top-`b` scored frames."""
    label = label.astype(bool)
    if label.sum() == 0:
        return np.full(len(budgets), np.nan)
    rng = np.random.default_rng(0)
    order = np.lexsort((rng.random(len(score)), -score))   # random tie-break
    cum = np.cumsum(label[order]) / label.sum()
    k = np.clip(np.round(budgets * len(score)).astype(int), 0, len(score))
    return np.where(k > 0, cum[np.maximum(k - 1, 0)], 0.0)


def analyze_frames(frames_csv: str | Path, out_dir: str | Path) -> dict:
    df = pd.read_csv(frames_csv)
    df["changed"] = df["changed"].astype(str).str.lower().isin(["true", "1"])
    has_gt = "correct_with" in df
    if has_gt:
        gain = (df["correct_with"] - df["correct_without"]) - (df["wrong_with"] - df["wrong_without"])
        df["helped"], df["hurt"] = gain > 0, gain < 0
    labels = ["changed"] + (["helped", "hurt"] if has_gt else [])
    budgets = np.linspace(0, 1, 101)
    res = {"frames": int(len(df)),
           "rate": {lab: float(df[lab].mean()) for lab in labels},
           "auc": {}, "recall_at_budget": {"budget": budgets.tolist()}}
    if has_gt:
        res["net_matches_fixed_by_depth"] = int(
            (df["correct_with"] - df["correct_without"]).sum())
    for s in SCORE_TYPES:
        if s not in df:
            continue
        sc = df[s].to_numpy(float)
        res["auc"][s] = {lab: roc_auc(sc, df[lab].to_numpy()) for lab in labels}
        tgt = "helped" if has_gt else "changed"
        res["recall_at_budget"][s] = recall_at_budget(sc, df[tgt].to_numpy(), budgets).tolist()
    res["recall_at_budget"]["target"] = "helped" if has_gt else "changed"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "score_analysis.json").write_text(json.dumps(res, indent=2))
    return res


def format_analysis(res: dict) -> str:
    lines = [f"Frames analysed: {res['frames']}"]
    for k, v in res["rate"].items():
        lines.append(f"  depth {k:<8s} the first association on {100 * v:5.1f}% of frames")
    if "net_matches_fixed_by_depth" in res:
        lines.append(f"  net correct matches gained with depth: {res['net_matches_fixed_by_depth']}")
    lines.append("ROC-AUC of each score for predicting those frames (0.5 = random):")
    for s, a in res["auc"].items():
        lines.append("  " + f"{s:<7s} " + "  ".join(f"{k}={v:.3f}" for k, v in a.items()))
    b = np.array(res["recall_at_budget"]["budget"])
    tgt = res["recall_at_budget"]["target"]
    for budget in (0.1, 0.2, 0.3):
        i = int(np.argmin(np.abs(b - budget)))
        parts = [f"{s}={100 * res['recall_at_budget'][s][i]:.0f}%" for s in res["auc"]]
        lines.append(f"  depth on {int(budget * 100)}% of frames catches '{tgt}' frames: "
                     + ", ".join(parts) + f" (fixed-rate ~{int(budget * 100)}%)")
    return "\n".join(lines)
