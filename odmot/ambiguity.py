"""Association-ambiguity scores: cheap signals that predict when depth could change a match.

All scores are computed from quantities the tracker has ALREADY computed for the frame
(the track-to-detection similarity matrix and the detection boxes), so they cost
microseconds. Each returns a float in [0, 1]; higher = more ambiguous.

margin  For every track, compare its best and second-best candidate detection under the
        tracker's own similarity (IoU x score). Ambiguity = second / best, but only when
        the second candidate is itself a *valid* match (similarity >= the tracker's
        match gate). Done for rows (tracks) and columns (detections); frame score = max.
        Intuition: if each track has one clear candidate, the Hungarian solution is the
        same with or without depth. Depth can only flip a decision when a track has two
        plausible candidates (or a detection two plausible tracks).

crowd   Max pairwise IoU between the frame's high-confidence detections. Intuition:
        overlapping people are where ID switches happen. Does not look at tracks, so it
        can fire one or two frames *before* the tracks start competing (useful to get a
        fresh depth reading for the tracks before the crossing).

hybrid  max(margin, crowd).
"""
from __future__ import annotations

import numpy as np

SCORE_TYPES = ("margin", "crowd", "hybrid")


def _row_ratio(sim: np.ndarray, gate: float) -> np.ndarray:
    if sim.shape[0] == 0 or sim.shape[1] < 2:
        return np.zeros(sim.shape[0])
    top2 = -np.partition(-sim, 1, axis=1)[:, :2]
    best, second = top2.max(axis=1), top2.min(axis=1)
    return np.where((best > 0) & (second >= gate), second / np.maximum(best, 1e-9), 0.0)


def margin_score(sim: np.ndarray, gate: float) -> float:
    """sim: (tracks, detections) similarity in [0,1]; gate: min similarity of a valid match."""
    if sim.size == 0:
        return 0.0
    r = max(_row_ratio(sim, gate).max(initial=0.0), _row_ratio(sim.T, gate).max(initial=0.0))
    return float(r)


def pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between two sets of xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def crowd_score(det_boxes: np.ndarray) -> float:
    if len(det_boxes) < 2:
        return 0.0
    iou = pairwise_iou(det_boxes, det_boxes)
    np.fill_diagonal(iou, 0.0)
    return float(iou.max())


def all_scores(sim: np.ndarray, det_boxes: np.ndarray, gate: float) -> dict[str, float]:
    m, c = margin_score(sim, gate), crowd_score(det_boxes)
    return {"margin": m, "crowd": c, "hybrid": max(m, c)}


def compute_score(kind: str, sim: np.ndarray, det_boxes: np.ndarray, gate: float) -> float:
    if kind == "margin":
        return margin_score(sim, gate)
    if kind == "crowd":
        return crowd_score(det_boxes)
    if kind == "hybrid":
        return max(margin_score(sim, gate), crowd_score(det_boxes))
    raise ValueError(f"Unknown ambiguity score '{kind}', choose from {SCORE_TYPES}")
