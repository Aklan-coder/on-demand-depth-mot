"""Run one depth policy over a split using the caches, with per-frame cost accounting.

Per-frame latency charged to a policy (all times measured on YOUR machine):

    latency(f) = det_time(f) + tracker_time(f) + [depth_time(f) if depth was used at f]

tracker_time includes the ambiguity-score computation (the controller runs inside the
tracker update), so on-demand policies pay for their own decision logic.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from .cache import SequenceCache
from .ambiguity import pairwise_iou
from .data import SeqInfo, frame_range, load_gt
from .policies import DepthController, PolicyConfig
from .tracker import ByteTracker, TrackerConfig


def gt_by_frame(seq: SeqInfo) -> dict[int, np.ndarray]:
    """{frame: (K,5) [id, x1, y1, x2, y2]} for evaluated pedestrians (consider=1, class=1)."""
    gt = load_gt(seq)
    gt = gt[(gt[:, 6] == 1) & (gt[:, 7] == 1)]
    out = {}
    for f in np.unique(gt[:, 0]).astype(int):
        g = gt[gt[:, 0] == f]
        out[f] = np.c_[g[:, 1], g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]]
    return out


def match_dets_to_gt(boxes: np.ndarray, gt: np.ndarray | None, iou_thr: float = 0.5):
    """GT identity for each detection (-1 if unmatched), Hungarian on IoU."""
    from scipy.optimize import linear_sum_assignment
    ids = np.full(len(boxes), -1, int)
    if gt is None or len(gt) == 0 or len(boxes) == 0:
        return ids
    iou = pairwise_iou(boxes, gt[:, 1:5])
    r, c = linear_sum_assignment(-iou)
    ok = iou[r, c] >= iou_thr
    ids[r[ok]] = gt[c[ok], 0].astype(int)
    return ids


def run_sequence(seq: SeqInfo, cache: SequenceCache, policy: PolicyConfig,
                 tcfg: TrackerConfig, split: str, out_txt: Path | None,
                 frame_log: list | None = None):
    if policy.uses_real_depth and not cache.has_depth:
        raise FileNotFoundError(
            f"Policy '{policy.name}' needs depth but there is no depth cache for {seq.name}. "
            "Run `python -m odmot cache-depth` first.")
    tracker = ByteTracker(tcfg, frame_rate=seq.fps)
    ctrl = DepthController(policy)
    fr = frame_range(seq.length, split)
    lat, used, trk_t = [], [], []
    rows = []
    gtf = gt_by_frame(seq) if (frame_log is not None and seq.gt_path.exists()) else None
    for local, f in enumerate(fr, start=1):
        boxes, scores, near = cache.frame(f)
        lookup = {"t": 0.0}

        def depth_fn(near=near):
            ts = time.perf_counter()
            v = near
            lookup["t"] += time.perf_counter() - ts
            return v

        log = {} if frame_log is not None else None
        det_gt = match_dets_to_gt(boxes, gtf.get(f)) if gtf is not None else None
        ts = time.perf_counter()
        tracks, dec = tracker.update(boxes, scores, seq.height, ctrl, depth_fn, log=log,
                                     det_gt=det_gt)
        t_track = time.perf_counter() - ts - lookup["t"]
        t = cache.det_time[f - 1] + t_track + (cache.depth_time[f - 1] if dec.real else 0.0)
        lat.append(t)
        used.append(dec.real)
        trk_t.append(t_track)
        for tid, tlwh, sc in tracks:
            w, h = tlwh[2], tlwh[3]
            if w * h > tcfg.min_box_area and w / max(h, 1e-6) <= tcfg.max_aspect:
                rows.append((local, tid, *tlwh, sc))
        if log is not None:
            log.update(seq=seq.name, frame=local, ondemand_score=dec.score)
            frame_log.append(log)
    if out_txt is not None:
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(out_txt, "w") as fh:
            for r in rows:
                fh.write(f"{r[0]},{r[1]},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},"
                         f"{r[6]:.4f},-1,-1,-1\n")
    return {"latency": np.array(lat), "depth_used": np.array(used, bool),
            "tracker_time": np.array(trk_t),
            "det_time": cache.det_time[fr.start - 1:fr.stop - 1],
            "depth_time": cache.depth_time[fr.start - 1:fr.stop - 1]}


def summarize_cost(per_seq: dict[str, dict]) -> dict:
    lat = np.concatenate([r["latency"] for r in per_seq.values()])
    used = np.concatenate([r["depth_used"] for r in per_seq.values()])
    trk = np.concatenate([r["tracker_time"] for r in per_seq.values()])
    dt = np.concatenate([r["depth_time"] for r in per_seq.values()])
    return {
        "frames": int(len(lat)),
        "depth_frames": int(used.sum()),
        "depth_rate": float(used.mean()) if len(used) else 0.0,
        "latency_mean_ms": float(lat.mean() * 1e3),
        "latency_p50_ms": float(np.percentile(lat, 50) * 1e3),
        "latency_p95_ms": float(np.percentile(lat, 95) * 1e3),
        "latency_max_ms": float(lat.max() * 1e3),
        "fps": float(1.0 / lat.mean()),
        "tracker_ms": float(trk.mean() * 1e3),
        "depth_compute_s": float(np.nansum(np.where(used, dt, 0.0))),
        "total_compute_s": float(lat.sum()),
    }


def run_policy(seqs: list[SeqInfo], caches: dict[str, SequenceCache], policy: PolicyConfig,
               tcfg: TrackerConfig, split: str, run_dir: Path,
               log_frames: bool = False, memory: dict | None = None) -> dict:
    """Writes run_dir/<policy>/data/<seq>.txt (TrackEval format) and cost.json."""
    pdir = Path(run_dir) / policy.name
    per_seq, frame_log = {}, ([] if log_frames else None)
    for seq in seqs:
        per_seq[seq.name] = run_sequence(seq, caches[seq.name], policy, tcfg, split,
                                         pdir / "data" / f"{seq.name}.txt", frame_log)
    cost = summarize_cost(per_seq)
    cost["policy"] = policy.name
    cost["type"] = policy.type
    cost["every"] = policy.every if policy.type == "fixed" else None
    cost["score"] = policy.score if policy.type == "ondemand" else None
    cost["threshold"] = policy.threshold if policy.type == "ondemand" else None
    cost["hold"] = policy.hold if policy.type == "ondemand" else None
    cost["fallback"] = policy.fallback
    if memory:
        # On-demand does NOT save memory: the depth model must stay loaded to be callable.
        mem = memory.get("det_mb") or 0.0
        if policy.uses_real_depth:
            mem += memory.get("depth_mb") or 0.0
        cost["model_memory_mb"] = mem
    cost["per_sequence"] = {k: {"depth_rate": float(v["depth_used"].mean()),
                                "latency_mean_ms": float(v["latency"].mean() * 1e3)}
                            for k, v in per_seq.items()}
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "cost.json").write_text(json.dumps(cost, indent=2))
    if frame_log is not None:
        keys = sorted({k for r in frame_log for k in r})
        with open(pdir / "frames.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(frame_log)
    return cost
