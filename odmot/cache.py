"""Run the expensive networks ONCE per frame and save the results.

Why: on a CPU-only laptop Depth Anything takes on the order of a second per frame. The
experiments need dozens of policies x thresholds; re-running depth for each would take
weeks. Instead we cache, for every frame:

  detections : boxes + scores from YOLOv8, and the measured detector time
  depth      : per-detection nearness from Depth Anything, and the measured depth time

A policy run then *looks up* depth for a frame only when the policy asks for it, and is
*charged* that frame's measured depth time. This gives exactly the accuracy the live
system would have, with simulated (but measured) cost. `odmot demo` runs the real
models live on a clip to validate that the simulated latency matches reality.

Files (one per sequence; finished sequences are skipped, so caching is resumable):
  <cache>/det/<detector_name>/<seq>.npz    frames, boxes, scores, det_time
  <cache>/depth/<detector_name>/<depth_name>/<seq>.npz   nearness, depth_time
  ... plus a meta.json next to them with model info, mean times and memory.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .data import SeqInfo
from .models import box_nearness, normalize_nearness


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except Exception:  # psutil missing - memory is reported as NaN
        return float("nan")


def _read_bgr(path: Path) -> np.ndarray:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _progress(seq: str, i: int, n: int, t0: float, every: int = 50):
    if i % every == 0 or i == n:
        el = time.perf_counter() - t0
        eta = el / max(i, 1) * (n - i)
        print(f"  {seq}: {i}/{n} frames  ({el / max(i, 1):.2f}s/frame, ETA {eta / 60:.1f} min)",
              flush=True)


def det_cache_path(cache_dir, det_name, seq_name) -> Path:
    return Path(cache_dir) / "det" / det_name / f"{seq_name}.npz"


def depth_cache_path(cache_dir, det_name, depth_name, seq_name) -> Path:
    return Path(cache_dir) / "depth" / det_name / depth_name / f"{seq_name}.npz"


def build_detection_cache(detector, seqs: list[SeqInfo], cache_dir, warmup: int = 3,
                          rss_baseline: float | None = None) -> None:
    """detector: callable(bgr) -> (N,5) [x1,y1,x2,y2,score]; must have `.name`.

    rss_baseline: process memory (MB) measured BEFORE the model was loaded, so the
    reported model memory includes weights + activations (peak RSS - baseline).
    """
    rss_before = _rss_mb() if rss_baseline is None else rss_baseline
    peak = _rss_mb()
    meta = {"detector": detector.name, "sequences": {}}
    meta_path = Path(cache_dir) / "det" / detector.name / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    for seq in seqs:
        out = det_cache_path(cache_dir, detector.name, seq.name)
        if out.exists():
            print(f"[det] {seq.name}: cached, skipping")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"[det] {seq.name}: {seq.length} frames")
        first = _read_bgr(seq.image_path(1))
        for _ in range(warmup):          # first calls include one-off setup; don't time them
            detector(first)
        frames, boxes, scores = [], [], []
        det_time = np.zeros(seq.length, np.float64)
        t0 = time.perf_counter()
        for f in range(1, seq.length + 1):
            img = _read_bgr(seq.image_path(f))   # disk read is not part of model latency
            ts = time.perf_counter()
            d = detector(img)
            det_time[f - 1] = time.perf_counter() - ts
            if f % 25 == 0:
                peak = max(peak, _rss_mb())
            frames.append(np.full(len(d), f, np.int32))
            boxes.append(d[:, :4])
            scores.append(d[:, 4])
            _progress(seq.name, f, seq.length, t0)
        np.savez_compressed(out, frames=np.concatenate(frames), boxes=np.concatenate(boxes),
                            scores=np.concatenate(scores), det_time=det_time)
        meta["sequences"][seq.name] = {"mean_time_s": float(det_time.mean()),
                                       "n_dets": int(sum(len(x) for x in frames))}
        peak = max(peak, _rss_mb())
        meta["rss_model_mb"] = peak - rss_before if np.isfinite(rss_before) else None
        meta_path.write_text(json.dumps(meta, indent=2))


def build_depth_cache(estimator, seqs: list[SeqInfo], cache_dir, det_name: str,
                      warmup: int = 2, save_maps: bool = False,
                      rss_baseline: float | None = None) -> None:
    """estimator: callable(rgb) -> (h,w) relative inverse depth; must have `.name`.

    Requires the detection cache for `det_name` (per-box depth is stored for every cached
    detection, so the runner never needs the full depth maps).
    """
    rss_before = _rss_mb() if rss_baseline is None else rss_baseline
    peak = _rss_mb()
    root = Path(cache_dir) / "depth" / det_name / estimator.name
    meta_path = root / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else \
        {"depth_model": estimator.name, "detector": det_name, "sequences": {}}
    for seq in seqs:
        out = depth_cache_path(cache_dir, det_name, estimator.name, seq.name)
        if out.exists():
            print(f"[depth] {seq.name}: cached, skipping")
            continue
        det = np.load(det_cache_path(cache_dir, det_name, seq.name))
        frames, boxes = det["frames"], det["boxes"]
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"[depth] {seq.name}: {seq.length} frames")
        first = _read_bgr(seq.image_path(1))[:, :, ::-1].copy()
        for _ in range(warmup):
            estimator(first)
        nearness = np.full(len(frames), np.nan, np.float32)
        depth_time = np.zeros(seq.length, np.float64)
        maps = [] if save_maps else None
        order = np.argsort(frames, kind="stable")
        starts = np.searchsorted(frames[order], np.arange(1, seq.length + 2))
        t0 = time.perf_counter()
        for f in range(1, seq.length + 1):
            rgb = _read_bgr(seq.image_path(f))[:, :, ::-1].copy()
            ts = time.perf_counter()
            disp = estimator(rgb)
            near = normalize_nearness(disp)     # included in the timed cost: it is part of
            idx = order[starts[f - 1]:starts[f]]  # "computing depth" for this frame
            nearness[idx] = box_nearness(near, boxes[idx], seq.width, seq.height)
            depth_time[f - 1] = time.perf_counter() - ts
            if f % 25 == 0:
                peak = max(peak, _rss_mb())
            if maps is not None:
                maps.append((near * 255).astype(np.uint8))
            _progress(seq.name, f, seq.length, t0, every=25)
        extra = {"maps": np.stack(maps)} if maps else {}
        np.savez_compressed(out, nearness=nearness, depth_time=depth_time, **extra)
        meta["sequences"][seq.name] = {"mean_time_s": float(depth_time.mean())}
        peak = max(peak, _rss_mb())
        meta["rss_model_mb"] = peak - rss_before if np.isfinite(rss_before) else None
        meta_path.write_text(json.dumps(meta, indent=2))


class SequenceCache:
    """Everything the runner needs for one sequence, indexed by 1-based frame."""

    def __init__(self, cache_dir, det_name: str, depth_name: str | None, seq: SeqInfo):
        det = np.load(det_cache_path(cache_dir, det_name, seq.name))
        self.seq = seq
        frames = det["frames"]
        order = np.argsort(frames, kind="stable")
        self._frames = frames[order]
        self._boxes = det["boxes"][order].astype(np.float32)
        self._scores = det["scores"][order].astype(np.float32)
        self.det_time = det["det_time"]
        self._starts = np.searchsorted(self._frames, np.arange(1, seq.length + 2))
        self.has_depth = False
        if depth_name is not None:
            p = depth_cache_path(cache_dir, det_name, depth_name, seq.name)
            if p.exists():
                dep = np.load(p)
                if len(dep["nearness"]) != len(frames):
                    raise ValueError(f"Depth cache {p} does not match detection cache")
                self._near = dep["nearness"][order].astype(np.float32)
                self.depth_time = dep["depth_time"]
                self.has_depth = True
        if not self.has_depth:
            self._near = np.full(len(frames), np.nan, np.float32)
            self.depth_time = np.full(seq.length, np.nan)

    def frame(self, f: int):
        """(boxes (N,4) xyxy, scores (N,), nearness (N,)) for 1-based frame f."""
        a, b = self._starts[f - 1], self._starts[f]
        return self._boxes[a:b], self._scores[a:b], self._near[a:b]
