"""Live end-to-end run: real models on every frame, no caches.

Two uses:
  1. A demo video for the presentation (boxes coloured by ID, banner showing when depth
     fires and the current ambiguity score).
  2. Validating the cached simulation: the latency measured here, frame by frame, should
     match what the runner charges from the caches for the same policy and frames.
"""
from __future__ import annotations

import time

import numpy as np

from .data import SeqInfo
from .models import box_nearness, normalize_nearness
from .policies import DepthController, PolicyConfig
from .tracker import ByteTracker, TrackerConfig


def _color(tid: int):
    rng = np.random.default_rng(tid * 7919)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def run_live(seq: SeqInfo, frames: range, detector, estimator, policy: PolicyConfig,
             tcfg: TrackerConfig, video_path: str | None = None, show_depth_inset: bool = True):
    import cv2
    tracker = ByteTracker(tcfg, frame_rate=seq.fps)
    ctrl = DepthController(policy)
    writer = None
    stats = {"det": [], "depth": [], "track": [], "total": [], "used": [], "score": []}
    last_near = None
    for f in frames:
        bgr = cv2.imread(str(seq.image_path(f)))
        if bgr is None:
            raise FileNotFoundError(seq.image_path(f))
        t0 = time.perf_counter()
        det = detector(bgr)
        t_det = time.perf_counter() - t0
        boxes, scores = det[:, :4], det[:, 4]
        dt = {"t": 0.0}

        def depth_fn():
            nonlocal last_near
            ts = time.perf_counter()
            near = normalize_nearness(estimator(np.ascontiguousarray(bgr[:, :, ::-1])))
            v = box_nearness(near, boxes, seq.width, seq.height)
            dt["t"] = time.perf_counter() - ts
            last_near = near
            return v

        ts = time.perf_counter()
        tracks, dec = tracker.update(boxes, scores, seq.height, ctrl, depth_fn)
        t_track = time.perf_counter() - ts - dt["t"]
        total = time.perf_counter() - t0
        for k, v in (("det", t_det), ("depth", dt["t"]), ("track", t_track), ("total", total),
                     ("used", dec.real), ("score", dec.score)):
            stats[k].append(v)

        if video_path:
            vis = bgr.copy()
            for tid, tlwh, _ in tracks:
                x, y, w, h = (int(round(v)) for v in tlwh)
                cv2.rectangle(vis, (x, y), (x + w, y + h), _color(tid), 3)
                cv2.putText(vis, str(tid), (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, _color(tid), 2)
            on = dec.real
            banner = (f"frame {f}   depth {'ON ' if on else 'off'}   "
                      f"ambiguity {dec.score:.2f}   {total * 1e3:.0f} ms" if np.isfinite(dec.score)
                      else f"frame {f}   depth {'ON ' if on else 'off'}   {total * 1e3:.0f} ms")
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 46), (0, 0, 0), -1)
            cv2.putText(vis, banner, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (80, 220, 255) if on else (230, 230, 230), 2)
            if show_depth_inset and last_near is not None:
                ih = vis.shape[0] // 4
                iw = int(ih * last_near.shape[1] / last_near.shape[0])
                inset = cv2.applyColorMap(cv2.resize((last_near * 255).astype(np.uint8),
                                                     (iw, ih)), cv2.COLORMAP_INFERNO)
                vis[vis.shape[0] - ih:, vis.shape[1] - iw:] = inset
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(video_path, fourcc, seq.fps, (vis.shape[1], vis.shape[0]))
            writer.write(vis)
    if writer is not None:
        writer.release()
    out = {k: np.array(v, float) for k, v in stats.items()}
    return out
