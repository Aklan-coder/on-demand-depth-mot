"""Generate a small synthetic dataset in exact MOT17 layout, plus matching caches.

Used to test the full pipeline (caches -> policies -> TrackEval -> report -> plots)
without the real dataset or models. Pedestrians walk at different depths with
non-constant velocity and cross each other; the far person in an overlap gets a lower
detection score or is missed (occlusion). Detections are jittered GT boxes plus a few
false positives; "real depth" is true nearness plus noise; timings are fake but shaped
like a CPU laptop (detector ~300 ms, depth ~1.2 s).

    python tests/make_synthetic.py --out /tmp/synth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from odmot.ambiguity import pairwise_iou  # noqa: E402
from odmot.cache import depth_cache_path, det_cache_path  # noqa: E402

W, H = 1280, 720


def simulate(seed: int, n_frames: int, n_people: int):
    rng = np.random.default_rng(seed)
    people = []
    for pid in range(1, n_people + 1):
        near = rng.uniform(0.15, 0.95)                # 1 = close to camera
        h = 90 + 330 * near                           # closer -> taller
        foot = 330 + 360 * near + rng.normal(0, 25)   # closer -> lower in image (+ noise:
        #                                               uneven ground breaks pseudo-depth)
        direction = rng.choice([-1, 1])
        x0 = rng.uniform(-200, W + 200)
        speed = rng.uniform(2, 7) * (0.5 + near)
        start = int(rng.integers(0, n_frames // 3))
        people.append(dict(id=pid, near=near, h=h, foot=foot, dir=direction, x=x0,
                           v=speed, start=start, phase=rng.uniform(0, 6.28)))
    gt, dets = [], []
    for f in range(1, n_frames + 1):
        boxes = []
        for p in people:
            if f < p["start"]:
                continue
            # non-constant velocity: speed oscillates and sometimes reverses briefly
            v = p["v"] * (1 + 0.8 * np.sin(0.05 * f + p["phase"]))
            p["x"] += p["dir"] * v
            if p["x"] < -300 or p["x"] > W + 300:
                p["dir"] *= -1
            w = 0.41 * p["h"]
            x1, y2 = p["x"] - w / 2, p["foot"] + 2 * np.sin(0.3 * f + p["phase"])
            boxes.append((p["id"], x1, y2 - p["h"], x1 + w, y2, p["near"]))
        if not boxes:
            continue
        b = np.array(boxes)
        iou = pairwise_iou(b[:, 1:5], b[:, 1:5])
        np.fill_diagonal(iou, 0)
        for k, (pid, x1, y1, x2, y2, near) in enumerate(boxes):
            inside = (x2 > 0) and (x1 < W)
            if not inside:
                continue
            # visibility: occluded by any nearer, overlapping person
            occ = max([iou[k, j] for j in range(len(boxes)) if boxes[j][5] > near] + [0.0])
            vis = float(np.clip(1 - 2.0 * occ, 0, 1))
            gt.append((f, pid, x1, y1, x2 - x1, y2 - y1, 1, 1, vis))
            if vis < 0.15 and rng.random() < 0.7:
                continue                              # missed under heavy occlusion
            score = float(np.clip(rng.normal(0.85, 0.07) * (0.35 + 0.65 * vis), 0.06, 0.99))
            j = rng.normal(0, 0.04 * (y2 - y1), 4)
            depth = float(np.clip(near + rng.normal(0, 0.03), 0, 1))
            dets.append((f, x1 + j[0], y1 + j[1], x2 + j[2], y2 + j[3], score, depth))
        for _ in range(rng.poisson(0.3)):             # false positives
            cx, cy, hh = rng.uniform(0, W), rng.uniform(300, 700), rng.uniform(80, 300)
            dets.append((f, cx - 0.2 * hh, cy - hh, cx + 0.2 * hh, cy, rng.uniform(0.1, 0.6),
                         rng.uniform(0, 1)))
    return np.array(gt), np.array(dets)


def write_sequence(root: Path, cache: Path, name: str, seed: int, n_frames: int, n_people: int,
                   det_name: str, depth_name: str):
    import cv2
    gt, dets = simulate(seed, n_frames, n_people)
    seq_dir = root / "train" / f"{name}-FRCNN"
    (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
    (seq_dir / "img1").mkdir(exist_ok=True)
    blank = np.full((36, 64, 3), 128, np.uint8)        # tiny placeholder images
    for f in range(1, n_frames + 1):
        cv2.imwrite(str(seq_dir / "img1" / f"{f:06d}.jpg"), blank)
    np.savetxt(seq_dir / "gt" / "gt.txt", gt,
               fmt=["%d", "%d", "%.2f", "%.2f", "%.2f", "%.2f", "%d", "%d", "%.4f"], delimiter=",")
    (seq_dir / "seqinfo.ini").write_text(
        f"[Sequence]\nname={name}-FRCNN\nimDir=img1\nframeRate=30\nseqLength={n_frames}\n"
        f"imWidth={W}\nimHeight={H}\nimExt=.jpg\n")
    rng = np.random.default_rng(seed + 1000)
    dp = det_cache_path(cache, det_name, name)
    dp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dp, frames=dets[:, 0].astype(np.int32),
                        boxes=dets[:, 1:5].astype(np.float32),
                        scores=dets[:, 5].astype(np.float32),
                        det_time=rng.normal(0.30, 0.02, n_frames).clip(0.2))
    zp = depth_cache_path(cache, det_name, depth_name, name)
    zp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(zp, nearness=dets[:, 6].astype(np.float32),
                        depth_time=rng.normal(1.2, 0.08, n_frames).clip(0.9))
    return len(gt), len(dets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=600)
    a = ap.parse_args()
    out = Path(a.out)
    seqs = {"MOT17-02": (1, 14), "MOT17-04": (2, 20), "MOT17-09": (3, 10)}
    for name, (seed, n) in seqs.items():
        g, d = write_sequence(out / "MOT17", out / "cache", name, seed, a.frames, n,
                              "yolov8s_1088", "Depth-Anything-V2-Small-hf_518")
        print(f"{name}: {g} GT boxes, {d} detections")
    cfg = (Path(__file__).resolve().parents[1] / "configs" / "default.yaml").read_text()
    cfg = cfg.replace("mot17_root: data/MOT17", f"mot17_root: {out / 'MOT17'}")
    cfg = cfg.replace("cache_dir: cache ", f"cache_dir: {out / 'cache'} ")
    cfg = cfg.replace("results_dir: results ", f"results_dir: {out / 'results'} ")
    cfg = cfg.replace("sequences: [MOT17-02, MOT17-04, MOT17-05, MOT17-09, MOT17-10, MOT17-11, MOT17-13]",
                      f"sequences: [{', '.join(seqs)}]")
    (out / "synthetic.yaml").write_text(cfg)
    print(f"Config: {out / 'synthetic.yaml'}")


if __name__ == "__main__":
    main()
