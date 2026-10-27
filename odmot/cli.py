"""Command-line entry point:  python -m odmot <command> [options]

  check        verify the dataset layout and write TrackEval ground truth for each split
  estimate     time a few frames of each model on this machine and project caching time
  cache-det    run YOLOv8 on every frame once and cache boxes + timings   (slow, resumable)
  cache-depth  run Depth Anything on every frame once and cache per-box depth (slowest)
  sweep        run all policies on a split from the caches, evaluate with TrackEval,
               write summary.csv and plots                                  (fast)
  analyze      score-quality analysis: does the ambiguity score predict where depth helps?
  plot         re-draw plots from an existing summary.csv / analysis
  demo         live end-to-end run with real models on a clip; optional annotated video
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

import numpy as np

from .data import SPLITS, find_sequence, frame_range, prepare_trackeval_gt
from .models import depth_model_name, detector_name
from .policies import expand_policies
from .tracker import TrackerConfig


# ----------------------------------------------------------------------------- helpers
def load_config(path: str) -> dict:
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


def get_seqs(cfg, names=None):
    names = names or cfg["sequences"]
    return [find_sequence(cfg["paths"]["mot17_root"], n) for n in names]


def names(cfg):
    d, z = cfg["detector"], cfg["depth"]
    return detector_name(d["weights"], d["imgsz"]), depth_model_name(z["model"], z["input_size"])


def gt_dir(cfg, split):
    return Path(cfg["paths"]["results_dir"]) / split / "_gt"


def ensure_gt(cfg, seqs, split):
    g = gt_dir(cfg, split)
    if not all((g / s.name / "gt" / "gt.txt").exists() for s in seqs):
        prepare_trackeval_gt(seqs, split, g)
    return g


def model_memory(cfg) -> dict:
    det_n, dep_n = names(cfg)
    cache = Path(cfg["paths"]["cache_dir"])
    out = {}
    for key, p in (("det_mb", cache / "det" / det_n / "meta.json"),
                   ("depth_mb", cache / "depth" / det_n / dep_n / "meta.json")):
        if p.exists():
            out[key] = json.loads(p.read_text()).get("rss_model_mb")
    return out


def select_policies(cfg, only: str | None):
    pols = expand_policies(cfg["policies"])
    if only:
        pats = [p.strip() for p in only.split(",") if p.strip()]
        pols = [p for p in pols if any(fnmatch.fnmatch(p.name, pat) or p.name == pat
                                       for pat in pats)]
        if not pols:
            sys.exit(f"No policies match --only {only}")
    return pols


# ---------------------------------------------------------------------------- commands
def cmd_check(cfg, args):
    seqs = get_seqs(cfg, args.seqs)
    total = 0
    for s in seqs:
        n_img = len(list((s.path / s.im_dir).glob(f"*{s.im_ext}")))
        ok = "OK" if n_img == s.length else f"MISSING IMAGES ({n_img}/{s.length})"
        gt = "gt OK" if s.gt_path.exists() else "NO GT"
        print(f"{s.name}: {s.length} frames {s.width}x{s.height} @{s.fps:g}fps  {ok}  {gt}")
        total += s.length
    for split in SPLITS:
        prepare_trackeval_gt(seqs, split, gt_dir(cfg, split))
    n_val = sum(len(frame_range(s.length, "val_half")) for s in seqs)
    print(f"Total {total} frames ({n_val} in val_half). TrackEval GT written under "
          f"{Path(cfg['paths']['results_dir'])}/<split>/_gt")


def _build_detector(cfg):
    from .models import YoloDetector
    d = cfg["detector"]
    return YoloDetector(d["weights"], d["imgsz"], d["conf"], d.get("iou", 0.7), d["device"],
                        d.get("threads", 0), d.get("max_det", 300))


def _build_depth(cfg):
    from .models import DepthAnythingEstimator
    z = cfg["depth"]
    return DepthAnythingEstimator(z["model"], z["input_size"], z["device"], z.get("threads", 0))


def cmd_estimate(cfg, args):
    import time

    import cv2
    seqs = get_seqs(cfg, args.seqs)
    s = seqs[0]
    imgs = [cv2.imread(str(s.image_path(f))) for f in range(1, args.frames + 3)]
    total_frames = sum(x.length for x in seqs)
    for label, build, conv in (("YOLOv8", _build_detector, lambda im: im),
                               ("Depth Anything", _build_depth,
                                lambda im: np.ascontiguousarray(im[:, :, ::-1]))):
        m = build(cfg)
        m(conv(imgs[0]))
        m(conv(imgs[1]))      # warm-up
        ts = []
        for im in imgs[2:]:
            t0 = time.perf_counter()
            m(conv(im))
            ts.append(time.perf_counter() - t0)
        t = float(np.median(ts))
        print(f"{label}: {t * 1e3:.0f} ms/frame -> {total_frames} frames ~ "
              f"{t * total_frames / 3600:.1f} h")
        del m


def cmd_cache_det(cfg, args):
    from .cache import _rss_mb, build_detection_cache
    seqs = get_seqs(cfg, args.seqs)
    base = _rss_mb()          # measured before the model is loaded
    build_detection_cache(_build_detector(cfg), seqs, cfg["paths"]["cache_dir"],
                          rss_baseline=base)


def cmd_cache_depth(cfg, args):
    from .cache import _rss_mb, build_depth_cache, det_cache_path
    det_n, _ = names(cfg)
    seqs = get_seqs(cfg, args.seqs)
    missing = [s.name for s in seqs
               if not det_cache_path(cfg["paths"]["cache_dir"], det_n, s.name).exists()]
    if missing:
        sys.exit(f"Detection cache missing for {missing}; run `python -m odmot cache-det` first.")
    base = _rss_mb()
    build_depth_cache(_build_depth(cfg), seqs, cfg["paths"]["cache_dir"], det_n,
                      save_maps=cfg["depth"].get("save_maps", False), rss_baseline=base)


def _load_caches(cfg, seqs, need_depth):
    from .cache import SequenceCache
    det_n, dep_n = names(cfg)
    caches = {s.name: SequenceCache(cfg["paths"]["cache_dir"], det_n, dep_n, s) for s in seqs}
    if need_depth:
        missing = [k for k, c in caches.items() if not c.has_depth]
        if missing:
            sys.exit(f"No depth cache for {missing}. Run `python -m odmot cache-depth` first, "
                     "or restrict to depth-free policies with --only no_depth,pseudo_depth")
    return caches


def cmd_sweep(cfg, args):
    from .evaluate import evaluate
    from .plots import plot_tradeoff
    from .report import build_summary, format_summary
    from .runner import run_policy
    seqs = get_seqs(cfg, args.seqs)
    pols = select_policies(cfg, args.only)
    caches = _load_caches(cfg, seqs, any(p.uses_real_depth for p in pols))
    tcfg = TrackerConfig.from_dict(cfg.get("tracker"))
    g = ensure_gt(cfg, seqs, args.split)
    run_dir = Path(cfg["paths"]["results_dir"]) / args.split / args.tag
    mem = model_memory(cfg)
    costs = {}
    for i, p in enumerate(pols, 1):
        c = run_policy(seqs, caches, p, tcfg, args.split, run_dir, memory=mem)
        costs[p.name] = c
        print(f"[{i}/{len(pols)}] {p.name:<28s} depth on {100 * c['depth_rate']:5.1f}% of "
              f"frames, {c['latency_mean_ms']:7.1f} ms/frame", flush=True)
    print("Evaluating with TrackEval ...", flush=True)
    metrics = evaluate(g, run_dir, list(costs), [s.name for s in seqs],
                       cfg["paths"].get("trackeval"), benchmark=cfg.get("benchmark", "MOT17"))
    df = build_summary(costs, metrics)
    df.to_csv(run_dir / "summary.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(format_summary(df))
    plot_tradeoff(df, run_dir, title_suffix=f"{cfg.get('benchmark', 'MOT17')} {args.split}")
    print(f"\nWrote {run_dir / 'summary.csv'} and plots in {run_dir}")


def cmd_analyze(cfg, args):
    from .analysis import analyze_frames, format_analysis
    from .plots import plot_score_analysis
    from .policies import PolicyConfig
    from .runner import run_policy
    seqs = get_seqs(cfg, args.seqs)
    caches = _load_caches(cfg, seqs, True)
    tcfg = TrackerConfig.from_dict(cfg.get("tracker"))
    run_dir = Path(cfg["paths"]["results_dir"]) / args.split / "analysis"
    p = PolicyConfig(name="always_depth_logged", type="always")
    run_policy(seqs, caches, p, tcfg, args.split, run_dir, log_frames=True)
    res = analyze_frames(run_dir / p.name / "frames.csv", run_dir)
    print(format_analysis(res))
    plot_score_analysis(res, run_dir, title_suffix=f"{cfg.get('benchmark', 'MOT17')} {args.split}")
    print(f"\nWrote {run_dir / 'score_analysis.json'} and plots in {run_dir}")


def cmd_plot(cfg, args):
    import pandas as pd

    from .plots import plot_score_analysis, plot_tradeoff
    base = Path(cfg["paths"]["results_dir"]) / args.split
    s = base / args.tag / "summary.csv"
    if s.exists():
        plot_tradeoff(pd.read_csv(s), base / args.tag, title_suffix=f"{cfg.get('benchmark', 'MOT17')} {args.split}")
        print(f"Plots written to {base / args.tag}")
    a = base / "analysis" / "score_analysis.json"
    if a.exists():
        plot_score_analysis(json.loads(a.read_text()), base / "analysis",
                            title_suffix=f"{cfg.get('benchmark', 'MOT17')} {args.split}")
        print(f"Plots written to {base / 'analysis'}")


def cmd_demo(cfg, args):
    from .live import run_live
    seq = get_seqs(cfg, [args.seq])[0]
    pols = {p.name: p for p in expand_policies(cfg["policies"])}
    if args.policy not in pols:
        sys.exit(f"Unknown policy {args.policy}. Available: {', '.join(pols)}")
    start = args.start
    frames = range(start, min(start + args.frames, seq.length + 1))
    det = _build_detector(cfg)
    dep = _build_depth(cfg) if pols[args.policy].uses_real_depth else None
    st = run_live(seq, frames, det, dep, pols[args.policy],
                  TrackerConfig.from_dict(cfg.get("tracker")), args.video)
    used = st["used"].astype(bool)
    print(f"{seq.name} frames {frames.start}-{frames.stop - 1}, policy {args.policy}")
    print(f"  depth on {100 * used.mean():.1f}% of frames")
    print(f"  live latency: mean {st['total'].mean() * 1e3:.0f} ms, "
          f"p95 {np.percentile(st['total'], 95) * 1e3:.0f} ms  "
          f"(det {st['det'].mean() * 1e3:.0f} ms, depth when used "
          f"{st['depth'][used].mean() * 1e3 if used.any() else 0:.0f} ms, "
          f"tracker {st['track'].mean() * 1e3:.1f} ms)")
    if args.video:
        print(f"  video: {args.video}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m odmot", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, split=False):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--seqs", nargs="+", default=None, help="subset of sequences")
        if split:
            p.add_argument("--split", default="val_half", choices=SPLITS)
            p.add_argument("--tag", default="main", help="results sub-folder name")
        return p

    add("check", cmd_check)
    add("estimate", cmd_estimate).add_argument("--frames", type=int, default=5)
    add("cache-det", cmd_cache_det)
    add("cache-depth", cmd_cache_depth)
    add("sweep", cmd_sweep, split=True).add_argument(
        "--only", default=None, help="comma-separated policy names / wildcards, e.g. 'no_depth,od_margin*'")
    add("analyze", cmd_analyze, split=True)
    add("plot", cmd_plot, split=True)
    d = add("demo", cmd_demo)
    d.add_argument("--seq", default="MOT17-09")
    d.add_argument("--policy", default="od_margin_thr0.5")
    d.add_argument("--start", type=int, default=1)
    d.add_argument("--frames", type=int, default=150)
    d.add_argument("--video", default=None, help="write an annotated .mp4 here")

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    args.fn(cfg, args)


if __name__ == "__main__":
    main()
