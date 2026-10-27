"""HOTA / IDF1 / AssA / ID-switch evaluation with the official TrackEval code."""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np


def _import_trackeval(extra_path: str | None = None):
    # TrackEval still uses np.float / np.int / np.bool, removed in numpy >= 1.24.
    for name, typ in (("float", float), ("int", int), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, typ)
    try:
        import trackeval
    except ImportError:
        if extra_path and Path(extra_path).is_dir():
            sys.path.insert(0, str(Path(extra_path).resolve()))
            import trackeval
        else:
            raise ImportError(
                "TrackEval not found. Install it with:\n  pip install "
                "https://github.com/JonathonLuiten/TrackEval/archive/refs/heads/master.zip\n"
                "or set paths.trackeval in the config to a cloned TrackEval folder.")
    return trackeval


def evaluate(gt_dir, run_dir, trackers: list[str], seqs: list[str],
             trackeval_path: str | None = None, verbose: bool = False,
             benchmark: str = "MOT17") -> dict[str, dict]:
    """Return {tracker: {"HOTA":..,"DetA":..,"AssA":..,"IDF1":..,"IDSW":..,"MOTA":..,
    "per_sequence": {seq: {...}}}}. Percent metrics are 0-100."""
    te = _import_trackeval(trackeval_path)
    ecfg = te.Evaluator.get_default_eval_config()
    ecfg.update(USE_PARALLEL=False, PRINT_RESULTS=verbose, PRINT_CONFIG=False,
                TIME_PROGRESS=False, OUTPUT_SUMMARY=True, OUTPUT_DETAILED=False,
                PLOT_CURVES=False, PRINT_ONLY_COMBINED=True, DISPLAY_LESS_PROGRESS=True,
                OUTPUT_EMPTY_CLASSES=False)
    dcfg = te.datasets.MotChallenge2DBox.get_default_dataset_config()
    dcfg.update(GT_FOLDER=str(gt_dir), TRACKERS_FOLDER=str(run_dir), OUTPUT_FOLDER=str(run_dir),
                TRACKERS_TO_EVAL=list(trackers), SKIP_SPLIT_FOL=True, BENCHMARK=benchmark,
                SPLIT_TO_EVAL="train", SEQ_INFO={s: None for s in seqs}, PRINT_CONFIG=False,
                TRACKER_SUB_FOLDER="data", DO_PREPROC=True)
    mcfg = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5, "PRINT_CONFIG": False}
    evaluator = te.Evaluator(ecfg)
    dataset = te.datasets.MotChallenge2DBox(dcfg)
    metrics = [te.metrics.HOTA(mcfg), te.metrics.CLEAR(mcfg), te.metrics.Identity(mcfg)]
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink if not verbose else sys.stdout):
        res, msg = evaluator.evaluate([dataset], metrics)
    out = {}
    for tr in trackers:
        if msg["MotChallenge2DBox"][tr] != "Success":
            raise RuntimeError(f"TrackEval failed for {tr}: {msg['MotChallenge2DBox'][tr]}")
        r_all = res["MotChallenge2DBox"][tr]
        out[tr] = _metrics(r_all["COMBINED_SEQ"]["pedestrian"])
        out[tr]["per_sequence"] = {s: _metrics(r_all[s]["pedestrian"]) for s in seqs}
    return out


def _metrics(r) -> dict:
    return {
        "HOTA": float(np.mean(r["HOTA"]["HOTA"]) * 100),
        "DetA": float(np.mean(r["HOTA"]["DetA"]) * 100),
        "AssA": float(np.mean(r["HOTA"]["AssA"]) * 100),
        "IDF1": float(r["Identity"]["IDF1"] * 100),
        "IDSW": int(r["CLEAR"]["IDSW"]),
        "MOTA": float(r["CLEAR"]["MOTA"] * 100),
        "Frag": int(r["CLEAR"]["Frag"]),
    }
