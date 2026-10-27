"""MOT17 dataset access, train/val half splits, and TrackEval ground-truth preparation.

Expected layout (the official MOT17.zip, unzipped):

    <mot17_root>/train/MOT17-02-FRCNN/{img1/, gt/gt.txt, det/, seqinfo.ini}
    <mot17_root>/train/MOT17-02-DPM/...   (same images, different public detections)

We only need ONE copy of each sequence's images and GT, so the "-FRCNN" variant is used
by default. We never use the public detections; detections come from our YOLOv8 cache.

Splits (the standard protocol from ByteTrack / FairMOT for MOT17 ablations):
    train_half : frames 1 .. floor(L/2)          -> tune thresholds here
    val_half   : frames floor(L/2)+1 .. L        -> report numbers here
    full       : all frames
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MOT17_TRAIN_SEQS = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09",
                    "MOT17-10", "MOT17-11", "MOT17-13"]
VARIANTS = ("FRCNN", "SDP", "DPM")
SPLITS = ("train_half", "val_half", "full")


@dataclass
class SeqInfo:
    name: str            # canonical name, e.g. "MOT17-02"
    path: Path           # sequence directory (contains img1/, gt/, seqinfo.ini)
    length: int
    width: int
    height: int
    fps: float
    im_dir: str = "img1"
    im_ext: str = ".jpg"

    def image_path(self, frame: int) -> Path:
        """Path of a 1-based frame index."""
        return self.path / self.im_dir / f"{frame:06d}{self.im_ext}"

    @property
    def gt_path(self) -> Path:
        return self.path / "gt" / "gt.txt"


def find_sequence(mot17_root: str | Path, name: str, subset: str = "train") -> SeqInfo:
    """Locate a sequence by canonical name ("MOT17-02") or full folder name."""
    root = Path(mot17_root)
    candidates = [root / subset / name, root / name]
    for v in VARIANTS:
        candidates += [root / subset / f"{name}-{v}", root / f"{name}-{v}"]
    for c in candidates:
        if (c / "seqinfo.ini").is_file():
            return read_seqinfo(c, canonical=_canonical(name))
    raise FileNotFoundError(
        f"Could not find sequence '{name}' under {root}. Expected e.g. "
        f"{root / subset / (name + '-FRCNN') / 'seqinfo.ini'}")


def _canonical(name: str) -> str:
    for v in VARIANTS:
        if name.endswith("-" + v):
            return name[: -(len(v) + 1)]
    return name


def read_seqinfo(seq_dir: Path, canonical: str | None = None) -> SeqInfo:
    ini = configparser.ConfigParser()
    ini.read(seq_dir / "seqinfo.ini")
    s = ini["Sequence"]
    return SeqInfo(
        name=canonical or _canonical(s.get("name", seq_dir.name)),
        path=seq_dir,
        length=int(s["seqLength"]),
        width=int(s["imWidth"]),
        height=int(s["imHeight"]),
        fps=float(s.get("frameRate", 30)),
        im_dir=s.get("imDir", "img1"),
        im_ext=s.get("imExt", ".jpg"),
    )


def frame_range(length: int, split: str) -> range:
    """1-based frame indices belonging to a split."""
    half = length // 2
    if split == "train_half":
        return range(1, half + 1)
    if split == "val_half":
        return range(half + 1, length + 1)
    if split == "full":
        return range(1, length + 1)
    raise ValueError(f"Unknown split '{split}', choose from {SPLITS}")


def load_gt(seq: SeqInfo) -> np.ndarray:
    """Raw gt.txt rows: frame, id, x, y, w, h, consider, class, visibility."""
    return np.loadtxt(seq.gt_path, delimiter=",", ndmin=2)


def prepare_trackeval_gt(seqs: list[SeqInfo], split: str, out_dir: str | Path) -> Path:
    """Write a TrackEval-compatible GT folder for one split.

    Frames are renumbered so the first frame of the split becomes frame 1; tracker
    outputs written by the runner use the same renumbering. Layout (SKIP_SPLIT_FOL=True):

        out_dir/<seq>/gt/gt.txt
        out_dir/<seq>/seqinfo.ini
    """
    out = Path(out_dir)
    for seq in seqs:
        fr = frame_range(seq.length, split)
        gt = load_gt(seq)
        keep = (gt[:, 0] >= fr.start) & (gt[:, 0] < fr.stop)
        gt = gt[keep].copy()
        gt[:, 0] -= fr.start - 1
        d = out / seq.name
        (d / "gt").mkdir(parents=True, exist_ok=True)
        fmt = ["%d", "%d", "%.2f", "%.2f", "%.2f", "%.2f", "%d", "%d", "%.6f"]
        np.savetxt(d / "gt" / "gt.txt", gt[:, :9], fmt=fmt[: gt.shape[1]], delimiter=",")
        (d / "seqinfo.ini").write_text(
            "[Sequence]\n"
            f"name={seq.name}\nimDir={seq.im_dir}\nframeRate={seq.fps:g}\n"
            f"seqLength={len(fr)}\nimWidth={seq.width}\nimHeight={seq.height}\n"
            f"imExt={seq.im_ext}\n")
    return out
