"""Cross-check odmot.tracker against the official ByteTrack implementation.

    git clone https://github.com/ifzhang/ByteTrack.git ../ByteTrack
    python tests/make_synthetic.py --out synth
    python scripts/compare_bytetrack.py synth --ref ../ByteTrack [--plus1]

The official code needs torch, lap and cython_bbox; small numpy shims replace them (the
tracking logic itself does not use torch). --plus1 uses cython_bbox's +1-pixel IoU
convention; without it both trackers use the same IoU and outputs should be identical.
Result when written: 100.00% identical frames (98.6% with --plus1).
"""
import sys, types, importlib.util
from pathlib import Path
import numpy as np
np.float = float; np.int = int
_ref = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else "../ByteTrack"
REF = Path(_ref) / "yolox" / "tracker"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from odmot.tracker import assign, ByteTracker, TrackerConfig
from odmot.policies import DepthController, PolicyConfig
from odmot.cache import SequenceCache
from odmot.data import find_sequence, frame_range

IOU_PLUS1 = "--plus1" in sys.argv
# --- shims for torch / lap / cython_bbox -------------------------------------------
torch = types.ModuleType("torch"); torch.nn = types.ModuleType("torch.nn")
torch.nn.functional = types.ModuleType("torch.nn.functional")
sys.modules.update({"torch": torch, "torch.nn": torch.nn, "torch.nn.functional": torch.nn.functional})
lap = types.ModuleType("lap")
def lapjv(cost, extend_cost=True, cost_limit=np.inf):
    c = np.where(cost > cost_limit, 1e6, cost)
    m, _, _ = assign(c, cost_limit / 2)
    x = -np.ones(cost.shape[0], int); y = -np.ones(cost.shape[1], int)
    for i, j in m: x[i] = j; y[j] = i
    return 0.0, x, y
lap.lapjv = lapjv; sys.modules["lap"] = lap
cb = types.ModuleType("cython_bbox")
def bbox_overlaps(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); k = 1.0 if IOU_PLUS1 else 0.0
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]) + k, 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]) + k, 0, None)
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0] + k) * (a[:, 3] - a[:, 1] + k); ab = (b[:, 2] - b[:, 0] + k) * (b[:, 3] - b[:, 1] + k)
    return inter / (aa[:, None] + ab[None, :] - inter)
cb.bbox_overlaps = bbox_overlaps; sys.modules["cython_bbox"] = cb
# --- load yolox.tracker modules without yolox/__init__ ----------------------------------
for pkg in ("yolox", "yolox.tracker"):
    m = types.ModuleType(pkg); m.__path__ = []; sys.modules[pkg] = m
def load(name):
    spec = importlib.util.spec_from_file_location(f"yolox.tracker.{name}", REF / f"{name}.py",
                                                  submodule_search_locations=None)
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
    spec.loader.exec_module(mod); setattr(sys.modules["yolox.tracker"], name, mod); return mod
for n in ("kalman_filter", "basetrack", "matching", "byte_tracker"): load(n)
BYTETracker = sys.modules["yolox.tracker.byte_tracker"].BYTETracker
BaseTrack = sys.modules["yolox.tracker.basetrack"].BaseTrack

S = Path(sys.argv[1])
tot = same = 0
for name in ("MOT17-02", "MOT17-04", "MOT17-09"):
    seq = find_sequence(S / "MOT17", name)
    cache = SequenceCache(S / "cache", "yolov8s_1088", None, seq)
    BaseTrack._count = 0
    ref = BYTETracker(types.SimpleNamespace(track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False), frame_rate=30)
    ours = ByteTracker(TrackerConfig()); ctrl = DepthController(PolicyConfig("n", "never"))
    for f in frame_range(seq.length, "full"):
        b, s, _ = cache.frame(f)
        r = ref.update(np.c_[b, s].astype(np.float64), (seq.height, seq.width), (seq.height, seq.width))
        o, _ = ours.update(b, s, seq.height, ctrl)
        A = sorted((t.track_id, *np.round(t.tlwh, 2)) for t in r)
        B = sorted((tid, *np.round(tlwh, 2)) for tid, tlwh, _ in o)
        tot += 1; same += A == B
    print(f"{name}: identical outputs on {same}/{tot} frames so far")
print(f"TOTAL identical frames: {same}/{tot} ({100*same/tot:.2f}%)")
