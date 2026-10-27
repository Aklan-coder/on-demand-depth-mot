"""Tests that run without PyTorch:  python -m pytest tests  (or: python tests/test_pipeline.py)"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from odmot.ambiguity import crowd_score, margin_score, pairwise_iou  # noqa: E402
from odmot.cache import SequenceCache, build_depth_cache, build_detection_cache  # noqa: E402
from odmot.data import find_sequence, frame_range  # noqa: E402
from odmot.models import box_nearness, normalize_nearness, pseudo_nearness  # noqa: E402
from odmot.policies import DepthController, PolicyConfig, expand_policies  # noqa: E402
from odmot.tracker import ByteTracker, TrackerConfig, assign  # noqa: E402


def test_splits():
    assert list(frame_range(10, "train_half")) == [1, 2, 3, 4, 5]
    assert list(frame_range(11, "val_half")) == [6, 7, 8, 9, 10, 11]
    assert len(frame_range(600, "full")) == 600


def test_margin_score():
    # one track, one clear candidate -> not ambiguous
    assert margin_score(np.array([[0.9, 0.0]]), gate=0.2) == 0.0
    # one track, two near-equal valid candidates -> very ambiguous
    assert margin_score(np.array([[0.6, 0.55]]), gate=0.2) > 0.9
    # second candidate below the match gate does not count
    assert margin_score(np.array([[0.6, 0.15]]), gate=0.2) == 0.0
    # two tracks competing for one detection (column direction)
    assert margin_score(np.array([[0.5], [0.45]]), gate=0.2) > 0.85
    assert margin_score(np.zeros((0, 3)), 0.2) == 0.0


def test_crowd_score():
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [100, 100, 110, 110]], float)
    assert abs(crowd_score(b) - pairwise_iou(b[:1], b[1:2])[0, 0]) < 1e-9


def test_assign_respects_unmatched_cost():
    # ByteTrack/lapjv semantics: prefer leaving a bad pair unmatched
    # track A: det1 0.7, det2 0.1; track B: only det2 0.75.
    # Max-cardinality would give A-det1 + B-det2 (1.45); lapjv semantics give
    # A-det2 and leave B and det1 unmatched (0.1 + 2 * 0.4 = 0.9).
    cost = np.array([[0.7, 0.1], [1e6, 0.75]])
    m, ur, uc = assign(cost, unmatched_cost=0.4)
    assert m == [(0, 1)] and ur == [1] and uc == [0]


def test_policies():
    pols = {p.name: p for p in expand_policies([
        {"name": "fixed", "type": "fixed", "every": [2, 5]},
        {"name": "od", "type": "ondemand", "threshold": [0.5], "hold": [0, 2]}])}
    assert set(pols) == {"fixed_N2", "fixed_N5", "od_thr0.5_hold0", "od_thr0.5_hold2"}
    c = DepthController(pols["fixed_N5"])
    used = [c.decide(np.zeros((0, 0)), np.zeros((0, 4)), 0.2).real for _ in range(10)]
    assert used == [True, False, False, False, False] * 2
    c = DepthController(pols["od_thr0.5_hold2"])
    amb, clear = np.array([[0.6, 0.55]]), np.array([[0.9, 0.0]])
    seq = [amb, clear, clear, clear, amb]
    used = [c.decide(s, np.zeros((0, 4)), 0.2).real for s in seq]
    assert used == [True, True, True, False, True]


def _gt_tracks(n_frames=60):
    """Two people walking towards each other and crossing, plus one static person."""
    out = []
    for f in range(1, n_frames + 1):
        out.append((f, 1, 100 + 8 * f, 300, 60, 160))
        out.append((f, 2, 600 - 8 * f, 310, 60, 160))
        out.append((f, 3, 900, 200, 80, 200))
    return np.array(out, float)


def test_bytetrack_perfect_detections():
    """With perfect detections and no crossing ambiguity, IDs must be stable."""
    gt = _gt_tracks()
    tr = ByteTracker(TrackerConfig())
    ctrl = DepthController(PolicyConfig("never", "never"))
    ids_per_gt = {1: set(), 2: set(), 3: set()}
    for f in range(1, 61):
        g = gt[gt[:, 0] == f]
        boxes = np.c_[g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]]
        tracks, _ = tr.update(boxes, np.full(len(g), 0.9), 720, ctrl)
        if f > 1:
            assert len(tracks) == 3
            for tid, tlwh, _ in tracks:
                k = int(np.argmin(np.abs(g[:, 2] - tlwh[0]) + np.abs(g[:, 3] - tlwh[1])))
                ids_per_gt[int(g[k, 1])].add(tid)
    # person 3 never interacts -> exactly one ID
    assert len(ids_per_gt[3]) == 1


def test_depth_breaks_crossing_tie():
    """Two equal-size people overlap; true depth differs. With depth the identities
    are preserved through the crossing at least as well as without depth."""
    def run(policy):
        tr = ByteTracker(TrackerConfig(depth_weight=1.0))
        ctrl = DepthController(policy)
        switches, last = 0, {}
        for f in range(1, 80):
            xa, xb = 200 + 6 * f, 680 - 6 * f
            boxes = np.array([[xa, 300, xa + 70, 470], [xb, 302, xb + 70, 472]], float)
            near = np.array([0.8, 0.3])
            tracks, _ = tr.update(boxes, np.array([0.9, 0.9]), 720, ctrl, lambda: near)
            for tid, tlwh, _ in tracks:
                who = 0 if abs(tlwh[0] - xa) < abs(tlwh[0] - xb) else 1
                if tid in last and last[tid] != who:
                    switches += 1
                last[tid] = who
        return switches
    assert run(PolicyConfig("a", "always")) <= run(PolicyConfig("n", "never"))


def test_zero_margin_means_depth_cannot_change_assignment():
    """Property behind the margin score: if no track or detection has two valid
    candidates (margin == 0), the gated assignment is identical for ANY depth values."""
    tr = ByteTracker(TrackerConfig(depth_weight=1.0))
    rng = np.random.default_rng(0)
    gate, thr = 0.2, 0.8
    checked = 0
    for _ in range(3000):
        n, m = rng.integers(1, 7), rng.integers(1, 7)
        sim = rng.uniform(0, 1, (n, m)) * (rng.uniform(0, 1, (n, m)) < 0.35)
        if margin_score(sim, gate) > 0:
            continue
        checked += 1
        dist = 1 - sim
        no_depth = tr._cost(dist, [], None, None, thr)
        base, _, _ = assign(no_depth, tr._unmatched_cost(thr, None))
        # emulate an arbitrary depth term on gated-valid pairs
        dd = rng.uniform(0, 1, (n, m))
        with_depth = np.where(dist > thr, 1e6, dist + 1.0 * dd)
        dep, _, _ = assign(with_depth, tr._unmatched_cost(thr, "real"))
        assert set(base) == set(dep)
    assert checked > 500


def test_nearness_helpers():
    m = np.tile(np.linspace(0, 10, 100, dtype=np.float32), (50, 1))
    n = normalize_nearness(m)
    assert n.min() >= 0 and n.max() <= 1
    v = box_nearness(n, np.array([[0, 0, 100, 100], [900, 0, 1000, 100]], float), 1000, 500)
    assert v[0] < v[1]
    assert np.allclose(pseudo_nearness(np.array([[0, 0, 1, 360]], float), 720), 0.5)


class _FakeDetector:
    name = "fake_640"

    def __call__(self, bgr):
        return np.array([[10, 10, 50, 120, 0.9], [60, 10, 100, 120, 0.4]], np.float32)


class _FakeDepth:
    name = "fakedepth_518"

    def __call__(self, rgb):
        h, w = rgb.shape[:2]
        return np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))


def test_cache_roundtrip_and_live():
    from make_synthetic import write_sequence
    from odmot.live import run_live
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        write_sequence(d / "MOT17", d / "cache0", "MOT17-02", 1, 20, 3, "x", "y")
        seq = find_sequence(d / "MOT17", "MOT17-02")
        build_detection_cache(_FakeDetector(), [seq], d / "cache", warmup=1)
        build_depth_cache(_FakeDepth(), [seq], d / "cache", "fake_640", warmup=1)
        c = SequenceCache(d / "cache", "fake_640", "fakedepth_518", seq)
        b, s, n = c.frame(5)
        assert b.shape == (2, 4) and c.has_depth and np.all(np.isfinite(n)) and n[0] < n[1]
        assert c.det_time.shape == (20,) and (c.depth_time > 0).all()
        st = run_live(seq, range(1, 21), _FakeDetector(), _FakeDepth(),
                      PolicyConfig("a", "always"), TrackerConfig(), str(d / "demo.mp4"))
        assert st["used"].all() and (d / "demo.mp4").exists()


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} tests passed")


def test_depth_swap_invariants():
    """Swap mode: same match count, same matched sets, gates respected, swap only when
    depth clearly prefers it; unmatched tracks never get a detection."""
    import numpy as np
    from odmot.tracker import ByteTracker, TrackerConfig, STrack
    rng = np.random.default_rng(0)
    tr = ByteTracker(TrackerConfig(depth_mode="swap", depth_weight=2.0))
    tr.frame_id = 5
    for trial in range(300):
        n, m = rng.integers(2, 7), rng.integers(2, 7)
        tracks = []
        for _ in range(n):
            t = STrack(np.array([0, 0, 10, 20.]), 0.9)
            if rng.random() < 0.8:
                t.depth["real"], t.depth_frame["real"] = rng.random(), 5
            tracks.append(t)
        dist = rng.random((n, m))
        dn = rng.random(m)
        base = tr._cost(dist, tracks, dn, None, 0.8)
        from odmot.tracker import assign
        m0, ut0, ud0 = assign(base, tr._unmatched_cost(0.8, None))
        m1, ut1, ud1 = tr._associate(dist, tracks, dn, "real", 0.8)
        assert len(m1) == len(m0)
        assert {i for i, _ in m1} == {i for i, _ in m0}
        assert {j for _, j in m1} == {j for _, j in m0}
        assert ut1 == ut0 and ud1 == ud0
        assert all(dist[i, j] <= 0.8 for i, j in m1)


def test_depth_swap_fixes_crossing():
    """The two-people-crossing example: ByteTrack swaps IDs, depth swaps them back."""
    import numpy as np
    from odmot.tracker import ByteTracker, TrackerConfig, STrack
    tr = ByteTracker(TrackerConfig(depth_mode="swap", depth_weight=2.0))
    tr.frame_id = 3
    tracks = [STrack(np.array([0, 0, 10, 20.]), 0.9) for _ in range(2)]
    for t, d in zip(tracks, (0.80, 0.40)):
        t.depth["real"], t.depth_frame["real"] = d, 3
    sim = np.array([[.50, .60], [.58, .45]])
    m_no, _, _ = tr._associate(1 - sim, tracks, np.array([.78, .42]), None, 0.8)
    m_d, _, _ = tr._associate(1 - sim, tracks, np.array([.78, .42]), "real", 0.8)
    assert sorted(m_no) == [(0, 1), (1, 0)]
    assert sorted(m_d) == [(0, 0), (1, 1)]


def test_fixed_offsets():
    import numpy as np
    from odmot.policies import expand_policies, DepthController
    ps = expand_policies([{"name": "fixed_N10", "type": "fixed", "every": 10, "offset": [0, 3]}])
    assert [p.name for p in ps] == ["fixed_N10_off0", "fixed_N10_off3"]
    on = []
    for p in ps:
        c = DepthController(p)
        on.append([t for t in range(20) if c.decide(np.zeros((0, 0)), np.zeros((0, 4)), 0.2).real])
    assert on[0] == [0, 10] and on[1] == [7, 17]
