"""ByteTrack, re-implemented in numpy, with a depth term in the association cost.

Faithful to ByteTrack (Zhang et al., ECCV 2022): Kalman prediction, two-stage
association (high-score detections first, then low-score detections against the
remaining tracks), unconfirmed-track association, track birth/death rules, duplicate
removal. With `depth_weight = 0` or a "never" policy it IS ByteTrack.

The one addition is how depth enters association:

    cost(track i, det j) = (1 - sim_ij) + depth_weight * |near_i - near_j|

  * The IoU gate is applied BEFORE depth is added: a pair that ByteTrack would not allow
    is never allowed. Depth therefore never creates matches; it only re-ranks the
    plausible ones. That is exactly the "tie-breaker for ambiguous association" role the
    project studies, and it means depth can only change decisions when a track has
    more than one plausible candidate - which is what the ambiguity score measures.
  * near_i is the track's depth state (EMA of its matched detections' depth); if the
    track has no recent depth reading the term is 0 for that track (neutral).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .ambiguity import pairwise_iou
from .kalman import KalmanFilter
from .models import pseudo_nearness

BIG = 1e6


@dataclass
class TrackerConfig:
    track_thresh: float = 0.5       # high-score detections (first association)
    low_thresh: float = 0.1         # low-score detections used in second association
    new_track_thresh: float = 0.6   # min score to start a new track
    match_thresh: float = 0.8       # max (1 - sim) for a valid first-stage match
    second_match_thresh: float = 0.5
    unconfirmed_match_thresh: float = 0.7
    track_buffer: int = 30          # frames a lost track is kept (at 30 fps)
    fuse_score: bool = True         # ByteTrack's MOT17 setting: sim = IoU * det score
    depth_weight: float = 0.5
    depth_momentum: float = 0.5     # EMA weight of the old track depth
    depth_stale: int = 30           # frames after which a track's depth reading is ignored
    depth_in_second: bool = True    # also use depth in the low-score association
    # how depth enters association:
    #   "cost" (original, reference): depth term added to the cost, unmatched cost raised
    #   "swap" (experimental): plain ByteTrack matching first; depth may only swap the
    #          detections of two already-matched tracks, when both swapped pairs pass the
    #          gate and the depth-augmented cost of the swap is lower. Match count, the set
    #          of matched tracks/detections and all gates are unchanged by construction.
    depth_mode: str = "cost"
    min_box_area: float = 100.0     # output filter (ByteTrack MOT17)
    max_aspect: float = 1.6         # output filter: drop boxes with w/h > this

    @classmethod
    def from_dict(cls, d: dict | None) -> "TrackerConfig":
        d = d or {}
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown tracker config keys: {sorted(unknown)}")
        return cls(**d)


class TrackState:
    New, Tracked, Lost, Removed = 0, 1, 2, 3


SOURCES = ("real", "pseudo")


class STrack:
    _kf = KalmanFilter()

    def __init__(self, tlwh, score, near_real=np.nan, near_pseudo=np.nan, gt_id=-1):
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.gt_id = int(gt_id)   # analysis only: GT identity of the last matched detection
        self.score = float(score)
        self.mean = self.cov = None
        self.is_activated = False
        self.state = TrackState.New
        self.track_id = 0
        self.frame_id = self.start_frame = 0
        self.tracklet_len = 0
        self.det_near = {"real": float(near_real), "pseudo": float(near_pseudo)}
        self.depth = {"real": np.nan, "pseudo": np.nan}
        self.depth_frame = {"real": -10**9, "pseudo": -10**9}

    # --- geometry -----------------------------------------------------------------------
    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        cx, cy, a, h = self.mean[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, w, h])

    @property
    def xyxy(self):
        t = self.tlwh
        return np.array([t[0], t[1], t[0] + t[2], t[1] + t[3]])

    @staticmethod
    def tlwh_to_xyah(t):
        return np.array([t[0] + t[2] / 2, t[1] + t[3] / 2, t[2] / max(t[3], 1e-6), t[3]])

    # --- lifecycle ------------------------------------------------------------------------
    def activate(self, frame_id, track_id):
        self.track_id = track_id
        self.mean, self.cov = self._kf.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = frame_id == 1
        self.frame_id = self.start_frame = frame_id
        self._absorb_depth(self, frame_id)

    def update(self, det: "STrack", frame_id: int, reactivate: bool = False):
        self.mean, self.cov = self._kf.update(self.mean, self.cov, self.tlwh_to_xyah(det._tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = det.score
        if det.gt_id >= 0:
            self.gt_id = det.gt_id
        self.tracklet_len = 0 if reactivate else self.tracklet_len + 1
        self._absorb_depth(det, frame_id)

    def _absorb_depth(self, det: "STrack", frame_id: int, momentum: float | None = None):
        m = STrack.depth_momentum if momentum is None else momentum
        for s in SOURCES:
            v = det.det_near[s]
            if np.isfinite(v):
                old = self.depth[s]
                fresh = frame_id - self.depth_frame[s] <= STrack.depth_stale
                self.depth[s] = v if (not np.isfinite(old) or not fresh) else m * old + (1 - m) * v
                self.depth_frame[s] = frame_id

    def track_depth(self, source: str, frame_id: int) -> float:
        if frame_id - self.depth_frame[source] > STrack.depth_stale:
            return np.nan
        return self.depth[source]

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def multi_predict(tracks: list["STrack"]):
        if not tracks:
            return
        means = np.array([t.mean.copy() for t in tracks])
        covs = np.array([t.cov for t in tracks])
        for i, t in enumerate(tracks):
            if t.state != TrackState.Tracked:
                means[i][7] = 0   # lost tracks: freeze height velocity (ByteTrack)
        means, covs = STrack._kf.multi_predict(means, covs)
        for t, m, c in zip(tracks, means, covs):
            t.mean, t.cov = m, c

    # class-level depth settings (set by the tracker)
    depth_momentum = 0.5
    depth_stale = 30


def _boxes(tracks):
    return np.array([t.xyxy for t in tracks]).reshape(-1, 4)


def assign(cost: np.ndarray, unmatched_cost: float):
    """Min-cost assignment where leaving a row/column unmatched costs `unmatched_cost`.

    Same semantics as ByteTrack's lap.lapjv(extend_cost=True, cost_limit=2*unmatched_cost);
    entries >= BIG are forbidden.
    """
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    A = np.full((n + m, n + m), BIG)
    A[:n, :m] = cost
    A[np.arange(n), m + np.arange(n)] = unmatched_cost
    A[n + np.arange(m), np.arange(m)] = unmatched_cost
    A[n:, m:] = 0.0
    r, c = linear_sum_assignment(A)
    matches = [(i, j) for i, j in zip(r, c) if i < n and j < m and cost[i, j] < BIG / 2]
    mr = {i for i, _ in matches}
    mc = {j for _, j in matches}
    return matches, [i for i in range(n) if i not in mr], [j for j in range(m) if j not in mc]


class ByteTracker:
    def __init__(self, cfg: TrackerConfig, frame_rate: float = 30.0):
        self.cfg = cfg
        STrack.depth_momentum = cfg.depth_momentum
        STrack.depth_stale = cfg.depth_stale
        self.max_time_lost = int(frame_rate / 30.0 * cfg.track_buffer)
        self.reset()

    def reset(self):
        self.tracked: list[STrack] = []
        self.lost: list[STrack] = []
        self.removed: list[STrack] = []
        self.frame_id = 0
        self._next_id = 0

    def _new_id(self):
        self._next_id += 1
        return self._next_id

    # --------------------------------------------------------------------------------------
    def _cost(self, dist, tracks, det_near, src, thresh):
        """Gate on appearance-free distance first, then add the depth term."""
        cost = dist.astype(np.float64).copy()
        invalid = dist > thresh
        if src is not None and self.cfg.depth_weight > 0 and cost.size:
            tn = np.array([t.track_depth(src, self.frame_id) for t in tracks])
            dd = np.abs(tn[:, None] - np.asarray(det_near)[None, :])
            cost += self.cfg.depth_weight * np.nan_to_num(dd, nan=0.0)
        cost[invalid] = BIG
        return cost

    def _associate(self, dist, tracks, det_near, src, thresh):
        """One association step. Returns (matches, unmatched_tracks, unmatched_dets)."""
        if self.cfg.depth_mode == "cost":
            return assign(self._cost(dist, tracks, det_near, src, thresh),
                          self._unmatched_cost(thresh, src))
        if self.cfg.depth_mode != "swap":
            raise ValueError(f"Unknown depth_mode '{self.cfg.depth_mode}' (cost | swap)")
        base = self._cost(dist, tracks, det_near, None, thresh)
        matches, ut, ud = assign(base, self._unmatched_cost(thresh, None))   # = ByteTrack
        if src is not None and self.cfg.depth_weight > 0:
            matches = self._depth_swap(matches, dist, tracks, det_near, src, thresh)
        return matches, ut, ud

    def _depth_swap(self, matches, dist, tracks, det_near, src, thresh):
        """Depth as a pure tie-breaker between two ByteTrack matches (i->j, k->l).

        Swap to (i->l, k->j) only if (a) both new pairs pass the same gate, (b) all four
        depth readings exist, and (c) the swap lowers (1 - sim) + w * |depth gap| summed
        over the two pairs. Swaps are applied greedily by gain, each match at most once.
        Only detections are exchanged between matched tracks, so the number of matches
        and the sets of matched tracks and detections never change."""
        if len(matches) < 2:
            return matches
        w = self.cfg.depth_weight
        tn = np.array([t.track_depth(src, self.frame_id) for t in tracks])
        dn = np.asarray(det_near, np.float64)
        m = list(matches)
        cands = []
        for a in range(len(m)):
            i, j = m[a]
            for b in range(a + 1, len(m)):
                k, l = m[b]
                if dist[i, l] > thresh or dist[k, j] > thresh:
                    continue
                v = (tn[i], tn[k], dn[j], dn[l])
                if not all(np.isfinite(v)):
                    continue
                d_sim = dist[i, l] + dist[k, j] - dist[i, j] - dist[k, l]
                d_dep = w * (abs(tn[i] - dn[l]) + abs(tn[k] - dn[j])
                             - abs(tn[i] - dn[j]) - abs(tn[k] - dn[l]))
                gain = -(d_sim + d_dep)
                if gain > 1e-9:
                    cands.append((gain, a, b))
        used = set()
        for gain, a, b in sorted(cands, reverse=True):
            if a in used or b in used:
                continue
            (i, j), (k, l) = m[a], m[b]
            m[a], m[b] = (i, l), (k, j)
            used.update((a, b))
        return m

    def _unmatched_cost(self, thresh, src):
        # every gated-valid pair must stay preferable to leaving both sides unmatched,
        # so that depth only re-ranks (never rejects) - see module docstring
        extra = self.cfg.depth_weight if src is not None else 0.0
        return (thresh + extra) / 2.0 + 1e-9

    def update(self, boxes, scores, img_h, controller, depth_fn=None, log=None, det_gt=None):
        """Process one frame.

        boxes: (N,4) xyxy; scores: (N,); img_h: image height (for pseudo-depth)
        controller: policies.DepthController (decides whether to use depth this frame)
        depth_fn:  () -> (N,) nearness for ALL N detections; called only if the
                   controller asks for real depth (that is when the cost is paid)
        log: optional dict; filled with per-frame diagnostics (scores, counterfactual)
        det_gt: optional (N,) GT identity per detection (-1 = none); analysis only
        Returns (tracks, decision): tracks = list of (track_id, tlwh, score).
        """
        cfg = self.cfg
        self.frame_id += 1
        boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
        scores = np.asarray(scores, np.float64).reshape(-1)
        hi = np.where(scores >= cfg.track_thresh)[0]
        lo = np.where((scores > cfg.low_thresh) & (scores < cfg.track_thresh))[0]
        pseudo_all = pseudo_nearness(boxes, img_h)

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        confirmed = [t for t in self.tracked if t.is_activated]
        pool = _joint(confirmed, self.lost)
        STrack.multi_predict(pool)

        # ---- first association: similarity, then the depth decision ----------------------
        iou1 = pairwise_iou(_boxes(pool), boxes[hi])
        sim1 = iou1 * scores[hi][None, :] if cfg.fuse_score else iou1
        gate1 = 1.0 - cfg.match_thresh
        decision = controller.decide(sim1, boxes[hi], gate1)
        real_all = None
        if decision.real:
            if depth_fn is None:
                raise RuntimeError("Policy requested depth but no depth_fn was given")
            real_all = np.asarray(depth_fn(), np.float64).reshape(-1)
        src = "real" if decision.real else ("pseudo" if decision.pseudo else None)
        near_all = real_all if src == "real" else (pseudo_all if src == "pseudo" else
                                                   np.full(len(boxes), np.nan))

        gts = np.full(len(boxes), -1) if det_gt is None else np.asarray(det_gt)

        def mk(idx):
            r = real_all[idx] if real_all is not None else np.full(len(idx), np.nan)
            return [STrack(_xyxy_to_tlwh(boxes[k]), scores[k], r[n], pseudo_all[k], gts[k])
                    for n, k in enumerate(idx)]

        dets_hi = mk(hi)
        matches, u_track, u_det = self._associate(1.0 - sim1, pool, near_all[hi], src,
                                                  cfg.match_thresh)

        if log is not None:
            from .ambiguity import all_scores
            log.update(all_scores(sim1, boxes[hi], gate1))
            log.update(n_tracks=len(pool), n_dets=len(hi), real=decision.real,
                       pseudo=decision.pseudo)
            if src is not None:
                base = self._cost(1.0 - sim1, pool, near_all[hi], None, cfg.match_thresh)
                m0, _, _ = assign(base, self._unmatched_cost(cfg.match_thresh, None))
                log["changed"] = set(m0) != set(matches)
                log["n_changed_pairs"] = len(set(m0) - set(matches))
                if det_gt is not None:
                    g = gts[hi]

                    def n_correct(ms):
                        return sum(1 for i, j in ms if g[j] >= 0 and pool[i].gt_id == g[j])

                    def n_wrong(ms):
                        return sum(1 for i, j in ms if g[j] >= 0 and pool[i].gt_id >= 0
                                   and pool[i].gt_id != g[j])
                    log["correct_with"], log["correct_without"] = n_correct(matches), n_correct(m0)
                    log["wrong_with"], log["wrong_without"] = n_wrong(matches), n_wrong(m0)

        activated, refind, lost, removed = [], [], [], []
        for i, j in matches:
            t, d = pool[i], dets_hi[j]
            if t.state == TrackState.Tracked:
                t.update(d, self.frame_id)
                activated.append(t)
            else:
                t.update(d, self.frame_id, reactivate=True)
                refind.append(t)

        # ---- second association: low-score detections vs remaining tracked tracks ----------
        dets_lo = mk(lo)
        r_tracked = [pool[i] for i in u_track if pool[i].state == TrackState.Tracked]
        iou2 = pairwise_iou(_boxes(r_tracked), boxes[lo])
        src2 = src if cfg.depth_in_second else None
        matches2, u_track2, _ = self._associate(1.0 - iou2, r_tracked, near_all[lo], src2,
                                                cfg.second_match_thresh)
        for i, j in matches2:
            t = r_tracked[i]
            if t.state == TrackState.Tracked:
                t.update(dets_lo[j], self.frame_id)
                activated.append(t)
            else:
                t.update(dets_lo[j], self.frame_id, reactivate=True)
                refind.append(t)
        for i in u_track2:
            t = r_tracked[i]
            if t.state != TrackState.Lost:
                t.mark_lost()
                lost.append(t)

        # ---- unconfirmed tracks (seen once) vs leftover high-score detections --------------
        rem = [dets_hi[j] for j in u_det]
        rem_idx = hi[u_det] if len(u_det) else np.zeros(0, int)
        iou3 = pairwise_iou(_boxes(unconfirmed), boxes[rem_idx])
        sim3 = iou3 * scores[rem_idx][None, :] if cfg.fuse_score else iou3
        matches3, u_unconf, u_det3 = self._associate(
            1.0 - sim3, unconfirmed, near_all[rem_idx], src, cfg.unconfirmed_match_thresh)
        for i, j in matches3:
            unconfirmed[i].update(rem[j], self.frame_id)
            activated.append(unconfirmed[i])
        for i in u_unconf:
            unconfirmed[i].mark_removed()
            removed.append(unconfirmed[i])

        # ---- births ---------------------------------------------------------------------------
        for j in u_det3:
            d = rem[j]
            if d.score >= cfg.new_track_thresh:
                d.activate(self.frame_id, self._new_id())
                activated.append(d)

        # ---- deaths ---------------------------------------------------------------------------
        for t in self.lost:
            if self.frame_id - t.end_frame > self.max_time_lost:
                t.mark_removed()
                removed.append(t)

        self.tracked = [t for t in self.tracked if t.state == TrackState.Tracked]
        self.tracked = _joint(self.tracked, activated)
        self.tracked = _joint(self.tracked, refind)
        self.lost = _sub(self.lost, self.tracked)
        self.lost.extend(lost)
        self.lost = _sub(self.lost, removed)
        self.removed.extend(removed)
        self.removed = self.removed[-1000:]  # bounded memory on long sequences
        self.tracked, self.lost = _remove_duplicates(self.tracked, self.lost)

        out = [(t.track_id, t.tlwh, t.score) for t in self.tracked if t.is_activated]
        return out, decision


def _xyxy_to_tlwh(b):
    return np.array([b[0], b[1], b[2] - b[0], b[3] - b[1]])


def _joint(a, b):
    seen = {t.track_id for t in a}
    return a + [t for t in b if t.track_id not in seen and not seen.add(t.track_id)]


def _sub(a, b):
    ids = {t.track_id for t in b}
    return [t for t in a if t.track_id not in ids]


def _remove_duplicates(a, b):
    if not a or not b:
        return a, b
    d = 1.0 - pairwise_iou(_boxes(a), _boxes(b))
    dup_a, dup_b = set(), set()
    for p, q in zip(*np.where(d < 0.15)):
        ta, tb = a[p], b[q]
        if ta.frame_id - ta.start_frame > tb.frame_id - tb.start_frame:
            dup_b.add(q)
        else:
            dup_a.add(p)
    return ([t for i, t in enumerate(a) if i not in dup_a],
            [t for i, t in enumerate(b) if i not in dup_b])
