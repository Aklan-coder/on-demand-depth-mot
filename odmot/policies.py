"""Depth policies: decide, once per frame, which depth signal the association may use.

    never      plain ByteTrack, no depth                               (lower bound on cost)
    pseudo     PD-SORT-style pseudo-depth from box bottoms, every frame (free depth cue)
    always     Depth Anything on every frame                           (upper bound on cost)
    fixed      Depth Anything every `every` frames                     (naive budget baseline)
    ondemand   Depth Anything when ambiguity score >= `threshold`,     (ours)
               then keep it on for `hold` more frames

Any real-depth policy can set `fallback: pseudo`, meaning frames without real depth use
pseudo-depth instead of no depth at all (tests whether the two cues are complementary).

The controller sees only information available *before* association in a live system:
the predicted-track vs detection similarity matrix and the detections themselves.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .ambiguity import compute_score

POLICY_TYPES = ("never", "pseudo", "always", "fixed", "ondemand")


@dataclass
class Decision:
    real: bool = False      # use Depth Anything on this frame
    pseudo: bool = False    # use pseudo-depth on this frame
    score: float = float("nan")
    score_time: float = 0.0


@dataclass
class PolicyConfig:
    name: str
    type: str
    every: int = 1
    offset: int = 0                 # fixed-rate phase: depth on frames where (t + offset) % every == 0
    score: str = "margin"
    threshold: float = 0.5
    hold: int = 0
    fallback: str = "none"          # "none" | "pseudo"
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.type not in POLICY_TYPES:
            raise ValueError(f"Policy '{self.name}': unknown type '{self.type}' "
                             f"(choose from {POLICY_TYPES})")
        if self.fallback not in ("none", "pseudo"):
            raise ValueError(f"Policy '{self.name}': fallback must be 'none' or 'pseudo'")

    @property
    def uses_real_depth(self) -> bool:
        return self.type in ("always", "fixed", "ondemand")


class DepthController:
    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self._t = -1
        self._hold_until = -1

    def decide(self, sim: np.ndarray, det_boxes: np.ndarray, gate: float) -> Decision:
        """Called once per frame, before the first association step."""
        self._t += 1
        c = self.cfg
        if c.type == "never":
            return Decision()
        if c.type == "pseudo":
            return Decision(pseudo=True)
        if c.type == "always":
            return Decision(real=True)
        if c.type == "fixed":
            real = (self._t + c.offset) % max(c.every, 1) == 0
            return Decision(real=real, pseudo=(not real and c.fallback == "pseudo"))
        # on-demand: the score computation is part of this policy's cost, so time it
        ts = time.perf_counter()
        s = compute_score(c.score, sim, det_boxes, gate)
        st = time.perf_counter() - ts
        if s >= c.threshold:
            self._hold_until = self._t + c.hold
        real = self._t <= self._hold_until
        return Decision(real=real, pseudo=(not real and c.fallback == "pseudo"),
                        score=s, score_time=st)


def expand_policies(entries: list[dict]) -> list[PolicyConfig]:
    """Expand config entries whose `every` / `threshold` / `hold` are lists into a sweep.

    {name: fixed, type: fixed, every: [2, 5]}  ->  fixed_N2, fixed_N5
    """
    out: list[PolicyConfig] = []
    for e in entries:
        e = dict(e)
        sweep_keys = [k for k in ("every", "offset", "threshold", "hold") if isinstance(e.get(k), list)]
        combos = [{}]
        for k in sweep_keys:
            combos = [dict(c, **{k: v}) for c in combos for v in e[k]]
        for c in combos:
            d = dict(e, **c)
            tag = {"every": "N", "offset": "off", "threshold": "thr", "hold": "hold"}
            suffix = "".join(f"_{tag[k]}{v:g}" for k, v in c.items())
            d["name"] = e["name"] + suffix
            known = {k: d.pop(k) for k in list(d) if k in PolicyConfig.__dataclass_fields__}
            out.append(PolicyConfig(**known, extra=d))
    names = [p.name for p in out]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"Duplicate policy names after expansion: {sorted(dup)}")
    return out
