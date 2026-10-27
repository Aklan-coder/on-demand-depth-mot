# Design notes: what was decided, and why

Written so you can defend each choice in the report and the Week 15 presentation. Every
choice below is also a place a reviewer could push back, so the trade-off is stated.

## 1. The question, made testable

> At the same depth-computation budget, does on-demand depth give better tracking than a
> fixed schedule?

"Better" is measured by HOTA, IDF1, AssA and ID switches (TrackEval, official code).
"Budget" is the **fraction of frames on which Depth Anything runs** (`depth_rate`), plus
the measured latency it costs. The fixed-rate policies (every 2, 3, 5, 10, 20, 30 frames),
together with no-depth (rate 0) and always-depth (rate 1), trace a reference curve. Every
policy is reported as its distance above or below that curve at its own rate
(`dHOTA_vs_fixed`, `dIDF1_vs_fixed`). One positive number then answers the question; a
single on-demand point next to a single fixed-rate point would not.

## 2. Cache-and-replay instead of running models per experiment

Depth Anything V2-Small takes on the order of a second per frame on a CPU laptop. A sweep
of about 30 policies × 2 splits × 5,316 frames would need weeks. So each network runs
**once** per frame. Its outputs (boxes, per-box depth) and its **measured wall-clock
time** are cached, and each policy replays the cached frames:

* **Accuracy is exact.** A policy that uses depth on frame *t* sees exactly what the live
  system would see on frame *t*.
* **Cost is measured.** The frame is charged the depth time actually measured on frame *t*
  on your laptop, plus detector time and the tracker's own time. The tracker's time
  includes the ambiguity-score computation, so on-demand pays for its own decision logic.
* **Caveat.** In the cache runs the two models ran in separate passes, not
  interleaved. `odmot demo` runs everything live; compare its latency with the cached
  numbers for the same frames to show the simulation is faithful.

## 3. Tracker: ByteTrack, re-implemented in numpy and verified

I wrote my own implementation instead of importing a library, for three reasons: it
needs a hook between "similarity computed" and "association solved"; it has to run on
Windows without compiled packages (`lap`, `cython_bbox`); and every line has to be
explainable. Fidelity is checked by `scripts/compare_bytetrack.py`, which runs the
official ByteTrack code on the same detections. The outputs (IDs and boxes) were
**identical on 100% of frames**, and 98.6% when the official code's +1-pixel IoU
convention is used.

Detector: COCO-pretrained YOLOv8 (person class), not a MOT-fine-tuned YOLOX like the
ByteTrack paper uses. Absolute HOTA will therefore be below published ByteTrack numbers.
That is fine, because all policies share the same detections and only relative
comparisons are claimed. Say this explicitly in the report.

BoT-SORT (named on the slides) adds camera-motion compensation and ReID. On a CPU, ReID
would dominate the runtime and muddy the depth-cost story, so it was left out. CMC would
be the most useful extension for the moving-camera sequences (05, 10, 11, 13).

## 4. How depth enters association: gate, then re-rank

```
cost(i, j) = (1 − sim_ij) + w · |near_i − near_j|     only for pairs that pass the IoU gate
```

* The IoU gate is applied **before** depth is added. Depth never allows a match that
  ByteTrack would forbid. It only changes the ranking among matches that are already
  plausible. This keeps the experiment about *disambiguation*: the tracker without depth
  is unchanged, and depth can only break ties.
* The cost of leaving a track unmatched is raised by `w/2`, so a large depth difference
  can't *reject* an otherwise valid match. Rejecting matches would cause fragmentation
  and confound the comparison.
* `near_i` is the track's depth state: an exponential moving average of the depth of its
  matched detections, ignored once older than `depth_stale` frames. Real depth and
  pseudo-depth are kept as **separate** track states, because their scales are not
  comparable.

**Provable property (tested in `test_zero_margin_means_depth_cannot_change_assignment`).**
If no track has two valid candidates and no detection has two valid tracks, the valid
pairs form a matching. Every valid pair is cheaper than leaving both sides unmatched, so
the assignment is the same for *any* depth values. Depth can change the first
association **only** on frames where the margin score is above 0. That is the formal
reason the margin score is the right signal: its zero set is exactly the set of frames
where computing depth is provably wasted.

## 5. Depth representation

* Depth Anything outputs **relative inverse depth** with an unknown per-frame scale and
  shift (it is affine-invariant). Raw values are not comparable across frames. Each map
  is normalized to [0, 1] "nearness" using its 2nd and 98th percentiles. This is good
  enough to compare a track (seen a few frames ago) with a detection now. It is an
  approximation: when the scene content changes, the normalization drifts. A metric
  depth model would remove this, at a higher cost.
* Per-box depth is the **median** over the central 40% of the box width and 20–80% of its
  height. That covers torso and upper legs and avoids the background at the sides and
  above the head.
* **Pseudo-depth** (PD-SORT style) is `y2 / H`, the box bottom, which costs nothing. It
  assumes a level camera and a flat ground plane, and fails on stairs, slopes, a tilted
  camera, or occluded feet. The `pseudo_depth` baseline asks the question every reviewer
  will ask: is a neural depth network needed at all?

## 6. The ambiguity scores

Each score uses only data the tracker has *already* computed before association, so it
costs microseconds:

* **margin**: for each track, the second-best valid candidate divided by the best
  (using the tracker's own fused IoU × score similarity); the same for each detection
  over tracks; the frame score is the maximum. It is 0 when every decision is clear-cut
  and close to 1 when two candidates are nearly tied.
* **crowd**: the maximum IoU between any two high-confidence detections. It doesn't need
  tracks and can fire a frame or two *before* tracks start competing.
* **hybrid**: max(margin, crowd).

`hold` keeps depth on for k frames after a trigger. Occlusion events last several frames,
and a fresh depth reading on the tracks *before* a crossing is what makes depth useful
*during* it.

**Known weakness, to discuss in the report:** the frame score is a maximum over all
tracks. In a crowded sequence (MOT17-04), almost every frame contains at least one
ambiguous pair, so the trigger rate saturates. Natural fixes, as future work: a count or
fraction of ambiguous tracks instead of the max, or running depth only on a crop around
the ambiguous tracks (region-level instead of frame-level).

## 7. The score-quality analysis (`odmot analyze`)

The tracking metrics show *whether* on-demand wins. The analysis shows *why*. In an
always-depth run, every frame's first association is solved twice on the same tracker
state, with and without depth. Each detection's ground-truth identity is known (Hungarian
matching at IoU ≥ 0.5), and so is each track's latest ground-truth identity. That makes
it possible to label each frame:

* **changed**: depth altered at least one match;
* **helped**: depth produced more correct matches (or fewer wrong ones);
* **hurt**: the reverse.

Two outputs follow. The first is the ROC-AUC of each score for predicting "helped"
frames. The second is the **recall at budget**: the share of "helped" frames caught when
depth runs on the top-k% scored frames. A fixed schedule catches about k%; that is the
diagonal. A score well above the diagonal is the clearest single piece of evidence for
the idea, and it holds even if the overall HOTA differences turn out small.

Expect depth to change very few frames. That is itself a finding: it quantifies how much
always-on depth computation is wasted.

## 8. System metrics, honestly

* **Mean latency / FPS** improve with on-demand. **p95 latency** does not: the frames that
  compute depth are as slow as ever. A real-time system has to be provisioned for p95, so
  report both.
* **Memory is not saved.** The depth model has to stay loaded to be callable on demand.
  `model_memory_mb` reports this. Say it plainly: the savings are in compute, energy and
  average throughput, not in RAM.
* `depth_compute_s` (total seconds spent in depth) is the closest thing to an energy proxy
  available on a laptop.

## 9. Positioning against related work

Make the claim about **when** depth is computed, not **whether** depth helps tracking.
Depth-aware association is established prior work (PD-SORT, DepthMOT, SparseTrack-style
depth cues, and the work cited on the slides), and the system builds on it. Before
writing the related-work section, check how each paper you cite computes its depth cue
and what it costs. The contribution to argue is the **compute-adaptive controller**, the
**budget-matched evaluation protocol**, and the evidence (section 7) that an
association-level signal predicts where depth matters.

## 10. Threats to validity and extensions

Threats:
* Untuned detector (COCO YOLOv8).
* No camera-motion compensation.
* Relative depth normalized per frame.
* Hyper-parameters (`depth_weight`, `depth_stale`) are tuned on train_half only; vary
  `depth_weight` in an ablation.
* MOT17 has only 7 training sequences, so report per-sequence numbers (`metrics.json`)
  as well as the combined score.

Extensions, in order of value:
* Region-level depth on crops around the ambiguous tracks.
* A budget controller that adapts the threshold to hit a target depth rate.
* Camera-motion compensation (BoT-SORT).
* A learned trigger trained on the "helped" labels from section 7.
