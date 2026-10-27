# When Is Depth Worth Computing for Multi-Object Tracking?

**On-demand monocular depth based on association difficulty.** CSCE 585 (Machine Learning Systems) project, Abdulahi Taiwo.

A ByteTrack tracker on YOLOv8 detections calls **Depth Anything V2** only when its
**ambiguity score** says depth could change an association. It is evaluated on MOT17
against no depth, depth on every frame, depth every N frames, and free pseudo-depth,
using tracking metrics (HOTA, IDF1, AssA, ID switches) and system metrics
(latency, FPS, memory, depth invocation rate).

```
frame ─► YOLOv8 ─► Kalman predict ─► similarity matrix ─► ambiguity score ─┬─ low ──► associate (IoU)
                                                                          └─ high ─► Depth Anything V2
                                                                                     ─► associate (IoU + depth)
```

Design rationale (useful for the report and the final presentation): **[DESIGN.md](DESIGN.md)**.

---

## 1. Setup (Windows laptop, CPU only)

Use Python 3.10, 3.11 or 3.12 (3.11 recommended). In PowerShell, from this folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python tests\test_pipeline.py        # should end with "10 tests passed"
```

The YOLOv8 weights (~22 MB) and the Depth Anything V2 Small weights (~100 MB) download
automatically on first use.

## 2. Get MOT17

1. Download **MOT17.zip** (about 5.5 GB) from <https://motchallenge.net/data/MOT17/>.
2. Unzip it so that this file exists: `data\MOT17\train\MOT17-02-FRCNN\seqinfo.ini`.
3. Optional, to save disk space: only the `-FRCNN` folders are used, so you can delete the
   `-DPM` and `-SDP` folders and the `test` folder.

```powershell
python -m odmot check
```

`check` lists the 7 training sequences (5,316 frames) and writes TrackEval ground truth for
each split.

## 3. Run the experiment

All commands are run from the repo root. `scripts\run_all.bat` runs the whole sequence.

| Step | Command | Time on a CPU laptop |
|---|---|---|
| Measure your speed | `python -m odmot estimate` | 2 min |
| Detect every frame once | `python -m odmot cache-det` | about 0.5 to 1 h |
| Depth on every frame once | `python -m odmot cache-depth` | about 2 to 4 h (run overnight) |
| Tune on the first half of each sequence | `python -m odmot sweep --split train_half` | about 5 to 15 min |
| Report on the second half | `python -m odmot sweep --split val_half` | about 5 to 15 min |
| Does the score find the right frames? | `python -m odmot analyze --split val_half` | about 2 min |
| Demo video | `python -m odmot demo --seq MOT17-09 --frames 150 --policy od_margin_thr0.7 --video results\demo.mp4` | about 5 min |

The times are rough guesses for an Intel i5 with no GPU. `estimate` prints the real
projection for your machine.

**Caching is resumable.** Each finished sequence is skipped on the next run, so if the
laptop sleeps or you press Ctrl+C, run the same command again. While caching, keep the
laptop plugged in and set Windows sleep to "Never". With 8 GB of RAM, close the browser:
each model needs about 1 to 1.5 GB.

**If depth caching is too slow**, set `depth.input_size: 392` in `configs/default.yaml`
(about 1.8 times faster). Likewise `detector.imgsz: 640` speeds up detection but misses
small, distant people. Changing either value creates a new cache folder, so old results
stay valid.

### Why it's organised this way

The neural networks run once per frame and their outputs and **measured times** are
cached. Every policy then replays the same frames. When a policy asks for depth on frame
*t*, it gets frame *t*'s cached depth and is charged frame *t*'s measured depth time. The
accuracy is exactly what the live system would get, and the cost is real measured cost.
This turns months of CPU time into minutes. `demo` runs the models live to confirm that
the simulated latency matches reality.

## 4. What you get

`results/<split>/main/`:

* **`summary.csv`**: one row per policy, with `depth_rate`, `HOTA`, `DetA`, `AssA`, `IDF1`,
  `IDSW`, `MOTA`, `latency_mean_ms`, `latency_p95_ms`, `fps`, `depth_compute_s`,
  `model_memory_mb`, and the **budget-matched comparison** `dHOTA_vs_fixed` /
  `dIDF1_vs_fixed`. These measure how far a policy sits above (+) or below (−) the
  fixed-rate curve at the same depth rate. This column answers the research question.
* **`tradeoff_depth_rate.png`**: HOTA, IDF1 and ID switches against the % of frames with
  depth. The gray dashed line is fixed-rate; colored lines are on-demand.
* **`tradeoff_latency.png`**: the same plots against measured mean latency.
* `<policy>/data/*.txt` (MOT-format tracks), `<policy>/cost.json`, and `metrics.json` (per-sequence).

`results/val_half/analysis/`:

* **`score_analysis.json`** and **`score_recall_at_budget.png`**: on every frame the
  tracker solves association with and without depth on the same state, and checks both
  against the ground-truth identities. This shows how often depth changes, helps, or
  hurts a decision. It also shows how well each score ranks those frames (ROC-AUC), and
  what share of the "depth helped" frames a given depth budget catches.

### Protocol (keep it honest)

1. Choose the on-demand threshold, score type and `depth_weight` on **`train_half`**.
2. Report that **same** configuration on **`val_half`**. It's fine to show the full curves
   for `val_half`, but don't pick the best `val_half` point and present it as the result.
3. Report `latency_p95_ms` next to mean FPS. On-demand makes the average faster, but the
   slowest frames are still the ones that compute depth.

## 5. Policies (in `configs/default.yaml`)

| Name | Depth usage |
|---|---|
| `no_depth` | never (plain ByteTrack) |
| `pseudo_depth` | PD-SORT-style pseudo-depth from box bottoms, every frame (free) |
| `always_depth` | Depth Anything on every frame |
| `fixed_N<k>` | Depth Anything every k frames |
| `od_margin_thr<t>` | **ours**: when the margin ambiguity score ≥ t |
| `od_crowd_thr<t>`, `od_hybrid_thr<t>` | alternative ambiguity scores |
| `od_margin_hold_…` | keep depth on for a few frames after a trigger |
| `od_margin_pd_…` | on-demand depth, with pseudo-depth on all other frames |

A list value in the config (for example `threshold: [0.5, 0.7]`) expands into a sweep. Use
`--only "no_depth,od_margin*"` to run a subset.

## 6. Tests and verification

* `python tests/test_pipeline.py` runs unit and integration tests. They don't need PyTorch
  or the dataset.
* `python tests/make_synthetic.py --out synth` builds a small synthetic dataset in MOT17
  layout with ready-made caches. Run it with
  `python -m odmot --config synth/synthetic.yaml sweep`. It exercises the whole pipeline,
  including TrackEval, in about 2 minutes. **The numbers it produces are not results**;
  it is a plumbing test.
* `scripts/compare_bytetrack.py` checks the tracker against the official ByteTrack code.
  The outputs were identical on 100% of frames (instructions are in the file).

## 7. Repository layout

```
odmot/
  data.py        MOT17 reading, half splits, TrackEval GT preparation
  models.py      YOLOv8 + Depth Anything V2 wrappers, depth-to-box helpers, pseudo-depth
  cache.py       run models once, cache outputs + measured timings (resumable)
  kalman.py      ByteTrack's Kalman filter
  tracker.py     ByteTrack with the gated depth term (verified against official code)
  ambiguity.py   margin / crowd / hybrid ambiguity scores
  policies.py    never / pseudo / always / fixed / on-demand controllers
  runner.py      replay a policy over a split, per-frame cost accounting
  evaluate.py    TrackEval wrapper (HOTA, IDF1, AssA, IDSW, MOTA)
  report.py      summary table, Pareto front, budget-matched deltas
  analysis.py    does the score predict the frames where depth helps?
  plots.py       presentation plots
  live.py        live run with real models + annotated demo video
  cli.py         python -m odmot <command>
configs/default.yaml
tests/           tests + synthetic-data generator
scripts/         run_all.bat / run_all.sh, ByteTrack cross-check
```
