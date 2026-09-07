# When Is Depth Worth Computing for Multi-Object Tracking?

## On-Demand Monocular Depth Based on Association Difficulty

**Author:** Abdulahi Taiwo  
**Course:** CSCE 585  
**Research Area:** Computer Vision · Multi-Object Tracking · Monocular Depth Estimation · Computational Efficiency  
**Project Status:** **Approved Research Proposal — Implementation Pending**

---

## Overview

Multi-object tracking (MOT) requires a tracking system to detect multiple objects and maintain consistent identities for those objects across video frames. This becomes particularly challenging when objects overlap, cross paths, or undergo occlusion, creating uncertainty when detections must be associated with existing tracks.

Depth information can provide additional spatial information that may help resolve difficult associations and preserve object identities. However, monocular depth estimation introduces additional computational cost. Computing depth for every frame may therefore be unnecessary when object associations are already straightforward.

This project proposes an **on-demand monocular depth strategy for multi-object tracking**. Rather than estimating depth continuously, the proposed system will use an **association-ambiguity score** to determine when additional depth information is needed.

When association ambiguity is low, depth estimation will be skipped. When ambiguity is high, monocular depth estimation will be invoked to provide additional spatial information for the association process.

The project will investigate whether this selective strategy can provide a better **tracking-performance versus computational-cost trade-off** than computing depth continuously or at predetermined fixed intervals.

---

## Problem Statement

Maintaining consistent object identities is a fundamental challenge in multi-object tracking.

Association becomes particularly difficult when objects:

- overlap;
- cross paths;
- undergo partial or temporary occlusion; or
- create multiple plausible detection-to-track associations.

These situations can increase the likelihood of incorrect associations and identity switches.

Depth information provides an additional spatial cue that may help distinguish objects during difficult association events. However, obtaining monocular depth estimates requires additional computation.

This creates an important efficiency problem:

> **Should a tracking system pay the computational cost of depth estimation on every frame when many frames may contain straightforward associations?**

This project investigates whether depth computation can instead be allocated according to **association difficulty**.

---

## Research Question

The primary research question is:

> **Can tracking ambiguity determine when depth is needed?**

The project investigates whether an association-difficulty signal can identify situations in which monocular depth information may be sufficiently useful to justify its computational cost.

A central experimental question is:

> **At the same depth-computation budget, does on-demand depth outperform fixed-rate depth?**

This comparison will investigate whether deciding **when** to compute depth according to tracking difficulty is more effective than allocating the same depth computation at predetermined temporal intervals.

---

## Motivation

Not all tracking situations have the same level of difficulty.

Some frames may contain well-separated objects with straightforward associations. Other frames may contain overlapping, crossing, or occluded objects for which association becomes considerably more difficult.

This creates a trade-off between the potential value of depth information and the computational cost required to obtain it.

Four general depth-computation strategies can therefore be considered:

1. **No Depth** — never compute depth.
2. **Always Depth** — compute depth for every frame.
3. **Fixed-Rate Depth** — compute depth every *N* frames.
4. **On-Demand Depth** — compute depth when association ambiguity indicates that additional spatial information may be useful.

The fourth strategy forms the central focus of this research.

Rather than asking only whether depth can assist multi-object tracking, this project investigates a more resource-aware question:

> **When is depth worth computing?**

---

## Proposed Methodology

The proposed framework consists of four primary components:

### 1. Object Detection

**YOLOv8** is planned as the object detector for identifying objects in each video frame.

### 2. Multi-Object Tracking

**ByteTrack or BoT-SORT** will provide the baseline tracking and identity-association framework.

### 3. Association-Ambiguity Estimation

A proposed **ambiguity score** will evaluate the difficulty of the current detection-to-track association problem and determine whether additional depth information should be requested.

### 4. On-Demand Monocular Depth Estimation

**Depth Anything V2** will be used to estimate monocular depth when the association process is determined to have high ambiguity.

### Proposed Architecture

```text
                     Input Video
                          |
                          v
                  Object Detection
                       YOLOv8
                          |
                          v
                Multi-Object Tracking
                ByteTrack / BoT-SORT
                          |
                          v
              Association Difficulty
                       Analysis
                          |
                          v
                   Ambiguity Score
                     /          \
                    /            \
                   v              v
          Low Ambiguity      High Ambiguity
                |                  |
                v                  v
           Skip Depth       Compute Monocular
                             Depth Information
                         (Depth Anything V2)
                |                  |
                \                  /
                 \                /
                  v              v
                Association / Track Update
                          |
                          v
                     Output Tracks
```

The fundamental decision policy is:

> **Low Ambiguity → Skip Depth**

> **High Ambiguity → Compute Depth**

The objective is not to eliminate depth estimation, but to allocate it selectively to tracking situations in which additional spatial information may be most useful.

---

## Association-Ambiguity Concept

A central component of the proposed research is determining how **association difficulty** should be quantified.

During multi-object tracking, detections in the current frame must be matched with existing tracks. Some matches may be clear, while others may involve several plausible associations.

The project will investigate an ambiguity measure capable of distinguishing between these conditions.

Potential indicators to be investigated during implementation may include:

- competing detection-to-track matches;
- object overlap;
- crossing trajectories;
- occlusion;
- similarity among candidate associations; and
- uncertainty within the track-to-detection matching process.

The exact mathematical formulation and threshold of the ambiguity score have **not yet been finalized** and remain part of the planned research.

Conceptually:

```text
Clear Association
       |
       v
Low Ambiguity
       |
       v
Skip Depth


Difficult Association
       |
       v
High Ambiguity
       |
       v
Invoke Depth Estimation
       |
       v
Use Additional Spatial Information
for Association
```

This allows depth computation to respond to the state of the tracking problem rather than following a fixed schedule.

---

## Research Objectives

The planned research objectives are to:

1. Establish a baseline multi-object tracking pipeline.
2. Investigate measurable indicators of association difficulty during multi-object tracking.
3. Develop an ambiguity-based mechanism for determining when monocular depth estimation should be invoked.
4. Integrate Depth Anything V2 with the tracking pipeline for selective depth estimation.
5. Implement an on-demand depth-triggering strategy based on association ambiguity.
6. Compare the proposed approach against **No Depth, Always Depth, and Fixed-Rate Depth** strategies.
7. Evaluate the effect of selective depth computation on tracking and identity association.
8. Measure the computational requirements associated with the different depth strategies.
9. Determine whether on-demand depth can provide a better balance between tracking quality and computational efficiency.

---

## Research Hypothesis

The project is based on the hypothesis that:

> **Depth information does not need to be computed uniformly across every video frame to provide useful information for multi-object tracking.**

Instead, selectively invoking monocular depth estimation during periods of high association ambiguity may provide useful spatial information during difficult associations while avoiding unnecessary depth computation during straightforward tracking conditions.

The experimental phase of this project will determine whether this hypothesis is supported.

---

## Experimental Plan

### Dataset

The planned experimental evaluation will use the **MOT17** benchmark dataset.

MOT17 provides established video sequences for evaluating multi-object tracking systems and will allow the proposed approach to be evaluated using recognized MOT performance metrics.

### Baselines and Comparison Strategies

| Method | Depth Usage |
|---|---|
| **No Depth** | Depth is never computed |
| **Always Depth** | Depth is computed for every frame |
| **Fixed-Rate Depth** | Depth is computed every *N* frames |
| **On-Demand Depth (Proposed)** | Depth is computed when association ambiguity is high |

These comparisons are intended to determine both whether depth is useful and whether **selective allocation** of depth computation provides advantages over conventional allocation strategies.

### Compute-Budget Comparison

A central part of the experimental design will compare **Fixed-Rate Depth** and **On-Demand Depth** under comparable depth-computation budgets.

For example, if an on-demand configuration invokes depth for a particular proportion of frames, a fixed-rate configuration can be selected to produce a comparable number of depth invocations.

Tracking performance can then be evaluated under approximately equivalent depth-computation budgets.

The resulting comparison addresses the central question:

> **Given approximately the same depth-computation budget, is it more effective to allocate depth according to association difficulty than according to fixed temporal intervals?**

---

## Evaluation Metrics

Evaluation will consider both **tracking performance** and **system performance**.

### Tracking Metrics

#### HOTA — Higher Order Tracking Accuracy

HOTA will provide a measure of overall multi-object tracking performance.

#### IDF1 — Identity F1 Score

IDF1 will be used to evaluate how consistently object identities are preserved throughout tracking sequences.

#### AssA — Association Accuracy

Association Accuracy is particularly relevant because the proposed method focuses directly on difficult object-association situations.

#### ID Switches

Identity switches will be measured to determine how frequently the tracker incorrectly changes an object's assigned identity.

### System Metrics

The computational behavior of each strategy will also be evaluated.

Planned system metrics include:

- **Latency**
- **Frames Per Second (FPS)**
- **Memory Usage**
- **Depth Invocation Frequency**

Depth invocation frequency will measure how often monocular depth estimation is activated and will support comparisons between fixed-rate and on-demand strategies under similar depth-computation budgets.

---

## Implementation Roadmap

### Phase 1 — Baseline Setup

- Prepare the development environment.
- Prepare the MOT17 dataset.
- Establish YOLOv8 object detection.
- Establish the baseline ByteTrack/BoT-SORT tracking pipeline.

### Phase 2 — Association Difficulty

- Study the track-to-detection association process.
- Identify measurable indicators of difficult associations.
- Develop the proposed ambiguity score.
- Determine an appropriate depth-triggering policy.

### Phase 3 — Depth Integration

- Integrate Depth Anything V2.
- Associate depth information with tracked objects.
- Establish the Always-Depth configuration.

### Phase 4 — On-Demand Depth

- Connect the ambiguity mechanism to the depth estimator.
- Implement selective depth invocation.
- Record depth-triggering events.

### Phase 5 — Baseline Experiments

Evaluate:

- No Depth
- Always Depth
- Fixed-Rate Depth
- On-Demand Depth

### Phase 6 — Performance Evaluation

Measure:

- HOTA
- IDF1
- AssA
- ID Switches
- Latency
- FPS
- Memory Usage
- Depth Invocation Frequency

### Phase 7 — Compute-Budget Analysis

Compare Fixed-Rate Depth and On-Demand Depth under comparable depth-computation budgets.

### Phase 8 — Analysis and Documentation

- Analyze experimental results.
- Generate figures and tables.
- Document limitations.
- Prepare the final CSCE 585 project report.
- Update the repository with experimental results and reproducibility instructions.

---

## Current Progress

**Current Stage: Approved Research Proposal**

This repository currently documents the proposed methodology and experimental plan. **Implementation and experimental evaluation have not yet been completed.**

### Progress Checklist

- [x] Research problem identified
- [x] Research question formulated
- [x] Initial literature review
- [x] Proposed methodology developed
- [x] Experimental strategy defined
- [x] Project proposal presented
- [x] Project proposal approved
- [ ] Development environment setup
- [ ] MOT17 dataset preparation
- [ ] Baseline tracker implementation
- [ ] Association ambiguity formulation
- [ ] Ambiguity scoring implementation
- [ ] Depth Anything V2 integration
- [ ] On-demand depth triggering
- [ ] No-Depth experiments
- [ ] Always-Depth experiments
- [ ] Fixed-Rate experiments
- [ ] On-Demand experiments
- [ ] Compute-budget comparison
- [ ] Tracking performance evaluation
- [ ] System performance evaluation
- [ ] Results analysis
- [ ] Final project report

The repository will be updated as implementation and experimentation progress.

---

## Related Work

The project is informed by recent research in **multi-object tracking, depth-aware association, occlusion handling, and adaptive depth computation**.

### PD-SORT

**PD-SORT: Occlusion-Robust Multi-Object Tracking Using Pseudo-Depth Cues** investigates the use of pseudo-depth cues for improving multi-object tracking, particularly under occlusion.

This work is relevant to the present project because it demonstrates how depth-related spatial information can contribute to object association. The proposed project differs in its central question: rather than simply making depth information available to the tracker, it investigates **when depth computation is worth performing**.

- **DOI:** [10.1109/TCE.2025.3541839](https://doi.org/10.1109/TCE.2025.3541839)

### FDTA

**From Detection to Association: Learning Discriminative Object Embeddings for Multi-Object Tracking (FDTA)** addresses association quality in multi-object tracking through feature refinement, including depth-aware spatial cues.

FDTA is relevant to this project because it further demonstrates the importance of spatial and depth-aware information for improving object association.

- **CVPR 2026:** [Official Paper Page](https://openaccess.thecvf.com/content/CVPR2026/html/Shao_From_Detection_to_Association_Learning_Discriminative_Object_Embeddings_for_Multi-Object_CVPR_2026_paper.html)

### CADENCE

**CADENCE: Context-Adaptive Depth Estimation for Navigation and Computational Efficiency** investigates adaptive depth estimation from the perspective of computational efficiency.

Its context-adaptive computation concept is particularly relevant to the motivation of this project: computationally expensive depth estimation may not need to operate uniformly when the value of depth varies with the current situation.

- **DOI:** [10.1109/AIIoT68874.2026.11569617](https://doi.org/10.1109/AIIoT68874.2026.11569617)

Together, these works motivate the intersection explored in this project:

> **Can the association state of a multi-object tracker itself determine when the computational cost of monocular depth estimation is justified?**

---

## References

1. Y. Wang, D. Zhang, R. Li, Z. Zheng, and M. Li, **“PD-SORT: Occlusion-Robust Multi-Object Tracking Using Pseudo-Depth Cues,”** *IEEE Transactions on Consumer Electronics*, 2025.  
   [DOI: 10.1109/TCE.2025.3541839](https://doi.org/10.1109/TCE.2025.3541839)

2. Y. Shao, Y. Yang, R. Yu, W. Li, X. Guo, H. Yan, W. Wang, and X. Sun, **“From Detection to Association: Learning Discriminative Object Embeddings for Multi-Object Tracking,”** *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 6878–6888, 2026.  
   [CVPR 2026 Official Paper Page](https://openaccess.thecvf.com/content/CVPR2026/html/Shao_From_Detection_to_Association_Learning_Discriminative_Object_Embeddings_for_Multi-Object_CVPR_2026_paper.html)

3. T. K. Johnsen and M. Levorato, **“CADENCE: Context-Adaptive Depth Estimation for Navigation and Computational Efficiency,”** *2026 IEEE World AI IoT Congress (AIIoT)*, 2026.  
   [DOI: 10.1109/AIIoT68874.2026.11569617](https://doi.org/10.1109/AIIoT68874.2026.11569617)

### Planned Technical Components

The proposed implementation will additionally use or evaluate:

- **YOLOv8** — object detection
- **ByteTrack** — multi-object tracking
- **BoT-SORT** — multi-object tracking
- **Depth Anything V2** — monocular depth estimation
- **MOT17** — multi-object tracking benchmark

Complete technical citations for the implementation components will be maintained as the implementation stage begins.

---

## Project Information

**Author:** Abdulahi Taiwo  
**Course:** CSCE 585  
**Research Area:** Computer Vision / Multi-Object Tracking / Monocular Depth Estimation / Computational Efficiency  
**Status:** **Approved Research Proposal — Implementation Pending**

> This repository will be updated throughout the implementation, experimentation, and evaluation stages of the CSCE 585 research project.
