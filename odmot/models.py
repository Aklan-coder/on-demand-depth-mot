"""Thin wrappers around the two neural networks in the pipeline.

Everything else in this package (tracker, ambiguity score, policies, evaluation) is
plain numpy, so it can be tested without PyTorch. These wrappers are the only place
that imports torch / ultralytics / transformers, and they import lazily.

    YoloDetector(bgr_image)          -> (N, 5) float32 [x1, y1, x2, y2, score], persons only
    DepthAnythingEstimator(rgb)      -> (h, w) float32 relative inverse depth (bigger = closer)
"""
from __future__ import annotations

import numpy as np


def detector_name(weights: str, imgsz: int) -> str:
    """Cache folder name for a detector setting (computable without loading the model)."""
    stem = str(weights).replace("\\", "/").split("/")[-1].replace(".pt", "")
    return f"{stem}_{imgsz}"


def depth_model_name(model_name: str, input_size: int) -> str:
    return f"{model_name.replace(chr(92), '/').rstrip('/').split('/')[-1]}_{input_size}"


class YoloDetector:
    """YOLOv8 person detector (COCO class 0) via the ultralytics package.

    `weights` can be any ultralytics checkpoint name ("yolov8n.pt", "yolov8s.pt", ...) -
    it downloads automatically the first time - or a local path to fine-tuned weights.
    """

    def __init__(self, weights: str = "yolov8s.pt", imgsz: int = 1088, conf: float = 0.05,
                 iou: float = 0.7, device: str = "cpu", threads: int = 0, max_det: int = 300):
        import torch
        from ultralytics import YOLO
        if threads:
            torch.set_num_threads(threads)
        self.model = YOLO(weights)
        self.imgsz, self.conf, self.iou, self.device = imgsz, conf, iou, device
        self.max_det = max_det          # ultralytics default 300; crowded MOT20 frames need more
        self.name = detector_name(weights, imgsz)

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        r = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                               classes=[0], device=self.device, max_det=self.max_det,
                               verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 5), np.float32)
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()[:, None]
        return np.concatenate([xyxy, conf], axis=1).astype(np.float32)


class DepthAnythingEstimator:
    """Depth Anything V2 (Hugging Face port) monocular relative-depth model.

    Pre-processing follows the official repo: resize so the SHORT side equals
    `input_size` (rounded to a multiple of the ViT patch size 14), ImageNet normalisation.
    We do it ourselves with OpenCV instead of the HF image processor, which is faster on
    CPU and avoids API drift between transformers versions.

    On a CPU-only laptop `input_size` is the main speed knob: 518 is the official
    setting; 392 or 308 are roughly 2x / 3x faster with somewhat blurrier depth.
    """

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
                 input_size: int = 518, device: str = "cpu", threads: int = 0, model=None):
        import torch
        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        if model is None:
            from transformers import AutoModelForDepthEstimation
            model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model = model.to(device).eval()
        self.input_size, self.device = input_size, device
        self.name = depth_model_name(model_name, input_size)

    def preprocess(self, rgb: np.ndarray):
        import cv2
        h, w = rgb.shape[:2]
        s = self.input_size / min(h, w)
        nh = max(14, int(round(h * s / 14)) * 14)
        nw = max(14, int(round(w * s / 14)) * 14)
        x = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        x = (x - self.MEAN) / self.STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        return self.torch.from_numpy(x).to(self.device)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        x = self.preprocess(rgb)
        with self.torch.inference_mode():
            out = self.model(pixel_values=x).predicted_depth  # (1, h, w)
        return out[0].float().cpu().numpy()


def normalize_nearness(disp: np.ndarray, lo_pct: float = 2, hi_pct: float = 98) -> np.ndarray:
    """Map a relative inverse-depth map to [0, 1] "nearness" (1 = closest).

    Depth Anything is affine-invariant: its raw output has an unknown per-frame scale and
    shift, so raw values are NOT comparable across frames. Robust per-frame min-max
    (2nd/98th percentile) makes values roughly comparable between neighbouring frames,
    which is all the tracker needs (it compares a track's depth to a detection's depth).
    """
    lo, hi = np.percentile(disp, [lo_pct, hi_pct])
    return np.clip((disp - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def box_nearness(near_map: np.ndarray, boxes_xyxy: np.ndarray, img_w: int, img_h: int,
                 x_frac: float = 0.4, y_range: tuple[float, float] = (0.2, 0.8)) -> np.ndarray:
    """Median nearness inside the central part of each box.

    Person boxes contain a lot of background at the sides and above the shoulders, so we
    sample only the central `x_frac` of the width and the `y_range` band of the height
    (torso / upper legs). Median is robust to the remaining background pixels.
    """
    if len(boxes_xyxy) == 0:
        return np.zeros((0,), np.float32)
    mh, mw = near_map.shape
    sx, sy = mw / img_w, mh / img_h
    out = np.empty(len(boxes_xyxy), np.float32)
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy[:, :4]):
        cx, bw, bh = (x1 + x2) / 2, (x2 - x1), (y2 - y1)
        xa, xb = (cx - bw * x_frac / 2) * sx, (cx + bw * x_frac / 2) * sx
        ya, yb = (y1 + bh * y_range[0]) * sy, (y1 + bh * y_range[1]) * sy
        xa, xb = int(np.clip(np.floor(xa), 0, mw - 1)), int(np.clip(np.ceil(xb), 1, mw))
        ya, yb = int(np.clip(np.floor(ya), 0, mh - 1)), int(np.clip(np.ceil(yb), 1, mh))
        patch = near_map[ya:max(yb, ya + 1), xa:max(xb, xa + 1)]
        out[i] = float(np.median(patch)) if patch.size else np.nan
    return out


def pseudo_nearness(boxes_xyxy: np.ndarray, img_h: int) -> np.ndarray:
    """PD-SORT-style pseudo-depth from box geometry, as nearness in [0, 1].

    With a roughly level camera looking at a ground plane, people whose feet are lower in
    the image are closer. PD-SORT uses the distance from the box bottom to the image
    bottom; we use y2 / H so that, like the real depth, 1 = closest. Cost: ~free.
    """
    if len(boxes_xyxy) == 0:
        return np.zeros((0,), np.float32)
    return np.clip(boxes_xyxy[:, 3] / img_h, 0.0, 1.0).astype(np.float32)
