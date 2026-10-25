"""Object detectors behind one interface (step 3.3): batch of BGR frames -> boxes, scores, COCO class ids.

Two backends, imported lazily so this module loads in an environment that has only one of them:
  - "ultralytics": Ultralytics YOLOv8 (AGPL-3.0). Letterbox, fp16 and NMS are done by the library.
  - "yolox": Megvii YOLOX (Apache-2.0). Preprocessing as YOLOX's own ValTransform(legacy=False): resize keeping the
    aspect ratio so the image fits imgsz x imgsz, pad bottom/right with 114, BGR, no normalisation. VERIFY (checked
    at 3.3 by comparing both settings on real frames): the 0.1.1rc0 release weights expect this non-legacy input.
Both return boxes in the pixels of the frame they were given (x1, y1, x2, y2), filtered to `classes`, with score
>= conf_thresh and per-class NMS at nms_iou.
"""

from dataclasses import dataclass

import numpy as np

# COCO-80 class names in the order both detectors use (dataset fact, not a tunable).
COCO_NAMES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
              "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
              "scissors", "teddy bear", "hair drier", "toothbrush"]
PAD_VALUE = 114   # YOLOX letterbox fill


def class_ids(names: list[str]) -> list[int]:
    unknown = [n for n in names if n not in COCO_NAMES]
    if unknown:
        raise ValueError(f"not COCO class names: {unknown}")
    return [COCO_NAMES.index(n) for n in names]


@dataclass
class Detections:
    boxes: np.ndarray    # (n, 4) float32 x1, y1, x2, y2 in input-frame pixels
    scores: np.ndarray   # (n,) float32
    classes: np.ndarray  # (n,) int COCO ids


class UltralyticsDetector:
    def __init__(self, weights: str, imgsz: int, conf_thresh: float, nms_iou: float, classes: list[str]):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.kw = dict(imgsz=imgsz, conf=conf_thresh, iou=nms_iou, classes=class_ids(classes), quantize="fp16",
                       device=0, verbose=False)

    def __call__(self, frames: list[np.ndarray]) -> list[Detections]:
        out = []
        for r in self.model.predict(frames, **self.kw):
            b = r.boxes
            out.append(Detections(b.xyxy.cpu().numpy().astype(np.float32), b.conf.cpu().numpy().astype(np.float32),
                                  b.cls.cpu().numpy().astype(int)))
        return out


class YoloxDetector:
    def __init__(self, exp_name: str, weights: str, imgsz: int, conf_thresh: float, nms_iou: float,
                 classes: list[str], legacy: bool = False):
        import torch
        from yolox.exp import get_exp
        from yolox.utils import postprocess
        self.torch, self.postprocess = torch, postprocess
        exp = get_exp(None, exp_name)
        self.model = exp.get_model()
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)   # official release .pth (VERIFY source)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval().cuda().half()
        self.num_classes, self.imgsz, self.legacy = exp.num_classes, imgsz, legacy
        self.conf, self.nms = conf_thresh, nms_iou
        self.keep = np.array(class_ids(classes))

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        import cv2
        h, w = frame.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = int(h * r), int(w * r)
        padded = np.full((self.imgsz, self.imgsz, 3), PAD_VALUE, np.uint8)
        padded[:nh, :nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        img = padded.transpose(2, 0, 1).astype(np.float32)
        if self.legacy:   # pre-0.2 weights: RGB, /255, ImageNet mean/std
            img = img[::-1] / 255.0
            img = (img - np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)) / np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return np.ascontiguousarray(img, np.float32), r

    def __call__(self, frames: list[np.ndarray]) -> list[Detections]:
        torch = self.torch
        prepped = [self.preprocess(f) for f in frames]
        batch = torch.from_numpy(np.stack([p[0] for p in prepped])).cuda().half()
        with torch.inference_mode():
            raw = self.model(batch).float()
        out = []
        for det, (_, r) in zip(self.postprocess(raw, self.num_classes, self.conf, self.nms), prepped):
            if det is None:
                out.append(Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)))
                continue
            det = det.cpu().numpy()
            cls = det[:, 6].astype(int)
            keep = np.isin(cls, self.keep)
            out.append(Detections((det[keep, :4] / r).astype(np.float32), (det[keep, 4] * det[keep, 5]).astype(np.float32),
                                  cls[keep]))
        return out


def make_detector(backend: str, weights: str, imgsz: int, conf_thresh: float, nms_iou: float, classes: list[str],
                  yolox_exp: str | None = None, legacy: bool = False):
    if backend == "ultralytics":
        return UltralyticsDetector(weights, imgsz, conf_thresh, nms_iou, classes)
    if backend == "yolox":
        if not yolox_exp:
            raise ValueError("yolox backend needs the YOLOX experiment name (e.g. yolox-s)")
        return YoloxDetector(yolox_exp, weights, imgsz, conf_thresh, nms_iou, classes, legacy)
    raise ValueError(f"unknown detector backend {backend!r}")
