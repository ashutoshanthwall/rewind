"""Vendored YOLOX inference code (Megvii, Apache-2.0); see NOTICE.md and LICENSE in this folder.

build_yolox() rebuilds the model exactly as yolox.exp.Exp.get_model() does for the released checkpoints.
"""

from torch import nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .yolox import YOLOX

# depth / width multipliers per released model (yolox-0.3.0 exps/default/*.py); all use SiLU and 80 COCO classes.
SIZES = {"yolox-nano": (0.33, 0.25), "yolox-tiny": (0.33, 0.375), "yolox-s": (0.33, 0.50), "yolox-m": (0.67, 0.75),
         "yolox-l": (1.0, 1.0), "yolox-x": (1.33, 1.25)}
NUM_CLASSES = 80


def build_yolox(name: str) -> YOLOX:
    if name not in SIZES or name in ("yolox-nano",):   # nano uses depthwise convs (not supported here)
        raise ValueError(f"unsupported YOLOX model {name!r}; expected one of {sorted(set(SIZES) - {'yolox-nano'})}")
    depth, width = SIZES[name]
    in_channels = [256, 512, 1024]
    model = YOLOX(YOLOPAFPN(depth, width, in_channels=in_channels, act="silu"),
                  YOLOXHead(NUM_CLASSES, width, in_channels=in_channels, act="silu"))
    for m in model.modules():          # as Exp.get_model's init_yolo: BatchNorm eps matters at inference
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
    return model
