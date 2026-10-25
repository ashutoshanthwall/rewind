"""Step 4.2: CLIP (open_clip) in fp16 with preprocessing done on the GPU.

open_clip's own transform resizes the short side and centre-crops a square, which cuts ~22 % off each side of a
16:9 frame (PROGRESS open-issue item 4). Rewind never centre-crops. Every image is first made square, then resized
with bicubic + antialias interpolation (torch's antialias mode is written to match PIL's, which open_clip uses) and
normalised with the model's own mean/std:
  - frame views: tile0 = the left square of side H (x 0..H), tile1 = the right square (x W-H..W); pad = the whole
    frame letterboxed into a W x W square (black bars, as open_clip's resize_mode='longest' pads with 0);
  - object crops: the box widened by appearance.crop_pad_frac of its width/height on each side, then grown to a
    square around its centre (more context, no distortion); parts outside the frame are black.
The difference to open_clip's PIL path is measured, not assumed (scripts/embed_appearance.py --check-preprocess).
"""

import warnings

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from open_clip.pretrained import get_pretrained_cfg
from torchvision.transforms import Normalize

FRAME_VIEWS = ("tile0", "tile1", "pad")


def check_pretrained_compatible(model_name: str, pretrained: str) -> None:
    """Raise unless open_clip knows this (model, tag) pair and their activations agree.

    open_clip itself only warns when weights trained with QuickGELU are loaded into a GELU model
    (or vice versa), then runs them with the wrong activation. E.g. tag 'openai' needs model
    'ViT-B-16-quickgelu', not 'ViT-B-16'. Uses only bundled metadata: no download, no GPU.
    """
    model_cfg = open_clip.get_model_config(model_name)
    if model_cfg is None:
        raise ValueError(f"Unknown open_clip model {model_name!r}")
    tag_cfg = get_pretrained_cfg(model_name, pretrained)
    if not tag_cfg:
        raise ValueError(f"open_clip has no pretrained tag {pretrained!r} for {model_name!r}; "
                         f"available: {open_clip.list_pretrained_tags_by_model(model_name)}")
    # Missing key means False: the same default open_clip 3.3.0 uses (factory.py, create_model).
    model_quick_gelu = model_cfg.get("quick_gelu", False)
    tag_quick_gelu = tag_cfg.get("quick_gelu", False)
    if model_quick_gelu != tag_quick_gelu:
        raise ValueError(
            f"QuickGELU mismatch: model {model_name!r} has quick_gelu={model_quick_gelu} but tag "
            f"{pretrained!r} was trained with quick_gelu={tag_quick_gelu}. Pick a matching model/tag pair."
        )


class Clip:
    def __init__(self, model_name: str, pretrained: str, device: str = "cuda"):
        check_pretrained_compatible(model_name, pretrained)
        warnings.filterwarnings("error", message="QuickGELU mismatch", category=UserWarning)
        # precision="fp16": open_clip mixed precision (step 0.2): Conv/Linear/attention fp16, LayerNorm fp32
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device, precision="fp16")
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = device
        norm = [t for t in self.preprocess.transforms if isinstance(t, Normalize)]
        if len(norm) != 1:
            raise ValueError(f"{model_name}/{pretrained}: expected one Normalize in {self.preprocess}")
        self.mean = torch.tensor(norm[0].mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(norm[0].std, device=device).view(1, 3, 1, 1)
        size = self.model.visual.image_size
        self.size = size[0] if isinstance(size, (tuple, list)) else size
        self.name = f"{model_name}/{pretrained}"

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        """L2-normalised fp32 text embeddings, (n, d)."""
        z = self.model.encode_text(self.tokenizer(texts).to(self.device)).float()
        return F.normalize(z, dim=-1).cpu().numpy()

    @torch.inference_mode()
    def encode_images(self, batch: torch.Tensor) -> np.ndarray:
        """batch: (n, 3, S, S) normalised fp32 on the device -> L2-normalised fp16 embeddings (n, d) on the CPU."""
        z = self.model.encode_image(batch.half()).float()
        return F.normalize(z, dim=-1).half().cpu().numpy()

    def to_input(self, square: torch.Tensor) -> torch.Tensor:
        """(3, s, s) float 0..255 RGB on the device -> (3, S, S) normalised model input."""
        x = F.interpolate(square[None], size=(self.size, self.size), mode="bicubic", antialias=True,
                          align_corners=False)
        x = x.clamp(0, 255).round() / 255.0          # PIL rounds to uint8 before ToTensor
        return ((x - self.mean) / self.std)[0]


def frame_tensor(frame_bgr: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """HxWx3 uint8 BGR -> (3, H, W) float 0..255 RGB on the device."""
    t = torch.from_numpy(np.ascontiguousarray(frame_bgr)).to(device)
    return t.flip(-1).permute(2, 0, 1).float()


def view_box(view: str, w: int, h: int) -> tuple[int, int, int, int]:
    """Frame region of a frame view, (x1, y1, x2, y2); tiles need a landscape frame."""
    if view == "pad":
        return 0, 0, w, h
    if w < h:
        raise ValueError(f"tiles need a landscape frame, got {w}x{h}")
    return (0, 0, h, h) if view == "tile0" else (w - h, 0, w, h)


def square_region(img: torch.Tensor, x1: int, y1: int, side: int) -> torch.Tensor:
    """The side x side square at (x1, y1) of img (3, H, W); parts outside the image are 0 (black)."""
    _, h, w = img.shape
    out = img.new_zeros((3, side, side))
    sx1, sy1, sx2, sy2 = max(x1, 0), max(y1, 0), min(x1 + side, w), min(y1 + side, h)
    if sx2 > sx1 and sy2 > sy1:
        out[:, sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = img[:, sy1:sy2, sx1:sx2]
    return out


def frame_view(img: torch.Tensor, view: str) -> torch.Tensor:
    _, h, w = img.shape
    if view == "pad":
        side = max(w, h)
        return square_region(img, (w - side) // 2, (h - side) // 2, side)
    x1, y1, x2, _ = view_box(view, w, h)
    return img[:, y1:y1 + h, x1:x2]


def crop_square(box: tuple, pad_frac: float) -> tuple[int, int, int]:
    """(x1, y1, side) of the square used for an object box: widened by pad_frac per side, grown to a square
    around the centre, whole pixels."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh) * (1 + 2 * pad_frac)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side_px = max(1, int(round(side)))
    return int(round(cx - side_px / 2)), int(round(cy - side_px / 2)), side_px
