"""Step 5.1: LanguageBind-Video (V1.5_FT) as a dual encoder: 8-frame windows and query texts embed independently.

Model code is vendored (rewind/embed/languagebind_vendor, NOTICE.md). Weights are loaded with
torch.load(weights_only=True); every key must match except the text position_ids buffer, which the checkpoint does not
store (a constant index, checked to be 0..76); any other missing or unexpected key stops the run. Inference in
fp16 (model.half()); embeddings L2-normalised here (get_image_features / get_text_features return raw projections).

Preprocessing (pre-registered in PROGRESS.md, step 5.1): every frame is the 'pad' view of rewind/embed/clip_model.py
(whole frame letterboxed into a square, black bars, no centre crop), resized to the model's image size with bicubic +
antialias, rounded to 0..255 like the CLIP path, scaled to 0..1 and normalised with the OpenAI CLIP mean/std.
LanguageBind's own transform is not used: it centre-crops and flips at random at inference.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer

from rewind.embed.clip_model import frame_tensor, frame_view
from rewind.embed.languagebind_vendor.configuration_video import LanguageBindVideoConfig
from rewind.embed.languagebind_vendor.modeling_video import LanguageBindVideo

# OpenAI CLIP normalisation, as in LanguageBind's processing_video.py (OPENAI_DATASET_MEAN / _STD)
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
TEXT_POSITION_IDS = "text_model.embeddings.position_ids"
# Files the loader reads (scripts/fetch_languagebind.sh fetches and checks all of them). transformers loads a missing
# config or tokenizer silently with defaults (pre-Phase-10 review, repro-1), so presence and basic shape are checked.
REQUIRED_FILES = ("config.json", "pytorch_model.bin", "tokenizer.json", "vocab.json", "merges.txt",
                  "tokenizer_config.json", "special_tokens_map.json")
CLIP_VOCAB = 49408                # CLIP BPE vocabulary (OpenAI CLIP / LanguageBind text tower)


class LanguageBind:
    def __init__(self, weights_dir: Path, device: str = "cuda", dtype: torch.dtype = torch.float16):
        weights_dir = Path(weights_dir)
        missing_files = [f for f in REQUIRED_FILES if not (weights_dir / f).is_file()]
        if missing_files:
            raise FileNotFoundError(f"{weights_dir}: missing {missing_files}; run bash scripts/fetch_languagebind.sh")
        config = LanguageBindVideoConfig.from_pretrained(weights_dir)
        model = LanguageBindVideo(config)
        state = torch.load(weights_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # The checkpoint does not store the text tower's position_ids buffer (a constant 0..76 index that newer
        # transformers stopped saving); every learned tensor must load. Anything else missing or extra is an error.
        if unexpected or set(missing) != {TEXT_POSITION_IDS}:
            raise RuntimeError(f"state dict mismatch: missing {missing}, unexpected {unexpected}")
        pos = model.get_buffer(TEXT_POSITION_IDS)
        if not torch.equal(pos, torch.arange(pos.shape[-1]).expand_as(pos)):
            raise RuntimeError(f"{TEXT_POSITION_IDS} is not 0..{pos.shape[-1] - 1}")
        self.model = model.to(device=device, dtype=dtype).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(weights_dir)
        if len(self.tokenizer) != CLIP_VOCAB:
            raise RuntimeError(f"tokenizer in {weights_dir} has {len(self.tokenizer)} tokens, expected {CLIP_VOCAB}")
        self.num_frames = config.vision_config.num_frames
        self.size = config.vision_config.image_size
        self.dim = config.projection_dim
        self.max_tokens = config.text_config.max_position_embeddings     # 77
        self.device, self.dtype = device, dtype
        self.mean = torch.tensor(MEAN, device=device).view(3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(3, 1, 1)

    def frame_input(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """HxWx3 uint8 BGR -> (3, S, S) normalised input on the device (pad view, model dtype)."""
        sq = frame_view(frame_tensor(frame_bgr, self.device), "pad")
        x = F.interpolate(sq[None], size=(self.size, self.size), mode="bicubic", antialias=True, align_corners=False)
        x = x.clamp(0, 255).round()[0] / 255.0
        return ((x - self.mean) / self.std).to(self.dtype)

    @torch.inference_mode()
    def encode_windows(self, windows: torch.Tensor) -> np.ndarray:
        """windows: (B, T, 3, S, S) frame inputs, T = num_frames -> L2-normalised fp16 embeddings (B, dim) on the CPU."""
        if windows.ndim != 5 or windows.shape[1] != self.num_frames:
            raise ValueError(f"expected (B, {self.num_frames}, 3, S, S) windows, got {tuple(windows.shape)}")
        z = self.model.get_image_features(pixel_values=windows.permute(0, 2, 1, 3, 4)).float()   # model: b c t h w
        return F.normalize(z, dim=-1).half().cpu().numpy()

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        """L2-normalised fp32 text embeddings, (n, dim). CLIP tokenizer, padded to max_tokens (pad = end of text);
        a text longer than max_tokens is an error, never cut silently."""
        too_long = [x for x in texts if len(self.tokenizer(x)["input_ids"]) > self.max_tokens]
        if too_long:
            raise ValueError(f"texts over {self.max_tokens} tokens: {too_long}")
        t = self.tokenizer(texts, max_length=self.max_tokens, padding="max_length", return_tensors="pt")
        z = self.model.get_text_features(input_ids=t["input_ids"].to(self.device),
                                         attention_mask=t["attention_mask"].to(self.device)).float()
        return F.normalize(z, dim=-1).cpu().numpy()
