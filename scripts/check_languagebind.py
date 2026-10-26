"""Step 5.1 debugging of the failed sanity check (3) (PROGRESS.md): is rewind/embed/languagebind.py correct?

  --known-answer DIR  LanguageBind's own sample videos (repo assets/video, 0.mp4 + 1.mp4) and the captions of its
                      inference.py: each video must match its own caption. 8 frames sampled uniformly over the whole
                      file (LanguageBind's sampling), with our pad view and with LanguageBind's centre crop (its
                      decord transform without the random flip).
  --centre-crop-auc   sanity check (3) again on the same tuning clips, frames centre-cropped instead of padded.
                      Diagnostic only, tuning only; the pre-registered pad view is not changed by it.

Run from the repo root:
  uv run python -m scripts.check_languagebind --known-answer <LanguageBind repo>/assets/video
  uv run python -m scripts.check_languagebind --centre-crop-auc
"""

import argparse
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from rewind.embed.languagebind import MEAN, STD
from rewind.eval.metrics import auc
from rewind.ingest.manifest import connect, load_videos
from rewind.paths import REPO_ROOT
from scripts.embed_action import clip_windows, load_model

CAPTIONS = ["Training a parakeet to climb up a ladder.", "A lion climbing a tree to catch a monkey."]  # inference.py
SANITY_CLIPS = ["2018-03-15.14-55-07.15-00-07.hospital.G436", "2018-03-15.14-50-00.14-55-00.school.G424"]  # (3)'s


def centre_crop(lb, bgr: np.ndarray) -> torch.Tensor:
    """Short side to the model size, centre square (bicubic + antialias, rounded like the pad path)."""
    x = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).to(lb.device).permute(2, 0, 1).float()
    h, w = x.shape[-2:]
    s = lb.size / min(h, w)
    x = F.interpolate(x[None], size=(round(h * s), round(w * s)), mode="bicubic", antialias=True,
                      align_corners=False)[0]
    t, l = (x.shape[-2] - lb.size) // 2, (x.shape[-1] - lb.size) // 2
    x = x[:, t:t + lb.size, l:l + lb.size].clamp(0, 255).round() / 255
    mean = torch.tensor(MEAN, device=lb.device).view(3, 1, 1)
    std = torch.tensor(STD, device=lb.device).view(3, 1, 1)
    return ((x - mean) / std).to(lb.dtype)


def known_answer(lb, video_dir: Path) -> None:
    zt = lb.encode_text(CAPTIONS)
    scale = float(lb.model.logit_scale.detach().exp())
    for name, prep in (("pad (ours)", lb.frame_input), ("centre crop (LanguageBind)", lambda f: centre_crop(lb, f))):
        zv = []
        for i in range(len(CAPTIONS)):
            with av.open(str(video_dir / f"{i}.mp4")) as c:
                frames = [f.to_ndarray(format="bgr24") for f in c.decode(video=0)]
            idx = np.linspace(0, len(frames) - 1, lb.num_frames).astype(int)
            zv.append(lb.encode_windows(torch.stack([prep(frames[j]) for j in idx])[None])[0])
        sim = np.stack(zv).astype(np.float32) @ zt.T
        soft = np.exp(scale * sim) / np.exp(scale * sim).sum(1, keepdims=True)
        ok = (sim.argmax(1) == np.arange(len(CAPTIONS))).all()
        print(f"{name}: cosine video x caption\n{sim.round(3)}\nsoftmax(logit_scale * cos)\n{soft.round(4)}\n"
              f"-> {'each video matches its own caption' if ok else 'WRONG MATCH'}")


def centre_crop_auc(lb, cfg: dict) -> None:
    act, s = cfg["action"], cfg["action"]["sanity"]
    conn = connect(REPO_ROOT / "data" / "rewind.sqlite")
    vids = {v.clip: v for v in load_videos(conn)}
    chosen = [vids[c] for c in SANITY_CLIPS]
    lb.frame_input = lambda f: centre_crop(lb, f)
    xs, ms = zip(*[clip_windows(lb, v, cfg, act["batch_windows"])[:2] for v in chosen])
    xb, mb = np.concatenate(xs).astype(np.float32), pd.concat(ms, ignore_index=True)
    types = list(s["prompts"])
    z = lb.encode_text([s["prompts"][t] for t in types])
    ev = pd.read_sql_query("SELECT video_id, activity_type, start_s, end_s FROM events "
                           "WHERE label_source != 'contrib'", conn)
    for i, t in enumerate(types):
        lab = np.zeros(len(mb), bool)
        for e in ev[(ev.activity_type == t) & ev.video_id.isin([v.video_id for v in chosen])].itertuples():
            lab |= ((mb.video_id == e.video_id) & (mb.window_start < e.end_s)
                    & (mb.window_start + act["window_s"] > e.start_s)).to_numpy()
        print(f"centre crop: {t}: {lab.sum()} / {len(lab)} windows positive, AUC (b) {auc(xb @ z[i], lab):.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    p.add_argument("--known-answer", type=Path, metavar="DIR")
    p.add_argument("--centre-crop-auc", action="store_true")
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    lb = load_model(cfg)
    if args.known_answer:
        known_answer(lb, args.known_answer)
    if args.centre_crop_auc:
        centre_crop_auc(lb, cfg)


if __name__ == "__main__":
    main()
