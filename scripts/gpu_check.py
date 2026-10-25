"""GPU smoke test (step 0.2): load the configured CLIP model on the GPU in fp16 and embed test images.

Beyond shapes, it checks the embeddings mean something: solid red/green/blue images must each
match their own colour prompt best. Exits non-zero on any failure.

Usage: uv run python scripts/gpu_check.py --pretrained <open_clip tag>
"""

import argparse
import sys
import warnings
from pathlib import Path

import open_clip
import torch
import yaml
from PIL import Image

from rewind.embed.clip_model import check_pretrained_compatible  # noqa: F401 (also imported from here by tests)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Test fixtures, not tunables: (name, RGB) of the solid-colour images and their prompts.
COLOURS = [("red", (255, 0, 0)), ("green", (0, 255, 0)), ("blue", (0, 0, 255))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pretrained", required=True,
                        help="open_clip pretrained tag, e.g. openai (config value is still TBD)")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    args = parser.parse_args()

    model_name = yaml.safe_load(args.config.read_text())["appearance"]["model"]
    if model_name == "TBD":
        sys.exit(f"appearance.model is TBD in {args.config}")
    check_pretrained_compatible(model_name, args.pretrained)

    if not torch.cuda.is_available():
        sys.exit("CUDA is not available to torch. No CPU fallback: fix the driver/torch build first.")
    device = torch.device("cuda")

    print(f"GPU:        {torch.cuda.get_device_name(device)}")
    print(f"torch:      {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"open_clip:  {open_clip.__version__}")
    print(f"model:      {model_name} / pretrained={args.pretrained}")

    # Second line of defence behind check_pretrained_compatible: if open_clip's own check disagrees
    # with ours, its warning becomes an error instead of scrolling past.
    warnings.filterwarnings("error", message="QuickGELU mismatch", category=UserWarning)

    # precision="fp16" is OpenAI-style mixed precision (checked in open_clip 3.3.0 factory.py):
    # Conv/Linear/attention weights become fp16; embeddings, LayerNorms and logit_scale stay fp32.
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=args.pretrained, device=device, precision="fp16"
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    matmul_weights = [m.weight for m in model.modules() if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))]
    not_fp16 = [w.dtype for w in matmul_weights if w.dtype != torch.float16]
    if not matmul_weights or not_fp16:
        sys.exit(f"Expected all Linear/Conv2d weights in float16; {len(not_fp16)} of "
                 f"{len(matmul_weights)} are not ({set(not_fp16)})")
    n_fp16 = sum(p.numel() for p in model.parameters() if p.dtype == torch.float16)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"weights:    {100 * n_fp16 / n_total:.1f}% of {n_total / 1e6:.1f}M params in fp16 (mixed precision)")

    images = torch.stack([preprocess(Image.new("RGB", (224, 224), rgb)) for _, rgb in COLOURS])
    prompts = [f"a plain {name} image" for name, _ in COLOURS]

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        img_emb = model.encode_image(images.to(device, dtype=torch.float16))
        txt_emb = model.encode_text(tokenizer(prompts).to(device))

    # Checked on open_clip 3.3.0: VisionTransformer.output_dim is 512 for ViT-B-16. Comes from the model
    # itself, so the shape check below is a consistency check; the colour test is the meaningful one.
    expected_dim = model.visual.output_dim
    print(f"embedding:  shape={tuple(img_emb.shape)} dtype={img_emb.dtype} (expected dim {expected_dim})")
    free_b, total_b = torch.cuda.mem_get_info(device)
    print(f"VRAM:       torch peak allocated {torch.cuda.max_memory_allocated(device) / 2**20:.0f} MiB; "
          f"device in use {(total_b - free_b) / 2**20:.0f} of {total_b / 2**20:.0f} MiB "
          f"(all processes, incl. CUDA context)")

    failures = []
    if img_emb.shape != (len(COLOURS), expected_dim) or txt_emb.shape != (len(COLOURS), expected_dim):
        failures.append(f"unexpected shapes image={tuple(img_emb.shape)} text={tuple(txt_emb.shape)}")
    if img_emb.dtype != torch.float16:
        failures.append(f"image embeddings are {img_emb.dtype}, expected float16")
    if not (torch.isfinite(img_emb).all() and torch.isfinite(txt_emb).all()):
        failures.append("embeddings contain NaN/inf (fp16 overflow?)")

    img_n = torch.nn.functional.normalize(img_emb.float(), dim=-1)
    txt_n = torch.nn.functional.normalize(txt_emb.float(), dim=-1)
    sim = (img_n @ txt_n.T).cpu()
    print("cosine similarity (rows = images, cols = prompts):")
    print("            " + "  ".join(f"{name:>6}" for name, _ in COLOURS))
    for (name, _), row in zip(COLOURS, sim):
        print(f"  {name:>6}    " + "  ".join(f"{v:6.3f}" for v in row.tolist()))
    for i, (name, _) in enumerate(COLOURS):
        best = int(sim[i].argmax())
        if best != i:
            failures.append(f"{name} image matched '{prompts[best]}' best")

    if failures:
        print("FAIL:\n  " + "\n  ".join(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
