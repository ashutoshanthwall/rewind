"""Step 2.2 review aid: a local HTML page with n random drafted queries next to their keyframe sheets.

The sample is drawn by seed (config queries.seed, queries.review_n) over all rows with a query, so the reviewer sees
an unbiased sample, not rows chosen by the writer. Open the page in a browser; images load from
eval/queries/keyframes/ (local only, nothing is uploaded).

Run from the repo root:
  uv run python -m scripts.review_queries
"""

import argparse
import csv
import html
import random
from pathlib import Path

import yaml

from rewind.eval.splits import SPLITS
from rewind.paths import display_path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUERIES_DIR = REPO_ROOT / "eval" / "queries"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    parser.add_argument("--out", type=Path, default=QUERIES_DIR / "review.html")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())["queries"]

    rows = [r for split in SPLITS for r in csv.DictReader(open(QUERIES_DIR / f"worksheet_{split}.csv", newline=""))]
    blank = [r["query_id"] for r in rows if not r["query"].strip()]
    if blank:
        raise SystemExit(f"{len(blank)} rows have no query yet: {blank}")
    sample = random.Random(cfg["seed"]).sample(rows, cfg["review_n"])
    cards = []
    for r in sample:
        img = Path(r["keyframe"]).relative_to("eval/queries")
        meta = (f"{r['query_id']} · {r['split']} · {r['kind']} · {r['qtype']} · {r['specificity']} · "
                f"{r['modality'].upper()} · label: {r['activity_type']} · {r['camera_id']} · "
                f"{float(r['start_s']):.1f}-{float(r['end_s']):.1f} s in {r['clip']}")
        cards.append(f'<section><p class="q">“{html.escape(r["query"])}”</p><p class="m">{html.escape(meta)}</p>'
                     f'<img src="{html.escape(str(img))}" loading="lazy" alt="keyframe sheet {r["query_id"]}"></section>')
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Query review (step 2.2)</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{{font-family:system-ui,sans-serif;max-width:1300px;margin:0 auto;padding:16px;background:#fafafa;color:#111}}
section{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:0 0 24px}}
.q{{font-size:1.3em;margin:0 0 4px}} .m{{color:#555;font-size:.85em;margin:0 0 8px}} img{{width:100%;height:auto}}
</style></head><body><h1>Step 2.2 — {len(sample)} random queries (seed {cfg['seed']}) of {len(rows)}</h1>
<p>Sheet: middle frame with label boxes (red = main actor) on top; first frame, last frame and a zoom below.
Check: is it the right activity, could a witness have seen these details, is the specificity level right?</p>
{''.join(cards)}</body></html>"""
    args.out.write_text(page)
    print(f"wrote {display_path(args.out)}: open it in a browser (file://{args.out.resolve()})")


if __name__ == "__main__":
    main()
