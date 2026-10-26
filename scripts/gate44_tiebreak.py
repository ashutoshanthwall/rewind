"""GATE 4.4 tie-break reading (decided by Ashutosh, 26 Oct; PROGRESS.md). Dev split only.

The pre-registered order of scripts/appearance_search.py --ablation (crop_source none -> motion ->
detection_then_motion, then mean_center, then frame_mode), with one change: when the two query families disagree
(specific seconds watched better, category AP worse, or the reverse), the paired rule's "not worse in either family"
is read as "the rule does not decide", and the family of the headline metric (seconds watched, CLAUDE.md 1.10)
decides; wins > losses is still required. Prints each step and the resulting variant.

Run from the repo root (~20 s): uv run python -m scripts.gate44_tiebreak
"""

import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rewind.embed.clip_model import Clip
from rewind.eval.gate import paired_decision
from rewind.ingest.manifest import connect
from rewind.paths import REPO_ROOT
from rewind.search.core import SearchConfig
from scripts.appearance_search import GRID, Index, query_rows, split_videos, vname
from scripts.collect_appearance import index_dir, load_means

ART = REPO_ROOT / "artifacts"


def tiebreak_decision(dec: dict, ap_key: str) -> tuple[bool, bool]:
    """(families conflict, adopt) for one paired_decision result under the tie-break reading."""
    f = dec["families"]
    sp, ca = f["specific_seconds"]["mean_diff"], f["category_" + ap_key]["mean_diff"]
    conflict = sp != 0 and ca != 0 and (sp > 0) != (ca > 0)
    return conflict, (sp > 0 and dec["wins"] > dec["losses"]) if conflict else dec["adopt"]


def main(config: Path = REPO_ROOT / "configs" / "config.yaml") -> None:
    cfg = yaml.safe_load(config.read_text())
    a, m = cfg["appearance"], cfg["metrics"]
    ap_key = f"AP@{max(m['category_k'])}"
    scfg = SearchConfig.from_config(cfg)
    conn = connect(REPO_ROOT / "data" / "rewind.sqlite")
    videos = split_videos(conn, "dev")
    d = index_dir(ART, a["model"], a["pretrained"])
    vecs, meta = np.load(d / "vectors.npy"), pd.read_parquet(d / "meta.parquet")
    clip = Clip(a["model"], a["pretrained"])
    results = {}
    for fm, cs, c in GRID:
        means = load_means(ART, a["model"], a["pretrained"], fm, cs) if c else None
        index = Index(vecs, meta, videos, fm, cs, a["max_crops_per_frame"], c, means)
        results[(fm, cs, c)] = query_rows(conn, "dev", index, clip, cfg, scfg)

    def step(simple: tuple, complex_: tuple) -> bool:
        dec = paired_decision(results[simple], results[complex_], ap_key, m["bootstrap_n"], m["bootstrap_seed"],
                              m["ci_level"])
        conflict, adopt = tiebreak_decision(dec, ap_key)
        f = dec["families"]
        print(f"{vname(complex_)} vs {vname(simple)}: W/T/L {dec['wins']}/{dec['ties']}/{dec['losses']}; "
              f"specific {f['specific_seconds']['mean_diff']:+.0f} s; category {f['category_' + ap_key]['mean_diff']:+.3f}; "
              f"literal rule adopt={dec['adopt']}; conflict={conflict}; tie-break adopt={adopt}")
        return adopt

    # Starting point as pre-registered: the config's frame_mode and mean_center BEFORE the gate (tiles2, centered).
    fm0, c0, cs = "tiles2", True, "none"
    for nxt in ("motion", "detection_then_motion"):
        if step((fm0, cs, c0), (fm0, nxt, c0)):
            cs = nxt
    c = step((fm0, cs, False), (fm0, cs, True))
    fm = "tiles2" if step(("pad", cs, c), ("tiles2", cs, c)) else "pad"
    final = (fm, cs, c)
    spec = [r for r in results[final] if r["kind"] == "specific"]
    print(f"FINAL: {vname(final)}: median s {statistics.median(r['seconds'] for r in spec):.0f}, "
          f"hits {sum(r['hit'] for r in spec)}/{len(spec)}, R@10 {statistics.mean(r['R@10'] for r in spec):.3f}, "
          f"category mAP@50 {statistics.mean(r[ap_key] for r in results[final] if r['kind'] == 'category'):.3f}")


if __name__ == "__main__":
    main()
