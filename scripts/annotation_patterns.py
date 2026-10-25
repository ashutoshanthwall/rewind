"""Print git sparse-checkout patterns for the Kitware box files (<clip>.geom.yml) of the selected clips (step 1.3).

The annotation checkout (docs/meva_notes.md) fetches activities and types for all clips but boxes (~1 GB for our
selection, far more for all clips) only for the clips in data/selection.txt. Run from the repo root:
  uv run python -m scripts.annotation_patterns >> data/annotations/meva-data-repo/.git/info/sparse-checkout
then `git -C data/annotations/meva-data-repo sparse-checkout reapply`. Contrib (IR) files, .json included, are
already covered by the recipe's contrib/**/*G47[4-9]* pattern.
"""

import argparse
from pathlib import Path

from rewind.ingest.manifest import VIDEO_SUFFIX
from scripts.download import parse_selection

REPO_ROOT = Path(__file__).resolve().parent.parent


def geom_patterns(selection: Path) -> list[str]:
    clips = [Path(key).name.removesuffix(VIDEO_SUFFIX) for key, _ in parse_selection(selection)]
    return [f"/annotation/DIVA-phase-2/MEVA/kitware*/**/{clip}.geom.yml" for clip in clips]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", type=Path, default=REPO_ROOT / "data" / "selection.txt")
    print("\n".join(geom_patterns(parser.parse_args().selection)))


if __name__ == "__main__":
    main()
