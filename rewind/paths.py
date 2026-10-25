"""Repository paths shared by the scripts."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def display_path(path: Path) -> str:
    """Path for messages: relative to the repo root when inside it, else as given (e.g. a scratch output path).

    Path.relative_to raises for a path outside the repo, which used to crash scripts at their final print, after
    their outputs (and experiments.csv rows) had already been written."""
    resolved = Path(path).resolve()
    return str(resolved.relative_to(REPO_ROOT)) if resolved.is_relative_to(REPO_ROOT) else str(path)
