"""Append one row per experiment to results/experiments.csv (CLAUDE.md rule 1.1.11).

Each row ties a result to the exact code (git hash) and settings (config hash) that produced it.
Every problem raises: a log row that can't be traced back is worse than no row.
"""

import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "results" / "experiments.csv"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"

COLUMNS = ["timestamp", "git_hash", "config_hash", "split", "metrics", "notes"]

# Fixed by the evaluation protocol (CLAUDE.md 1.10), not a tunable.
# "none" is for runs that don't use a split (throughput, scaling benchmarks).
VALID_SPLITS = ("tuning", "dev", "test", "none")

# Experiment outputs (this log, reports, figures). Changes here don't make the code dirty;
# otherwise every row after the first would be marked dirty by the previous row's append.
OUTPUT_DIR = "results"


def git_hash(repo_dir: Path) -> str:
    """HEAD commit hash, with '-dirty' appended if the working tree differs from it.

    Counts modified tracked files and new untracked (non-ignored) files, outside OUTPUT_DIR.
    """
    head = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "--verify", "HEAD"],
        capture_output=True, text=True,
    )
    if head.returncode != 0:
        raise RuntimeError(
            f"Cannot read git HEAD in {repo_dir} (not a git repo, or no commits yet): "
            f"{head.stderr.strip()}"
        )
    status = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain",
         "--", ":(top)", f":(top,exclude){OUTPUT_DIR}"],
        capture_output=True, text=True,
    )
    if status.returncode != 0:
        raise RuntimeError(f"'git status' failed in {repo_dir}: {status.stderr.strip()}")
    suffix = "-dirty" if status.stdout.strip() else ""
    return head.stdout.strip() + suffix


def config_hash(config_path: Path) -> str:
    """SHA-256 of the config file's raw bytes (comments included)."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def run_stamp(config_path: Path = DEFAULT_CONFIG, repo_dir: Path = REPO_ROOT) -> dict:
    """Git hash and config hash at the START of a run.

    Long jobs take this before they begin and pass it to log_experiment, so the row names the code and settings
    that ran, not whatever is committed when the job ends (a commit during a run was logged as the run's code
    twice, 25 Oct)."""
    return {"git_hash": git_hash(Path(repo_dir)), "config_hash": config_hash(Path(config_path))}


def log_experiment(
    split: str,
    metrics: dict,
    notes: str = "",
    *,
    csv_path: Path = DEFAULT_CSV,
    config_path: Path = DEFAULT_CONFIG,
    repo_dir: Path = REPO_ROOT,
    stamp: dict | None = None,
) -> dict:
    """Append one experiment row and return it as a dict. `stamp` (from run_stamp at the start of a long run)
    replaces the git and config hashes read now."""
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError(f"metrics must be a non-empty dict, got {metrics!r}")
    for name, value in metrics.items():
        if not isinstance(name, str):
            raise ValueError(f"metric key must be a string, got {name!r}")
        # A None, string, bool or NaN metric means something upstream broke; refuse to log it.
        # Nested breakdowns (per-qtype, per-query) belong in results/reports/, not in this log.
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"metric {name!r} must be a finite number, got {value!r}")
    if not isinstance(notes, str):
        raise TypeError(f"notes must be a string, got {notes!r}")
    metrics_json = json.dumps(metrics, sort_keys=True, allow_nan=False)
    if stamp is None:
        stamp = run_stamp(config_path, repo_dir)
    elif set(stamp) != {"git_hash", "config_hash"}:
        raise ValueError(f"stamp must come from run_stamp, got keys {sorted(stamp)}")

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_hash": stamp["git_hash"],
        "config_hash": stamp["config_hash"],
        "split": split,
        "metrics": metrics_json,
        "notes": notes,
    }

    csv_path = Path(csv_path)
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    if not needs_header:
        with csv_path.open(newline="") as f:
            existing_header = next(csv.reader(f))
        if existing_header != COLUMNS:
            raise ValueError(
                f"{csv_path} has header {existing_header}, expected {COLUMNS}. "
                "Refusing to append to a log with a different layout."
            )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
    return row
