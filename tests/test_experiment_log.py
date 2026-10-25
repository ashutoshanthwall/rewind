import csv
import json
import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from rewind.experiment_log import COLUMNS, log_experiment

CONFIG_BYTES = b"search:\n  k_per_index: 200\n"
# Computed outside this code with: printf 'search:\n  k_per_index: 200\n' | sha256sum
CONFIG_SHA256 = "856c2b423a24a029f607c17d7a5eae9dceea27eaf74fbfa5ff393eebc90e3b77"


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.com",
         "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A throwaway git repo with one committed config file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "config.yaml").write_bytes(CONFIG_BYTES)
    git(repo, "add", "config.yaml")
    git(repo, "commit", "-q", "-m", "init")
    return repo


def log(repo, **kwargs):
    kwargs.setdefault("split", "dev")
    kwargs.setdefault("metrics", {"recall@5": 0.4, "seconds_watched": 812.5})
    return log_experiment(
        csv_path=repo / "results" / "experiments.csv", config_path=repo / "config.yaml",
        repo_dir=repo, **kwargs
    )


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.reader(f))


def test_writes_one_correct_row(repo):
    notes = 'dummy run, with "quotes"\nand a second line'
    before = datetime.now(timezone.utc).replace(microsecond=0)
    log(repo, notes=notes)
    after = datetime.now(timezone.utc)

    rows = read_rows(repo / "results" / "experiments.csv")
    assert rows[0] == COLUMNS
    assert len(rows) == 2
    row = dict(zip(COLUMNS, rows[1]))

    assert row["git_hash"] == git(repo, "rev-parse", "HEAD")
    assert row["config_hash"] == CONFIG_SHA256
    assert row["split"] == "dev"
    assert json.loads(row["metrics"]) == {"recall@5": 0.4, "seconds_watched": 812.5}
    assert row["notes"] == notes
    ts = datetime.fromisoformat(row["timestamp"])
    assert ts.utcoffset() == timedelta(0)
    assert before <= ts <= after


def test_second_call_appends_without_repeating_header(repo):
    log(repo, split="tuning", notes="first")
    log(repo, split="dev", notes="second")
    rows = read_rows(repo / "results" / "experiments.csv")
    assert len(rows) == 3
    assert rows.count(COLUMNS) == 1
    assert [r[COLUMNS.index("notes")] for r in rows[1:]] == ["first", "second"]


def test_uncommitted_change_marks_hash_dirty(repo):
    head = git(repo, "rev-parse", "HEAD")
    (repo / "config.yaml").write_bytes(CONFIG_BYTES + b"  ef_search: 400\n")
    row = log(repo)
    assert row["git_hash"] == head + "-dirty"
    assert row["config_hash"] != CONFIG_SHA256


def test_untracked_code_file_marks_hash_dirty(repo):
    # A new, never-committed script that produced the result must not be invisible.
    (repo / "new_metric.py").write_text("x = 1\n")
    assert log(repo)["git_hash"].endswith("-dirty")


def test_logging_does_not_dirty_the_next_row(repo):
    # Commit the log, as the real repo does; appending to it (or writing reports under
    # results/) is output, not a code change, so the next row must stay clean.
    log(repo, notes="first")
    git(repo, "add", "results/experiments.csv")
    git(repo, "commit", "-q", "-m", "log")
    (repo / "results" / "report.md").write_text("numbers\n")
    log(repo, notes="second")
    row = log(repo, notes="third")
    assert row["git_hash"] == git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize("split", ["train", "Test", ""])
def test_rejects_invalid_split(repo, split):
    with pytest.raises(ValueError, match="split"):
        log(repo, split=split)


@pytest.mark.parametrize("metrics", [{}, None, [("recall@5", 0.4)]])
def test_rejects_empty_or_non_dict_metrics(repo, metrics):
    with pytest.raises(ValueError, match="metrics"):
        log(repo, metrics=metrics)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None, "0.4", True, {"rgb": 0.4}])
def test_rejects_non_finite_or_non_numeric_metric_values(repo, bad):
    # None would be logged as JSON null and "0.4" as a string: both hide a broken metric.
    with pytest.raises(ValueError, match="recall@5"):
        log(repo, metrics={"recall@5": bad})
    assert not (repo / "results" / "experiments.csv").exists()


def test_rejects_non_string_metric_keys(repo):
    # json.dumps would silently turn 5 into "5".
    with pytest.raises(ValueError, match="key"):
        log(repo, metrics={5: 0.4})


def test_rejects_non_string_notes(repo):
    # csv.DictWriter would silently write None as an empty cell.
    with pytest.raises(TypeError, match="notes"):
        log(repo, notes=None)


def test_missing_config_raises_and_writes_nothing(repo):
    with pytest.raises(FileNotFoundError):
        log_experiment("dev", {"m": 1}, csv_path=repo / "e.csv",
                       config_path=repo / "missing.yaml", repo_dir=repo)
    assert not (repo / "e.csv").exists()


def test_not_a_git_repo_raises_and_writes_nothing(tmp_path):
    (tmp_path / "config.yaml").write_bytes(CONFIG_BYTES)
    with pytest.raises(RuntimeError, match="git"):
        log_experiment("dev", {"m": 1}, csv_path=tmp_path / "e.csv",
                       config_path=tmp_path / "config.yaml", repo_dir=tmp_path)
    assert not (tmp_path / "e.csv").exists()


def test_mismatched_header_raises_and_leaves_file_untouched(repo):
    path = repo / "results" / "experiments.csv"
    path.parent.mkdir()
    path.write_text("timestamp,split,metrics\n")
    with pytest.raises(ValueError, match="header"):
        log(repo)
    assert path.read_text() == "timestamp,split,metrics\n"


def test_display_path_inside_and_outside_repo(tmp_path):
    # inside the repo -> relative; outside (e.g. a scratch dir) -> as given, no ValueError (was a crash, 25 Oct)
    from rewind.paths import REPO_ROOT, display_path
    assert display_path(REPO_ROOT / "results" / "reports" / "x.md") == "results/reports/x.md"
    assert display_path(tmp_path / "x.md") == str(tmp_path / "x.md")
