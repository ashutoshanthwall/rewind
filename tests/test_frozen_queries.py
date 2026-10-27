"""Gate 2.2: the frozen query files never change, and their content keeps the protocol's shape.

Fails if any file listed in eval/queries/HASHES is edited, removed or missing from HASHES.
"""

import csv
import re
from collections import Counter
from pathlib import Path

import pytest

from rewind.eval.queries import FROZEN_COLUMNS, KINDS, QTYPES, check_frozen, load_queries, read_hashes, sha256
from rewind.eval.splits import SPLITS

QUERIES_DIR = Path(__file__).resolve().parent.parent / "eval" / "queries"


def test_frozen_files_match_hashes():
    hashes = read_hashes(QUERIES_DIR)
    expected = {f"queries_{s}.csv" for s in SPLITS} | {f"worksheet_{s}.csv" for s in SPLITS} | {"replacements.csv"}
    assert set(hashes) == expected
    for name, digest in hashes.items():
        assert sha256(QUERIES_DIR / name) == digest, f"{name} changed after Gate 2.2"


def test_no_unhashed_csv_in_queries_dir():
    hashed = set(read_hashes(QUERIES_DIR))
    assert {p.name for p in QUERIES_DIR.glob("*.csv")} <= hashed


def test_check_frozen_detects_an_edit(tmp_path):
    (tmp_path / "queries_dev.csv").write_text("a\n")
    (tmp_path / "HASHES").write_text(f"{sha256(tmp_path / 'queries_dev.csv')}  queries_dev.csv\n")
    check_frozen(tmp_path)
    (tmp_path / "queries_dev.csv").write_text("b\n")
    with pytest.raises(ValueError, match="changed"):
        check_frozen(tmp_path)
    (tmp_path / "queries_dev.csv").unlink()
    with pytest.raises(FileNotFoundError):
        check_frozen(tmp_path)


def test_frozen_content_shape():
    rows = []
    for split in SPLITS:
        with open(QUERIES_DIR / f"queries_{split}.csv", newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == FROZEN_COLUMNS
            rows += list(reader)
    assert len({r["query_id"] for r in rows}) == len(rows) == 60
    assert all(r["query"].strip() and r["kind"] in KINDS and r["qtype"] in QTYPES for r in rows)
    assert Counter(r["kind"] for r in rows) == Counter({"category": 18, "specific": 42})
    # category ground truth = all activities of the type in the split: no single event attached
    assert all(not r["event_id"] and not r["clip"] for r in rows if r["kind"] == "category")
    assert all(r["event_id"] and r["clip"] and r["src_ids"] for r in rows if r["kind"] == "specific")


def test_load_queries_guards_test_split():
    assert len(load_queries(QUERIES_DIR, "dev")) == 17
    with pytest.raises(PermissionError):
        load_queries(QUERIES_DIR, "test")


# Code that may see every split, test included: the tools that build the splits and the queries (steps 2.1-2.2).
# Step 8.2 adds its own script here, explicitly. Any other file must go through split_video_ids / split_event_ids /
# load_queries without allow_test.
TEST_ACCESS_ALLOWED = {"rewind/eval/splits.py", "rewind/eval/queries.py", "scripts/make_splits.py",
                       "scripts/make_worksheets.py", "scripts/freeze_queries.py", "scripts/review_queries.py",
                       "scripts/test_run.py"}   # step 8.2: the single test run
TEST_ACCESS = re.compile(
    r"allow_test=(?!False)"                                  # passing allow_test (anything but False)
    r"|queries_\{|queries_test|worksheet_\{|worksheet_test"  # opening query / worksheet files directly
    r"|\bevent_splits\b|\bvideo_blocks\b|\b(?:FROM|JOIN) blocks\b")  # raw SQL on the split tables


def test_no_unguarded_test_split_access_outside_allowed_files():
    root = QUERIES_DIR.parent.parent
    offenders = {}
    for path in sorted([*root.glob("rewind/**/*.py"), *root.glob("scripts/**/*.py")]):
        rel = str(path.relative_to(root))
        hits = [m.group(0) for m in TEST_ACCESS.finditer(path.read_text())]
        if hits and rel not in TEST_ACCESS_ALLOWED:
            offenders[rel] = sorted(set(hits))
    assert offenders == {}, f"test-split access outside the allowed files: {offenders}"


def test_test_access_pattern_catches_bypasses():
    for code in ['load_queries(d, "test", allow_test=True)', 'open(d / "queries_test.csv")',
                 'open(d / f"worksheet_{split}.csv")', 'conn.execute("SELECT * FROM event_splits")',
                 'conn.execute("SELECT split FROM blocks")']:
        assert TEST_ACCESS.search(code), code
    for code in ["def f(split, allow_test: bool = False):", 'load_queries(d, "dev")', "split_video_ids(conn, 'dev')"]:
        assert not TEST_ACCESS.search(code), code
