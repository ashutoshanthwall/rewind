import hashlib

import pytest

from scripts.download import download_one, etag_of, make_read_only, parse_selection, plan, stray_files


def write_selection(tmp_path, body):
    path = tmp_path / "selection.txt"
    path.write_text("# header comment\n# Format: s3_key<TAB>size_bytes\n" + body)
    return path


def test_parse_selection_reads_keys_and_sizes(tmp_path):
    sel = write_selection(tmp_path, "drops/a/x.avi\t10\ndrops/b/y.avi\t3\n")
    assert parse_selection(sel) == [("drops/a/x.avi", 10), ("drops/b/y.avi", 3)]


@pytest.mark.parametrize("body, match", [
    ("drops/a/x.avi 10\n", "expected"),                     # space instead of tab
    ("drops/a/x.avi\t-5\n", "expected"),                    # negative size
    ("drops/a/x.avi\t0\n", "expected"),                     # zero size
    ("drops/a/x.avi\t10\ndrops/b/x.avi\t11\n", "Duplicate"),  # same file name in two folders
    ("", "no files"),
])
def test_parse_selection_rejects_bad_input(tmp_path, body, match):
    with pytest.raises(ValueError, match=match):
        parse_selection(write_selection(tmp_path, body))


def test_plan_skips_complete_files_and_ignores_partials(tmp_path):
    (tmp_path / "done.avi").write_bytes(b"x" * 10)        # complete
    (tmp_path / "half.avi.part").write_bytes(b"x" * 4)    # interrupted earlier: must be redone
    entries = [("k/done.avi", 10), ("k/half.avi", 10), ("k/new.avi", 7)]
    assert plan(entries, tmp_path) == [("k/half.avi", 10), ("k/new.avi", 7)]


def test_plan_refuses_final_file_with_wrong_size(tmp_path):
    (tmp_path / "bad.avi").write_bytes(b"x" * 9)
    with pytest.raises(RuntimeError, match="9 bytes, selection says 10"):
        plan([("k/bad.avi", 10)], tmp_path)
    assert (tmp_path / "bad.avi").stat().st_size == 9      # left untouched


def test_make_read_only(tmp_path):
    dest = tmp_path / "raw"
    dest.mkdir()
    (dest / "a.avi").write_bytes(b"x")
    make_read_only(dest, [("k/a.avi", 1)])
    assert (dest / "a.avi").stat().st_mode & 0o777 == 0o444
    assert dest.stat().st_mode & 0o777 == 0o555
    dest.chmod(0o755)  # let pytest clean up tmp_path


# --- content check (S3 ETag) ---------------------------------------------------------------------------

def test_etag_single_part_is_plain_md5(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    # RFC 1321 test vector: MD5("abc")
    assert etag_of(f, "900150983cd24fb0d6963f7d28e17f72") == "900150983cd24fb0d6963f7d28e17f72"


def test_etag_multipart_is_md5_of_part_md5s(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"abcdefghij")
    # part_size 4 -> parts "abcd", "efgh", "ij" (last part short); ETag = MD5(concat of the 3 binary digests)-3
    expected = hashlib.md5(b"".join(hashlib.md5(p).digest() for p in (b"abcd", b"efgh", b"ij"))).hexdigest() + "-3"
    assert etag_of(f, "whatever-3", part_size=4) == expected
    assert etag_of(f, "whatever-3", part_size=4) != hashlib.md5(b"abcdefghij").hexdigest() + "-3"


def test_etag_refuses_part_count_that_does_not_fit(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"abcdefghij")
    with pytest.raises(ValueError, match="part size is unknown"):
        etag_of(f, "whatever-2", part_size=4)   # 10 bytes / 4 -> 3 parts, not 2


def fake_copy(content):
    def copy(key, out):
        out.write_bytes(content)
    return copy


def test_download_one_renames_only_after_checks(tmp_path):
    good = hashlib.md5(b"abc").hexdigest()
    final = download_one("k/x.avi", 3, good, tmp_path, copy=fake_copy(b"abc"))
    assert final == tmp_path / "x.avi" and final.read_bytes() == b"abc"
    assert not (tmp_path / "x.avi.part").exists()


@pytest.mark.parametrize("content, match", [
    (b"abd", "Content mismatch"),   # same size, different bytes: only the ETag catches it
    (b"ab", "Size mismatch"),
])
def test_download_one_keeps_bad_download_as_part(tmp_path, content, match):
    good = hashlib.md5(b"abc").hexdigest()
    with pytest.raises(RuntimeError, match=match):
        download_one("k/x.avi", 3, good, tmp_path, copy=fake_copy(content))
    assert not (tmp_path / "x.avi").exists()
    assert (tmp_path / "x.avi.part").read_bytes() == content


def test_stray_files(tmp_path):
    (tmp_path / "a.avi").write_bytes(b"x")
    (tmp_path / "old.avi.part").write_bytes(b"x")
    assert stray_files(tmp_path, [("k/a.avi", 1)]) == ["old.avi.part"]
