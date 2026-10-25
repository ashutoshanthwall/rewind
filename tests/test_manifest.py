from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pytest

from rewind.ingest.manifest import (Video, connect, frame_to_seconds, load_videos, parse_video_filename, probe_video,
                                    seconds_to_frame, sha256_file)
from scripts.build_manifest import (annotation_index, build, camera_modality, check_name_vs_container,
                                    selected_files, sessions, validate_names, verify)

# --- filename parsing -----------------------------------------------------------------------------


def test_regular_name():
    v = parse_video_filename("2018-03-07.10-55-06.11-00-05.school.G336.r13.avi")
    assert v.clip == "2018-03-07.10-55-06.11-00-05.school.G336"
    assert (v.camera_id, v.site) == ("G336", "school")
    assert v.start_time == datetime(2018, 3, 7, 10, 55, 6)
    assert v.name_end_time == datetime(2018, 3, 7, 11, 0, 5)  # 299 s by name; the container says 300.0 s


def test_truncated_end_time_in_real_name():
    # Real selected file: container 300.333 s, but the name ends at 13:54:59 (299 s).
    v = parse_video_filename("2018-03-11.13-50-00.13-54-59.school.G330.r13.avi")
    assert (v.name_end_time - v.start_time).total_seconds() == 299


def test_name_crossing_midnight_ends_next_day():
    v = parse_video_filename("2018-03-11.23-55-09.00-00-09.bus.G340.r13.avi")
    assert v.start_time == datetime(2018, 3, 11, 23, 55, 9)
    assert v.name_end_time == datetime(2018, 3, 12, 0, 0, 9)  # 300 s later, next day


@pytest.mark.parametrize("name", [
    "2018-03-07.10-55-06.11-00-05.school.G336",           # no suffix
    "2018-03-07.10-55-06.11-00-05.school.G336.r12.avi",   # other release
    "2018-03-07.10-55-06.school.G336.r13.avi",            # missing end time
    "2018-03-07.10-55-06.11-00-05.school.336.r13.avi",    # camera without G
    "2018-03-07.10-55-06.11-00-05.School.G336.r13.avi",   # upper-case site
    "x2018-03-07.10-55-06.11-00-05.school.G336.r13.avi",  # junk before the date
])
def test_malformed_names_raise(name):
    with pytest.raises(ValueError, match="Not a MEVA video name"):
        parse_video_filename(name)


@pytest.mark.parametrize("name", [
    "2018-02-30.10-55-06.11-00-05.school.G336.r13.avi",  # 30 February
    "2018-03-07.25-55-06.11-00-05.school.G336.r13.avi",  # hour 25
    "2018-03-07.10-60-06.11-00-05.school.G336.r13.avi",  # minute 60
])
def test_impossible_dates_and_times_raise(name):
    with pytest.raises(ValueError):
        parse_video_filename(name)


# --- frame <-> seconds ----------------------------------------------------------------------------


@pytest.mark.parametrize("frame, fps, seconds", [
    (0, 30.0, 0.0),
    (30, 30.0, 1.0),
    (9000, 30.0, 300.0),   # one 5-min clip
    (45, 15.0, 3.0),
])
def test_frame_to_seconds(frame, fps, seconds):
    assert frame_to_seconds(frame, fps) == seconds


@pytest.mark.parametrize("seconds, fps, frame", [
    (0.0, 30.0, 0),
    (0.0333, 30.0, 0),      # 0.999 frames: still frame 0
    (1 / 30, 30.0, 1),      # exactly the start of frame 1
    (0.7, 30.0, 21),        # 0.7 * 30 = 21.000000000000004 in floating point -> 21
    (299.999, 30.0, 8999),  # 8999.97 -> last frame of a 9000-frame clip
    (2.0, 15.0, 30),
])
def test_seconds_to_frame(seconds, fps, frame):
    assert seconds_to_frame(seconds, fps) == frame


def test_negative_inputs_raise():
    with pytest.raises(ValueError):
        frame_to_seconds(-1, 30.0)
    with pytest.raises(ValueError):
        seconds_to_frame(-0.001, 30.0)


# --- Video: frame <-> wall-clock, bounded ---------------------------------------------------------


def make_video(start=datetime(2018, 3, 7, 10, 55, 6), n_frames=9000, fps=30.0):
    return Video(1, "clip", "data/raw/clip.r13.avi", "G336", "rgb", start, n_frames, fps)


def test_frame_time_hand_computed():
    v = make_video()
    assert v.frame_time(0) == datetime(2018, 3, 7, 10, 55, 6)
    assert v.frame_time(300) == datetime(2018, 3, 7, 10, 55, 16)               # 300 / 30 = 10 s
    assert v.frame_time(8999) == datetime(2018, 3, 7, 11, 0, 5, 966667)        # 8999 / 30 = 299.966667 s
    assert v.end_time == datetime(2018, 3, 7, 11, 0, 6)                        # 9000 / 30 = 300 s, exclusive


def test_frame_at_hand_computed():
    v = make_video()
    assert v.frame_at(datetime(2018, 3, 7, 10, 55, 6)) == 0
    assert v.frame_at(datetime(2018, 3, 7, 10, 55, 16)) == 300
    assert v.frame_at(datetime(2018, 3, 7, 10, 55, 16, 30000)) == 300   # 10.03 s = 300.9 frames -> 300
    assert v.frame_at(datetime(2018, 3, 7, 11, 0, 5, 999000)) == 8999   # 299.999 s = 8999.97 frames


@pytest.mark.parametrize("frame", [-1, 9000])
def test_frame_time_out_of_range_raises(frame):
    with pytest.raises(ValueError, match="outside"):
        make_video().frame_time(frame)


@pytest.mark.parametrize("t", [
    datetime(2018, 3, 7, 10, 55, 5, 999999),  # 1 us before frame 0
    datetime(2018, 3, 7, 11, 0, 6),           # end_time is exclusive
])
def test_frame_at_out_of_range_raises(t):
    with pytest.raises(ValueError, match="outside"):
        make_video().frame_at(t)


def test_frame_round_trip_every_frame():
    # datetime stores 1/30 s as 33333 us; without snapping, frame_at(frame_time(1)) would floor 0.99999 to 0.
    v = make_video()
    assert all(v.frame_at(v.frame_time(f)) == f for f in range(v.n_frames))


def test_wall_clock_across_midnight():
    v = make_video(start=datetime(2018, 3, 11, 23, 59, 50))
    assert v.frame_time(600) == datetime(2018, 3, 12, 0, 0, 10)  # 600 / 30 = 20 s
    assert v.frame_at(datetime(2018, 3, 12, 0, 0, 10)) == 600


# --- container and hashing ------------------------------------------------------------------------


def write_avi(path, n_frames, fps=30, size=(64, 48)):
    with av.open(str(path), "w", format="avi") as out:
        stream = out.add_stream("mpeg4", rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            frame = av.VideoFrame.from_ndarray(np.full((size[1], size[0], 3), i * 8 % 256, np.uint8), format="rgb24")
            out.mux(stream.encode(frame))
        out.mux(stream.encode())


def test_probe_reads_header(tmp_path):
    path = tmp_path / "toy.avi"
    write_avi(path, n_frames=15, fps=30)
    info = probe_video(path)
    assert info == {"n_frames": 15, "fps": 30.0, "duration_s": 0.5, "width": 64, "height": 48, "codec": "mpeg4"}


def test_probe_rejects_two_streams(tmp_path):
    path = tmp_path / "two.avi"
    with av.open(str(path), "w", format="avi") as out:
        streams = [out.add_stream("mpeg4", rate=Fraction(30)) for _ in range(2)]
        for s in streams:
            s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), np.uint8), format="rgb24")
        for s in streams:
            out.mux(s.encode(frame))
            out.mux(s.encode())
    with pytest.raises(ValueError, match="exactly one"):
        probe_video(path)


def test_sha256_known_vector(tmp_path):
    path = tmp_path / "abc"
    path.write_bytes(b"abc")
    # FIPS 180-2 test vector for "abc".
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


# --- cross-checks ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name_s, container_s, ok", [
    (300, 300.333, True),
    (300, 299.233, True),   # G474 clips run up to 0.77 s short
    (299, 300.0, True),
    (300, 301.5, False),
    (300, 298.9, False),
])
def test_name_vs_container(name_s, container_s, ok):
    if ok:
        check_name_vs_container("f", name_s, container_s, max_diff_s=1.0)
    else:
        with pytest.raises(ValueError, match="container duration"):
            check_name_vs_container("f", name_s, container_s, max_diff_s=1.0)


def write_clip_table(path, lines):
    path.write_text("\n".join(lines) + "\n")


def test_camera_modality(tmp_path):
    table = tmp_path / "table.txt"
    write_clip_table(table, [
        "2018-03-07.10-55-00.11-00-00.school.G474 2018-03-07.10-55-00 no-camera-model IR no-reference-clip 0 -1",
        "2018-03-07.10-55-06.11-00-05.school.G336 2018-03-07.10-55-00 x.krtd 3-336 self 0 0",
        "2018-03-07.11-00-06.11-05-05.school.G336 2018-03-07.11-00-00 x.krtd 3-336 self 0 0",
    ])
    assert camera_modality(table, {"G474", "G336"}) == {"G336": "rgb", "G474": "ir"}


def test_camera_modality_mixed_or_unknown_raises(tmp_path):
    table = tmp_path / "table.txt"
    write_clip_table(table, [
        "2018-03-07.10-55-00.11-00-00.school.G474 2018-03-07.10-55-00 no-camera-model IR no-reference-clip 0 -1",
        "2018-03-07.11-00-00.11-05-00.school.G474 2018-03-07.11-00-00 x.krtd 3-336 self 0 0",
    ])
    with pytest.raises(ValueError, match="both IR and non-IR"):
        camera_modality(table, {"G474"})
    with pytest.raises(ValueError, match="no clip"):
        camera_modality(table, {"G999"})


def test_camera_modality_malformed_line_raises(tmp_path):
    table = tmp_path / "table.txt"
    write_clip_table(table, ["2018-03-07.10-55-00.11-00-00.school.G474 IR"])
    with pytest.raises(ValueError, match="7 fields"):
        camera_modality(table, {"G474"})


def test_selected_files_must_match_selection(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.r13.avi").write_bytes(b"x")
    sel = tmp_path / "selection.txt"
    sel.write_text("drops/a.r13.avi\t1\ndrops/b.r13.avi\t1\n")
    with pytest.raises(ValueError, match="missing"):
        selected_files(raw, sel)
    (raw / "b.r13.avi").write_bytes(b"x")
    assert [p.name for p in selected_files(raw, sel)] == ["a.r13.avi", "b.r13.avi"]


def test_sessions_counts_breaks_and_overlaps():
    # Clips start at 0, 300 and 900 s, each 300.2 s long.
    # gaps: 300 - 300.2 = -0.2 (overlap, same session); 900 - 600.2 = 299.8 (new session).
    t0 = datetime(2018, 3, 7, 10, 0, 0)
    df = pd.DataFrame({"start": [t0 + timedelta(seconds=s) for s in (900, 0, 300)], "duration_s": [300.2] * 3})
    n, gaps = sessions(df, max_gap_s=1.0)
    assert n == 2
    assert gaps == pytest.approx([-0.2])


# --- database -------------------------------------------------------------------------------------


ROW = ("c", "data/raw/c.r13.avi", 10, "ab", "G336", "school", "rgb", "2018-03-07T10:55:06", "2018-03-07T11:00:05",
       300.0, 9000, 30.0, 1920, 1080, "h264", "2026-10-25T00:00:00+00:00")
INSERT = ("INSERT INTO videos (clip, path, size_bytes, sha256, camera_id, site, modality, start_time, name_end_time,"
          " duration_s, n_frames, fps, width, height, codec, ingested_at) VALUES (" + ", ".join("?" * 16) + ")")


def test_db_round_trip(tmp_path):
    conn = connect(tmp_path / "m.sqlite")
    conn.execute(INSERT, ROW)
    (v,) = load_videos(conn)
    assert (v.clip, v.camera_id, v.modality, v.n_frames, v.fps) == ("c", "G336", "rgb", 9000, 30.0)
    assert v.start_time == datetime(2018, 3, 7, 10, 55, 6)


def test_db_rejects_unknown_modality_and_duplicate_clip(tmp_path):
    import sqlite3
    conn = connect(tmp_path / "m.sqlite")
    conn.execute(INSERT, ROW)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(INSERT, ROW)  # same clip twice
    bad = list(ROW)
    bad[0], bad[1], bad[6] = "d", "data/raw/d.r13.avi", "thermal"
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(INSERT, bad)


# --- build / verify end to end on toy files -------------------------------------------------------

RGB_NAME = "2018-03-07.10-00-00.10-00-00.school.G336.r13.avi"  # 0 s by name; the toy file is 0.5 s
IR_NAME = "2018-03-07.10-00-00.10-00-00.school.G474.r13.avi"


def toy_raw(tmp_path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    write_avi(raw / RGB_NAME, n_frames=15)
    write_avi(raw / IR_NAME, n_frames=12)
    return [raw / IR_NAME, raw / RGB_NAME]


def test_build_inserts_rows_and_resumes(tmp_path, capsys):
    files = toy_raw(tmp_path)
    conn = connect(tmp_path / "m.sqlite")
    build(conn, files, tmp_path, {"G336": "rgb", "G474": "ir"}, max_diff_s=2.0)
    rows = {r["camera_id"]: r for r in conn.execute("SELECT * FROM videos")}
    assert rows["G336"]["path"] == f"data/raw/{RGB_NAME}"
    assert (rows["G336"]["n_frames"], rows["G336"]["duration_s"], rows["G336"]["modality"]) == (15, 0.5, "rgb")
    assert (rows["G474"]["n_frames"], rows["G474"]["modality"]) == (12, "ir")
    assert rows["G336"]["sha256"] == sha256_file(files[1])
    assert rows["G336"]["start_time"] == "2018-03-07T10:00:00"
    build(conn, files, tmp_path, {"G336": "rgb", "G474": "ir"}, max_diff_s=2.0)  # second run: nothing new
    assert "Inserted 0, already present 2." in capsys.readouterr().out
    assert conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 2


def test_build_rejects_changed_size_on_resume(tmp_path):
    files = toy_raw(tmp_path)
    conn = connect(tmp_path / "m.sqlite")
    build(conn, files, tmp_path, {"G336": "rgb", "G474": "ir"}, max_diff_s=2.0)
    write_avi(files[1], n_frames=20)  # same name, different content and size
    with pytest.raises(ValueError, match="differs from manifest"):
        build(conn, files, tmp_path, {"G336": "rgb", "G474": "ir"}, max_diff_s=2.0)


def test_build_rejects_duration_mismatch(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw / "2018-03-07.10-00-00.10-00-05.school.G336.r13.avi"  # 5 s by name, 0.5 s of video
    write_avi(path, n_frames=15)
    conn = connect(tmp_path / "m.sqlite")
    with pytest.raises(ValueError, match="container duration"):
        build(conn, [path], tmp_path, {"G336": "rgb"}, max_diff_s=2.0)
    assert conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 0


def test_verify_detects_changed_file(tmp_path):
    files = toy_raw(tmp_path)
    conn = connect(tmp_path / "m.sqlite")
    build(conn, files, tmp_path, {"G336": "rgb", "G474": "ir"}, max_diff_s=2.0)
    assert verify(conn, files, tmp_path)
    data = bytearray(files[1].read_bytes())
    data[-100] ^= 0xFF  # flip one byte, same size
    files[1].write_bytes(bytes(data))
    assert not verify(conn, files, tmp_path)


def test_verify_detects_missing_row(tmp_path):
    files = toy_raw(tmp_path)
    conn = connect(tmp_path / "m.sqlite")
    build(conn, files[:1], tmp_path, {"G336": "rgb", "G474": "ir"}, max_diff_s=2.0)
    assert not verify(conn, files, tmp_path)


# --- annotation cross-checks ----------------------------------------------------------------------


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


def test_annotation_index(tmp_path):
    touch(tmp_path / "kitware" / "x" / "a.activities.yml")
    touch(tmp_path / "kitware-meva-training" / "x" / "b.activities.yml")
    touch(tmp_path / "contrib" / "CMU-v2" / "c-activities.yml")
    kitware, contrib = annotation_index(tmp_path)
    assert kitware == {"a": "kitware", "b": "kitware-meva-training"}
    assert contrib == {"c"}


def test_annotation_index_rejects_clip_in_both_sources(tmp_path):
    touch(tmp_path / "kitware" / "x" / "a.activities.yml")
    touch(tmp_path / "kitware-meva-training" / "y" / "a.activities.yml")
    touch(tmp_path / "contrib" / "CMU-v2" / "c-activities.yml")
    with pytest.raises(ValueError, match="both"):
        annotation_index(tmp_path)


def test_validate_names(tmp_path):
    rgb, ir = Path(RGB_NAME), Path(IR_NAME)
    modality = {"G336": "rgb", "G474": "ir"}
    clip = RGB_NAME.removesuffix(".r13.avi")
    validate_names([rgb, ir], modality, {clip: "kitware"})  # IR clips need no Kitware file
    with pytest.raises(ValueError, match="without a Kitware annotation"):
        validate_names([rgb, ir], modality, {})
    other_site = Path(RGB_NAME.replace("school", "hospital").replace("10-00-00.10-00-00", "11-00-00.11-00-00"))
    with pytest.raises(ValueError, match="more than one site"):
        validate_names([rgb, other_site], modality, {clip: "kitware"})
