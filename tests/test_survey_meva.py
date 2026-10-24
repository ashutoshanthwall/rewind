import pandas as pd
import pytest

from scripts.select_footage import select
from scripts.survey_meva import attach_ir_labels, combine_contrib, ir_by_camera, parse_activities, parse_clip_name


def test_regular_clip():
    c = parse_clip_name("2018-03-07.17-35-06.17-40-06.school.G339")
    assert (c["date"], c["site"], c["camera"]) == ("2018-03-07", "school", "G339")
    assert c["start"] == "17:35:06"
    assert c["start_s"] == 17 * 3600 + 35 * 60 + 6  # 63306
    assert c["duration_s"] == 300


def test_partial_clip():
    # Real name from the release: 09:49:37 -> 09:50:00 is 23 s.
    assert parse_clip_name("2018-03-05.09-49-37.09-50-00.school.G474")["duration_s"] == 23


def test_clip_crossing_midnight():
    # 23:55:09 -> 00:00:09 next day = 300 s, not -86100 s.
    assert parse_clip_name("2018-03-11.23-55-09.00-00-09.bus.G340")["duration_s"] == 300


def test_zero_length_clip_kept_with_zero_duration():
    assert parse_clip_name("2018-03-05.10-01-05.10-01-05.school.G421")["duration_s"] == 0


@pytest.mark.parametrize("name", [
    "2018-03-07.17-35-06.17-40-06.school.G339.r13.avi",  # suffix not stripped
    "2018-03-07.17-35-06.school.G339",                   # missing end time
    "2018-03-07.17-35-06.17-40-06.school.339",           # camera without G
])
def test_malformed_names_raise(name):
    with pytest.raises(ValueError):
        parse_clip_name(name)


# --- KPF activity parsing -------------------------------------------------------------------------

KNOWN = {"vehicle_stops", "vehicle_starts", "person_sits_down"}


def act(name, start=0, end=10):
    return {"act": {"act2": {name: 1.0}, "id2": 1, "timespan": [{"tsr0": [start, end]}]}}


def test_parse_activities_counts_per_name_and_ignores_meta():
    items = [{"meta": "note"}, act("vehicle_stops"), act("vehicle_stops"), act("person_sits_down")]
    counts, empty = parse_activities(items, KNOWN, "toy")
    assert counts == {"vehicle_stops": 2, "person_sits_down": 1}
    assert empty is False


def test_parse_activities_empty_marker_is_not_an_event():
    counts, empty = parse_activities([{"meta": "x was marked empty-37"}, act("empty_37", 0, 9000)], KNOWN, "toy")
    assert counts == {} and empty is True


@pytest.mark.parametrize("items, match", [
    ([act("vehicle_teleports")], "not in activity-names"),
    ([{"geom": {}}], "Unexpected KPF record"),
    ([act("empty_37"), act("vehicle_stops")], "marked empty_37 but also has"),
])
def test_parse_activities_rejects_bad_files(items, match):
    with pytest.raises(ValueError, match=match):
        parse_activities(items, KNOWN, "toy")


# --- IR flag ----------------------------------------------------------------------------------------

def test_ir_flag_is_per_camera():
    sets = pd.DataFrame({"clip": ["d.a.b.s.G1", "d.c.e.s.G1", "d.a.b.s.G2"], "camera_set": ["IR", "IR", "3-424"]})
    videos = pd.DataFrame({"camera": ["G1", "G2"]})
    assert ir_by_camera(sets, videos).to_dict() == {"G1": True, "G2": False}


def test_ir_flag_mixed_or_unknown_camera_raises():
    mixed = pd.DataFrame({"clip": ["d.a.b.s.G1", "d.c.e.s.G1"], "camera_set": ["IR", "skip"]})
    with pytest.raises(ValueError, match="both IR and non-IR"):
        ir_by_camera(mixed, pd.DataFrame({"camera": ["G1"]}))
    known = pd.DataFrame({"clip": ["d.a.b.s.G1"], "camera_set": ["IR"]})
    with pytest.raises(ValueError, match="IR status unknown"):
        ir_by_camera(known, pd.DataFrame({"camera": ["G1", "G9"]}))


# --- selection --------------------------------------------------------------------------------------

def clip(camera, start_s, duration_s=300, date="2018-03-11", ir=False, annotated=True):
    return {"clip": f"{date}.{camera}.{start_s}", "s3_key": f"k/{date}.{camera}.{start_s}", "camera": camera,
            "date": date, "start_s": start_s, "duration_s": duration_s, "ir": ir, "annotated": annotated}


SURVEY = pd.DataFrame([
    clip("G1", 1000), clip("G1", 1300),                      # RGB covers [1000, 1600)
    clip("G1", 2000, annotated=False),                      # unannotated: never selected
    clip("G1", 3000, duration_s=0),                         # zero-length: never selected
    clip("G1", 1000, date="2018-03-12"),                    # other date
    clip("IR1", 700, ir=True),                              # [700, 1000): touches RGB start, no overlap
    clip("IR1", 999, ir=True),                              # [999, 1299): overlaps by 1 s
    clip("IR1", 1599, ir=True),                             # [1599, 1899): overlaps last second
    clip("IR1", 1600, ir=True),                             # [1600, 1900): starts exactly at RGB end
    clip("IR1", 1000, ir=True, date="2018-03-12"),          # same time, wrong date
])


def test_select_rgb_takes_annotated_nonzero_clips_on_chosen_dates():
    sel = select(SURVEY, ["G1"], ["2018-03-11"], None)
    assert sorted(sel["start_s"]) == [1000, 1300]


def test_select_ir_uses_half_open_overlap_on_same_date():
    # Hand-checked: overlap iff ir_start < rgb_end and rgb_start < ir_end, intervals [start, start+duration).
    sel = select(SURVEY, ["G1"], ["2018-03-11"], "IR1")
    assert sorted(sel.loc[sel["ir"], "start_s"]) == [999, 1599]


def test_select_rejects_missing_camera_date_pair_and_non_rgb_camera():
    with pytest.raises(ValueError, match="No annotated clips"):
        select(SURVEY, ["G1"], ["2018-03-11", "2018-03-13"], None)
    with pytest.raises(ValueError, match="Not RGB cameras"):
        select(SURVEY, ["IR1"], ["2018-03-11"], None)


# --- contrib (third-party) labels: IR only -----------------------------------------------------------

def test_parse_activities_contrib_set_and_old_names():
    # Contrib files use a YAML !!set for act2 and old activity names.
    items = [{"meta": "x.avi"}, {"act": {"act2": {"vehicle_stopping"}}}, {"act": {"act2": {"vehicle_stopping"}}},
             {"act": {"act2": {"vehicle_starting"}}}]
    rename = {"vehicle_stopping": "vehicle_stops", "vehicle_starting": "vehicle_starts"}
    counts, empty = parse_activities(items, KNOWN, "toy", rename)
    assert counts == {"vehicle_stops": 2, "vehicle_starts": 1} and empty is False
    with pytest.raises(ValueError, match="not in activity-names"):
        parse_activities([{"act": {"act2": {"vehicle_flying"}}}], KNOWN, "toy", rename)


def test_combine_contrib_takes_max_over_sources_not_sum():
    per_source = pd.DataFrame([
        {"clip": "c1", "source": "UMD-IBM", "records": 5, "n_vehicle_starts": 1, "n_vehicle_stops": 0},
        {"clip": "c1", "source": "CMU-v2", "records": 3, "n_vehicle_starts": 0, "n_vehicle_stops": 2},
        {"clip": "c2", "source": "CMU-v2", "records": 4, "n_vehicle_starts": 0, "n_vehicle_stops": 0},
    ])
    out = combine_contrib(per_source).set_index("clip")
    assert out.loc["c1", "ir_label_sources"] == "CMU-v2+UMD-IBM"
    assert out.loc["c1", "ir_label_records"] == 5  # max(5, 3), not 8
    assert (out.loc["c1", "ir_vehicle_starts"], out.loc["c1", "ir_vehicle_stops"]) == (1, 2)
    assert out.loc["c2", "ir_label_records"] == 4


IR_LABELS = pd.DataFrame([{"clip": "ir1", "ir_label_sources": "CMU-v2", "ir_label_records": 3,
                           "ir_vehicle_starts": 1, "ir_vehicle_stops": 1}])


def test_attach_ir_labels_only_on_ir_clips():
    df = pd.DataFrame({"clip": ["ir1", "ir2", "rgb1"], "ir": [True, True, False]})
    out = attach_ir_labels(df, IR_LABELS).set_index("clip")
    assert out["ir_labelled"].to_dict() == {"ir1": True, "ir2": False, "rgb1": False}
    assert out.loc["ir1", "ir_label_records"] == 3 and out.loc["rgb1", "ir_label_records"] == 0


@pytest.mark.parametrize("df", [
    pd.DataFrame({"clip": ["ir1"], "ir": [False]}),   # label on an RGB clip: must never happen
    pd.DataFrame({"clip": ["other"], "ir": [True]}),  # label for a clip with no video
])
def test_attach_ir_labels_rejects_rgb_or_unknown_clip(df):
    with pytest.raises(ValueError, match="not IR videos"):
        attach_ir_labels(df, IR_LABELS)
