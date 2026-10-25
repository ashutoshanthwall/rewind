import json
from collections import Counter
from datetime import datetime

import pytest

from rewind.ingest.annotations import (Event, best_iou, box_iou, boundary_links, group_ids, kitware_events, load_kpf,
                                       mean_box_iou, merge_contrib, parse_contrib_json, parse_geom_line,
                                       parse_kitware_activities, parse_kitware_types, piece, read_geom)
from rewind.ingest.manifest import Video, connect, load_videos
from scripts.annotation_patterns import geom_patterns
from scripts.build_events import build, kitware_index

KNOWN = {"vehicle_stops", "vehicle_starts", "person_talks_to_person", "vehicle_turns_left"}
RENAME = {"vehicle_turning_left": "vehicle_turns_left", "Stopping": "vehicle_stops"}

# Real lines from the selected files (both KPF styles), shortened ids where they don't matter.
REPR_GEOM = "- {'geom': {'g0': '298 294 382 507', 'id0': 777, 'id1': 111, 'keyframe': True, 'ts0': 2315}}"
FLOW_GEOM = ("- { geom: {id1: 5573913994148029640, id0: 1, ts0: 1040, ts1: 34.67, g0: 490 259 545 290, "
             "keyframe: true } }")
FLOW_GEOM_NO_KF = "- { geom: {id1: 7, id0: 2, ts0: 0, ts1: 0.0, g0: -12 0 40 1081 } }"


# --- geom lines -----------------------------------------------------------------------------------


def test_geom_line_both_styles():
    assert parse_geom_line(REPR_GEOM) == (111, 2315, (298, 294, 382, 507))
    assert parse_geom_line(FLOW_GEOM) == (5573913994148029640, 1040, (490, 259, 545, 290))
    assert parse_geom_line(FLOW_GEOM_NO_KF) == (7, 0, (-12, 0, 40, 1081))  # stored as annotated, not clipped


def test_geom_meta_lines_and_bad_lines():
    assert parse_geom_line('- { meta: "2018-03-07.16-55-06.17-00-06.school.G336" }') is None
    assert parse_geom_line("- {'meta': {'team': 'Kitware'}}") is None
    with pytest.raises(ValueError, match="Not a geom or meta"):
        parse_geom_line("- {'act': {'act2': {'vehicle_stops': 1.0}}}")
    with pytest.raises(ValueError, match=r"without \['ts0'\]"):
        parse_geom_line("- {'geom': {'g0': '1 2 3 4', 'id0': 1, 'id1': 5}}")
    with pytest.raises(ValueError, match=r"without \['g0'\]"):  # only 3 numbers: not a box
        parse_geom_line("- {'geom': {'g0': '1 2 3', 'id0': 1, 'id1': 5, 'ts0': 9}}")


def test_read_geom_mixed_styles_and_duplicates(tmp_path):
    p = tmp_path / "c.geom.yml"
    p.write_text("- { meta: \"c\" }\n" + REPR_GEOM + "\n" + REPR_GEOM.replace("2315", "2316") + "\n\n" + FLOW_GEOM + "\n")
    assert read_geom(p) == {111: {2315: (298, 294, 382, 507), 2316: (298, 294, 382, 507)},
                            5573913994148029640: {1040: (490, 259, 545, 290)}}
    p.write_text(REPR_GEOM + "\n" + REPR_GEOM + "\n")
    with pytest.raises(ValueError, match="second box for id1 111 at frame 2315"):
        read_geom(p)


# --- Kitware activities / types / join --------------------------------------------------------------


def act_line(name, s, e, id2=9, actors=((3, None),), style="repr", status="good"):
    if style == "repr":
        acts = ", ".join(f"{{'id1': {i}, 'timespan': [{{'tsr0': [{a or s}, {b or e}]}}]}}"
                         for i, span in actors for a, b in [span or (None, None)])
        return (f"- {{'act': {{'act2': {{'{name}': 1.0}}, 'actors': [{acts}], 'id2': {id2}, "
                f"'src_status': '{status}', 'timespan': [{{'tsr0': [{s}, {e}]}}]}}}}")
    acts = ", ".join(f"{{id1: {i}, timespan: [{{tsr0: [{(span or (s, e))[0]}, {(span or (s, e))[1]}]}}]}}"
                     for i, span in actors)
    return (f"- {{ act: {{ act2: {{{name}: 1.0}}, id2: {id2}, timespan: [{{tsr0: [{s}, {e}]}}], "
            f"src_status: {status}, actors: [{acts}] }} }}")


def parse_acts(tmp_path, *lines):
    p = tmp_path / "x.activities.yml"
    p.write_text("\n".join(lines) + "\n")
    return parse_kitware_activities(load_kpf(p), KNOWN, p.name)


def test_activities_both_styles(tmp_path):
    acts, empty = parse_acts(tmp_path, "- { meta: { team: \"Kitware\" } }",
                             act_line("vehicle_stops", 2301, 2380, id2=9, actors=((3, None),)),
                             act_line("person_talks_to_person", 10, 50, id2=10, actors=((4, (10, 30)), (5, None)),
                                      style="flow", status="janitor"))
    assert not empty
    assert acts == [
        {"name": "vehicle_stops", "start": 2301, "end": 2380, "id2": 9, "src_status": "good", "actors": [(3, 2301, 2380)]},
        {"name": "person_talks_to_person", "start": 10, "end": 50, "id2": 10, "src_status": "janitor",
         "actors": [(4, 10, 30), (5, 10, 50)]}]


def test_both_empty_marker_styles(tmp_path):
    act_marker = "- { act: { act2: { empty_37: 1.0 }, id2: 1, timespan: [{tsr0: [0, 9000]}], src: truth } }"
    assert parse_acts(tmp_path, act_marker) == ([], True)
    assert parse_acts(tmp_path, '- { meta: "2018-03-07.10-55-00.11-00-00.school.G336 was marked empty-37" }') == ([], True)
    assert parse_acts(tmp_path, "- {'meta': '2018-03-11.11-20-00.11-25-00.bus.G340 was marked empty_37'}") == ([], True)
    acts, empty = parse_acts(tmp_path, '- { meta: "2018-03-07.16-55-06.17-00-06.school.G336" }',
                             '- { meta: "vehicle_stops 1 instances" }', act_line("vehicle_stops", 1, 2))
    assert (len(acts), empty) == (1, False)  # other string metas are not markers
    with pytest.raises(ValueError, match="marked empty_37 but has 1 activities"):
        parse_acts(tmp_path, "- {'meta': 'c was marked empty_37'}", act_line("vehicle_stops", 1, 2))


def test_activity_errors(tmp_path):
    with pytest.raises(ValueError, match="unknown activity 'person_flies'"):
        parse_acts(tmp_path, act_line("person_flies", 1, 2))
    with pytest.raises(ValueError, match="duplicate activity ids"):
        parse_acts(tmp_path, act_line("vehicle_stops", 1, 2, id2=5), act_line("vehicle_starts", 3, 4, id2=5))
    two_spans = act_line("vehicle_stops", 1, 2).replace("'timespan': [{'tsr0': [1, 2]}]}}",
                                                        "'timespan': [{'tsr0': [1, 2]}, {'tsr0': [5, 6]}]}}")
    with pytest.raises(ValueError, match="exactly one timespan"):
        parse_acts(tmp_path, two_spans)
    no_actors = act_line("vehicle_stops", 1, 2).replace("'actors': [{'id1': 3, 'timespan': [{'tsr0': [1, 2]}]}]",
                                                        "'actors': []")
    with pytest.raises(ValueError, match="has no actors"):
        parse_acts(tmp_path, no_actors)


def test_types(tmp_path):
    p = tmp_path / "x.types.yml"
    p.write_text("- {'types': {'cset3': {'vehicle': 1.0}, 'id1': 3}}\n- { types: { id1: 4, cset3: { person: 1.0 } } }\n")
    assert parse_kitware_types(load_kpf(p), "x") == {3: "vehicle", 4: "person"}
    p.write_text("- {'types': {'cset3': {'vehicle': 1.0}, 'id1': 3}}\n- {'types': {'cset3': {'person': 1.0}, 'id1': 3}}\n")
    with pytest.raises(ValueError, match="two types records for id1 3"):
        parse_kitware_types(load_kpf(p), "x")


def kit_act(start, end, actors):
    return {"name": "vehicle_stops", "start": start, "end": end, "id2": 1, "src_status": "good", "actors": actors}


def test_kitware_join_dense_geom():
    track = {f: (f, 0, f + 10, 10) for f in range(5, 9)}
    (ev,) = kitware_events("c", "kitware-eval", "kitware", [kit_act(5, 8, [(3, 5, 8)])], {3: "vehicle"}, {3: track}, 10)
    assert (ev.start_frame, ev.end_frame, ev.actor_types, ev.boxes, ev.src_ids) == (5, 8, ["vehicle"], [track], ["1"])


def test_kitware_boxes_cut_to_event_span():
    # The actor's track runs 2 frames past the activity (as in 1 real case): only frames 5..8 are kept.
    track = {f: (f, 0, f + 10, 10) for f in range(5, 11)}
    (ev,) = kitware_events("c", "kitware-eval", "kitware", [kit_act(5, 8, [(3, 5, 10)])], {3: "vehicle"}, {3: track}, 12)
    assert sorted(ev.boxes[0]) == [5, 6, 7, 8]


def test_kitware_join_errors():
    full = {3: {f: (0, 0, 1, 1) for f in range(5, 9)}}
    with pytest.raises(ValueError, match="does not cover exactly frames 5..8"):
        gap = {3: {f: (0, 0, 1, 1) for f in (5, 6, 8)}}
        kitware_events("c", "kitware-eval", "kitware", [kit_act(5, 8, [(3, 5, 8)])], {3: "vehicle"}, gap, 10)
    with pytest.raises(ValueError, match="geom tracks for ids that no activity uses"):
        kitware_events("c", "kitware-eval", "kitware", [kit_act(5, 8, [(3, 5, 8)])], {3: "vehicle"},
                       {**full, 4: {0: (0, 0, 1, 1)}}, 10)
    with pytest.raises(ValueError, match=r"span 5-10 outside 0..9"):  # inclusive end must be < n_frames
        kitware_events("c", "kitware-eval", "kitware", [kit_act(5, 10, [(3, 5, 8)])], {3: "vehicle"}, full, 10)
    with pytest.raises(ValueError, match="no types record"):
        kitware_events("c", "kitware-eval", "kitware", [kit_act(5, 8, [(3, 5, 8)])], {}, full, 10)
    with pytest.raises(ValueError, match=r"actor 3 span 5-10 outside 0..9"):
        kitware_events("c", "kitware-eval", "kitware", [kit_act(5, 9, [(3, 5, 10)])], {3: "vehicle"},
                       {3: {f: (0, 0, 1, 1) for f in range(5, 11)}}, 10)
    with pytest.raises(ValueError, match="actor id appears in more than one activity"):
        kitware_events("c", "kitware-eval", "kitware", [kit_act(5, 8, [(3, 5, 8)]), kit_act(5, 8, [(3, 5, 8)])],
                       {3: "vehicle"}, full, 10)


# --- contrib JSON --------------------------------------------------------------------------------


def contrib_act(aid, name, start, end, x=0, obj_frames=None):
    frames = obj_frames if obj_frames is not None else range(start, end + 1)
    return {"activityID": aid, "activity": name, "localization": {"c.avi": {str(start): 1, str(end): 0}},
            "objects": [{"objectType": "Person", "objectID": "0", "localization": {"c.avi": {
                str(f): {"presenceConf": 1.0, "boundingBox": {"x": x, "y": 10, "w": 20, "h": 5}} for f in frames}}}]}


def test_contrib_json():
    data = {"activities": [
        contrib_act(842, "vehicle_turning_left", 2, 4),
        {"activityID": 7, "activity": "Stopping", "localization": {"c.avi": {"5": 0}}, "objects": []}]}  # zero-length
    events, dropped = parse_contrib_json(data, "c", "UMD-IBM", RENAME, KNOWN, n_frames=10)
    assert dropped == 1
    (ev,) = events
    assert (ev.activity_type, ev.start_frame, ev.end_frame, ev.label_source, ev.src_ids) == (
        "vehicle_turns_left", 2, 4, "contrib", ["UMD-IBM:842"])
    assert ev.boxes == [{2: (0, 10, 20, 15), 3: (0, 10, 20, 15), 4: (0, 10, 20, 15)}]  # x2 = x + w, y2 = y + h
    assert ev.actor_types is None


def test_contrib_json_errors():
    with pytest.raises(ValueError, match=r"has boxes on 2 frames \(2..4\), expected exactly the frames 2..4"):
        parse_contrib_json({"activities": [contrib_act(1, "Stopping", 2, 4, obj_frames=[2, 4])]}, "c", "CMU-v2",
                           RENAME, KNOWN, 10)
    with pytest.raises(ValueError, match=r"has boxes on 9 frames \(0..8\)"):  # boxes outside the span: not trimmed
        parse_contrib_json({"activities": [contrib_act(1, "Stopping", 2, 4, obj_frames=range(0, 9))]}, "c", "CMU-v2",
                           RENAME, KNOWN, 10)
    bad_shape = contrib_act(1, "Stopping", 2, 4)
    bad_shape["localization"]["c.avi"] = {"2": 1, "4": 1}
    with pytest.raises(ValueError, match="is not {start: 1, end: 0}"):
        parse_contrib_json({"activities": [bad_shape]}, "c", "CMU-v2", RENAME, KNOWN, 10)
    other_clip = contrib_act(1, "Stopping", 2, 4)
    other_clip["localization"]["d.avi"] = {"2": 1, "4": 0}
    with pytest.raises(ValueError, match=r"localization for \['c.avi', 'd.avi'\], expected only c.avi"):
        parse_contrib_json({"activities": [other_clip]}, "c", "CMU-v2", RENAME, KNOWN, 10)
    with pytest.raises(ValueError, match=r"span 2-10 outside 0..9"):
        parse_contrib_json({"activities": [contrib_act(1, "Stopping", 2, 10)]}, "c", "CMU-v2", RENAME, KNOWN, 10)
    with pytest.raises(ValueError, match="unknown activity 'Flying'"):
        parse_contrib_json({"activities": [contrib_act(1, "Flying", 2, 4)]}, "c", "CMU-v2", RENAME, KNOWN, 10)


# --- IoU and merging -----------------------------------------------------------------------------


def test_box_iou_hand_computed():
    assert box_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)  # overlap 5x5, union 100+100-25
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0  # touching edges


def ev(clip, name, s, e, box=None, source="CMU-v2", aid=1):
    boxes = [] if box is None else [{f: box for f in range(s, e + 1)}]
    return Event(clip, name, s, e, "contrib", [source], [f"{source}:{aid}"], boxes=boxes)


def test_mean_box_iou_over_shared_frames():
    a = ev("c", "vehicle_stops", 0, 3, (0, 0, 10, 10))
    b = Event("c", "vehicle_stops", 2, 5, "contrib", ["UMD-IBM"], ["UMD-IBM:1"],
              boxes=[{2: (0, 0, 10, 10), 3: (5, 5, 15, 15), 4: (0, 0, 1, 1), 5: (0, 0, 1, 1)}])
    assert mean_box_iou(a, b) == pytest.approx((1 + 25 / 175) / 2)  # shared frames 2 and 3 only
    assert mean_box_iou(a, ev("c", "vehicle_stops", 0, 3)) == 0.0  # no boxes: never merged


def test_merge_contrib():
    car, other_car = (0, 0, 10, 10), (50, 50, 60, 60)
    events = [ev("c", "vehicle_starts", 10, 20, car, "CMU-v2", 1),
              ev("c", "vehicle_starts", 15, 30, car, "UMD-IBM", 2),       # same car, overlaps -> merged with 1
              ev("c", "vehicle_starts", 28, 40, car, "UMD-IBM", 3),       # overlaps 2 only -> joins the chain
              ev("c", "vehicle_starts", 12, 18, other_car, "CMU-v2", 4),  # overlaps in time, other box -> separate
              ev("c", "vehicle_stops", 10, 20, car, "UMD-IBM", 5),        # other type -> separate
              ev("c", "vehicle_starts", 50, 60, car, "UMD-IBM", 6)]       # no time overlap -> separate
    merged = merge_contrib(events, min_box_iou=0.3)
    assert [(m.activity_type, m.start_frame, m.end_frame, m.sources, m.src_ids) for m in merged] == [
        ("vehicle_stops", 10, 20, ["UMD-IBM"], ["UMD-IBM:5"]),                  # sorted by (start, end, type)
        ("vehicle_starts", 10, 40, ["CMU-v2", "UMD-IBM"], ["CMU-v2:1", "UMD-IBM:2", "UMD-IBM:3"]),
        ("vehicle_starts", 12, 18, ["CMU-v2"], ["CMU-v2:4"]),
        ("vehicle_starts", 50, 60, ["UMD-IBM"], ["UMD-IBM:6"])]
    assert len(merged[1].boxes) == 3  # actors of all merged records kept


def test_merge_contrib_transitive_any_order():
    # X and Y don't overlap each other; Z (listed last) overlaps both -> one event 0..30.
    car = (0, 0, 10, 10)
    xyz = [ev("c", "vehicle_stops", 0, 10, car, "CMU-v2", "X"), ev("c", "vehicle_stops", 20, 30, car, "CMU-v2", "Y"),
           ev("c", "vehicle_stops", 5, 25, car, "UMD-IBM", "Z")]
    (m,) = merge_contrib(xyz, min_box_iou=0.3)
    assert (m.start_frame, m.end_frame, m.src_ids) == (0, 30, ["CMU-v2:X", "CMU-v2:Y", "UMD-IBM:Z"])
    with pytest.raises(ValueError, match="one clip"):
        merge_contrib([ev("c", "vehicle_stops", 0, 1), ev("d", "vehicle_stops", 0, 1)], 0.3)


# --- clip-boundary linking -------------------------------------------------------------------------


def test_boundary_links_one_to_one():
    box = (0, 0, 10, 10)
    a = [piece(1, ev("A", "person_talks_to_person", 50, 99, box)),
         piece(2, ev("A", "person_talks_to_person", 60, 99, (1, 0, 11, 10))),  # IoU with b's box: 9x10 / (100+100-90) = 0.818
         piece(3, ev("A", "vehicle_stops", 70, 99, box)),                       # no same-type piece in B
         piece(4, ev("A", "person_talks_to_person", 70, 98, box))]              # does not reach the last frame
    b = [piece(5, ev("B", "person_talks_to_person", 0, 20, box)),
         piece(6, ev("B", "person_talks_to_person", 1, 20, box))]               # does not start on frame 0
    # 1 and 2 both match 5; 1 has IoU 1.0, so 1-5 wins and 2 stays unlinked (one-to-one).
    assert boundary_links(a, 100, b, min_box_iou=0.5) == [(1, 5)]
    assert boundary_links(a[1:2], 100, b, min_box_iou=0.8) == [(2, 5)]   # 0.818 >= 0.8
    assert boundary_links(a[1:2], 100, b, min_box_iou=0.85) == []
    assert boundary_links([a[2]], 100, b, min_box_iou=0.5) == []   # other type, alone
    assert boundary_links([a[3]], 100, b, min_box_iou=0.5) == []   # ends on frame 98, alone
    # One A piece, two B pieces on frame 0: only the better one is linked (one-to-one on the A side).
    b2 = [piece(5, ev("B", "person_talks_to_person", 0, 20, box)),
          piece(7, ev("B", "person_talks_to_person", 0, 9, (1, 0, 11, 10)))]
    assert boundary_links(a[:1], 100, b2, min_box_iou=0.5) == [(1, 5)]


def test_boundary_uses_last_and_first_frame_boxes():
    # A's car is at (100,100,...) until frame 98 and at (0,0,10,10) on its last frame 99.
    moving = Event("A", "vehicle_stops", 90, 99, "kitware-eval", ["kitware"], ["1"],
                   boxes=[{**{f: (100, 100, 110, 110) for f in range(90, 99)}, 99: (0, 0, 10, 10)}])
    assert boundary_links([piece(1, moving)], 100, [piece(5, ev("B", "vehicle_stops", 0, 5, (0, 0, 10, 10)))], 0.5) == [(1, 5)]
    assert boundary_links([piece(1, moving)], 100, [piece(5, ev("B", "vehicle_stops", 0, 5, (100, 100, 110, 110)))], 0.5) == []


def test_group_ids_chain():
    assert group_ids([1, 2, 3, 4], [(2, 3), (3, 4)]) == {1: 1, 2: 2, 3: 2, 4: 2}
    assert group_ids([5, 6], []) == {5: 5, 6: 6}


def test_best_iou_needs_boxes_at_those_frames():
    assert best_iou([{5: (0, 0, 10, 10)}], 5, [{0: (0, 0, 10, 10)}], 0) == 1.0
    assert best_iou([{5: (0, 0, 10, 10)}], 4, [{0: (0, 0, 10, 10)}], 0) == 0.0


# --- frame_end_time and the whole build on a toy tree ---------------------------------------------


def test_frame_end_time():
    v = Video(1, "c", "p", "G336", "rgb", datetime(2018, 3, 7, 10, 55, 6), 9000, 30.0)
    assert v.frame_end_time(0) == datetime(2018, 3, 7, 10, 55, 6, 33333)
    assert v.frame_end_time(8999) == datetime(2018, 3, 7, 11, 0, 6)  # = v.end_time, to the microsecond
    w = Video(1, "c", "p", "G424", "rgb", datetime(2018, 3, 7, 16, 55, 6), 9008, 30.0)
    assert w.frame_end_time(9007) == w.end_time == datetime(2018, 3, 7, 17, 0, 6, 266667)  # 300.2666... s, rounded once
    with pytest.raises(ValueError):
        v.frame_end_time(9000)


INSERT = ("INSERT INTO videos (clip, path, size_bytes, sha256, camera_id, site, modality, start_time, name_end_time,"
          " duration_s, n_frames, fps, width, height, codec, ingested_at) VALUES (?, ?, 1, 'x', ?, 'bus', ?, ?, ?, ?, ?,"
          " 30.0, 1920, 1080, 'h264', 'now')")


def write_kitware(root, clip, acts_lines, types_lines, geom_lines):
    d = root / "kitware" / "2018-03-07" / "10"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{clip}.activities.yml").write_text("\n".join(acts_lines) + "\n")
    (d / f"{clip}.types.yml").write_text("\n".join(types_lines) + "\n")
    (d / f"{clip}.geom.yml").write_text("\n".join(geom_lines) + "\n")


def geom(id1, frames, box="0 0 100 50"):
    return [f"- {{'geom': {{'g0': '{box}', 'id0': {f}, 'id1': {id1}, 'keyframe': False, 'ts0': {f}}}}}" for f in frames]


def test_build_end_to_end(tmp_path):
    # Two back-to-back 10-frame clips of G340 (A: 10:00:00, B: 10:00:00.333 = A's end) and an empty one.
    a, b, e = ("2018-03-07.10-00-00.10-00-00.bus.G340", "2018-03-07.10-00-01.10-00-01.bus.G340",
               "2018-03-07.11-00-00.11-00-00.bus.G340")
    conn = connect(tmp_path / "m.sqlite")
    conn.execute(INSERT, (a, "raw/a", "G340", "rgb", "2018-03-07T10:00:00", "2018-03-07T10:00:00", 1 / 3, 10))
    conn.execute(INSERT, (b, "raw/b", "G340", "rgb", "2018-03-07T10:00:00.333333", "2018-03-07T10:00:01", 1 / 3, 10))
    conn.execute(INSERT, (e, "raw/e", "G340", "rgb", "2018-03-07T11:00:00", "2018-03-07T11:00:00", 1 / 3, 10))
    root = tmp_path / "annot"
    # A: a car stops on frames 6..9 (the last frame) -> continues in B on frames 0..3 at the same place.
    write_kitware(root, a, [act_line("vehicle_stops", 6, 9, id2=1, actors=((1, None),))],
                  ["- {'types': {'cset3': {'vehicle': 1.0}, 'id1': 1}}"], geom(1, range(6, 10)))
    write_kitware(root, b, [act_line("vehicle_stops", 0, 3, id2=1, actors=((8, None),), style="flow"),
                            act_line("person_talks_to_person", 2, 2, id2=2, actors=((9, None),))],
                  ["- { types: { id1: 8, cset3: { vehicle: 1.0 } } }", "- {'types': {'cset3': {'person': 1.0}, 'id1': 9}}"],
                  geom(8, range(0, 4)) + geom(9, [2], "500 500 520 560"))
    write_kitware(root, e, ["- {'meta': 'x was marked empty_37'}"], [], [])
    videos = load_videos(conn)
    stats = Counter()
    with conn:
        stats = build(conn, videos, kitware_index(root), {}, KNOWN, {},
                      {"contrib_merge_min_box_iou": 0.3, "boundary_merge_min_box_iou": 0.5}, max_gap_s=2.0)
    rows = [dict(r) for r in conn.execute("SELECT * FROM events ORDER BY event_id")]
    assert [(r["event_id"], r["activity_type"], r["start_frame"], r["end_frame"], r["at_clip_start"],
             r["at_clip_end"], r["group_id"]) for r in rows] == [
        (1, "vehicle_stops", 6, 9, 0, 1, 1),
        (2, "vehicle_stops", 0, 3, 1, 0, 1),        # linked to its piece in A
        (3, "person_talks_to_person", 2, 2, 0, 0, 3)]
    # Inclusive frames -> seconds [s/30, (e+1)/30) and wall-clock; hand-computed.
    assert rows[0]["start_s"] == pytest.approx(0.2) and rows[0]["end_s"] == pytest.approx(1 / 3)
    assert (rows[0]["start_time"], rows[0]["end_time"]) == ("2018-03-07T10:00:00.200000", "2018-03-07T10:00:00.333333")
    assert (rows[1]["start_time"], rows[1]["end_time"]) == ("2018-03-07T10:00:00.333333", "2018-03-07T10:00:00.466666")
    assert json.loads(rows[1]["src_ids"]) == ["1"] and json.loads(rows[1]["actor_types"]) == ["vehicle"]
    assert conn.execute("SELECT COUNT(*) FROM event_boxes").fetchone()[0] == 4 + 4 + 1
    assert tuple(conn.execute("SELECT x1, y1, x2, y2 FROM event_boxes WHERE event_id = 3").fetchone()) == (500, 500, 520, 560)
    labels = conn.execute("SELECT v.clip, l.label_status, l.label_source, l.n_events FROM clip_labels l "
                          "JOIN videos v USING (video_id) ORDER BY v.clip").fetchall()
    assert [tuple(r) for r in labels] == [(a, "events", "kitware-eval", 1), (b, "events", "kitware-eval", 2),
                                         (e, "empty", "kitware-eval", 0)]
    assert stats["boundary_links"] == 1 and stats["back_to_back_clip_pairs"] == 1  # 10:00:00.67 -> 11:00 is not
    assert {len(r["start_time"]) for r in rows} == {len(r["end_time"]) for r in rows} == {26}  # fixed width


def contrib_json(path, clip, acts):
    """Write a contrib .json for `clip`; acts are contrib_act() dicts made for clip 'c'."""
    text = json.dumps({"activities": acts, "filesProcessed": [f"{clip}.avi"]}).replace('"c.avi"', f'"{clip}.avi"')
    path.write_text(text)
    return path


def test_build_ir_contrib_branch(tmp_path):
    ir1, ir2, ir3 = ("2018-03-11.11-00-00.11-05-00.school.G474", "2018-03-11.11-10-00.11-15-00.school.G474",
                     "2018-03-11.11-20-00.11-25-00.school.G474")
    conn = connect(tmp_path / "m.sqlite")
    for clip, t in ((ir1, "11:00:00"), (ir2, "11:10:00"), (ir3, "11:20:00")):
        conn.execute(INSERT, (clip, f"raw/{clip}", "G474", "ir", f"2018-03-11T{t}", f"2018-03-11T{t}", 1 / 3, 10))
    # ir1: CMU and UMD label the same stop (same box) -> one event from both sources; ir2: nothing; ir3: Kitware marker.
    cmu = contrib_json(tmp_path / "cmu.json", ir1, [contrib_act(1, "Stopping", 2, 6)])
    umd = contrib_json(tmp_path / "umd.json", ir1, [contrib_act(9, "Stopping", 4, 8)])
    root = tmp_path / "annot"
    write_kitware(root, ir3, ["- { act: { act2: { empty_37: 1.0 }, id2: 1, timespan: [{tsr0: [0, 9000]}], src: truth } }"],
                  [], [])
    cfg = {"contrib_merge_min_box_iou": 0.3, "boundary_merge_min_box_iou": 0.5}
    with conn:
        stats = build(conn, load_videos(conn), kitware_index(root), {ir1: [("CMU-v2", cmu), ("UMD-IBM", umd)]},
                      KNOWN, RENAME, cfg, max_gap_s=2.0)
    rows = [dict(r) for r in conn.execute("SELECT * FROM events")]
    assert [(r["activity_type"], r["start_frame"], r["end_frame"], r["label_source"], json.loads(r["sources"]),
             r["actor_types"], r["src_status"]) for r in rows] == [
        ("vehicle_stops", 2, 8, "contrib", ["CMU-v2", "UMD-IBM"], None, None)]
    labels = [tuple(r) for r in conn.execute("SELECT v.clip, l.label_status, l.label_source, l.n_events FROM clip_labels l "
                                             "JOIN videos v USING (video_id) ORDER BY v.clip")]
    assert labels == [(ir1, "events", "contrib", 1), (ir2, "none", "none", 0), (ir3, "none", "none", 0)]
    assert (stats["contrib_records_merged_away"], stats["ir_none_no_annotation_file"],
            stats["ir_none_kitware_empty37_marker"]) == (1, 1, 1)


def test_build_refuses_kitware_activities_on_ir(tmp_path):
    clip = "2018-03-11.11-00-00.11-05-00.school.G474"
    conn = connect(tmp_path / "m.sqlite")
    conn.execute(INSERT, (clip, "raw/x", "G474", "ir", "2018-03-11T11:00:00", "2018-03-11T11:00:00", 1 / 3, 10))
    root = tmp_path / "annot"
    write_kitware(root, clip, [act_line("vehicle_stops", 1, 2)], [], [])
    with pytest.raises(ValueError, match="IR clip .* has Kitware activities"):
        build(conn, load_videos(conn), kitware_index(root), {}, KNOWN, RENAME,
              {"contrib_merge_min_box_iou": 0.3, "boundary_merge_min_box_iou": 0.5}, max_gap_s=2.0)


def test_build_never_links_across_cameras(tmp_path):
    # G341's clip starts exactly when G340's ends, with a same-type piece at the same place: still not linked.
    a, c = "2018-03-07.10-00-00.10-00-00.bus.G340", "2018-03-07.10-00-01.10-00-01.bus.G341"
    conn = connect(tmp_path / "m.sqlite")
    conn.execute(INSERT, (a, "raw/a", "G340", "rgb", "2018-03-07T10:00:00", "2018-03-07T10:00:00", 1 / 3, 10))
    conn.execute(INSERT, (c, "raw/c", "G341", "rgb", "2018-03-07T10:00:00.333333", "2018-03-07T10:00:01", 1 / 3, 10))
    root = tmp_path / "annot"
    write_kitware(root, a, [act_line("vehicle_stops", 6, 9, actors=((1, None),))],
                  ["- {'types': {'cset3': {'vehicle': 1.0}, 'id1': 1}}"], geom(1, range(6, 10)))
    write_kitware(root, c, [act_line("vehicle_stops", 0, 3, actors=((1, None),))],
                  ["- {'types': {'cset3': {'vehicle': 1.0}, 'id1': 1}}"], geom(1, range(0, 4)))
    with conn:
        stats = build(conn, load_videos(conn), kitware_index(root), {}, KNOWN, {},
                      {"contrib_merge_min_box_iou": 0.3, "boundary_merge_min_box_iou": 0.5}, max_gap_s=2.0)
    assert [tuple(r) for r in conn.execute("SELECT event_id, group_id FROM events ORDER BY event_id")] == [(1, 1), (2, 2)]
    assert (stats["boundary_links"], stats["back_to_back_clip_pairs"]) == (0, 0)


def test_kitware_index_refuses_a_clip_in_both_levels(tmp_path):
    clip = "2018-03-07.10-00-00.10-00-00.bus.G340"
    for folder in ("kitware", "kitware-meva-training"):
        d = tmp_path / folder / "2018-03-07" / "10"
        d.mkdir(parents=True)
        (d / f"{clip}.activities.yml").write_text("")
    with pytest.raises(ValueError, match="annotated in both kitware and kitware-meva-training"):
        kitware_index(tmp_path)


def test_geom_patterns(tmp_path):
    sel = tmp_path / "selection.txt"
    sel.write_text("# header\ndrops-123-r13/2018-03-07/10/2018-03-07.10-55-06.11-00-06.bus.G340.r13.avi\t182000000\n")
    assert geom_patterns(sel) == ["/annotation/DIVA-phase-2/MEVA/kitware*/**/2018-03-07.10-55-06.11-00-06.bus.G340.geom.yml"]
