"""MEVA activity annotations -> events (step 1.3). One event = one annotated activity record within one clip.

Conventions, checked on the selected clips on 2026-10-25 (details and numbers: docs/meva_notes.md):
  - Frame numbers are 0-based and equal our decoded-frame index (1.1 frame index rule); timespans tsr0 [start, end]
    have an INCLUSIVE end. An event covers [start / fps, (end + 1) / fps) seconds of its clip.
  - Kitware (RGB ground truth): <clip>.activities.yml, .types.yml, .geom.yml with one KPF record per line, written
    in two styles: python-repr "- {'act': ...}" and YAML flow "- { act: ...}" (64-bit hash ids). Activities and
    types files are small and loaded with yaml. Geom files (~1 GB in total) are read line by line with key regexes
    that accept both styles; any other line raises. g0 = "x1 y1 x2 y2" (KPF spec p6). Geom is dense: one box per
    frame of each actor's span, nothing outside it (checked here on every actor; violations raise).
  - Empty markers: an act record 'empty_37' with tsr0 [0, 9000] (a fixed placeholder, not the clip length) or a
    meta line "... was marked empty_37 / empty-37". The clip is then 'empty'. VERIFY: whether that means "checked,
    none of the 37 activities present" or "not annotated".
  - Contrib (third-party; used for IR clips only, Gate 0.3): the per-clip NIST-style .json files are read, not the
    -geom.yml files, which lack boxes for most actors. Activity localization {start: 1, end: 0}; objects have a box
    {x, y, w, h} on every frame start..end; x2 = x + w, y2 = y + h. VERIFY: NIST's JSON spec says frames begin at 1
    and the '0' key is the first absent frame; the data (start frame 0 used, boxes on the end frame, ends never
    reach n_frames, the YAML tsr0 equal) fit 0-based and inclusive like Kitware. The difference is 1 frame.
    Contrib object types are all 'Person' (also for vehicles) and are not used. Zero-length records (a localization
    with one key) are dropped and counted. Overlapping records of one type (duplicates within and across sources)
    are merged by `merge_contrib`.
  - Activities crossing a 5-min clip boundary are annotated in each clip (MEVA-Annotation-Definitions.pdf p3) with
    unrelated ids. Both pieces stay events; `boundary_groups` gives them one group id.
  - Actor ids (id1) are unique per actor per activity, never persistent object identities; they are not stored.
  - Boxes are stored for the event's own frames only (an actor's span can run past its activity: 1 actor, 18 frames).
  VERIFY (box timing, measured on a few tracks only): Kitware boxes follow the object to within a few frames (optical
  flow on 3 vehicle tracks: best shift 0 to +1 frame, i.e. the box at frame f may match frame f+1). CMU-v2 contrib
  boxes can be out of time with the object (seen 20-40 frames ahead on 1 IR track; shifts vs UMD-IBM vary in sign).
  Do not use contrib boxes as frame-exact; event time spans are not affected.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

EMPTY_ACTIVITY = "empty_37"
VEHICLE_START_STOP = ("vehicle_starts", "vehicle_stops")  # documents/activity-names.txt; the event branch's labels
EMPTY_META_RE = re.compile(r"was marked empty[-_]37")

# Geom line fields; quotes are optional so that both KPF line styles match.
GEOM_LINE_RE = re.compile(r"^- \{ ?'?geom'?:")
META_LINE_RE = re.compile(r"^- \{ ?'?meta'?:")
GEOM_FIELD_RE = {
    "id1": re.compile(r"[{,]\s*'?id1'?:\s*(-?\d+)\s*[,}]"),
    "ts0": re.compile(r"[{,]\s*'?ts0'?:\s*(-?\d+)\s*[,}]"),
    "g0": re.compile(r"[{,]\s*'?g0'?:\s*'?(-?\d+) (-?\d+) (-?\d+) (-?\d+)'?\s*[,}]"),
}

Box = tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class Event:
    clip: str
    activity_type: str
    start_frame: int
    end_frame: int
    label_source: str                  # kitware-eval | kitware-train | contrib
    sources: list[str]
    src_ids: list[str]
    src_status: str | None = None
    actor_types: list[str] | None = None
    boxes: list[dict[int, Box]] = field(default_factory=list)  # one {frame: box} per actor; may be empty


def load_kpf(path: Path) -> list[dict]:
    """A small KPF file (activities, types) as a list of records; yaml reads both line styles."""
    records = yaml.load(path.read_text(), Loader=yaml.CSafeLoader) or []
    for rec in records:
        if not isinstance(rec, dict) or len(rec) != 1:
            raise ValueError(f"{path.name}: KPF record is not a one-key mapping: {rec!r}")
    return records


def single(items: list, what: str, where: str):
    if len(items) != 1:
        raise ValueError(f"{where}: expected exactly one {what}, got {items!r}")
    return items[0]


def parse_kitware_activities(records: list[dict], known: set[str], where: str) -> tuple[list[dict], bool]:
    """Returns (activities, marked_empty). Each activity: name, start, end, id2, src_status, actors [(id1, s, e)]."""
    acts, empty = [], False
    for rec in records:
        (kind, body), = rec.items()
        if kind == "meta":
            empty |= isinstance(body, str) and bool(EMPTY_META_RE.search(body))
        elif kind == "act":
            name = single(list(body["act2"]), "activity name", where)
            if name == EMPTY_ACTIVITY:
                empty = True
                continue
            if name not in known:
                raise ValueError(f"{where}: unknown activity {name!r}")
            start, end = single(body["timespan"], "timespan", where)["tsr0"]
            actors = []
            for actor in body["actors"]:
                s, e = single(actor["timespan"], "actor timespan", where)["tsr0"]
                actors.append((actor["id1"], s, e))
            if not actors:
                raise ValueError(f"{where}: activity {body['id2']} ({name}) has no actors")
            acts.append({"name": name, "start": start, "end": end, "id2": body["id2"],
                         "src_status": body["src_status"], "actors": actors})
        else:
            raise ValueError(f"{where}: unexpected record kind {kind!r} in an activities file")
    if empty and acts:
        raise ValueError(f"{where}: marked {EMPTY_ACTIVITY} but has {len(acts)} activities")
    ids = [a["id2"] for a in acts]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{where}: duplicate activity ids")
    return acts, empty


def parse_kitware_types(records: list[dict], where: str) -> dict[int, str]:
    types = {}
    for rec in records:
        (kind, body), = rec.items()
        if kind == "meta":
            continue
        if kind != "types":
            raise ValueError(f"{where}: unexpected record kind {kind!r} in a types file")
        if body["id1"] in types:
            raise ValueError(f"{where}: two types records for id1 {body['id1']}")
        types[body["id1"]] = single(list(body["cset3"]), "object type", where)
    return types


def parse_geom_line(line: str) -> tuple[int, int, Box] | None:
    """(id1, frame, box) for a geom line, None for a meta line; raises on anything else."""
    if META_LINE_RE.match(line):
        return None
    if not GEOM_LINE_RE.match(line):
        raise ValueError(f"Not a geom or meta KPF line: {line[:120]!r}")
    found = {k: r.search(line) for k, r in GEOM_FIELD_RE.items()}
    missing = [k for k, m in found.items() if m is None]
    if missing:
        raise ValueError(f"Geom line without {missing}: {line[:120]!r}")
    return int(found["id1"].group(1)), int(found["ts0"].group(1)), tuple(int(v) for v in found["g0"].groups())


def read_geom(path: Path) -> dict[int, dict[int, Box]]:
    """{id1: {frame: box}} streamed line by line. Raises on a duplicate (id1, frame)."""
    tracks: dict[int, dict[int, Box]] = defaultdict(dict)
    with path.open() as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parsed = parse_geom_line(line)
            if parsed is None:
                continue
            id1, frame, box = parsed
            if frame in tracks[id1]:
                raise ValueError(f"{path.name}:{n}: second box for id1 {id1} at frame {frame}")
            tracks[id1][frame] = box
    return dict(tracks)


def kitware_events(clip: str, label_source: str, folder: str, acts: list[dict], types: dict[int, str],
                   tracks: dict[int, dict[int, Box]], n_frames: int) -> list[Event]:
    """Join activities, types and geom of one clip; every check failure raises."""
    actor_ids = [a[0] for act in acts for a in act["actors"]]
    if len(actor_ids) != len(set(actor_ids)):
        raise ValueError(f"{clip}: an actor id appears in more than one activity")
    if set(tracks) - set(actor_ids):
        raise ValueError(f"{clip}: geom tracks for ids that no activity uses: {sorted(set(tracks) - set(actor_ids))[:5]}")
    events = []
    for act in acts:
        if not 0 <= act["start"] <= act["end"] < n_frames:
            raise ValueError(f"{clip}: activity {act['id2']} span {act['start']}-{act['end']} outside 0..{n_frames - 1}")
        boxes, actor_types = [], []
        for id1, s, e in act["actors"]:
            if id1 not in types:
                raise ValueError(f"{clip}: actor {id1} has no types record")
            if not 0 <= s <= e < n_frames:
                raise ValueError(f"{clip}: actor {id1} span {s}-{e} outside 0..{n_frames - 1}")
            track = tracks.get(id1, {})
            if sorted(track) != list(range(s, e + 1)):
                raise ValueError(f"{clip}: geom of actor {id1} does not cover exactly frames {s}..{e} "
                                 f"({len(track)} boxes)")
            boxes.append({f: b for f, b in track.items() if act["start"] <= f <= act["end"]})
            actor_types.append(types[id1])
        events.append(Event(clip, act["name"], act["start"], act["end"], label_source, [folder], [str(act["id2"])],
                            act["src_status"], actor_types, boxes))
    return events


def parse_contrib_json(data: dict, clip: str, source: str, rename: dict[str, str], known: set[str],
                       n_frames: int) -> tuple[list[Event], int]:
    """Events of one contrib .json for one clip, and the number of zero-length records dropped."""
    avi = f"{clip}.avi"
    events, dropped = [], 0
    for act in data["activities"]:
        name = rename.get(act["activity"], act["activity"])
        if name not in known:
            raise ValueError(f"{source} {clip}: unknown activity {act['activity']!r}")
        only_this_clip(act["localization"], avi, f"{source} {clip} activity {act['activityID']}")
        loc = {int(k): v for k, v in act["localization"][avi].items()}
        if len(loc) == 1:  # a single key: zero-length record with no end (3 in UMD-IBM)
            dropped += 1
            continue
        if sorted(loc.values()) != [0, 1]:
            raise ValueError(f"{source} {clip}: activity {act['activityID']} localization {loc} is not {{start: 1, end: 0}}")
        start, end = (k for k, v in sorted(loc.items(), key=lambda kv: -kv[1]))
        if not 0 <= start <= end < n_frames:
            raise ValueError(f"{source} {clip}: activity {act['activityID']} span {start}-{end} outside 0..{n_frames - 1}")
        boxes = []
        for obj in act["objects"]:
            only_this_clip(obj["localization"], avi, f"{source} {clip} object {obj['objectID']}")
            frames = {int(k): v for k, v in obj["localization"][avi].items()}
            if sorted(frames) != list(range(start, end + 1)):
                raise ValueError(f"{source} {clip}: object {obj['objectID']} of activity {act['activityID']} "
                                 f"has boxes on {len(frames)} frames ({min(frames, default=None)}..{max(frames, default=None)}), "
                                 f"expected exactly the frames {start}..{end}")
            boxes.append({f: (b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"])
                          for f, b in ((f, v["boundingBox"]) for f, v in frames.items())})
        events.append(Event(clip, name, start, end, "contrib", [source], [f"{source}:{act['activityID']}"],
                            boxes=boxes))
    return events, dropped


def only_this_clip(localization: dict, avi: str, where: str) -> None:
    if list(localization) != [avi]:
        raise ValueError(f"{where}: localization for {list(localization)}, expected only {avi}")


def box_iou(a: Box, b: Box) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def best_iou(boxes_a: list[dict[int, Box]], frame_a: int, boxes_b: list[dict[int, Box]], frame_b: int) -> float:
    """Max IoU over actor pairs, box of A at frame_a vs box of B at frame_b; 0 if either has no box there."""
    a = [t[frame_a] for t in boxes_a if frame_a in t]
    b = [t[frame_b] for t in boxes_b if frame_b in t]
    return max((box_iou(x, y) for x in a for y in b), default=0.0)


def mean_box_iou(a: Event, b: Event) -> float:
    """Mean over frames where both events have a box of the best actor-pair IoU; 0 if no such frame."""
    frames = sorted(set().union(*a.boxes) & set().union(*b.boxes)) if a.boxes and b.boxes else []
    if not frames:
        return 0.0
    return sum(best_iou(a.boxes, f, b.boxes, f) for f in frames) / len(frames)


def find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def merge_contrib(events: list[Event], min_box_iou: float) -> list[Event]:
    """Merge contrib records of one clip that have the same type, overlap in time and mean box IoU >= min_box_iou.

    Applied within and across sources (UMD-IBM has exact duplicates; CMU-v2 and UMD-IBM label some events twice).
    Time overlap alone is not enough: two CMU vehicle_starts records overlapping 56% in time are different vehicles
    (box IoU 0). Merging is transitive (union-find). A merged event spans the union of its records.
    """
    if len({e.clip for e in events}) > 1:
        raise ValueError("merge_contrib works on the events of one clip")
    parent = list(range(len(events)))
    for i, a in enumerate(events):
        for j in range(i + 1, len(events)):
            b = events[j]
            if (a.activity_type == b.activity_type and a.start_frame <= b.end_frame and b.start_frame <= a.end_frame
                    and mean_box_iou(a, b) >= min_box_iou):
                parent[find(parent, j)] = find(parent, i)
    groups = defaultdict(list)
    for i in range(len(events)):
        groups[find(parent, i)].append(events[i])
    merged = []
    for members in groups.values():
        first = members[0]
        merged.append(Event(first.clip, first.activity_type, min(m.start_frame for m in members),
                            max(m.end_frame for m in members), "contrib",
                            sorted({s for m in members for s in m.sources}),
                            [i for m in members for i in m.src_ids],
                            boxes=[t for m in members for t in m.boxes]))
    return sorted(merged, key=lambda e: (e.start_frame, e.end_frame, e.activity_type))


@dataclass(frozen=True)
class Piece:
    """What the boundary linking needs from one stored event."""
    event_id: int
    activity_type: str
    start_frame: int
    end_frame: int
    first_boxes: tuple[Box, ...]   # boxes of all actors at start_frame
    last_boxes: tuple[Box, ...]    # boxes of all actors at end_frame


def piece(event_id: int, ev: Event) -> Piece:
    return Piece(event_id, ev.activity_type, ev.start_frame, ev.end_frame,
                 tuple(t[ev.start_frame] for t in ev.boxes if ev.start_frame in t),
                 tuple(t[ev.end_frame] for t in ev.boxes if ev.end_frame in t))


def boundary_links(pieces_a: list[Piece], n_frames_a: int, pieces_b: list[Piece], min_box_iou: float) -> list[tuple[int, int]]:
    """Link pieces of clip A that end on its last frame to pieces of the next clip B (same camera, back to back)
    that start on frame 0: same type and best actor-box IoU (A's last frame vs B's first) >= min_box_iou.
    One-to-one, greedy by IoU (ties by event ids). Returns (event_id_a, event_id_b) pairs."""
    candidates = []
    for a in pieces_a:
        if a.end_frame != n_frames_a - 1:
            continue
        for b in pieces_b:
            if b.start_frame != 0 or b.activity_type != a.activity_type:
                continue
            iou = max((box_iou(x, y) for x in a.last_boxes for y in b.first_boxes), default=0.0)
            if iou >= min_box_iou:
                candidates.append((-iou, a.event_id, b.event_id))
    links, used_a, used_b = [], set(), set()
    for _, ia, ib in sorted(candidates):
        if ia not in used_a and ib not in used_b:
            links.append((ia, ib))
            used_a.add(ia)
            used_b.add(ib)
    return links


def group_ids(event_ids: list[int], links: list[tuple[int, int]]) -> dict[int, int]:
    """{event_id: smallest event_id of its linked chain}."""
    index = {e: i for i, e in enumerate(event_ids)}
    parent = list(range(len(event_ids)))
    for a, b in links:
        ra, rb = find(parent, index[a]), find(parent, index[b])
        parent[max(ra, rb)] = min(ra, rb)
    return {e: event_ids[find(parent, index[e])] for e in event_ids}


def activity_names(meva_repo: Path) -> tuple[set[str], dict[str, str]]:
    """Known activity names (documents/activity-names.txt) and the old->new mapping used by contrib files."""
    known = set((meva_repo / "documents" / "activity-names.txt").read_text().split())
    rename = {}
    for line in (meva_repo / "documents" / "activity-name-mapping.csv").read_text().splitlines()[1:]:
        old, new = line.split(",")[:2]
        rename[old.strip()] = new.strip()
    unmapped = set(rename.values()) - known
    if unmapped:
        raise ValueError(f"activity-name-mapping.csv maps to unknown names: {sorted(unmapped)}")
    return known, rename


def contrib_files(contrib_root: Path, sources: list[str], clips: set[str]) -> dict[str, list[tuple[str, Path]]]:
    """{clip: [(source, <clip>.json)]} for the given clips. Raises if a source folder is missing or has a clip twice."""
    out = defaultdict(list)
    for source in sources:
        folder = contrib_root / source
        if not folder.is_dir():
            raise FileNotFoundError(f"Contrib source folder not found: {folder}")
        seen = set()
        for path in folder.rglob("*.json"):
            clip = path.name.removesuffix(".json")
            if clip in clips:
                if clip in seen:
                    raise ValueError(f"{source}: two .json files for {clip}")
                seen.add(clip)
                out[clip].append((source, path))
    return dict(out)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())
