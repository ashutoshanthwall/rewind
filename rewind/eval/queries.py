"""Query worksheets (step 2.2): which events get a query, of which kind / qtype / specificity, fixed by seed.

Everything is decided before any keyframe is looked at or any query is written (config section `queries`), so the
writer cannot pick easy events or save the vague wording for hard ones.

  - Category queries (many correct answers: every activity of the type in the split): per split the configured
    vehicle types plus n_other random other types with >= min_category_instances eligible activities. qtype is
    vehicle_event for vehicle_starts / vehicle_stops, action otherwise. A random instance is the keyframe example.
  - Specific queries (one correct answer: one activity): per split, qtype quotas. IR slots (ir of them, qtypes
    taken in turn from ir_qtypes) use third-party-labelled IR events; the rest use Kitware RGB events.
    vehicle_event = vehicle_starts / vehicle_stops (alternating); action and appearance = any other type. Within a
    pool the type is drawn uniformly over types not yet used in the split (then the event uniformly), so frequent
    types (person_talks_to_person) don't crowd out the rest.
  - Specificity (vague / moderate / detailed) in equal thirds per split, shuffled by the seed.

Eligible event: the activity's only piece (not cut at a clip edge, so event_id == group_id), at least min_event_s
long, with boxes, and not boundary-flagged (rewind.eval.splits.split_event_ids leaves those out).
"""

import csv
import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path

from rewind.eval.splits import check_split_access
from rewind.ingest.annotations import VEHICLE_START_STOP

KINDS = ("category", "specific")
QTYPES = ("appearance", "action", "vehicle_event")  # CLAUDE.md 1.10


@dataclass(frozen=True)
class Candidate:
    event_id: int
    activity_type: str
    camera_id: str
    modality: str        # 'rgb' (Kitware labels) or 'ir' (third-party labels)


@dataclass(frozen=True)
class Slot:
    query_id: str
    split: str
    kind: str
    qtype: str
    specificity: str
    modality: str
    event_id: int        # specific: the one correct answer; category: an example instance for the keyframe
    activity_type: str


def pick_event(rng: random.Random, pool: list[Candidate], used_types: set[str], used_events: set[int],
               types: tuple[str, ...] | None = None) -> Candidate:
    """A random event: type uniformly among the pool's types not yet in used_types (all types once exhausted), then
    an unused event of that type uniformly. `types` restricts the type (e.g. one vehicle type)."""
    free = [c for c in pool if c.event_id not in used_events and (types is None or c.activity_type in types)]
    if not free:
        raise ValueError(f"no unused eligible event (types {types}); pool of {len(pool)}")
    all_types = sorted({c.activity_type for c in free})
    fresh = [t for t in all_types if t not in used_types]
    kind = rng.choice(fresh or all_types)
    return rng.choice([c for c in free if c.activity_type == kind])


def replacement(slot: Slot, candidates: list[Candidate], used_events: set[int], seed: int) -> Candidate:
    """A new event for a slot whose event turned out unusable (subject not visible, label clearly wrong): same
    modality and activity type, an event not used anywhere, drawn by a seed fixed per query id so the writer
    cannot choose it."""
    rng = random.Random(f"{seed}-replace-{slot.query_id}")
    return pick_event(rng, [c for c in candidates if c.modality == slot.modality], set(), used_events,
                      (slot.activity_type,))


def plan_split(split: str, candidates: list[Candidate], counts: dict[str, int], cfg: dict,
               rng: random.Random) -> list[Slot]:
    """Slots of one split. candidates: its eligible events. counts: activity_type -> eligible RGB activities
    (category queries are scored on RGB labels; IR labels are incomplete, recall only)."""
    prefix = split[:3]
    used_events: set[int] = set()
    rows: list[tuple] = []  # (kind, qtype, modality, candidate)

    cat = cfg["category"][split]
    others = sorted(t for t, n in counts.items() if n >= cfg["min_category_instances"] and t not in VEHICLE_START_STOP)
    n_other = cat["n_other"]
    if len(others) < n_other:
        raise ValueError(f"{split}: {len(others)} eligible category types, {n_other} wanted")
    for kind in list(cat["vehicle_types"]) + rng.sample(others, n_other):
        if counts.get(kind, 0) < cfg["min_category_instances"]:
            raise ValueError(f"{split}: category {kind} has {counts.get(kind, 0)} eligible activities")
        example = pick_event(rng, [c for c in candidates if c.modality == "rgb"], set(), used_events, (kind,))
        used_events.add(example.event_id)
        rows.append(("category", "vehicle_event" if kind in VEHICLE_START_STOP else "action", "rgb", example))

    spec = cfg["specific"][split]
    ir_slots = [cfg["ir_qtypes"][i % len(cfg["ir_qtypes"])] for i in range(spec["ir"])]
    for qtype in QTYPES:
        n_ir = ir_slots.count(qtype)
        if n_ir > spec[qtype]:
            raise ValueError(f"{split}: {n_ir} IR {qtype} slots but only {spec[qtype]} {qtype} queries")
        used_types: set[str] = set()
        for i in range(spec[qtype]):
            modality = "ir" if i < n_ir else "rgb"
            pool = [c for c in candidates if c.modality == modality
                    and (c.activity_type in VEHICLE_START_STOP) == (qtype == "vehicle_event")]
            types = (VEHICLE_START_STOP[i % 2],) if qtype == "vehicle_event" else None
            c = pick_event(rng, pool, used_types, used_events, types)
            used_types.add(c.activity_type)
            used_events.add(c.event_id)
            rows.append(("specific", qtype, modality, c))

    levels = cfg["specificity"]
    specificity = [levels[i % len(levels)] for i in range(len(rows))]
    rng.shuffle(specificity)
    slots, n = [], {"category": 0, "specific": 0}
    for (kind, qtype, modality, c), level in zip(rows, specificity):
        n[kind] += 1
        slots.append(Slot(f"{prefix}-{kind[0]}{n[kind]:02d}", split, kind, qtype, level, modality, c.event_id,
                          c.activity_type))
    return slots


# --- frozen query files (step 2.2 part 2) -----------------------------------------------------------------------

FROZEN_COLUMNS = ["query_id", "split", "kind", "qtype", "specificity", "modality", "query", "activity_type",
                  "event_id", "clip", "src_ids", "camera_id", "label_source", "start_time", "end_time", "start_s",
                  "end_s"]
# Ground truth: specific -> the one activity (event_id; clip + src_ids re-resolve it if event ids are rebuilt);
# category -> every activity of activity_type in the query's split, so its event columns are left empty.
EVENT_COLUMNS = FROZEN_COLUMNS[8:]
# Words a witness would not use, or that leak the label / the system (checked at freezing).
# "box" alone is allowed: a witness may see someone carry a box.
BANNED = re.compile(r"_|\b(camera|footage|video|frame|cctv|thermal|infrared|ir|bounding|timestamp|"
                    r"photo|image|g\d{3})\b", re.IGNORECASE)
# "light" is left out: "light poles" is not a colour.
COLOURS = re.compile(r"\b(white|black|red|blue|green|yellow|orange|grey|gray|silver|brown|tan|maroon|pink|purple|"
                     r"beige|dark|bright)\b", re.IGNORECASE)


def text_problems(row: dict) -> list[str]:
    """Wording rules a frozen query must meet (CLAUDE.md 1.10 + the witness-style rules of Gate 2.2)."""
    q, out = row["query"], []
    if not q.strip():
        out.append("empty query")
    if BANNED.search(q):
        out.append(f"banned word {BANNED.search(q).group(0)!r}")
    if row["modality"] == "ir" and COLOURS.search(q):
        out.append(f"colour/brightness word {COLOURS.search(q).group(0)!r} on an IR (thermal) query")
    return out


def sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_hashes(queries_dir: Path) -> dict[str, str]:
    """{file name: sha256} from HASHES (sha256sum format: '<hash>  <name>')."""
    out = {}
    for line in (queries_dir / "HASHES").read_text().splitlines():
        digest, name = line.split("  ", 1)
        out[name] = digest
    return out


def check_frozen(queries_dir: Path) -> None:
    """Raise if any file listed in HASHES changed or is missing (queries are never edited after Gate 2.2)."""
    for name, digest in read_hashes(queries_dir).items():
        path = queries_dir / name
        if not path.exists():
            raise FileNotFoundError(f"frozen file {path} is missing")
        if sha256(path) != digest:
            raise ValueError(f"frozen file {path} changed since Gate 2.2 (sha256 differs from HASHES)")


def load_queries(queries_dir: Path, split: str, allow_test: bool = False) -> list[dict]:
    """The frozen queries of one split, after checking every frozen file's hash. Test refused unless allow_test
    (step 8.2 only)."""
    check_split_access(split, allow_test)
    check_frozen(queries_dir)
    with open(queries_dir / f"queries_{split}.csv", newline="") as f:
        return list(csv.DictReader(f))
