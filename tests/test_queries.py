"""Step 2.2 worksheet plan: quotas, IR slots, distinct events, type spread, specificity thirds, determinism."""

import random
from collections import Counter

import pytest

from rewind.eval.queries import Candidate, pick_event, plan_split

CFG = {"min_category_instances": 2, "ir_qtypes": ["vehicle_event", "action"],
       "specificity": ["vague", "moderate", "detailed"],
       "category": {"dev": {"vehicle_types": ["vehicle_stops"], "n_other": 2}},
       "specific": {"dev": {"vehicle_event": 2, "action": 2, "appearance": 2, "ir": 2}}}


def pool() -> list[Candidate]:
    """RGB: 3 events of each of 4 person types and of vehicle_starts / vehicle_stops; IR: 2 of each of 3 types."""
    out, i = [], 0
    for kind in ["p_a", "p_b", "p_c", "p_d", "vehicle_starts", "vehicle_stops"]:
        for _ in range(3):
            out.append(Candidate(i, kind, "G1", "rgb"))
            i += 1
    for kind in ["vehicle_starts", "vehicle_stops", "p_ir"]:
        for _ in range(2):
            out.append(Candidate(i, kind, "G9", "ir"))
            i += 1
    return out


def plan(seed: int = 1):
    cands = pool()
    counts = Counter(c.activity_type for c in cands if c.modality == "rgb")  # as scripts.make_worksheets
    return plan_split("dev", cands, counts, CFG, random.Random(seed)), cands


def test_quotas_ids_and_ir_slots():
    slots, cands = plan()
    by_id = {c.event_id: c for c in cands}
    # 1 vehicle + 2 other categories, 6 specific -> 9 slots; ids dev-c01..c03, dev-s01..s06
    assert [s.query_id for s in slots] == ["dev-c01", "dev-c02", "dev-c03"] + [f"dev-s0{i}" for i in range(1, 7)]
    assert Counter((s.kind, s.qtype) for s in slots) == Counter(
        {("category", "vehicle_event"): 1, ("category", "action"): 2, ("specific", "vehicle_event"): 2,
         ("specific", "action"): 2, ("specific", "appearance"): 2})
    # ir 2 with ir_qtypes [vehicle_event, action]: one IR vehicle_event, one IR action; event modality matches slot
    assert Counter((s.qtype, s.modality) for s in slots if s.modality == "ir") == Counter(
        {("vehicle_event", "ir"): 1, ("action", "ir"): 1})
    assert all(by_id[s.event_id].modality == s.modality for s in slots)
    assert all(s.modality == "rgb" for s in slots if s.kind == "category")


def test_events_distinct_vehicle_rules_and_type_spread():
    slots, _ = plan()
    assert len({s.event_id for s in slots}) == len(slots)
    vehicle = [s for s in slots if s.kind == "specific" and s.qtype == "vehicle_event"]
    assert sorted(s.activity_type for s in vehicle) == ["vehicle_starts", "vehicle_stops"]  # i = 0, 1 alternate
    assert all(s.activity_type not in ("vehicle_starts", "vehicle_stops")
               for s in slots if s.kind == "specific" and s.qtype != "vehicle_event")
    # appearance draws 2 RGB events from 4 unused types: never the same type twice
    appearance = [s.activity_type for s in slots if s.qtype == "appearance"]
    assert len(set(appearance)) == 2
    assert slots[0].activity_type == "vehicle_stops" and slots[0].kind == "category"


def test_specificity_in_equal_thirds_and_seed_reproducible():
    slots, _ = plan()
    assert Counter(s.specificity for s in slots) == Counter({"vague": 3, "moderate": 3, "detailed": 3})
    assert plan(1)[0] == slots
    assert any(plan(s)[0] != slots for s in range(2, 10))


def test_pick_event_refuses_when_nothing_is_left():
    with pytest.raises(ValueError, match="no unused"):
        pick_event(random.Random(0), [Candidate(1, "x", "G1", "rgb")], set(), {1})


def test_category_type_needs_enough_instances():
    cands = pool()
    counts = Counter(c.activity_type for c in cands if c.modality == "rgb")
    counts["vehicle_stops"] = 1  # below min_category_instances = 2
    with pytest.raises(ValueError, match="vehicle_stops"):
        plan_split("dev", cands, counts, CFG, random.Random(1))


@pytest.mark.parametrize("query, modality, problem", [
    ("A man carrying a box toward the gym.", "rgb", None),                 # "box" is a witness word
    ("A car stops by the light poles.", "ir", None),                        # "light" is not a colour here
    ("Someone seen on camera 3 opening a door.", "rgb", "banned word 'camera'"),
    ("person_opens_trunk near the lot.", "rgb", "banned word '_'"),
    ("A white SUV pulls away.", "ir", "colour/brightness word 'white'"),
    ("A white SUV pulls away.", "rgb", None),                               # colours are fine on RGB
    ("  ", "rgb", "empty query"),
])
def test_text_problems(query, modality, problem):
    from rewind.eval.queries import text_problems
    found = text_problems({"query": query, "modality": modality})
    assert (found == []) if problem is None else any(problem in p for p in found)
