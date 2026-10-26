"""Event-branch intent and ranking (rewind/search/events.py) on toy cases."""

import numpy as np
import pandas as pd
import pytest

from rewind.search.core import SearchConfig
from rewind.search.events import event_clips, intent

KW = {"vehicle_stops": ["stop", "stops", "stopped", "pulls over", "park"],
      "vehicle_starts": ["start", "drives off", "pulls away", "leaves"]}
BOTH = ["vehicle_stops", "vehicle_starts"]


def test_intent_whole_words_and_phrases():
    assert intent("Some car pulled up and STOPPED by the roundabout.", KW, BOTH) == ["vehicle_stops"]
    assert intent("A car drives off.", KW, BOTH) == ["vehicle_starts"]
    assert intent("A car pulling away on the far road.", KW, BOTH) == []        # no stemming
    assert intent("An unstoppable truck", KW, BOTH) == []                        # inside a word
    assert intent("A van pulls over, then drives off", KW, BOTH) == BOTH
    assert intent("A van pulls  over", KW, BOTH) == []                           # phrase needs the exact spacing
    assert intent("parked car", KW, BOTH) == []                                  # 'park' is not 'parked'


def test_intent_only_enabled_types():
    assert intent("A car drives off.", KW, ["vehicle_stops"]) == []
    with pytest.raises(ValueError):
        intent("x", KW, ["vehicle_turns"])


def test_event_clips_filters_type_split_and_modality():
    cfg = SearchConfig(k=10, merge_gap_s=5, max_clip_s=60, context_s=5)
    q = np.array([1.0, 0.0])
    vecs = np.array([[1.0, 0.0], [0.8, 0.6], [0.6, 0.8], [1.0, 0.0], [0.0, 1.0]], dtype=np.float16)
    meta = pd.DataFrame({"video_id": [1, 1, 2, 3, 1], "t_s": [100.0, 103.0, 50.0, 10.0, 200.0],
                         "event_type": ["vehicle_stops", "vehicle_stops", "vehicle_stops", "vehicle_stops",
                                        "vehicle_starts"],
                         "modality": ["rgb", "rgb", "ir", "rgb", "rgb"]})
    durations = {1: 300.0, 2: 300.0, 3: 300.0}
    # stops in videos {1, 2}: rows 0 + 1 (video 1, 3 s apart -> one clip, hits 1.0 and 0.8 -> mean 0.9,
    # span [95, 108]) and row 2 (video 2, 0.6). Video 3 is outside the split; row 4 is a start.
    clips = event_clips(q, vecs, meta, ["vehicle_stops"], {1, 2}, False, durations, cfg)
    assert [(c.video_id, c.start_s, c.end_s) for c in clips] == [(1, 95.0, 108.0), (2, 45.0, 55.0)]
    assert clips[0].score == pytest.approx(0.9, abs=1e-3)
    # RGB only drops the IR event of video 2
    assert [c.video_id for c in event_clips(q, vecs, meta, ["vehicle_stops"], {1, 2}, True, durations, cfg)] == [1]
    # no candidates -> empty list (the caller then shows the appearance list only)
    assert event_clips(q, vecs, meta, ["vehicle_starts"], {2}, False, durations, cfg) == []


def test_event_index_puts_event_clips_first_only_with_intent():
    """scripts/event_gate.EventIndex: ON = event clips (real event_clips on toy vectors) + the OFF list; without
    intent the OFF list unchanged. The OFF ranker is a fixed list (its own logic is tested in test_search_core)."""
    from datetime import datetime

    from rewind.eval.metrics import Span
    from rewind.ingest.manifest import Video
    from scripts.event_gate import EventIndex

    v = Video(1, "c", "p", "A", "rgb", datetime(2018, 3, 7, 10, 0, 0), 9000, 30.0)
    off_list = [Span("A", datetime(2018, 3, 7, 10, 2), datetime(2018, 3, 7, 10, 2, 30))]

    class FixedOff:
        by_id, durations, footage_s = {1: v}, {1: 300.0}, 300.0

        def ranked(self, q, cfg, rgb_only, query=None):
            return list(off_list)

    meta = pd.DataFrame({"video_id": [1], "t_s": [100.0], "event_type": ["vehicle_stops"], "modality": ["rgb"]})
    idx = EventIndex(FixedOff(), np.array([[1.0, 0.0]], np.float16), meta, KW, BOTH)
    cfg = SearchConfig(k=10, merge_gap_s=5, max_clip_s=60, context_s=5)
    q = np.array([1.0, 0.0])
    got = idx.ranked(q, cfg, False, {"query_id": "x", "query": "A car stops."})
    # event at 100 s -> clip [95, 105] s = 10:01:35 .. 10:01:45, then the OFF list
    assert got == [Span("A", datetime(2018, 3, 7, 10, 1, 35), datetime(2018, 3, 7, 10, 1, 45))] + off_list
    assert idx.ranked(q, cfg, False, {"query_id": "y", "query": "A red car."}) == off_list
    assert idx.n_event_clips == {"x": 1, "y": 0}
    with pytest.raises(ValueError):
        idx.ranked(q, cfg, False)
