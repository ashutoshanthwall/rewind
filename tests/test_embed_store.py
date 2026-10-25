"""variant_mask (rewind/embed/store.py): which stored rows each crop_source / frame_mode variant searches."""

import pandas as pd
import pytest

from rewind.embed.store import variant_mask

ROWS = [  # video, frame, type, view, crop_source, crop_rank, dup
    (1, 0, "frame", "tile0", "", -1, False),        # 0
    (1, 0, "frame", "tile1", "", -1, False),        # 1
    (1, 0, "frame", "pad", "", -1, False),          # 2
    (1, 0, "crop", "", "detection", 0, False),      # 3
    (1, 0, "crop", "", "motion", 0, True),          # 4  same object as the detection
    (1, 0, "crop", "", "motion", 1, False),         # 5
    (1, 0, "crop", "", "motion", 2, False),         # 6
    (1, 30, "frame", "tile0", "", -1, False),       # 7
    (1, 30, "frame", "tile1", "", -1, False),       # 8
    (1, 30, "frame", "pad", "", -1, False),         # 9
    (1, 30, "crop", "", "detection", 0, False),     # 10
    (1, 30, "crop", "", "detection", 1, False),     # 11
    (1, 30, "crop", "", "motion", 0, False),        # 12
]
META = pd.DataFrame(ROWS, columns=["video_id", "frame", "type", "view", "crop_source", "crop_rank", "dup"])


def rows(frame_mode, crop_source, max_crops):
    return [i for i, k in enumerate(variant_mask(META, frame_mode, crop_source, max_crops)) if k]


def test_frame_modes_and_no_crops():
    assert rows("tiles2", "none", 2) == [0, 1, 7, 8]
    assert rows("pad", "none", 2) == [2, 9]


def test_motion_variant_takes_top_motion_crops_including_dups():
    assert rows("pad", "motion", 2) == [2, 4, 5, 9, 12]


def test_detection_then_motion_fills_with_non_dup_motion():
    # frame 0: 1 detection + 1 fill slot -> motion rank 1 (rank 0 is a dup); frame 30: 2 detections fill both slots
    assert rows("pad", "detection_then_motion", 2) == [2, 3, 5, 9, 10, 11]
    # 3 slots: frame 0 gets motion ranks 1 and 2; frame 30 gets motion rank 0
    assert rows("pad", "detection_then_motion", 3) == [2, 3, 5, 6, 9, 10, 11, 12]


def test_unknown_variant_refused():
    with pytest.raises(ValueError, match="frame_mode"):
        variant_mask(META, "tiles3", "none", 2)
    with pytest.raises(ValueError, match="crop_source"):
        variant_mask(META, "pad", "detections", 2)
