# Vendored from ByteTrack yolox/tracker/basetrack.py (https://github.com/FoundationVision/ByteTrack, commit d1bf019), MIT,
# Copyright (c) 2021 Yifu Zhang: full text in LICENSE. MODIFIED by Rewind (lines marked "Rewind"): see NOTICE.md.
import numpy as np
from collections import OrderedDict


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    # Rewind: no global id counter (BaseTrack._count / next_id removed); ids come from each BYTETracker, so two trackers
    # (two clips) never share state.

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed
