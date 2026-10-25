"""Frame sampling from the original videos (decoded on the fly, never written to disk; CLAUDE.md 1.4).

Frame index rule (step 1.1): the i-th decoded frame (0-based) is frame i of the manifest; its clip time is i / fps.
"""

from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np


def keep_step(fps: float, rate: float) -> int:
    """Keep every k-th decoded frame to sample `rate` frames per second; k must be a whole number."""
    k = fps / rate
    if k != int(k) or k < 1:
        raise ValueError(f"{rate} fps is not a whole-frame step at {fps} fps")
    return int(k)


def scaled_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    """(width, height) scaled down to max_width, aspect kept, height even; never scaled up."""
    if max_width >= width:
        return width, height
    return max_width, 2 * round(height * max_width / width / 2)


def sampled_frames(path: Path, n_frames: int, step: int, max_width: int,
                   thread_type: str = "AUTO") -> Iterator[tuple[int, np.ndarray]]:
    """Decode every frame sequentially and yield (index, BGR uint8 array) for frames 0, step, 2*step, ...

    Frames are scaled down to max_width (aspect kept; never up). Raises at the end if the decoded frame count
    differs from the manifest's n_frames. AUTO (frame) threading: step 1.4 found it 2x faster than slice threading
    with identical frames."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = thread_type
        width, height = scaled_size(stream.codec_context.width, stream.codec_context.height, max_width)
        n = 0
        for n, frame in enumerate(container.decode(stream), start=1):
            if (n - 1) % step == 0:
                yield n - 1, frame.to_ndarray(format="bgr24", width=width, height=height)
    if n != n_frames:
        raise ValueError(f"{path.name}: decoded {n} frames, manifest says {n_frames}")
