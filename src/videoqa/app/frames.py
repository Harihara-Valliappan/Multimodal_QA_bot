"""A rolling buffer that turns a live webcam stream into a model-ready clip.

The model was trained on 8 frames sampled at segment midpoints of a clip
(``videoqa.data.video.uniform_indices``). For a live camera there is no
"clip", so we keep the last ``window_s`` seconds of frames and, when a
question arrives, select 8 of them with the *same* midpoint rule over
that window. The demo therefore feeds the model input that is shaped the
way it was trained, rather than an arbitrary handful of frames.

Frames are stored already resized to the model's input size, so memory is
bounded (``max_frames`` x 448 x 448 x 3 bytes, ~36 MB at the default).
Thread-safe: Gradio pushes frames and reads clips from different workers.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image

from videoqa.data.video import uniform_indices


@dataclass(slots=True)
class FrameSelection:
    frames: list[Image.Image]
    timestamps: list[float]   # seconds before "now", oldest first
    window_s: float           # time actually covered by the selected frames


class FrameBuffer:
    def __init__(self, *, window_s: float = 8.0, max_frames: int = 60,
                 input_size: int = 448, min_interval_s: float = 0.2) -> None:
        if window_s <= 0 or max_frames < 1:
            raise ValueError("window_s must be > 0 and max_frames >= 1")
        self.window_s = window_s
        self.input_size = input_size
        self.min_interval_s = min_interval_s
        self._frames: deque[tuple[float, Image.Image]] = deque(maxlen=max_frames)
        self._lock = threading.Lock()

    def push(self, frame: np.ndarray | Image.Image, t: float | None = None) -> bool:
        """Add one RGB frame. Returns False if it was dropped as too soon."""
        if frame is None:
            return False
        t = time.monotonic() if t is None else t
        with self._lock:
            if self._frames and t - self._frames[-1][0] < self.min_interval_s:
                return False
        img = frame if isinstance(frame, Image.Image) else Image.fromarray(np.asarray(frame).astype(np.uint8))
        img = img.convert("RGB").resize((self.input_size, self.input_size), Image.BICUBIC)
        with self._lock:
            self._frames.append((t, img))
        return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    def select(self, num_frames: int = 8, now: float | None = None) -> FrameSelection:
        """Pick ``num_frames`` from the last ``window_s`` seconds, midpoint rule.

        Fewer stored frames than requested is not an error: frames repeat,
        exactly as ``uniform_indices`` does for short training clips, so the
        prompt always carries ``num_frames`` image slots.
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            recent = [(t, f) for t, f in self._frames if now - t <= self.window_s]
        if not recent:
            raise LookupError("no camera frames in the last %.1f s" % self.window_s)
        idx = uniform_indices(len(recent), num_frames)
        chosen = [recent[i] for i in idx]
        return FrameSelection(
            frames=[f for _, f in chosen],
            timestamps=[round(now - t, 2) for t, _ in chosen],
            window_s=round(chosen[-1][0] - chosen[0][0], 2),
        )
