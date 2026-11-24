"""Tests for the live-camera frame buffer (no GPU, no browser)."""

from __future__ import annotations

import numpy as np
import pytest

from videoqa.app.frames import FrameBuffer


def _frame(v: int) -> np.ndarray:
    return np.full((240, 320, 3), v, dtype=np.uint8)


def test_frames_are_resized_to_model_input() -> None:
    b = FrameBuffer(min_interval_s=0)
    b.push(_frame(10), t=0.0)
    sel = b.select(1, now=0.0)
    assert sel.frames[0].size == (448, 448)


def test_select_returns_requested_count_in_time_order() -> None:
    b = FrameBuffer(window_s=10, min_interval_s=0)
    for i in range(20):
        b.push(_frame(i), t=i * 0.5)          # 0.0 .. 9.5 s
    sel = b.select(8, now=9.5)
    assert len(sel.frames) == 8
    values = [np.asarray(f)[0, 0, 0] for f in sel.frames]
    assert values == sorted(values), "frames must stay in temporal order"


def test_select_ignores_frames_outside_window() -> None:
    b = FrameBuffer(window_s=2.0, min_interval_s=0)
    for i in range(10):
        b.push(_frame(i), t=float(i))         # 0 .. 9 s
    sel = b.select(4, now=9.0)
    assert max(sel.timestamps) <= 2.0
    assert {int(np.asarray(f)[0, 0, 0]) for f in sel.frames} <= {7, 8, 9}


def test_few_frames_repeat_like_training_sampler() -> None:
    b = FrameBuffer(min_interval_s=0)
    b.push(_frame(1), t=0.0); b.push(_frame(2), t=1.0)
    sel = b.select(8, now=1.0)
    assert len(sel.frames) == 8               # prompt always gets 8 image slots


def test_empty_buffer_raises() -> None:
    with pytest.raises(LookupError):
        FrameBuffer().select(8)


def test_too_frequent_frames_are_dropped() -> None:
    b = FrameBuffer(min_interval_s=0.2)
    assert b.push(_frame(1), t=0.0)
    assert not b.push(_frame(2), t=0.1)
    assert b.push(_frame(3), t=0.25)
    assert len(b) == 2


def test_memory_is_bounded() -> None:
    b = FrameBuffer(max_frames=5, min_interval_s=0)
    for i in range(50):
        b.push(_frame(i), t=float(i))
    assert len(b) == 5


def test_none_frame_is_ignored() -> None:
    assert FrameBuffer().push(None) is False
