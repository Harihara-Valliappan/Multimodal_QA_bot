"""Tests for frame sampling.

The index maths is pure and tested exhaustively; one test decodes a real
clip from the manifest, because the only way to know decord agrees with
us about frame counts is to ask it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from videoqa.config import PROJECT_ROOT
from videoqa.data.video import (
    FrameSampler,
    VideoDecodeError,
    build_transform,
    build_video_prompt,
    uniform_indices,
)


@pytest.mark.parametrize("total,n", [(100, 8), (8, 8), (3, 8), (1, 8), (1000, 16), (7, 1)])
def test_indices_are_in_range_and_the_right_count(total, n) -> None:
    idx = uniform_indices(total, n)
    assert len(idx) == n
    assert all(0 <= i < total for i in idx)


def test_indices_are_nondecreasing() -> None:
    idx = uniform_indices(500, 8)
    assert idx == sorted(idx), "frames must be sampled in temporal order"


def test_indices_are_deterministic() -> None:
    assert uniform_indices(997, 8) == uniform_indices(997, 8)


def test_indices_avoid_the_very_first_and_last_frame_on_long_clips() -> None:
    # Segment midpoints, not endpoints: clip edges are often black frames.
    idx = uniform_indices(1000, 8)
    assert idx[0] > 0
    assert idx[-1] < 999


def test_short_clip_repeats_rather_than_failing() -> None:
    # A 3-frame clip must still yield 8 entries, or the prompt's image
    # token count would no longer match the pixel tensor.
    assert len(uniform_indices(3, 8)) == 8


def test_zero_frames_is_an_error() -> None:
    with pytest.raises(VideoDecodeError):
        uniform_indices(0, 8)


def test_prompt_has_one_placeholder_per_frame() -> None:
    prompt = build_video_prompt("what colour is the car", 8)
    assert prompt.count("<image>") == 8
    assert prompt.endswith("what colour is the car")


def test_transform_outputs_normalized_chw_tensor() -> None:
    from PIL import Image

    tf = build_transform(448)
    out = tf(Image.new("RGB", (640, 360), "red"))
    assert out.shape == (3, 448, 448)
    assert out.dtype.is_floating_point


def test_missing_file_raises_decode_error(tmp_path) -> None:
    sampler = FrameSampler(num_frames=8, input_size=448)
    with pytest.raises(VideoDecodeError, match="missing file"):
        sampler.load(tmp_path / "nope.mp4")


def test_garbage_file_raises_decode_error(tmp_path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video")
    sampler = FrameSampler(num_frames=8, input_size=448)
    with pytest.raises(VideoDecodeError):
        sampler.load(bad)


def test_unsupported_backend_is_rejected() -> None:
    with pytest.raises(NotImplementedError):
        FrameSampler(backend="ffmpeg")


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data/manifests/val.jsonl").is_file(),
    reason="manifests not built",
)
def test_decodes_a_real_clip_to_the_configured_shape() -> None:
    row = json.loads((PROJECT_ROOT / "data/manifests/val.jsonl").read_text().splitlines()[0])
    path = PROJECT_ROOT / row["video_path"]
    if not path.is_file():
        pytest.skip("video not present")
    clip = FrameSampler(num_frames=8, input_size=448).load(path)
    assert clip.pixel_values.shape == (8, 3, 448, 448)
    assert clip.num_patches_list == [1] * 8
    assert len(clip.frame_indices) == 8
    assert clip.total_frames > 0
