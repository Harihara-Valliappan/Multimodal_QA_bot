"""Tests for audio conversion (pure numpy; ASR/TTS models are tested on hardware)."""

from __future__ import annotations

import numpy as np

from videoqa.app.audio import to_mono_float


def test_stereo_int16_becomes_mono_float_in_range() -> None:
    # Regression: averaging channels first produced float64 and skipped scaling.
    stereo = np.stack([np.full(1000, 32767, np.int16), np.full(1000, -32767, np.int16)], axis=1)
    sr, a = to_mono_float(48000, stereo)
    assert sr == 48000 and a.dtype == np.float32 and a.shape == (1000,)
    assert np.abs(a).max() <= 1.0


def test_mono_int16_full_scale_maps_to_one() -> None:
    _, a = to_mono_float(16000, np.full(10, 32767, np.int16))
    assert np.allclose(a, 1.0)


def test_float_input_passes_through() -> None:
    x = np.linspace(-0.5, 0.5, 100, dtype=np.float32)
    _, a = to_mono_float(16000, x)
    assert np.allclose(a, x)


def test_random_stereo_stays_in_range() -> None:
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((4800, 2)) * 8000).clip(-32767, 32767).astype(np.int16)
    _, a = to_mono_float(48000, x)
    assert np.abs(a).max() <= 1.0
