"""Small, dependency-light audio helpers shared by VAD, ASR and TTS."""

from __future__ import annotations

import numpy as np


def to_mono_float(sample_rate: int, audio: np.ndarray) -> tuple[int, np.ndarray]:
    """Browser audio (sr, int16, maybe stereo) -> (sr, mono float32 in [-1, 1]).

    Integers are scaled *before* channels are averaged: ``mean()`` returns
    float64, which would otherwise skip the scaling and leave int16-sized
    values (~1e4) -- a bug caught in testing.
    """
    a = np.asarray(audio)
    if np.issubdtype(a.dtype, np.integer):
        a = a.astype(np.float32) / np.iinfo(a.dtype).max
    if a.ndim == 2:
        a = a.mean(axis=1)
    return int(sample_rate), a.astype(np.float32)


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample a mono float32 signal (no-op when the rates already match)."""
    if src_rate == dst_rate or audio.size == 0:
        return audio.astype(np.float32)
    import torch
    import torchaudio.functional as F

    return F.resample(torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)),
                      src_rate, dst_rate).numpy()
