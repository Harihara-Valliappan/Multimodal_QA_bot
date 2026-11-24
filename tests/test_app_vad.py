"""Endpointer tests with a scripted scorer: deterministic, no model, no microphone."""

from __future__ import annotations

import numpy as np
import pytest

from videoqa.app.vad import SAMPLE_RATE, WINDOW, Endpointer, EndpointerConfig


class Scripted:
    """Speech probability from the window's amplitude: loud = speech."""

    def __call__(self, window: np.ndarray) -> float:
        return 0.9 if np.abs(window).max() > 0.1 else 0.05


def seconds(s: float, loud: bool) -> np.ndarray:
    n = int(s * SAMPLE_RATE)
    return np.full(n, 0.5 if loud else 0.0, dtype=np.float32)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def ep(**kw) -> Endpointer:
    return Endpointer(Scripted(), EndpointerConfig(**kw), clock=Clock())


def test_one_utterance_after_trailing_silence() -> None:
    e = ep(silence_ms=800)
    out = e.feed(np.concatenate([seconds(1, False), seconds(1.5, True), seconds(1, False)]))
    assert len(out) == 1
    dur = len(out[0]) / SAMPLE_RATE
    assert 1.5 <= dur <= 1.5 + 0.3 + 0.1, dur  # speech + pre-roll + one tail window


def test_no_utterance_until_silence_long_enough() -> None:
    e = ep(silence_ms=800)
    assert e.feed(np.concatenate([seconds(1.0, True), seconds(0.5, False)])) == []
    assert e.speaking, "a 0.5 s pause must not end the utterance"
    assert len(e.feed(seconds(0.5, False))) == 1


def test_short_pause_inside_sentence_is_kept_together() -> None:
    e = ep(silence_ms=800)
    audio = np.concatenate([seconds(0.8, True), seconds(0.4, False), seconds(0.8, True), seconds(1.0, False)])
    out = e.feed(audio)
    assert len(out) == 1 and len(out[0]) / SAMPLE_RATE >= 2.0


def test_chunking_does_not_change_the_result() -> None:
    audio = np.concatenate([seconds(0.5, False), seconds(1.2, True), seconds(1.0, False)])
    whole = ep().feed(audio)
    e = ep(); parts = []
    for chunk in np.array_split(audio, 37):  # odd sizes, not multiples of 512
        parts += e.feed(chunk)
    assert len(whole) == len(parts) == 1 and np.array_equal(whole[0], parts[0])


def test_pre_roll_keeps_audio_before_onset() -> None:
    e = ep(pre_roll_ms=300)
    out = e.feed(np.concatenate([seconds(1.0, False), seconds(1.0, True), seconds(1.0, False)]))
    assert np.abs(out[0][:WINDOW]).max() == 0.0, "utterance should start with pre-roll silence"


def test_too_short_blip_is_dropped() -> None:
    e = ep(min_speech_ms=250)
    assert e.feed(np.concatenate([seconds(0.1, True), seconds(1.0, False)])) == []


def test_max_length_forces_a_cut() -> None:
    e = ep(max_utterance_s=2.0)
    out = e.feed(seconds(5.0, True))
    assert len(out) >= 2 and all(len(u) / SAMPLE_RATE <= 2.0 + 1e-6 for u in out)


def test_mute_ignores_input_then_recovers() -> None:
    clock = Clock()
    e = Endpointer(Scripted(), EndpointerConfig(mute_margin_s=0.5), clock=clock)
    e.mute(2.0)  # muted until t = 2.5
    assert e.feed(np.concatenate([seconds(1, True), seconds(1, False)])) == []
    clock.t = 3.0
    assert not e.muted
    assert len(e.feed(np.concatenate([seconds(1, True), seconds(1, False)]))) == 1


def test_mute_discards_a_half_spoken_utterance() -> None:
    clock = Clock()
    e = Endpointer(Scripted(), EndpointerConfig(), clock=clock)
    e.feed(seconds(1.0, True))
    assert e.speaking
    e.mute(0.1)
    assert not e.speaking


@pytest.mark.parametrize("bad", [np.zeros(0, np.float32), np.zeros(100, np.float32)])
def test_tiny_chunks_are_buffered_not_dropped(bad) -> None:
    assert ep().feed(bad) == []
