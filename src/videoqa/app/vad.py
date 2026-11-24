"""Hands-free voice input: voice activity detection + end-of-utterance detection.

The browser streams microphone audio continuously. ``SileroVAD`` scores
every 32 ms window (512 samples @ 16 kHz) with a speech probability, and
``Endpointer`` turns that stream of probabilities into whole utterances:

    idle --(speech prob > start_threshold)--> speaking
    speaking --(prob < end_threshold for silence_ms)--> emit utterance, idle

* Hysteresis (start 0.5 / end 0.35) stops a single quiet syllable from
  ending the utterance.
* ``pre_roll_ms`` of audio before the onset is kept, so the first word is
  not clipped (detection fires ~60 ms after speech actually starts).
* Utterances shorter than ``min_speech_ms`` are dropped as noise (coughs,
  clicks); ``max_utterance_s`` caps a runaway recording.
* ``mute(seconds)``: while the assistant is speaking its answer, the mic
  would hear the loudspeaker and "ask" the answer back to itself. The
  assistant mutes the endpointer for the answer's duration plus a margin.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

SAMPLE_RATE = 16000
WINDOW = 512  # samples per VAD decision (32 ms) -- fixed by Silero for 16 kHz


class SpeechScorer(Protocol):
    def __call__(self, window: np.ndarray) -> float: ...


class SileroVAD:
    """Silero VAD v6 (MIT). Weights ship inside the pip package, so it works offline."""

    def __init__(self) -> None:
        import torch
        from silero_vad import load_silero_vad

        torch.set_num_threads(1)  # ~4 ms per window on one core; leave the rest alone
        self._torch = torch
        self._model = load_silero_vad(onnx=False)

    def __call__(self, window: np.ndarray) -> float:
        return float(self._model(self._torch.from_numpy(window), SAMPLE_RATE).item())

    def reset(self) -> None:
        self._model.reset_states()


@dataclass(slots=True)
class EndpointerConfig:
    start_threshold: float = 0.5
    end_threshold: float = 0.35
    silence_ms: int = 800
    min_speech_ms: int = 250
    pre_roll_ms: int = 300
    max_utterance_s: float = 15.0
    mute_margin_s: float = 0.6


class Endpointer:
    """Feed 16 kHz mono float32 chunks of any size; get back finished utterances."""

    def __init__(self, scorer: SpeechScorer, config: EndpointerConfig | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.scorer = scorer
        self.cfg = config or EndpointerConfig()
        self._clock = clock
        ms_to_windows = lambda ms: max(1, round(ms / 1000 * SAMPLE_RATE / WINDOW))  # noqa: E731
        self._silence_windows = ms_to_windows(self.cfg.silence_ms)
        self._min_windows = ms_to_windows(self.cfg.min_speech_ms)
        self._max_windows = ms_to_windows(self.cfg.max_utterance_s * 1000)
        self._pre_roll: deque[np.ndarray] = deque(maxlen=ms_to_windows(self.cfg.pre_roll_ms))
        self._pending = np.zeros(0, dtype=np.float32)
        self._muted_until = 0.0
        self._reset_utterance()

    # -- state -------------------------------------------------------------
    def _reset_utterance(self) -> None:
        self._speaking = False
        self._windows: list[np.ndarray] = []
        self._speech_windows = 0
        self._silent_run = 0

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def muted(self) -> bool:
        return self._clock() < self._muted_until

    def mute(self, seconds: float) -> None:
        """Ignore input for ``seconds`` (+ margin), e.g. while the answer is playing."""
        self._muted_until = self._clock() + seconds + self.cfg.mute_margin_s
        self._reset_utterance()
        self._pre_roll.clear()
        self._pending = np.zeros(0, dtype=np.float32)
        if hasattr(self.scorer, "reset"):
            self.scorer.reset()

    # -- streaming ---------------------------------------------------------
    def feed(self, chunk: np.ndarray) -> list[np.ndarray]:
        """Consume audio; return any utterances that ended inside this chunk."""
        if self.muted:
            return []
        audio = np.concatenate([self._pending, np.asarray(chunk, dtype=np.float32)])
        n_full = len(audio) // WINDOW
        self._pending = audio[n_full * WINDOW:]
        done: list[np.ndarray] = []
        for i in range(n_full):
            utterance = self._step(audio[i * WINDOW:(i + 1) * WINDOW])
            if utterance is not None:
                done.append(utterance)
        return done

    def _step(self, window: np.ndarray) -> np.ndarray | None:
        p = self.scorer(window)
        if not self._speaking:
            if p >= self.cfg.start_threshold:
                self._speaking = True
                self._windows = list(self._pre_roll) + [window]
                self._speech_windows, self._silent_run = 1, 0
            else:
                self._pre_roll.append(window)
            return None

        self._windows.append(window)
        if p >= self.cfg.end_threshold:
            self._speech_windows += 1
            self._silent_run = 0
        else:
            self._silent_run += 1

        ended = self._silent_run >= self._silence_windows
        too_long = len(self._windows) >= self._max_windows
        if not (ended or too_long):
            return None
        keep = self._speech_windows >= self._min_windows
        # trim the trailing silence that triggered the end, keep ~1 window of tail
        tail = max(0, self._silent_run - 1)
        frames = self._windows[: len(self._windows) - tail] if tail else self._windows
        utterance = np.concatenate(frames) if keep else None
        self._reset_utterance()
        self._pre_roll.clear()
        return utterance
