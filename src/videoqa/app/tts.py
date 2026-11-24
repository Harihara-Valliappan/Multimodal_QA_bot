"""Speech synthesis: an interface, NVIDIA Magpie (GPU) and Kokoro (CPU)."""

from __future__ import annotations

import logging
import os
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class Speech:
    audio: np.ndarray | None   # mono float32
    sample_rate: int
    seconds: float             # synthesis time

    @property
    def duration(self) -> float:
        return 0.0 if self.audio is None else len(self.audio) / self.sample_rate

    def as_gradio(self):
        return None if self.audio is None else (self.sample_rate, self.audio)


class SpeechSynthesizer(ABC):
    name: str = "tts"

    @abstractmethod
    def _synthesize(self, text: str) -> tuple[np.ndarray, int]: ...

    def speak(self, text: str) -> Speech:
        if not text.strip():
            return Speech(None, 22050, 0.0)
        t = time.perf_counter()
        audio, sr = self._synthesize(text)
        return Speech(np.asarray(audio, dtype=np.float32), int(sr), time.perf_counter() - t)

    def warm_up(self) -> None:
        self.speak("Warming up.")


class MagpieTTS(SpeechSynthesizer):
    """NVIDIA Magpie TTS Multilingual 357M (Nemotron Speech family), on the GPU.

    Measured on an A40: ~2.3 s for a 3 s sentence once warm (RTF ~0.8),
    3.8 GB of GPU memory. English voices: Aria, Jason, John, Leo, Sofia.
    """

    MODEL_ID = "nvidia/magpie_tts_multilingual_357m"
    SPEAKERS = {"Aria": 0, "Jason": 1, "John": 2, "Leo": 3, "Sofia": 4}

    def __init__(self, voice: str = "Sofia", device: str = "cuda") -> None:
        if voice not in self.SPEAKERS:
            raise ValueError(f"Magpie voice must be one of {sorted(self.SPEAKERS)}")
        warnings.filterwarnings("ignore")
        from nemo.collections.tts.models import MagpieTTSModel
        from nemo.utils import logging as nemo_logging

        nemo_logging.setLevel(logging.ERROR)
        self.model = MagpieTTSModel.from_pretrained(self.MODEL_ID).eval().to(device)
        self.speaker = self.SPEAKERS[voice]
        self.sample_rate = int(getattr(self.model, "sample_rate", 22050))
        self.name = f"Magpie TTS ({voice})"

    def _synthesize(self, text: str) -> tuple[np.ndarray, int]:
        import torch

        with torch.inference_mode():
            audio, length = self.model.do_tts(text, language="en", apply_TN=False, speaker_index=self.speaker)
        return audio[0, : int(length[0])].float().cpu().numpy(), self.sample_rate


class KokoroTTS(SpeechSynthesizer):
    """Kokoro-82M via ONNX on CPU: the low-latency option (~0.8 s per sentence)."""

    def __init__(self, model_dir: Path, voice: str = "af_heart", threads: int | None = None) -> None:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        # onnxruntime sizes its pool from the machine's physical cores (52 on the
        # GPU nodes), not the 8 cores Slurm gives the job: 52 threads on 8 cores
        # made synthesis ~10x slower (9.6 s vs 0.92 s; scripts/bench_tts_threads.py).
        # Use the job's CPU affinity; 4 threads measured fastest (0.81 s).
        available = len(os.sched_getaffinity(0))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or min(4, max(1, available - 1))
        opts.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model_dir / "kokoro-v1.0.onnx"), sess_options=opts,
                                       providers=["CPUExecutionProvider"])
        self.engine = Kokoro.from_session(session, str(model_dir / "voices-v1.0.bin"))
        self.voice = voice
        self.threads = opts.intra_op_num_threads
        self.name = f"Kokoro-82M ({voice})"

    def _synthesize(self, text: str) -> tuple[np.ndarray, int]:
        audio, sr = self.engine.create(text, voice=self.voice, speed=1.0, lang="en-us")
        return audio, sr
