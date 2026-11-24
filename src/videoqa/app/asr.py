"""Speech recognition: an interface plus the NVIDIA Nemotron implementation."""

from __future__ import annotations

import logging
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from videoqa.app.audio import resample, to_mono_float


@dataclass(slots=True)
class Transcript:
    text: str
    seconds: float


class SpeechRecognizer(ABC):
    sample_rate: int = 16000

    @abstractmethod
    def transcribe(self, sample_rate: int, audio: np.ndarray) -> Transcript: ...

    def warm_up(self) -> None:
        """The first calls compile kernels (25.9 s cold vs 0.06 s warm on an A40)."""
        for _ in range(2):
            self.transcribe(self.sample_rate, np.zeros(self.sample_rate, dtype=np.float32))


class NemotronASR(SpeechRecognizer):
    """``nvidia/nemotron-speech-streaming-en-0.6b`` (cache-aware FastConformer + RNNT).

    Whole utterances from the endpointer are transcribed with
    ``att_context_size=[70, 13]`` -- 1.12 s chunks, the model card's most
    accurate setting (6.93% average WER). Punctuation and capitalisation
    come out natively, so the text goes straight to the VLM.
    """

    MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
    ATT_CONTEXT = [70, 13]

    def __init__(self, device: str = "cuda") -> None:
        warnings.filterwarnings("ignore")
        import nemo.collections.asr as nemo_asr
        from nemo.utils import logging as nemo_logging

        nemo_logging.setLevel(logging.ERROR)
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.MODEL_ID).eval().to(device)
        self.model.encoder.set_default_att_context_size(self.ATT_CONTEXT)
        self.sample_rate = int(self.model.cfg.preprocessor.sample_rate)  # 16000, read from the model

    def transcribe(self, sample_rate: int, audio: np.ndarray) -> Transcript:
        sr, a = to_mono_float(sample_rate, audio)
        if a.size < sr * 0.2:
            return Transcript("", 0.0)  # under 200 ms: nothing was said
        a = resample(a, sr, self.sample_rate)
        t = time.perf_counter()
        out = self.model.transcribe([a], verbose=False)
        text = out[0].text if hasattr(out[0], "text") else str(out[0])
        return Transcript(text.strip(), time.perf_counter() - t)
