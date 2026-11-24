"""The orchestrator: visual input + question (typed or spoken) -> answer + speech.

Framework-free on purpose: the Gradio UI is a thin layer over this class,
so the whole pipeline can be exercised in tests without a browser.

    camera --> FrameBuffer --\\
    image -------------------+--> VideoQAModel --> answer --> SpeechSynthesizer
    video file --------------/          ^
    mic --> Endpointer --> SpeechRecognizer --> question
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from PIL import Image

from videoqa.app.asr import SpeechRecognizer
from videoqa.app.audio import resample, to_mono_float
from videoqa.app.frames import FrameBuffer
from videoqa.app.tts import Speech, SpeechSynthesizer
from videoqa.app.vad import SAMPLE_RATE, Endpointer
from videoqa.app.vlm import VideoQAModel


class Source(str, Enum):
    CAMERA = "Live camera"
    IMAGE = "Image"
    VIDEO = "Video file"


class InputError(LookupError):
    """A user-fixable problem (no frames yet, nothing uploaded, empty question)."""


@dataclass(slots=True)
class Reply:
    question: str
    answer: str
    speech: Speech
    frames: list[Image.Image]
    timings: dict[str, float] = field(default_factory=dict)
    detail: str = ""


class MultimodalAssistant:
    def __init__(self, vqa: VideoQAModel, asr: SpeechRecognizer, tts: SpeechSynthesizer,
                 frames: FrameBuffer, endpointer: Endpointer | None = None) -> None:
        self.vqa, self.asr, self.tts = vqa, asr, tts
        self.frames = frames
        self.endpointer = endpointer

    # -- visual input --------------------------------------------------------
    def push_camera_frame(self, frame) -> int:
        self.frames.push(frame)
        return len(self.frames)

    def _visual(self, source: Source, image, video) -> tuple[list[Image.Image], str]:
        if source is Source.CAMERA:
            sel = self.frames.select(self.vqa.num_frames)   # raises LookupError if empty
            return sel.frames, f"{len(sel.frames)} frames over the last {sel.window_s:.1f}s"
        if source is Source.IMAGE:
            if image is None:
                raise InputError("upload an image first")
            img = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
            return [img.convert("RGB")], "1 image"
        if not video:
            raise InputError("upload a video first")
        frames = self.vqa.sample_video(video)
        return frames, f"{len(frames)} frames sampled from the video"

    # -- questions -----------------------------------------------------------
    def ask(self, question: str, source: Source, image=None, video=None,
            asr_seconds: float | None = None) -> Reply:
        question = (question or "").strip()
        if not question:
            raise InputError("ask something first (type it, or just speak)")
        try:
            frames, detail = self._visual(source, image, video)
        except LookupError as e:
            raise InputError(str(e)) from e
        rec = self.vqa.answer_frames(frames, question)
        answer = rec.text or "(no answer)"
        speech = self.tts.speak(answer)
        if self.endpointer is not None and speech.duration:
            self.endpointer.mute(speech.duration)   # don't hear our own answer
        timings = {"vlm": rec.latency_s, "tts": speech.seconds}
        if asr_seconds is not None:
            timings = {"asr": asr_seconds, **timings}
        return Reply(question, answer, speech, frames, timings, detail)

    def transcribe(self, sample_rate: int, audio: np.ndarray):
        return self.asr.transcribe(sample_rate, audio)

    def hear(self, sample_rate: int, chunk: np.ndarray) -> list[np.ndarray]:
        """Hands-free mode: feed a mic chunk, get back finished 16 kHz utterances."""
        if self.endpointer is None:
            return []
        sr, mono = to_mono_float(sample_rate, chunk)
        return self.endpointer.feed(resample(mono, sr, SAMPLE_RATE))

    def ask_spoken(self, utterance_16k: np.ndarray, source: Source, image=None, video=None) -> Reply:
        heard = self.asr.transcribe(SAMPLE_RATE, utterance_16k)
        return self.ask(heard.text, source, image, video, asr_seconds=heard.seconds)

    # -- lifecycle -----------------------------------------------------------
    def warm_up(self) -> dict[str, float]:
        """First calls are slow (kernel compile, decoder JIT): pay that cost at startup.

        Measured cold vs warm on an A40: VLM 5.2 s -> 1.1 s, ASR 25.9 s -> 0.06 s.
        """
        out = {}
        for name, part in (("vlm", self.vqa), ("asr", self.asr), ("tts", self.tts)):
            t = time.perf_counter()
            part.warm_up()
            out[name] = time.perf_counter() - t
        return out
