"""MultimodalAssistant end to end with fake models: fast, deterministic, no GPU."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from videoqa.app.asr import SpeechRecognizer, Transcript
from videoqa.app.assistant import InputError, MultimodalAssistant, Source
from videoqa.app.frames import FrameBuffer
from videoqa.app.tts import SpeechSynthesizer
from videoqa.app.vad import SAMPLE_RATE, Endpointer, EndpointerConfig
from videoqa.app.vlm import StubVideoQA


class FakeASR(SpeechRecognizer):
    def __init__(self, text="what is the person doing"):
        self.text, self.calls = text, 0

    def transcribe(self, sample_rate, audio):
        self.calls += 1
        return Transcript(self.text, 0.01)


class FakeTTS(SpeechSynthesizer):
    name = "fake tts"

    def __init__(self, seconds_of_audio=2.0):
        self.n = int(seconds_of_audio * 22050)

    def _synthesize(self, text):
        return np.zeros(self.n, np.float32), 22050


class Loud:
    def __call__(self, w):
        return 0.9 if np.abs(w).max() > 0.1 else 0.05


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def make(hands_free=True, clock=None):
    ep = Endpointer(Loud(), EndpointerConfig(mute_margin_s=0.5), clock=clock or Clock()) if hands_free else None
    return MultimodalAssistant(StubVideoQA(), FakeASR(), FakeTTS(), FrameBuffer(min_interval_s=0), ep)


def speech(seconds, loud, rate=48000):
    return (np.full(int(seconds * rate), 16000 if loud else 0, np.int16))   # browser-style int16


def test_camera_question_uses_eight_buffered_frames():
    a = make()
    for i in range(10):
        a.frames.push(np.full((240, 320, 3), i, np.uint8))  # real timestamps: "recent"
    r = a.ask("what is happening", Source.CAMERA)
    assert len(r.frames) == 8 and "8 frames" in r.detail
    assert r.speech.duration == pytest.approx(2.0, abs=0.01)


def test_image_and_video_sources():
    a = make()
    assert len(a.ask("what is this", Source.IMAGE, image=Image.new("RGB", (64, 64))).frames) == 1
    assert len(a.ask("what happens", Source.VIDEO, video="clip.mp4").frames) == 8


@pytest.mark.parametrize("source,kw,msg", [
    (Source.CAMERA, {}, "no camera frames"),
    (Source.IMAGE, {}, "upload an image"),
    (Source.VIDEO, {}, "upload a video"),
])
def test_missing_inputs_raise_user_errors(source, kw, msg):
    with pytest.raises(InputError, match=msg):
        make().ask("q", source, **kw)


def test_empty_question_rejected():
    with pytest.raises(InputError, match="ask something"):
        make().ask("  ", Source.IMAGE, image=Image.new("RGB", (8, 8)))


def test_hands_free_speech_becomes_a_spoken_reply():
    a = make()
    a.frames.push(np.zeros((64, 64, 3), np.uint8), t=0.0)
    chunks = np.array_split(np.concatenate([speech(0.5, False), speech(1.2, True), speech(1.0, False)]), 12)
    utterances = [u for c in chunks for u in a.hear(48000, c)]
    assert len(utterances) == 1
    a.frames.push(np.zeros((64, 64, 3), np.uint8))       # fresh frame "now"
    reply = a.ask_spoken(utterances[0], Source.CAMERA)
    assert reply.question == "what is the person doing"
    assert set(reply.timings) == {"asr", "vlm", "tts"}


def test_assistant_does_not_hear_its_own_answer():
    # The loop the mute exists to prevent: answer plays -> mic hears it -> new "question".
    clock = Clock()
    a = make(clock=clock)
    a.frames.push(np.zeros((64, 64, 3), np.uint8))
    a.ask("q", Source.CAMERA)                           # 2.0 s answer -> muted until 2.5 s
    echo = np.concatenate([speech(1.5, True), speech(1.0, False)])
    assert a.hear(48000, echo) == [], "the loudspeaker echo must be ignored"
    clock.t = 3.0                                       # answer finished
    assert len(a.hear(48000, echo)) == 1, "listening must resume afterwards"


def test_push_to_talk_mode_has_no_endpointer():
    a = make(hands_free=False)
    assert a.hear(48000, speech(1.0, True)) == []
    assert a.transcribe(48000, speech(1.0, True)).text == "what is the person doing"
