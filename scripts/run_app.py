#!/usr/bin/env python3
"""Launch the live multimodal video-QA assistant.

    python scripts/run_app.py                               # Magpie TTS on GPU, hands-free voice
    python scripts/run_app.py --tts kokoro                  # fastest speech (CPU TTS)
    python scripts/run_app.py --push-to-talk                # no VAD: record/release
    python scripts/run_app.py --stub-vlm --device cpu       # UI test without the 8B model

On the cluster use slurm/05_app.sbatch and open it through an SSH tunnel at
http://localhost:7860 (a localhost origin is what lets the browser grant
camera and microphone access).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from videoqa.app.config import AppConfig  # noqa: E402


def parse_args(argv=None) -> AppConfig:
    # Defaults come from an *instance*: AppConfig is a slots dataclass, so the
    # class attribute AppConfig.model_config is a slot descriptor, not a value.
    d = AppConfig()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-c", "--model-config", default=d.model_config)
    p.add_argument("--adapter", default=d.adapter, help="'none' for the base model")
    p.add_argument("--stub-vlm", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tts", choices=["magpie", "kokoro"], default="magpie")
    p.add_argument("--voice", default=None, help="Magpie: Aria/Jason/John/Leo/Sofia; Kokoro: e.g. af_heart")
    p.add_argument("--push-to-talk", action="store_true", help="disable hands-free VAD")
    p.add_argument("--window", type=float, default=8.0, help="seconds of live video per question")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--auth-file", default=None, help="file containing user:password")
    a = p.parse_args(argv)
    return AppConfig(model_config=a.model_config, adapter=None if a.adapter in (None, "none") else a.adapter,
                     stub_vlm=a.stub_vlm, device=a.device, tts=a.tts, voice=a.voice,
                     hands_free=not a.push_to_talk, window_s=a.window, host=a.host, port=a.port,
                     auth_file=a.auth_file)


def build_assistant(cfg: AppConfig):
    """Construct every component from the config (dependency injection, one place)."""
    from videoqa.app.asr import NemotronASR
    from videoqa.app.assistant import MultimodalAssistant
    from videoqa.app.frames import FrameBuffer
    from videoqa.app.tts import KokoroTTS, MagpieTTS
    from videoqa.app.vad import Endpointer, SileroVAD
    from videoqa.app.vlm import InternVLVideoQA, StubVideoQA
    from videoqa.config import PROJECT_ROOT

    log = lambda msg: print(msg, flush=True)  # noqa: E731
    t = time.perf_counter()
    vqa = StubVideoQA() if cfg.stub_vlm else InternVLVideoQA(cfg.model_config, cfg.adapter)
    log(f"VLM  ready: {vqa.describe()} ({time.perf_counter() - t:.0f}s)")
    asr = NemotronASR(device=cfg.device)
    log(f"ASR  ready: {asr.MODEL_ID}")
    if cfg.tts == "magpie":
        tts = MagpieTTS(voice=cfg.voice or "Sofia", device=cfg.device)
    else:
        tts = KokoroTTS(PROJECT_ROOT / "models/tts", voice=cfg.voice or "af_heart")
    log(f"TTS  ready: {tts.name}")
    endpointer = Endpointer(SileroVAD(), cfg.endpointer) if cfg.hands_free else None
    log(f"VAD  {'on (Silero, hands-free)' if endpointer else 'off (push-to-talk)'}")
    assistant = MultimodalAssistant(vqa, asr, tts, FrameBuffer(window_s=cfg.window_s), endpointer)
    warm = assistant.warm_up()
    log("warm-up done: " + ", ".join(f"{k} {v:.1f}s" for k, v in warm.items()))
    return assistant


def main(argv=None) -> int:
    try:
        cfg = parse_args(argv)          # validates host/auth before any model loads
    except ValueError as e:
        raise SystemExit(str(e)) from e
    auth = None
    if cfg.auth_file:
        user, _, pw = Path(cfg.auth_file).read_text().strip().partition(":")
        auth = (user, pw)
    from videoqa.app.ui import AssistantUI

    AssistantUI(build_assistant(cfg), hands_free=cfg.hands_free).launch(cfg.host, cfg.port, auth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
