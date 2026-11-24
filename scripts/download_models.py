#!/usr/bin/env python3
"""Pre-download every model the app uses (needed on offline GPU nodes).

    python scripts/download_models.py            # into ./hf_cache and ./models/tts

Hugging Face models (InternVL3.5, Nemotron ASR, Magpie TTS) go to ./hf_cache;
the optional Kokoro CPU TTS files come from the kokoro-onnx GitHub release.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_MODELS = [
    ("OpenGVLab/InternVL3_5-8B", "9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79"),
    ("nvidia/nemotron-speech-streaming-en-0.6b", None),
    ("nvidia/magpie_tts_multilingual_357m", None),
]
KOKORO = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"


def main() -> int:
    os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
    from huggingface_hub import snapshot_download

    for repo, rev in HF_MODELS:
        print(f"-> {repo}", flush=True)
        snapshot_download(repo, revision=rev,
                          cache_dir=str(ROOT / "hf_cache") if repo.startswith("OpenGVLab") else None,
                          allow_patterns=None if repo.startswith("OpenGVLab") else ["*.nemo", "*.json", "*.md"])
    out = ROOT / "models/tts"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
        if not (out / name).is_file():
            print(f"-> Kokoro {name}", flush=True)
            urllib.request.urlretrieve(KOKORO + name, out / name)
    print("all models ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
