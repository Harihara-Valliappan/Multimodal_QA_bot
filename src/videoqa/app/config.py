"""Every setting of the live assistant in one typed place."""

from __future__ import annotations

from dataclasses import dataclass, field

from videoqa.app.vad import EndpointerConfig


@dataclass(slots=True)
class AppConfig:
    # vision-language model
    model_config: str = "configs/variant_monteclora.yaml"
    adapter: str | None = "adapters/monteclora"   # shipped in the repo (30 MB)
    stub_vlm: bool = False
    # live camera
    window_s: float = 8.0             # seconds of live video behind each question
    # speech
    device: str = "cuda"
    tts: str = "magpie"               # "magpie" (NVIDIA, GPU) or "kokoro" (CPU, fastest)
    voice: str | None = None          # default per engine: Sofia / af_heart
    hands_free: bool = True           # VAD auto-submit; push-to-talk otherwise
    endpointer: EndpointerConfig = field(default_factory=EndpointerConfig)
    # server
    host: str = "127.0.0.1"
    port: int = 7860
    auth_file: str | None = None

    def __post_init__(self) -> None:
        if self.tts not in {"magpie", "kokoro"}:
            raise ValueError("tts must be 'magpie' or 'kokoro'")
        if self.host != "127.0.0.1" and not self.auth_file:
            # Bound to the node's network interface, the app is reachable by
            # other cluster users; refuse to run it without a password.
            raise ValueError("host is not localhost: an auth_file (user:password) is required")
