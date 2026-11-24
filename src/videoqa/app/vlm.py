"""Video question answering models: an interface, InternVL3.5 (+ adapter) and a stub.

``InternVLVideoQA`` reuses the training/evaluation code paths on purpose --
``build_transform``, ``SampledClip``, ``FrameSampler`` and
``InternVLRunner.answer`` -- so the live assistant prompts the model exactly
as it was trained and scored (same preprocessing, "FrameK: <image>" prompt,
short-answer instruction, greedy decoding).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image

from videoqa.models.internvl import GenerationRecord


class VideoQAModel(ABC):
    num_frames: int = 8

    @abstractmethod
    def answer_frames(self, frames: list[Image.Image], question: str) -> GenerationRecord: ...

    @abstractmethod
    def sample_video(self, path: str) -> list[Image.Image]: ...

    @abstractmethod
    def describe(self) -> str: ...

    def warm_up(self) -> None:
        self.answer_frames([Image.new("RGB", (448, 448))] * self.num_frames, "what is in the video")


class InternVLVideoQA(VideoQAModel):
    def __init__(self, config_path: str, adapter_dir: str | None) -> None:
        import torch

        from videoqa.config import PROJECT_ROOT, load_config
        from videoqa.data.video import FrameSampler, build_transform
        from videoqa.models.internvl import InternVLRunner

        self._torch = torch
        self.config = load_config(config_path)
        base = InternVLRunner.load(self.config)
        model, self.adapter = base.model, None
        path = None
        if adapter_dir:
            path = Path(adapter_dir)
            path = path if path.is_absolute() else PROJECT_ROOT / path
            if not (path / "adapter_config.json").is_file():
                print(f"[warn] no adapter at {path}; using the base model", flush=True)
                path = None
        if path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(base.model, str(path), is_trainable=False).eval()
            self.adapter = path.parent.name if path.name in ("best", "last") else path.name
            model = model.base_model.model  # chat() lives on the inner InternVLChatModel
        self.runner = InternVLRunner(model, base.tokenizer, self.config)
        self.num_frames = self.config.video.num_frames
        self._transform = build_transform(self.config.video.input_size)
        self._sampler = FrameSampler(num_frames=self.num_frames, input_size=self.config.video.input_size,
                                     backend=self.config.video.backend)

    def answer_frames(self, frames: list[Image.Image], question: str) -> GenerationRecord:
        from videoqa.data.video import SampledClip

        pixels = self._torch.stack([self._transform(f.convert("RGB")) for f in frames])
        clip = SampledClip(pixels, [1] * len(frames), list(range(len(frames))), len(frames))
        return self.runner.answer(clip, question)

    def sample_video(self, path: str) -> list[Image.Image]:
        """Decode a file with the training sampler and return its frames as images."""
        from decord import VideoReader, cpu

        from videoqa.data.video import uniform_indices

        vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        idx = uniform_indices(len(vr), self.num_frames)
        size = self.config.video.input_size
        return [Image.fromarray(f).resize((size, size), Image.BICUBIC) for f in vr.get_batch(idx).asnumpy()]

    def describe(self) -> str:
        name = self.config.model.model_id.split("/")[-1]
        return f"{name} + {self.adapter}" if self.adapter else f"{name} (base)"


class StubVideoQA(VideoQAModel):
    """Stands in for InternVL so the UI and pipeline can be tested on a CPU node."""

    def answer_frames(self, frames, question):
        return GenerationRecord(f"stub answer about {len(frames)} frame(s)", 0.01)

    def sample_video(self, path):
        return [Image.new("RGB", (448, 448))] * self.num_frames

    def describe(self) -> str:
        return "stub model (UI test)"
