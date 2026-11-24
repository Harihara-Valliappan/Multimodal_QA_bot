"""Typed configuration objects for every stage of the experiment.

Why dataclasses instead of passing raw dicts around:

* a typo in a YAML key raises at load time, not three hours into training;
* defaults live in one readable place;
* ``ProjectConfig.to_dict()`` is dumped next to every result, which is
  what makes a run reproducible.

Nothing here imports torch or transformers, so the config can be loaded
on a login node without a GPU.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repository root = three levels above this file (src/videoqa/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or inconsistent."""


@dataclass(slots=True)
class PathsConfig:
    """Every filesystem location the project touches.

    Relative paths are resolved against the repository root so a config
    behaves the same whether it is launched from the repo root or from a
    Slurm working directory.
    """

    data_root: str = "data"
    annotations_dir: str = "data/activitynet_qa/annotations"
    videos_dir: str = "data/activitynet_qa/videos"
    manifests_dir: str = "data/manifests"
    reports_dir: str = "reports"
    checkpoints_dir: str = "checkpoints"
    hf_cache_dir: str = "hf_cache"

    def resolve(self, key: str) -> Path:
        """Return an absolute :class:`Path` for one of the fields above."""
        if not hasattr(self, key):
            raise ConfigError(f"Unknown path key: {key!r}")
        value = Path(getattr(self, key)).expanduser()
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    def all_resolved(self) -> dict[str, Path]:
        return {f.name: self.resolve(f.name) for f in dataclasses.fields(self)}


@dataclass(slots=True)
class ModelConfig:
    """Which pretrained checkpoint we adapt, pinned exactly.

    ``revision`` must be a commit SHA before any result is reported: a
    moving branch name makes the experiment unreproducible.
    """

    model_id: str = "OpenGVLab/InternVL3_5-8B"
    revision: str = "main"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    attn_implementation: str | None = None
    max_new_tokens: int = 32
    do_sample: bool = False

    def __post_init__(self) -> None:
        allowed = {"bfloat16", "float16", "float32"}
        if self.torch_dtype not in allowed:
            raise ConfigError(f"torch_dtype must be one of {sorted(allowed)}, got {self.torch_dtype!r}")


@dataclass(slots=True)
class VideoConfig:
    """Deterministic frame sampling policy.

    Determinism matters more than cleverness here: the base model and the
    adapted model must see byte-identical frames, or the comparison in
    Milestone 5 measures the sampler instead of the adapter.
    """

    num_frames: int = 8
    max_tiles_per_frame: int = 1
    input_size: int = 448
    sampling: str = "uniform"
    backend: str = "decord"
    min_decodable_frames: int = 1

    def __post_init__(self) -> None:
        if self.num_frames < 1:
            raise ConfigError("num_frames must be >= 1")
        if self.sampling != "uniform":
            raise ConfigError(f"Only deterministic 'uniform' sampling is supported, got {self.sampling!r}")
        if self.backend not in {"decord", "pyav", "opencv"}:
            raise ConfigError(f"Unsupported video backend: {self.backend!r}")


@dataclass(slots=True)
class DoraConfig:
    """PEFT settings. ``use_dora=True`` is the whole point of the project.

    The brief forbids a silent fallback to plain LoRA, so this flag is
    asserted at runtime rather than merely requested.
    """

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_dora: bool = True
    # The only way to run a plain-LoRA arm. It must be set explicitly in a
    # separately named config, so a run can never end up as LoRA by
    # accident and then get written up as DoRA. A2 in the brief ("is it
    # DoRA specifically, or just fine-tuning?") is what this exists for.
    allow_lora_control: bool = False
    # Optional LoRA variant from PEFT >= 0.21 ("kasa", "monteclora",
    # "bdlora", "velora"), plus its options. Variants are LoRA-family, so
    # they require use_dora=False and allow_lora_control=True like the
    # plain control. None means plain LoRA (or DoRA, per use_dora).
    lora_variant: str | None = None
    # PEFT tuner family. "lora" covers LoRA, DoRA and the LoRA variants;
    # "shira" (sparse high-rank) and "peanut" (non-linear, weight-aware)
    # are separate families with their own parameter names. They also
    # require use_dora=False + allow_lora_control=True.
    tuner: str = "lora"
    variant_options: dict[str, Any] = field(default_factory=dict)
    bias: str = "none"
    # Deliberately None, not "CAUSAL_LM". PEFT's PeftModelForCausalLM
    # wrapper passes ``inputs_embeds`` into forward(), which
    # InternVLChatModel.forward() does not accept -- it takes
    # ``pixel_values`` and ``image_flags`` and builds the embeddings
    # itself. The generic PeftModel passes through only the kwargs we
    # supply, which is what this architecture needs. Verified on the
    # real checkpoint: job 42947 failed with CAUSAL_LM, 42949 passed
    # without it. Do not "fix" this back.
    task_type: str | None = None
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    freeze_vision_encoder: bool = True
    freeze_projector: bool = True

    def __post_init__(self) -> None:
        if not self.use_dora and not self.allow_lora_control:
            raise ConfigError(
                "use_dora=False requested without allow_lora_control=True. This project "
                "compares DoRA against the base checkpoint; run a LoRA control through a "
                "separate explicitly named config that sets allow_lora_control: true."
            )
        if self.allow_lora_control and self.use_dora:
            raise ConfigError(
                "allow_lora_control=True with use_dora=True is contradictory: the flag "
                "exists only to permit the plain-LoRA control arm."
            )
        allowed_variants = {None, "kasa", "monteclora", "bdlora", "velora"}
        if self.lora_variant not in allowed_variants:
            raise ConfigError(f"lora_variant must be one of {sorted(v for v in allowed_variants if v)} or null")
        if self.lora_variant and (self.use_dora or not self.allow_lora_control):
            raise ConfigError(
                f"lora_variant={self.lora_variant!r} is a LoRA-family method: set use_dora: false "
                "and allow_lora_control: true"
            )
        if self.tuner not in {"lora", "shira", "peanut"}:
            raise ConfigError(f"tuner must be lora, shira or peanut, got {self.tuner!r}")
        if self.tuner != "lora" and (self.use_dora or not self.allow_lora_control or self.lora_variant):
            raise ConfigError(
                f"tuner={self.tuner!r} needs use_dora: false, allow_lora_control: true, no lora_variant"
            )
        if self.variant_options and not self.lora_variant and self.tuner == "lora":
            raise ConfigError("variant_options given without a lora_variant")
        if not self.target_modules:
            raise ConfigError("target_modules is empty; inspect model.named_modules() first")


@dataclass(slots=True)
class TrainConfig:
    """Optimisation settings for the short first experiment."""

    seed: int = 42
    learning_rate: float = 2e-5
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    max_train_samples: int | None = 3000
    max_grad_norm: float = 1.0


@dataclass(slots=True)
class EvalConfig:
    """Evaluation protocol, shared verbatim by base and adapted runs."""

    smoke_subset_size: int = 20
    val_subset_size: int = 500
    batch_size: int = 1
    seed: int = 42
    report_question_types: bool = True
    measure_latency: bool = True


@dataclass(slots=True)
class TemporalConfig:
    """Our temporal adapter in the visual pathway (see models/temporal.py).

    ``learning_rate`` is separate from the LoRA one on purpose: the gate
    starts at zero, and at LoRA's 2e-5 Adam could move it by only ~0.006
    in 300 steps, which would leave the adapter effectively switched off
    and make "temporal mixing does not help" a foregone conclusion.
    ``freeze_lora`` gives the temporal-only ablation arm: LoRA is attached
    but frozen at its zero-initialised B, i.e. an exact no-op.
    """

    enabled: bool = False
    kernel_size: int = 3
    learning_rate: float = 1.0e-3
    freeze_lora: bool = False

    def __post_init__(self) -> None:
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ConfigError("temporal.kernel_size must be a positive odd integer")
        if self.freeze_lora and not self.enabled:
            raise ConfigError("temporal.freeze_lora without temporal.enabled would train nothing")


@dataclass(slots=True)
class ProjectConfig:
    """Root config object handed to every stage."""

    experiment_name: str = "a1_dora_internvl35_8b"
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    dora: DoraConfig = field(default_factory=DoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view, dumped alongside results for provenance."""
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return out


_SECTION_TYPES: dict[str, type] = {
    "paths": PathsConfig,
    "model": ModelConfig,
    "video": VideoConfig,
    "dora": DoraConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
    "temporal": TemporalConfig,
}


def _build_section(name: str, raw: Any) -> Any:
    """Instantiate one config section, rejecting unknown keys loudly."""
    cls = _SECTION_TYPES[name]
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ConfigError(f"Section {name!r} must be a mapping, got {type(raw).__name__}")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"Unknown keys in section {name!r}: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Load a YAML config into a :class:`ProjectConfig`.

    ``path=None`` returns the documented defaults, which is what the
    environment audit uses before any experiment config exists.
    """
    if path is None:
        return ProjectConfig()

    cfg_path = Path(path).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()
    if not cfg_path.is_file():
        raise ConfigError(f"Config file not found: {cfg_path}")

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Top level of {cfg_path} must be a mapping")

    unknown = set(raw) - set(_SECTION_TYPES) - {"experiment_name"}
    if unknown:
        raise ConfigError(f"Unknown top-level sections: {sorted(unknown)}")

    sections = {name: _build_section(name, raw.get(name)) for name in _SECTION_TYPES}
    return ProjectConfig(experiment_name=raw.get("experiment_name", "unnamed"), **sections)
