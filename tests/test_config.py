"""Unit tests for the configuration layer.

These run in under a second with no GPU and no dataset - the point is to
catch config mistakes on the login node instead of inside a queued job.
"""

from __future__ import annotations

import pytest
import yaml

from videoqa.config import ConfigError, DoraConfig, ProjectConfig, VideoConfig, load_config


def test_defaults_are_self_consistent() -> None:
    cfg = ProjectConfig()
    assert cfg.dora.use_dora is True
    assert cfg.model.do_sample is False, "greedy decoding keeps base vs adapted comparable"
    assert cfg.video.sampling == "uniform"


def test_base_yaml_loads() -> None:
    cfg = load_config("configs/base.yaml")
    assert cfg.experiment_name == "a1_dora_internvl35_8b"
    assert cfg.model.model_id == "OpenGVLab/InternVL3_5-8B"
    assert cfg.dora.use_dora is True


def test_unknown_key_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"video": {"num_framez": 8}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown keys"):
        load_config(path)


def test_disabling_dora_is_refused() -> None:
    """The brief forbids a silent downgrade to LoRA; make it loud."""
    with pytest.raises(ConfigError, match="use_dora"):
        DoraConfig(use_dora=False)


def test_non_deterministic_sampling_is_refused() -> None:
    with pytest.raises(ConfigError, match="uniform"):
        VideoConfig(sampling="random")


def test_paths_resolve_to_absolute() -> None:
    cfg = ProjectConfig()
    resolved = cfg.paths.all_resolved()
    assert all(p.is_absolute() for p in resolved.values())
    assert resolved["reports_dir"].name == "reports"


def test_roundtrip_through_yaml(tmp_path) -> None:
    original = ProjectConfig(experiment_name="roundtrip")
    saved = original.save(tmp_path / "cfg.yaml")
    reloaded = load_config(saved)
    assert reloaded.to_dict() == original.to_dict()


def test_lora_control_differs_from_base_only_in_the_adapter_variant() -> None:
    """A2 is only a controlled comparison if nothing else drifted.

    This test is the guard against someone tuning the LoRA arm's learning
    rate (or rank, or targets) and leaving the DoRA arm alone, which would
    turn "is DoRA better?" into "is this learning rate better?".
    """
    import dataclasses

    base = load_config("configs/base.yaml")
    lora = load_config("configs/lora_control.yaml")

    assert base.dora.use_dora is True and lora.dora.use_dora is False
    assert lora.dora.allow_lora_control is True

    # every adapter field except the two variant flags must match
    for f in dataclasses.fields(base.dora):
        if f.name in {"use_dora", "allow_lora_control"}:
            continue
        assert getattr(base.dora, f.name) == getattr(lora.dora, f.name), f"dora.{f.name} drifted"

    # model, video, training and eval protocol must match exactly
    for section in ("model", "video", "train", "eval"):
        assert dataclasses.asdict(getattr(base, section)) == dataclasses.asdict(getattr(lora, section)), (
            f"{section} config drifted between the DoRA and LoRA arms"
        )

    # the names must differ, so artifacts cannot collide
    assert base.experiment_name != lora.experiment_name


@pytest.mark.parametrize("variant", ["kasa", "monteclora", "bdlora", "velora"])
def test_variant_configs_differ_from_lora_control_only_in_variant(variant) -> None:
    """Each A4 arm must be the A2 control plus one variant, nothing else."""
    import dataclasses

    lora = load_config("configs/lora_control.yaml")
    arm = load_config(f"configs/variant_{variant}.yaml")
    assert arm.dora.lora_variant == variant
    for f in dataclasses.fields(lora.dora):
        if f.name in {"lora_variant", "variant_options"}:
            continue
        assert getattr(lora.dora, f.name) == getattr(arm.dora, f.name), f"dora.{f.name} drifted"
    for section in ("model", "video", "train", "eval"):
        assert dataclasses.asdict(getattr(lora, section)) == dataclasses.asdict(getattr(arm, section)), section
