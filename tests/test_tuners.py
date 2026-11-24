"""Tests for the non-LoRA tuner families: SHiRA and PEANuT.

The dangerous failure is a save/reload mismatch. SHiRA chooses a random
sparse mask; if it were regenerated differently on reload, the adapter
evaluated on test would not be the adapter that was trained, and every
number would be silently wrong.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from videoqa.config import ConfigError, DoraConfig, ProjectConfig, load_config
from videoqa.models.dora import (
    adapter_variant,
    apply_adapter,
    assert_only_adapters_trainable,
    describe_trainable,
    parameters_without_grad,
)


class Shaped(nn.Module):
    """q_proj / v_proj with InternVL3.5-8B's real shapes, scaled down 16x."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(256, 256, bias=False)
        self.v_proj = nn.Linear(256, 64, bias=False)

    def forward(self, x):
        return self.v_proj(self.q_proj(x))


def _cfg(tuner: str, r: int = 16) -> ProjectConfig:
    cfg = ProjectConfig()
    cfg.dora = DoraConfig(use_dora=False, allow_lora_control=True, tuner=tuner, r=r)
    return cfg


@pytest.mark.parametrize("tuner", ["shira", "peanut"])
def test_tuner_attaches_and_only_adapters_train(tuner) -> None:
    cfg = _cfg(tuner)
    m = apply_adapter(Shaped(), cfg)
    report = assert_only_adapters_trainable(m, cfg)
    assert report.unexpected == [] and report.trainable > 0
    assert adapter_variant(cfg) == tuner


@pytest.mark.parametrize("tuner", ["shira", "peanut"])
def test_tuner_receives_gradients_and_changes_output(tuner) -> None:
    torch.manual_seed(0)
    m = apply_adapter(Shaped(), _cfg(tuner))
    x = torch.randn(8, 256)
    with torch.no_grad():
        before = m(x).clone()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        m(x).pow(2).mean().backward()
        opt.step()
    assert parameters_without_grad(m) == []
    with torch.no_grad():
        assert not torch.allclose(m(x), before), "training did not change the output"


@pytest.mark.parametrize("tuner", ["shira", "peanut"])
def test_save_and_reload_reproduces_outputs_exactly(tuner, tmp_path) -> None:
    from peft import PeftModel

    torch.manual_seed(0)
    base_state = Shaped().state_dict()
    trained_base = Shaped(); trained_base.load_state_dict(base_state)
    m = apply_adapter(trained_base, _cfg(tuner))
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randn(8, 256)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        m(x).pow(2).mean().backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        expected = m(x)
    m.save_pretrained(tmp_path / "adapter")

    torch.manual_seed(12345)  # a different global RNG state must not matter
    fresh = Shaped(); fresh.load_state_dict(base_state)
    reloaded = PeftModel.from_pretrained(fresh, tmp_path / "adapter").eval()
    with torch.no_grad():
        got = reloaded(x)
    assert torch.equal(got, expected), f"{tuner} reload changed the outputs"


def test_shira_matches_lora_parameter_count_at_same_rank() -> None:
    lora_cfg = ProjectConfig()
    lora_cfg.dora = DoraConfig(use_dora=False, allow_lora_control=True)
    lora = describe_trainable(apply_adapter(Shaped(), lora_cfg)).trainable
    shira = describe_trainable(apply_adapter(Shaped(), _cfg("shira"))).trainable
    assert shira == lora


def test_peanut_r21_is_within_two_percent_of_lora_r16() -> None:
    lora_cfg = ProjectConfig()
    lora_cfg.dora = DoraConfig(use_dora=False, allow_lora_control=True)
    lora = describe_trainable(apply_adapter(Shaped(), lora_cfg)).trainable
    peanut = describe_trainable(apply_adapter(Shaped(), _cfg("peanut", r=21))).trainable
    assert abs(peanut - lora) / lora < 0.02, (peanut, lora)


def test_tuner_requires_lora_family_flags() -> None:
    with pytest.raises(ConfigError, match="needs use_dora: false"):
        DoraConfig(tuner="shira")


def test_unknown_tuner_rejected() -> None:
    with pytest.raises(ConfigError, match="tuner must be"):
        DoraConfig(use_dora=False, allow_lora_control=True, tuner="poly")


@pytest.mark.parametrize("tuner,r", [("shira", 16), ("peanut", 21)])
def test_tuner_configs_differ_from_lora_control_only_where_intended(tuner, r) -> None:
    import dataclasses

    lora = load_config("configs/lora_control.yaml")
    arm = load_config(f"configs/tuner_{tuner}.yaml")
    assert arm.dora.tuner == tuner and arm.dora.r == r
    for f in dataclasses.fields(lora.dora):
        if f.name in {"tuner", "r"}:
            continue
        assert getattr(lora.dora, f.name) == getattr(arm.dora, f.name), f"dora.{f.name} drifted"
    for section in ("model", "video", "train", "eval"):
        assert dataclasses.asdict(getattr(lora, section)) == dataclasses.asdict(getattr(arm, section)), section
