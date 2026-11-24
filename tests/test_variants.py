"""Tests for the PEFT >= 0.21 LoRA variants (KaSA, MonteCLoRA, BD-LoRA, VeLoRA).

The failure these guard against is silent: KaSA and MonteCLoRA need extra
loss terms that PEFT does not add on its own, so a run that "works" can
still be a weakened version of the method. Each variant must also pass the
same adapter assertions as plain LoRA.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from videoqa.config import ConfigError, DoraConfig, ProjectConfig
from videoqa.models.dora import (
    adapter_variant,
    apply_adapter,
    assert_only_adapters_trainable,
    auxiliary_loss,
    parameters_without_grad,
)

VARIANT_OPTIONS = {
    "kasa": {},
    "monteclora": {},
    "bdlora": {"target_modules_bd_b": ["q_proj", "v_proj"], "nblocks": 4},
    "velora": {},
}


class Toy(nn.Module):
    def __init__(self, d: int = 64) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)

    def forward(self, x):
        return self.v_proj(self.q_proj(x))


def _cfg(variant: str | None) -> ProjectConfig:
    cfg = ProjectConfig()
    if variant is None:
        cfg.dora = DoraConfig(use_dora=False, allow_lora_control=True)
    else:
        cfg.dora = DoraConfig(use_dora=False, allow_lora_control=True,
                              lora_variant=variant, variant_options=VARIANT_OPTIONS[variant])
    return cfg


@pytest.mark.parametrize("variant", list(VARIANT_OPTIONS))
def test_variant_attaches_and_passes_adapter_checks(variant) -> None:
    torch.manual_seed(0)
    cfg = _cfg(variant)
    m = apply_adapter(Toy(), cfg)  # also asserts it is not DoRA
    report = assert_only_adapters_trainable(m, cfg)
    assert report.unexpected == []
    assert adapter_variant(cfg) == variant


@pytest.mark.parametrize("variant", list(VARIANT_OPTIONS))
def test_variant_gets_gradients_after_two_steps(variant) -> None:
    torch.manual_seed(0)
    m = apply_adapter(Toy(), _cfg(variant))
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        m.train()
        loss = m(torch.randn(8, 64)).pow(2).mean() + auxiliary_loss(m)
        loss.backward()
        opt.step()
    assert parameters_without_grad(m) == []


def test_auxiliary_loss_is_zero_for_plain_lora() -> None:
    m = apply_adapter(Toy(), _cfg(None))
    m(torch.randn(4, 64))
    aux = auxiliary_loss(m)
    assert float(aux) == 0.0


@pytest.mark.parametrize("variant", ["kasa", "monteclora"])
def test_auxiliary_loss_is_active_and_differentiable(variant) -> None:
    # The whole point: without this term the method is not what it claims.
    torch.manual_seed(0)
    m = apply_adapter(Toy(), _cfg(variant))
    m.train()
    m(torch.randn(8, 64))
    aux = auxiliary_loss(m)
    assert torch.is_tensor(aux) and float(aux) > 0.0, f"{variant} auxiliary loss is missing"
    aux.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.parameters() if p.requires_grad), "auxiliary loss carries no gradient"


def test_bdlora_trains_fewer_parameters_than_lora_at_same_rank() -> None:
    # Expected from the block-diagonal structure; recorded so the report
    # states BD-LoRA's capacity next to its score.
    lora = assert_only_adapters_trainable(apply_adapter(Toy(), _cfg(None)), _cfg(None))
    bd = assert_only_adapters_trainable(apply_adapter(Toy(), _cfg("bdlora")), _cfg("bdlora"))
    assert bd.trainable < lora.trainable


def test_variant_requires_lora_family_flags() -> None:
    with pytest.raises(ConfigError, match="LoRA-family"):
        DoraConfig(lora_variant="kasa")  # use_dora defaults to True


def test_unknown_variant_rejected() -> None:
    with pytest.raises(ConfigError, match="lora_variant must be one of"):
        DoraConfig(use_dora=False, allow_lora_control=True, lora_variant="qlora")


def test_options_without_variant_rejected() -> None:
    with pytest.raises(ConfigError, match="without a lora_variant"):
        DoraConfig(use_dora=False, allow_lora_control=True, variant_options={"nblocks": 4})
