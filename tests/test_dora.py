"""Tests for the DoRA attachment and its assertions.

A toy two-layer model stands in for the 8B checkpoint: the question
these tests answer is whether our *checks* are correct, not whether
InternVL trains. The GPU smoke test answers that.

The most important test here is the negative one: plain LoRA must be
rejected by assert_is_dora, or the whole guarantee is theatre.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from videoqa.config import ConfigError, ProjectConfig
from videoqa.models.dora import (
    DoraError,
    apply_dora,
    assert_is_dora,
    assert_only_adapters_trainable,
    build_lora_config,
    changed_parameters,
    describe_trainable,
    parameters_with_zero_grad,
    parameters_without_grad,
    snapshot_adapter,
)


class Toy(nn.Module):
    """Two linears named like the real attention projections."""

    def __init__(self, d: int = 32) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.other = nn.Linear(d, d)

    def forward(self, x):
        return self.other(self.v_proj(self.q_proj(x)))


def _toy_dora(r: int = 4):
    return get_peft_model(Toy(), LoraConfig(r=r, target_modules=["q_proj", "v_proj"], use_dora=True))


def _toy_lora(r: int = 4):
    return get_peft_model(Toy(), LoraConfig(r=r, target_modules=["q_proj", "v_proj"], use_dora=False))


# --- the central guarantee -------------------------------------------------

def test_dora_produces_magnitude_vectors() -> None:
    assert assert_is_dora(_toy_dora()) == 2


def test_plain_lora_is_rejected() -> None:
    # If this ever passes, the project's DoRA claim is unverifiable.
    with pytest.raises(DoraError, match="plain LoRA"):
        assert_is_dora(_toy_lora())


def test_config_refuses_use_dora_false() -> None:
    with pytest.raises(ConfigError, match="use_dora=False"):
        ProjectConfig().dora.__class__(use_dora=False)


def test_build_lora_config_sets_use_dora() -> None:
    cfg = build_lora_config(ProjectConfig())
    assert cfg.use_dora is True
    assert cfg.r == 16 and cfg.lora_alpha == 32
    assert set(cfg.target_modules) == {"q_proj", "v_proj"}  # PEFT stores these as a set


# --- trainable accounting --------------------------------------------------

def test_only_adapter_parameters_are_trainable() -> None:
    m = _toy_dora()
    for n, p in m.named_parameters():
        if "lora_" not in n:
            p.requires_grad_(False)
    report = describe_trainable(m)
    assert report.unexpected == []
    assert report.dora_magnitude_params > 0
    assert report.lora_params > 0
    assert 0 < report.percent < 100


def test_unexpected_trainable_parameter_is_flagged() -> None:
    m = _toy_dora()
    m.base_model.model.other.weight.requires_grad_(True)
    report = describe_trainable(m)
    assert any("other" in n for n in report.unexpected)


def test_assert_only_adapters_raises_on_unfrozen_base() -> None:
    m = _toy_dora()
    m.base_model.model.other.weight.requires_grad_(True)
    with pytest.raises(DoraError, match="non-adapter parameters are trainable"):
        assert_only_adapters_trainable(m, ProjectConfig())


def test_no_trainable_parameters_is_an_error() -> None:
    m = _toy_dora()
    for p in m.parameters():
        p.requires_grad_(False)
    with pytest.raises(DoraError, match="not attached"):
        assert_only_adapters_trainable(m, ProjectConfig())


# --- the training step the smoke test performs -----------------------------

def _freeze_non_adapter(m):
    for n, p in m.named_parameters():
        p.requires_grad_("lora_" in n)
    return m


def test_gradients_reach_every_trainable_parameter() -> None:
    m = _freeze_non_adapter(_toy_dora())
    m(torch.randn(4, 32)).sum().backward()
    assert parameters_without_grad(m) == [], "every adapter tensor must get a finite gradient"


def test_lora_A_has_zero_gradient_on_the_first_step() -> None:
    # PEFT inits lora_B to zeros so the adapter starts as a no-op; the
    # update is B @ A, so dL/dA = B.T @ dL/dy is identically zero until B
    # moves. This is correct, and the smoke test must not treat it as a
    # detached parameter -- hence the split between the two helpers.
    m = _freeze_non_adapter(_toy_dora())
    m(torch.randn(4, 32)).sum().backward()
    zero = parameters_with_zero_grad(m)
    assert any("lora_A" in n for n in zero)
    assert not any("lora_B" in n for n in zero)


def test_zero_gradient_resolves_after_a_step() -> None:
    m = _freeze_non_adapter(_toy_dora())
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    m(torch.randn(8, 32)).sum().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    m(torch.randn(8, 32)).sum().backward()
    assert parameters_with_zero_grad(m) == [], "B has moved, so A must now receive gradient"


def test_missing_gradient_is_reported() -> None:
    m = _freeze_non_adapter(_toy_dora())
    for p in m.parameters():
        p.grad = None
    assert len(parameters_without_grad(m)) > 0


def test_non_finite_gradient_is_reported() -> None:
    m = _freeze_non_adapter(_toy_dora())
    m(torch.randn(4, 32)).sum().backward()
    for n, p in m.named_parameters():
        if p.requires_grad and "lora_B" in n:
            p.grad[0] = float("nan")
            break
    assert any("lora_B" in n for n in parameters_without_grad(m))


def test_optimizer_steps_change_every_trainable_value() -> None:
    # Two steps, as the smoke test defaults to: after one step lora_A has
    # seen only a zero gradient and would barely have moved.
    m = _freeze_non_adapter(_toy_dora())
    before = snapshot_adapter(m)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        m(torch.randn(8, 32)).sum().backward()
        opt.step()
    moved, static = changed_parameters(before, m)
    assert static == [], f"these did not move: {static}"
    assert any("lora_magnitude_vector" in n for n in moved), "DoRA magnitudes must train too"


def test_snapshot_is_a_copy_not_a_view() -> None:
    m = _freeze_non_adapter(_toy_dora())
    before = snapshot_adapter(m)
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad:
                p.add_(1.0)
    moved, static = changed_parameters(before, m)
    assert static == [] and moved, "snapshot must not alias live tensors"


def test_adapter_changes_the_output() -> None:
    m = _freeze_non_adapter(_toy_dora())
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.add_(0.5)  # lora_B starts at zero, so the adapter is a no-op until it moves
    x = torch.randn(4, 32)
    with torch.no_grad():
        adapted = m(x)
        with m.disable_adapter():
            base = m(x)
    assert not torch.allclose(adapted, base)


def test_task_type_is_none_for_this_architecture() -> None:
    # Regression guard for job 42947: PeftModelForCausalLM injects
    # inputs_embeds into forward(), which InternVLChatModel rejects.
    assert ProjectConfig().dora.task_type is None
    assert build_lora_config(ProjectConfig()).task_type is None


def test_apply_dora_freezes_the_base_model() -> None:
    cfg = ProjectConfig()
    m = apply_dora(Toy(), cfg)
    report = assert_only_adapters_trainable(m, cfg)
    assert report.unexpected == []
    assert report.dora_magnitude_params > 0
