"""Tests for our temporal adapter (models/temporal.py).

The properties that make the A7 ablation interpretable:
* exact identity at initialisation -> "LoRA + temporal" starts as LoRA;
* frames genuinely influence each other once the gate opens;
* the gate is not stuck at zero (it receives gradient on step 1);
* save -> reload is bit-exact, and frame-count mismatches are refused.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from videoqa.config import ConfigError, TemporalConfig
from videoqa.models.temporal import (
    TemporalAdapter,
    TemporalAdapterError,
    attach_temporal_adapter,
    find_temporal_adapter,
    load_temporal_adapter,
    save_temporal_adapter,
)

T, N, C_IN, C = 8, 16, 32, 24


class FakeInternVL(nn.Module):
    """Just enough of InternVLChatModel: an mlp1 projector."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp1 = nn.Sequential(nn.LayerNorm(C_IN), nn.Linear(C_IN, C), nn.GELU(), nn.Linear(C, C))
        for p in self.mlp1.parameters():
            p.requires_grad_(False)  # frozen, as in training


def _attached(seed: int = 0):
    torch.manual_seed(seed)
    m = FakeInternVL()
    a = attach_temporal_adapter(m, num_frames=T)
    return m, a


def test_identity_at_initialisation() -> None:
    m, a = _attached()
    x = torch.randn(T, N, C_IN)
    assert torch.equal(m.mlp1(x), a.base(x)), "zero gate must make the adapter an exact no-op"


def test_gate_gets_gradient_on_first_step() -> None:
    # Like LoRA's B = 0: the zero-initialised part must still be trainable.
    m, a = _attached()
    m.mlp1(torch.randn(T, N, C_IN)).pow(2).sum().backward()
    assert a.temporal_gate.grad is not None and a.temporal_gate.grad.abs().sum() > 0


def test_all_adapter_params_get_gradient_after_one_step() -> None:
    m, a = _attached()
    opt = torch.optim.AdamW([p for p in a.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randn(T, N, C_IN)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        m.mlp1(x).pow(2).sum().backward()
        opt.step()
    for name in ("temporal_gate", "temporal_conv", "temporal_pos"):
        g = getattr(a, name).grad
        assert g is not None and g.abs().sum() > 0, f"{name} received no gradient"


def test_projector_stays_frozen() -> None:
    m, a = _attached()
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all("temporal_" in n for n in trainable), trainable


def test_frames_influence_neighbours_once_gate_is_open() -> None:
    # The whole point: changing frame 3 must change the tokens of frame 4.
    m, a = _attached()
    with torch.no_grad():
        a.temporal_gate.fill_(1.0)
    x = torch.randn(T, N, C_IN)
    x2 = x.clone(); x2[3] += 5.0
    with torch.no_grad():
        y, y2 = m.mlp1(x), m.mlp1(x2)
    assert not torch.allclose(y[4], y2[4]), "frame 4 ignores frame 3: no cross-frame mixing"
    assert torch.allclose(y[6], y2[6]), "kernel 3 should not reach 3 frames away"


def test_frame_position_vectors_break_frame_symmetry() -> None:
    # Identical frames must become distinguishable once pos is learned.
    m, a = _attached()
    with torch.no_grad():
        a.temporal_gate.fill_(1.0)
        a.temporal_conv.zero_()
        a.temporal_pos.copy_(torch.randn(T, C))
    x = torch.randn(1, N, C_IN).repeat(T, 1, 1)
    with torch.no_grad():
        y = m.mlp1(x)
    assert not torch.allclose(y[0], y[5])


def test_wrong_frame_count_is_refused() -> None:
    m, _ = _attached()
    with pytest.raises(TemporalAdapterError, match="one tile per frame"):
        m.mlp1(torch.randn(T + 1, N, C_IN))


def test_double_attach_is_refused() -> None:
    m, _ = _attached()
    with pytest.raises(TemporalAdapterError, match="already attached"):
        attach_temporal_adapter(m, num_frames=T)


def test_save_and_reload_is_bit_exact(tmp_path) -> None:
    m, a = _attached(seed=0)
    base_state = {k: v.clone() for k, v in a.base.state_dict().items()}
    opt = torch.optim.AdamW([p for p in a.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randn(T, N, C_IN)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        m.mlp1(x).pow(2).sum().backward()
        opt.step()
    with torch.no_grad():
        expected = m.mlp1(x)
    save_temporal_adapter(a, tmp_path)

    torch.manual_seed(999)
    fresh = FakeInternVL(); fresh.mlp1.load_state_dict(base_state)
    reloaded = load_temporal_adapter(fresh, tmp_path, num_frames=T)
    assert reloaded is not None
    with torch.no_grad():
        assert torch.equal(fresh.mlp1(x), expected)


def test_reload_refuses_frame_mismatch(tmp_path) -> None:
    _, a = _attached()
    save_temporal_adapter(a, tmp_path)
    with pytest.raises(TemporalAdapterError, match="trained for 8 frames"):
        load_temporal_adapter(FakeInternVL(), tmp_path, num_frames=16)


def test_load_returns_none_when_no_adapter_saved(tmp_path) -> None:
    assert load_temporal_adapter(FakeInternVL(), tmp_path, num_frames=T) is None


def test_find_temporal_adapter() -> None:
    m, a = _attached()
    assert find_temporal_adapter(m) is a
    assert find_temporal_adapter(FakeInternVL()) is None


def test_works_in_bf16_with_fp32_master_weights() -> None:
    m, a = _attached()
    m.mlp1.base.to(torch.bfloat16)
    y = m.mlp1(torch.randn(T, N, C_IN, dtype=torch.bfloat16))
    assert y.dtype == torch.bfloat16 and a.temporal_gate.dtype == torch.float32


def test_parameter_count() -> None:
    _, a = _attached()
    n = sum(p.numel() for p in a.parameters() if p.requires_grad)
    assert n == T * C + C * 3 + C  # pos + depthwise conv + gate


def test_config_validation() -> None:
    with pytest.raises(ConfigError, match="odd"):
        TemporalConfig(kernel_size=4)
    with pytest.raises(ConfigError, match="train nothing"):
        TemporalConfig(freeze_lora=True)
    assert TemporalConfig().enabled is False, "must default off so existing configs are unchanged"


def test_even_kernel_refused_by_module() -> None:
    with pytest.raises(TemporalAdapterError, match="odd"):
        TemporalAdapter(nn.Linear(4, 4), 4, T, kernel_size=2)
