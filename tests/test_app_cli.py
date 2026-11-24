"""CLI -> AppConfig mapping (regression: slots-dataclass defaults were descriptors)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("run_app", Path(__file__).parents[1] / "scripts/run_app.py")
run_app = importlib.util.module_from_spec(spec); spec.loader.exec_module(run_app)


def test_defaults_are_real_values():
    cfg = run_app.parse_args([])
    assert isinstance(cfg.model_config, str) and cfg.model_config.endswith(".yaml")
    assert isinstance(cfg.adapter, str) and "monteclora" in cfg.adapter
    assert cfg.tts == "magpie" and cfg.hands_free is True


def test_flags_map_onto_config():
    cfg = run_app.parse_args(["--tts", "kokoro", "--push-to-talk", "--adapter", "none", "--window", "5"])
    assert cfg.tts == "kokoro" and cfg.hands_free is False and cfg.adapter is None and cfg.window_s == 5


def test_network_host_requires_auth():
    with pytest.raises(ValueError, match="auth_file"):
        run_app.parse_args(["--host", "0.0.0.0"])
