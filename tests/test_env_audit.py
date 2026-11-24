"""Unit tests for the auditor's control flow.

We do not assert on hardware here - that is the auditor's job at runtime.
We assert that the auditor behaves correctly *around* the checks.
"""

from __future__ import annotations

from videoqa.config import ProjectConfig
from videoqa.env_audit import CheckResult, EnvironmentAuditor, Status


def test_offline_mode_skips_hub_call() -> None:
    auditor = EnvironmentAuditor(ProjectConfig(), check_remote=False)
    result = auditor.check_checkpoint_metadata()
    assert result.status is Status.WARN
    assert "offline" in result.detail


def test_a_raising_check_becomes_a_failure(monkeypatch) -> None:
    """One broken check must not abort the other fifteen."""
    auditor = EnvironmentAuditor(ProjectConfig(), check_remote=False)

    def boom() -> CheckResult:
        raise RuntimeError("simulated")

    boom.__name__ = "check_explosive"
    monkeypatch.setattr(auditor, "_checks", lambda: [boom, auditor.check_python])

    results = auditor.run()
    assert len(results) == 2
    assert results[0].status is Status.FAIL
    assert "simulated" in results[0].detail
    assert auditor.summary()["verdict"] == "FAIL"


def test_paths_check_creates_directories() -> None:
    auditor = EnvironmentAuditor(ProjectConfig(), check_remote=False)
    result = auditor.check_paths()
    assert result.status is Status.PASS
    assert ProjectConfig().paths.resolve("reports_dir").is_dir()


def test_summary_counts_statuses() -> None:
    auditor = EnvironmentAuditor(ProjectConfig(), check_remote=False)
    auditor.results = [
        CheckResult("a", Status.PASS, ""),
        CheckResult("b", Status.WARN, ""),
        CheckResult("c", Status.WARN, ""),
    ]
    summary = auditor.summary()
    assert (summary["n_pass"], summary["n_warn"], summary["n_fail"]) == (1, 2, 0)
    assert summary["verdict"] == "WARN"
