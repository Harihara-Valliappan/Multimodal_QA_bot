"""Milestone 0 - environment and compatibility audit.

The brief says: do not launch training until the stack has been proven.
This module turns that sentence into executable checks. Each check is a
small method that returns a :class:`CheckResult`; the auditor runs them
all, prints a table, and writes a JSON report under ``reports/``.

Design notes
------------
* Checks never raise: a crash in one check must not hide the other
  fifteen results. Exceptions are captured and reported as FAIL.
* Checks are ordered cheapest-first, so a login-node run still yields
  useful output even where no GPU is attached.
* ``status`` is tri-state. WARN means "works, but record this"; FAIL
  means "do not proceed to the next milestone".
"""

from __future__ import annotations

import enum
import importlib
import inspect
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from videoqa.config import ProjectConfig
from videoqa.logging_utils import get_logger, write_report

LOGGER = get_logger(__name__)

#: Packages whose exact versions must appear in the reproducibility record.
TRACKED_PACKAGES: tuple[str, ...] = (
    "torch",
    "torchvision",
    "transformers",
    "peft",
    "accelerate",
    "datasets",
    "huggingface_hub",
    "decord",
    "av",
    "cv2",
    "timm",
    "einops",
    "sentencepiece",
    "numpy",
    "yaml",
)


class Status(enum.StrEnum):
    """Outcome of a single check."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(slots=True)
class CheckResult:
    """One audited fact about the environment."""

    name: str
    status: Status
    detail: str
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": str(self.status), "detail": self.detail, "data": self.data}


def _version_of(module_name: str) -> str | None:
    """Import a module and return its ``__version__``, or None if absent."""
    try:
        module = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 - a missing optional dep is data, not an error
        return None
    return str(getattr(module, "__version__", "unknown"))


class EnvironmentAuditor:
    """Runs every compatibility check and produces a single report.

    Parameters
    ----------
    config:
        Project configuration; supplies the checkpoint id and the paths
        that must exist.
    check_remote:
        When True, the auditor contacts the Hugging Face Hub to confirm
        the checkpoint exists and to resolve its commit SHA. Set to
        False on an offline node.
    """

    def __init__(self, config: ProjectConfig, *, check_remote: bool = True) -> None:
        self.config = config
        self.check_remote = check_remote
        self.results: list[CheckResult] = []

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def _checks(self) -> list[Callable[[], CheckResult]]:
        """Checks in execution order (cheapest and most fundamental first)."""
        return [
            self.check_python,
            self.check_packages,
            self.check_cuda,
            self.check_bf16,
            self.check_video_backend,
            self.check_peft_dora_support,
            self.check_disk_space,
            self.check_paths,
            self.check_nvidia_smi,
            self.check_checkpoint_metadata,
        ]

    def run(self) -> list[CheckResult]:
        """Execute every check, capturing failures as results."""
        self.results = []
        for check in self._checks():
            name = check.__name__.removeprefix("check_")
            try:
                result = check()
            except Exception as exc:  # noqa: BLE001 - intentionally broad, see module docstring
                result = CheckResult(name, Status.FAIL, f"check raised {type(exc).__name__}: {exc}")
            self.results.append(result)
            LOGGER.info("[%-4s] %-22s %s", result.status, result.name, result.detail)
        return self.results

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.status is Status.FAIL]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if r.status is Status.WARN]

    def summary(self) -> dict[str, Any]:
        """Aggregate verdict plus the recommended starting configuration."""
        return {
            "milestone": "0_environment_audit",
            "experiment_name": self.config.experiment_name,
            "verdict": "FAIL" if self.failures else ("WARN" if self.warnings else "PASS"),
            "n_pass": sum(r.status is Status.PASS for r in self.results),
            "n_warn": len(self.warnings),
            "n_fail": len(self.failures),
            "checks": [r.as_dict() for r in self.results],
            "config": self.config.to_dict(),
        }

    def save(self, filename: str = "milestone0_environment_audit.json") -> Path:
        path = self.config.paths.resolve("reports_dir") / filename
        return write_report(self.summary(), path)

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------
    def check_python(self) -> CheckResult:
        """Python 3.10+ is required for the typing syntax used here."""
        info = sys.version_info
        data = {
            "version": platform.python_version(),
            "executable": sys.executable,
            "in_virtualenv": sys.prefix != sys.base_prefix,
        }
        if info < (3, 10):
            return CheckResult("python", Status.FAIL, f"Python {data['version']} is too old (need >= 3.10)", data)
        if not data["in_virtualenv"]:
            return CheckResult("python", Status.WARN, f"Python {data['version']} outside a virtualenv", data)
        return CheckResult("python", Status.PASS, f"Python {data['version']} in project venv", data)

    def check_packages(self) -> CheckResult:
        """Record versions of every library that can change a result."""
        versions = {name: _version_of(name) for name in TRACKED_PACKAGES}
        required = ("torch", "transformers", "peft", "accelerate")
        missing = [name for name in required if versions[name] is None]
        if missing:
            return CheckResult("packages", Status.FAIL, f"missing required packages: {missing}", versions)
        optional_missing = [k for k, v in versions.items() if v is None]
        status = Status.WARN if optional_missing else Status.PASS
        detail = f"torch={versions['torch']} transformers={versions['transformers']} peft={versions['peft']}"
        if optional_missing:
            detail += f" (optional missing: {optional_missing})"
        return CheckResult("packages", status, detail, versions)

    def check_cuda(self) -> CheckResult:
        """GPU visibility from inside this process, not just from the node."""
        try:
            import torch
        except ImportError:
            return CheckResult("cuda", Status.FAIL, "torch is not installed")

        data: dict[str, Any] = {
            "torch_version": torch.__version__,
            "cuda_compiled": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        if not data["cuda_available"]:
            return CheckResult(
                "cuda",
                Status.WARN,
                "no CUDA device visible (expected on a login node; rerun under sbatch)",
                data,
            )
        devices = []
        for idx in range(data["device_count"]):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1024**3, 2),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
        data["devices"] = devices
        return CheckResult("cuda", Status.PASS, f"{len(devices)}x {devices[0]['name']}", data)

    def check_bf16(self) -> CheckResult:
        """bfloat16 avoids the loss-scaling pitfalls of fp16; confirm it."""
        try:
            import torch
        except ImportError:
            return CheckResult("bf16", Status.FAIL, "torch is not installed")
        if not torch.cuda.is_available():
            return CheckResult("bf16", Status.WARN, "no GPU present; bf16 support unverified")
        supported = torch.cuda.is_bf16_supported()
        data = {"bf16_supported": supported, "requested_dtype": self.config.model.torch_dtype}
        if supported:
            return CheckResult("bf16", Status.PASS, "bfloat16 supported on this GPU", data)
        if self.config.model.torch_dtype == "bfloat16":
            return CheckResult("bf16", Status.FAIL, "config requests bfloat16 but GPU cannot do it", data)
        return CheckResult("bf16", Status.WARN, "bfloat16 unsupported; config already uses another dtype", data)

    def check_video_backend(self) -> CheckResult:
        """At least one video decoder must be importable."""
        candidates = {"decord": "decord", "pyav": "av", "opencv": "cv2"}
        available = {label: _version_of(mod) for label, mod in candidates.items()}
        working = [label for label, ver in available.items() if ver is not None]
        data = {"available": available, "configured": self.config.video.backend}
        if not working:
            return CheckResult("video_backend", Status.FAIL, "no video decoder installed", data)
        if self.config.video.backend not in working:
            return CheckResult(
                "video_backend",
                Status.WARN,
                f"configured backend {self.config.video.backend!r} missing; available: {working}",
                data,
            )
        return CheckResult("video_backend", Status.PASS, f"{self.config.video.backend} available", data)

    def check_peft_dora_support(self) -> CheckResult:
        """Hard gate: the installed PEFT must actually expose DoRA.

        The brief forbids a silent downgrade to LoRA, so we inspect the
        ``LoraConfig`` signature and instantiate one with ``use_dora=True``
        instead of trusting the version number.
        """
        try:
            import peft
            from peft import LoraConfig
        except ImportError:
            return CheckResult("peft_dora_support", Status.FAIL, "peft is not installed")

        params = inspect.signature(LoraConfig.__init__).parameters
        data: dict[str, Any] = {"peft_version": peft.__version__, "has_use_dora_param": "use_dora" in params}
        if "use_dora" not in params:
            return CheckResult(
                "peft_dora_support",
                Status.FAIL,
                f"peft {peft.__version__} has no use_dora parameter; upgrade peft",
                data,
            )
        cfg = LoraConfig(
            r=self.config.dora.r,
            lora_alpha=self.config.dora.lora_alpha,
            target_modules=list(self.config.dora.target_modules),
            use_dora=True,
            task_type=self.config.dora.task_type,
        )
        data["instantiated_use_dora"] = bool(cfg.use_dora)
        if not cfg.use_dora:
            return CheckResult("peft_dora_support", Status.FAIL, "LoraConfig silently dropped use_dora", data)
        return CheckResult("peft_dora_support", Status.PASS, f"peft {peft.__version__} accepts use_dora=True", data)

    def check_disk_space(self) -> CheckResult:
        """ActivityNet videos are large; fail early rather than mid-download."""
        root = self.config.paths.resolve("data_root")
        root.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(root)
        free_gb = round(usage.free / 1024**3, 1)
        data = {"path": str(root), "free_gb": free_gb, "total_gb": round(usage.total / 1024**3, 1)}
        if free_gb < 100:
            return CheckResult("disk_space", Status.FAIL, f"only {free_gb} GB free under {root}", data)
        if free_gb < 300:
            return CheckResult("disk_space", Status.WARN, f"{free_gb} GB free; plan a video subset", data)
        return CheckResult("disk_space", Status.PASS, f"{free_gb} GB free under {root}", data)

    def check_paths(self) -> CheckResult:
        """Create and report every directory the pipeline will write to."""
        resolved = self.config.paths.all_resolved()
        for path in resolved.values():
            path.mkdir(parents=True, exist_ok=True)
        data = {key: str(path) for key, path in resolved.items()}
        return CheckResult("paths", Status.PASS, f"{len(resolved)} project directories present", data)

    def check_nvidia_smi(self) -> CheckResult:
        """Driver-level view of the GPU, independent of torch."""
        binary = shutil.which("nvidia-smi")
        if binary is None:
            return CheckResult("nvidia_smi", Status.WARN, "nvidia-smi not on PATH (login node?)")
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [binary, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode != 0:
            return CheckResult("nvidia_smi", Status.WARN, f"nvidia-smi exited {proc.returncode}", {"stderr": proc.stderr.strip()})
        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        return CheckResult("nvidia_smi", Status.PASS, "; ".join(lines), {"gpus": lines})

    def check_checkpoint_metadata(self) -> CheckResult:
        """Resolve the checkpoint's commit SHA so the run can be pinned.

        Only metadata is fetched here - no weights - so this stays cheap
        enough to run on every audit.
        """
        if not self.check_remote:
            return CheckResult("checkpoint_metadata", Status.WARN, "remote check disabled (--offline)")
        try:
            from huggingface_hub import HfApi
        except ImportError:
            return CheckResult("checkpoint_metadata", Status.FAIL, "huggingface_hub is not installed")

        model_id = self.config.model.model_id
        info = HfApi().model_info(model_id, revision=self.config.model.revision)
        data = {
            "model_id": model_id,
            "requested_revision": self.config.model.revision,
            "resolved_sha": info.sha,
            "tags": list(info.tags or [])[:20],
        }
        status = Status.PASS if self.config.model.revision != "main" else Status.WARN
        detail = f"{model_id} @ {info.sha}"
        if status is Status.WARN:
            detail += "  (pin this SHA in the config before reporting results)"
        return CheckResult("checkpoint_metadata", status, detail, data)
