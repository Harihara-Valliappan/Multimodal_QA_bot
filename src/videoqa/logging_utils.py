"""Logging and machine-readable report helpers.

Two separate concerns live here on purpose:

* ``get_logger``  -> human-facing stream of what is happening, for the
  Slurm ``.out`` files.
* ``write_report`` -> machine-facing JSON artifact, so every milestone
  leaves behind evidence that can be diffed between runs.

The project brief demands reproducibility artifacts; the cheapest way to
guarantee them is to make writing one easier than not writing one.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int | str = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stdout exactly once.

    Repeated calls with the same ``name`` return the same logger without
    stacking duplicate handlers (a classic source of triplicated Slurm
    logs when a module is imported from several entry points).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def utc_timestamp() -> str:
    """ISO-8601 UTC timestamp, used to stamp every report."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_context() -> dict[str, Any]:
    """Where and how this process is running.

    Recorded in every report so a result can always be traced back to a
    specific node and Slurm job.
    """
    return {
        "timestamp_utc": utc_timestamp(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "python_executable": sys.executable,
        "cwd": str(Path.cwd()),
    }


def write_report(payload: dict[str, Any], path: str | Path) -> Path:
    """Write ``payload`` as pretty JSON, creating parent directories.

    Returns the resolved path so callers can log it.
    """
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    document = {"run_context": run_context(), **payload}
    out_path.write_text(json.dumps(document, indent=2, default=str) + "\n", encoding="utf-8")
    return out_path
