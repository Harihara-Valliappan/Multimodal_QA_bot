#!/usr/bin/env python3
"""Entry point for Milestone 0: audit the environment, write a report.

Usage
-----
    python scripts/run_env_audit.py                      # defaults
    python scripts/run_env_audit.py -c configs/base.yaml
    python scripts/run_env_audit.py --offline            # skip Hub call

Exit code is 1 when any check FAILs, so Slurm and CI can gate on it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src/`` importable without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from videoqa.config import load_config  # noqa: E402
from videoqa.env_audit import EnvironmentAuditor, Status  # noqa: E402
from videoqa.logging_utils import get_logger  # noqa: E402

LOGGER = get_logger("env_audit")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Milestone 0 environment and compatibility audit")
    parser.add_argument("-c", "--config", default=None, help="YAML config (defaults to built-in defaults)")
    parser.add_argument("--offline", action="store_true", help="skip Hugging Face Hub metadata lookup")
    parser.add_argument(
        "-o",
        "--output",
        default="milestone0_environment_audit.json",
        help="report filename, written under the configured reports_dir",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    LOGGER.info("Experiment: %s", config.experiment_name)
    LOGGER.info("Checkpoint: %s @ %s", config.model.model_id, config.model.revision)

    auditor = EnvironmentAuditor(config, check_remote=not args.offline)
    auditor.run()
    report_path = auditor.save(args.output)

    summary = auditor.summary()
    LOGGER.info("-" * 72)
    LOGGER.info(
        "Verdict: %s  (pass=%d warn=%d fail=%d)",
        summary["verdict"],
        summary["n_pass"],
        summary["n_warn"],
        summary["n_fail"],
    )
    LOGGER.info("Report written to %s", report_path)

    for result in auditor.results:
        if result.status is Status.FAIL:
            LOGGER.error("BLOCKER: %s - %s", result.name, result.detail)

    return 1 if auditor.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
