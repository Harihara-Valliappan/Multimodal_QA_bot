#!/usr/bin/env python3
"""Fetch the pretrained checkpoint (and optionally dataset annotations).

Downloading is a separate, resumable step on purpose: the weights are
~16 GB and a login-node download should never be entangled with a
queued GPU job. ``snapshot_download`` resumes, so re-running is cheap.

Usage
-----
    python scripts/download_assets.py --model
    python scripts/download_assets.py --model --annotations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from videoqa.config import ProjectConfig, load_config  # noqa: E402
from videoqa.logging_utils import get_logger, write_report  # noqa: E402

LOGGER = get_logger("download")

#: ActivityNet-QA ships its QA pairs as small JSON files in this repo.
ANNOTATION_REPO = "https://github.com/MILVLG/activitynet-qa.git"


def download_model(config: ProjectConfig) -> dict[str, str]:
    """Snapshot the pinned checkpoint into the project HF cache.

    The revision is pinned to a commit SHA in the config, so this
    returns byte-identical weights on every machine.
    """
    from huggingface_hub import snapshot_download

    cache_dir = config.paths.resolve("hf_cache_dir")
    cache_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Downloading %s @ %s", config.model.model_id, config.model.revision)
    LOGGER.info("Cache: %s", cache_dir)

    local_path = snapshot_download(
        repo_id=config.model.model_id,
        revision=config.model.revision,
        cache_dir=str(cache_dir),
        max_workers=8,
    )
    size_gb = sum(f.stat().st_size for f in Path(local_path).rglob("*") if f.is_file()) / 1024**3
    LOGGER.info("Done: %s (%.1f GB)", local_path, size_gb)
    return {"model_id": config.model.model_id, "revision": config.model.revision,
            "local_path": local_path, "size_gb": round(size_gb, 2)}


def download_annotations(config: ProjectConfig) -> dict[str, str]:
    """Clone the ActivityNet-QA annotation repository (QA pairs only).

    Only annotations live here - the videos are a separate acquisition
    problem handled in milestone 1.
    """
    import subprocess

    target = config.paths.resolve("annotations_dir")
    target.parent.mkdir(parents=True, exist_ok=True)

    if (target / ".git").exists():
        LOGGER.info("Annotations already present at %s", target)
    else:
        LOGGER.info("Cloning %s -> %s", ANNOTATION_REPO, target)
        subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "clone", "--depth", "1", ANNOTATION_REPO, str(target)],
            check=True,
        )
    files = sorted(p.name for p in target.rglob("*.json"))
    LOGGER.info("Annotation JSON files: %s", files or "none found at top level")
    return {"repo": ANNOTATION_REPO, "local_path": str(target), "json_files": files}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download project assets")
    parser.add_argument("-c", "--config", default="configs/base.yaml")
    parser.add_argument("--model", action="store_true", help="download the pinned checkpoint")
    parser.add_argument("--annotations", action="store_true", help="clone ActivityNet-QA annotations")
    args = parser.parse_args(argv)

    if not (args.model or args.annotations):
        parser.error("choose at least one of --model / --annotations")

    config = load_config(args.config)
    record: dict[str, object] = {"milestone": "0_asset_download"}

    if args.model:
        record["model"] = download_model(config)
    if args.annotations:
        record["annotations"] = download_annotations(config)

    path = write_report(record, config.paths.resolve("reports_dir") / "asset_download.json")
    LOGGER.info("Report written to %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
