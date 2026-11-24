#!/usr/bin/env python3
"""Entry point for Milestone 1: build the dataset manifests, write a report.

Usage
-----
    python scripts/build_manifests.py                         # defaults
    python scripts/build_manifests.py -c configs/base.yaml
    python scripts/build_manifests.py --no-verify-decode      # fast dry run
    python scripts/build_manifests.py --train-videos 400 --val-videos 100

Exit code is 1 when the manifests cannot be built or an invariant fails,
so Slurm and CI can gate on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make ``src/`` importable without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from videoqa.config import load_config  # noqa: E402
from videoqa.logging_utils import get_logger  # noqa: E402
from videoqa.manifest import ManifestError, build_manifests  # noqa: E402

LOGGER = get_logger("build_manifests")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Milestone 1 dataset manifest builder")
    parser.add_argument("-c", "--config", default=None, help="YAML config (defaults to built-in defaults)")
    parser.add_argument("--pool-split", default="test", help="annotation split the pool is drawn from")
    parser.add_argument("--train-videos", type=int, default=400, help="videos dealt to the train split")
    parser.add_argument("--val-videos", type=int, default=100, help="videos dealt to the val split")
    parser.add_argument("--seed", type=int, default=None, help="override the config's train seed")
    parser.add_argument(
        "--no-verify-decode",
        action="store_true",
        help="skip the per-clip decode check (faster, but Milestone 1 is not satisfied without it)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="milestone1_manifest_report.json",
        help="report filename, written under the configured reports_dir",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.train.seed

    annotations_dir = config.paths.resolve("annotations_dir")
    videos_dir = config.paths.resolve("videos_dir")
    manifests_dir = config.paths.resolve("manifests_dir")
    reports_dir = config.paths.resolve("reports_dir")

    LOGGER.info("annotations: %s", annotations_dir)
    LOGGER.info("videos:      %s", videos_dir)
    LOGGER.info("manifests:   %s", manifests_dir)
    LOGGER.info("pool split=%s  train=%d val=%d videos  seed=%d",
                args.pool_split, args.train_videos, args.val_videos, seed)
    if args.no_verify_decode:
        LOGGER.warning("decode verification disabled: this run does NOT satisfy Milestone 1")

    try:
        manifests, exclusions, stats = build_manifests(
            annotations_dir=annotations_dir,
            videos_dir=videos_dir,
            out_dir=manifests_dir,
            pool_split=args.pool_split,
            train_videos=args.train_videos,
            val_videos=args.val_videos,
            seed=seed,
            min_frames=config.video.min_decodable_frames,
            verify_decode=not args.no_verify_decode,
        )
    except ManifestError as exc:
        LOGGER.error("manifest build failed: %s", exc)
        return 1

    for split in ("train", "val", "test"):
        LOGGER.info(
            "%-5s %5d QA pairs over %4d videos",
            split, stats.split_counts[split], stats.split_video_counts[split],
        )
    LOGGER.info("excluded %d examples: %s", len(exclusions), stats.exclusion_counts or "none")

    report = {
        "milestone": 1,
        "seed": seed,
        "verify_decode": not args.no_verify_decode,
        "sources": {
            "annotations": "MILVLG/activitynet-qa (official QA annotations)",
            "videos": "VLM2Vec/ActivityNetQA (per-video mirror)",
            "caveat": (
                "Video pool is drawn from the official ActivityNet-QA TEST split, because no "
                "public mirror distributes the train videos. Our train/val/test partition is "
                "disjoint by video, so held-out sets are unseen, but results are NOT comparable "
                "to published ActivityNet-QA test numbers."
            ),
        },
        "config": config.to_dict(),
        "stats": stats.to_dict(),
        "manifest_files": {s: str(manifests_dir / f"{s}.jsonl") for s in manifests},
        "exclusions_file": str(manifests_dir / "exclusions.jsonl"),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / args.output
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("report written to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
