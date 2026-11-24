#!/usr/bin/env python3
"""Entry point for Milestone 4: train one adapter arm (A1 DoRA / A2 LoRA).

Usage
-----
    python scripts/run_training.py -c configs/lora_control.yaml   # A2 control
    python scripts/run_training.py -c configs/base.yaml           # A1 DoRA
    python scripts/run_training.py -c configs/base.yaml --val-size 100

Both arms run the same loop; the config decides the adapter. The variant
is printed and written into every artifact, so no report can be
mislabelled.

Exit code is 1 if training aborted (NaN loss, lost gradients, collapsing
validation), so Slurm and CI can gate on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from videoqa.config import PROJECT_ROOT, load_config  # noqa: E402
from videoqa.evaluation.runner import load_manifest, select_subset  # noqa: E402
from videoqa.logging_utils import get_logger  # noqa: E402
from videoqa.models.dora import (  # noqa: E402
    adapter_variant,
    apply_adapter,
    assert_only_adapters_trainable,
)
from videoqa.models.internvl import InternVLRunner  # noqa: E402
from videoqa.training.trainer import train  # noqa: E402

LOGGER = get_logger("run_training")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Milestone 4 adapter training")
    p.add_argument("-c", "--config", default="configs/base.yaml")
    p.add_argument("--val-size", type=int, default=200,
                   help="validation examples per in-training evaluation (small on purpose)")
    p.add_argument("--max-train-samples", type=int, default=None, help="override the config")
    p.add_argument("--run-name", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.max_train_samples is not None:
        config.train.max_train_samples = args.max_train_samples

    variant = adapter_variant(config)
    run_name = args.run_name or f"a1_{variant}"
    if config.temporal.enabled:
        variant = f"{variant}+temporal" if not config.temporal.freeze_lora else "temporal_only"
    LOGGER.info("=== run %s | adapter variant: %s ===", run_name, variant.upper())

    manifests_dir = config.paths.resolve("manifests_dir")
    reports_dir = config.paths.resolve("reports_dir")
    out_dir = config.paths.resolve("checkpoints_dir") / run_name

    train_examples = load_manifest(manifests_dir / "train.jsonl")
    val_examples = select_subset(
        load_manifest(manifests_dir / "val.jsonl"), args.val_size, seed=config.eval.seed
    )
    LOGGER.info("train pool %d | in-training val %d (seed %d)",
                len(train_examples), len(val_examples), config.eval.seed)

    LOGGER.info("loading %s", config.model.model_id)
    runner = InternVLRunner.load(config)
    peft_model = apply_adapter(runner.model, config)
    if config.temporal.enabled:
        from videoqa.models.temporal import attach_temporal_adapter

        adapter = attach_temporal_adapter(
            peft_model.base_model.model,
            num_frames=config.video.num_frames,
            kernel_size=config.temporal.kernel_size,
        )
        n_temporal = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        LOGGER.info("temporal adapter attached: %d params, lr %.1e", n_temporal, config.temporal.learning_rate)
        if config.temporal.freeze_lora:
            # Temporal-only ablation: LoRA stays at B = 0, an exact no-op.
            for name, param in peft_model.named_parameters():
                if "lora_" in name:
                    param.requires_grad_(False)
            LOGGER.info("LoRA frozen: temporal-only arm")
    report = assert_only_adapters_trainable(peft_model, config)
    LOGGER.info("variant=%s trainable=%d (%.4f%%) magnitudes=%d lora=%d",
                variant, report.trainable, report.percent,
                report.dora_magnitude_params, report.lora_params)

    state = train(
        peft_model=peft_model,
        tokenizer=runner.tokenizer,
        config=config,
        train_examples=train_examples,
        val_examples=val_examples,
        project_root=PROJECT_ROOT,
        out_dir=out_dir,
        reports_dir=reports_dir,
        device=runner.device,
        dtype=runner.dtype,
        run_name=run_name,
    )

    out = reports_dir / f"{run_name}_training_report.json"
    out.write_text(json.dumps({
        "milestone": 4,
        "run": run_name,
        "variant": variant,
        "status": "aborted" if state.aborted else "completed",
        "trainable": report.to_dict(),
        "training": state.to_dict(),
        "checkpoints": {"best": str(out_dir / "best"), "last": str(out_dir / "last")},
        "config": config.to_dict(),
    }, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("report -> %s", out)
    return 1 if state.aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
