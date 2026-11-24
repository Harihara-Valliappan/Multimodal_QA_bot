#!/usr/bin/env python3
"""Evaluate a checkpoint, with or without an adapter (A0 / A1 / A2).

Usage
-----
    python scripts/run_baseline_eval.py --smoke              # 20 examples
    python scripts/run_baseline_eval.py --split val          # base, full val subset
    python scripts/run_baseline_eval.py --split val --resume # continue a killed job
    python scripts/run_baseline_eval.py --split val \
        --adapter checkpoints/a1_lora/best --run-name a2_lora_val

Why one script for both
-----------------------
Milestone 5 requires the base and adapted models to be scored on the
*same* examples with *identical* decoding settings. The surest way to
guarantee that is to run the same code: the adapter is loaded on top of
the same checkpoint, and every other knob comes from the same config.

The smoke run exists to catch a broken prompt or a dtype mistake in two
minutes instead of two hours. Milestone 2 is not satisfied until the
validation run finishes.

Exit code is 1 on failure so Slurm and CI can gate on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from videoqa.config import PROJECT_ROOT, load_config  # noqa: E402
from videoqa.data.video import FrameSampler  # noqa: E402
from videoqa.evaluation.runner import (  # noqa: E402
    build_report,
    load_manifest,
    run_evaluation,
    select_subset,
)
from videoqa.logging_utils import get_logger  # noqa: E402
from videoqa.models.internvl import InternVLRunner  # noqa: E402

LOGGER = get_logger("baseline_eval")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Milestone 2 baseline evaluation (run A0)")
    p.add_argument("-c", "--config", default="configs/base.yaml")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--smoke", action="store_true", help="run the small smoke subset instead")
    p.add_argument("--limit", type=int, default=None, help="override the subset size")
    p.add_argument("--resume", action="store_true", help="skip example_ids already in the predictions file")
    p.add_argument("--run-name", default=None, help="defaults to a0_base_<split>[_smoke]")
    p.add_argument("--adapter", default=None,
                   help="path to a PEFT adapter directory; omit to evaluate the base checkpoint")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    run_name = args.run_name or f"a0_base_{args.split}{'_smoke' if args.smoke else ''}"
    size = args.limit if args.limit is not None else (
        config.eval.smoke_subset_size if args.smoke else config.eval.val_subset_size
    )

    manifests_dir = config.paths.resolve("manifests_dir")
    reports_dir = config.paths.resolve("reports_dir")
    examples = select_subset(
        load_manifest(manifests_dir / f"{args.split}.jsonl"), size, seed=config.eval.seed
    )
    LOGGER.info("run=%s split=%s examples=%d seed=%d", run_name, args.split, len(examples), config.eval.seed)
    LOGGER.info("videos touched: %d", len({e["video_name"] for e in examples}))

    LOGGER.info("loading %s (%s)", config.model.model_id, config.model.torch_dtype)
    runner = InternVLRunner.load(config)
    LOGGER.info("loaded on %s as %s", runner.device, runner.dtype)

    adapter_info: dict[str, object] = {"adapter": None}
    if args.adapter:
        from peft import PeftModel

        from videoqa.models.dora import DORA_MAGNITUDE_MARKER

        adapter_path = Path(args.adapter)
        if not adapter_path.is_absolute():
            adapter_path = PROJECT_ROOT / adapter_path
        if not (adapter_path / "adapter_config.json").is_file():
            LOGGER.error("no adapter_config.json under %s", adapter_path)
            return 1
        adapted = PeftModel.from_pretrained(runner.model, str(adapter_path), is_trainable=False).eval()
        # Report which variant was actually loaded, rather than trusting
        # the directory name: a mislabelled arm would invalidate A2.
        from videoqa.models.dora import ADAPTER_MARKERS

        n_mag = sum(1 for n, _ in adapted.named_parameters() if DORA_MAGNITUDE_MARKER in n)
        peft_type = json.loads((adapter_path / "adapter_config.json").read_text()).get("peft_type", "LORA")
        variant = "dora" if n_mag else peft_type.lower()
        n_adapter = sum(1 for n, _ in adapted.named_parameters() if any(m in n for m in ADAPTER_MARKERS))
        LOGGER.info("adapter loaded from %s -> variant=%s (%d magnitude vectors, %d adapter tensors)",
                    adapter_path, variant, n_mag, n_adapter)
        # chat() lives on the inner InternVLChatModel; adapters are
        # injected into its submodules, so this object is the adapted one.
        from videoqa.models.temporal import load_temporal_adapter

        temporal = load_temporal_adapter(adapted.base_model.model, adapter_path,
                                         num_frames=config.video.num_frames)
        if temporal is not None:
            variant = f"{variant}+temporal"
            LOGGER.info("temporal adapter re-attached (mean |gate| = %.4g)",
                        float(temporal.temporal_gate.abs().mean()))
        runner = InternVLRunner(adapted.base_model.model, runner.tokenizer, config)
        adapter_info = {
            "adapter": str(adapter_path),
            "variant_detected": variant,
            "magnitude_vectors": n_mag,
            "adapter_tensors": n_adapter,
        }

    sampler = FrameSampler(
        num_frames=config.video.num_frames,
        input_size=config.video.input_size,
        backend=config.video.backend,
    )

    predictions_path = reports_dir / f"{run_name}_predictions.jsonl"
    outcome = run_evaluation(
        runner=runner,
        sampler=sampler,
        examples=examples,
        predictions_path=predictions_path,
        project_root=PROJECT_ROOT,
        resume=args.resume,
    )

    report = build_report(
        outcome=outcome,
        runner=runner,
        split=args.split,
        subset_size=size,
        seed=config.eval.seed,
        config_dict=config.to_dict(),
        run_name=run_name,
    )
    report["adapter"] = adapter_info
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{run_name}_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    m = outcome.metrics
    LOGGER.info("exact match (official) : %.4f", m["official_exact_match"]["overall"])
    LOGGER.info("normalized             : %.4f", m["normalized"]["overall"])
    LOGGER.info("formatting gap         : %.4f", m["formatting_gap"])
    LOGGER.info("latency mean/median    : %.2fs / %.2fs", outcome.latency["mean"], outcome.latency["median"])
    LOGGER.info("peak GPU memory        : %.2f GB", outcome.peak_memory_gb)
    LOGGER.info("decode failures        : %d", outcome.decode_failures)
    LOGGER.info("predictions -> %s", predictions_path)
    LOGGER.info("report      -> %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
