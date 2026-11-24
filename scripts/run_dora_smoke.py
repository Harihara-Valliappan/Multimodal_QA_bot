#!/usr/bin/env python3
"""Entry point for Milestone 3: prove DoRA actually works on this model.

Usage
-----
    python scripts/run_dora_smoke.py
    python scripts/run_dora_smoke.py --steps 2 --lr 1e-4

Every check is an assertion with a named failure. The brief is explicit
that if DoRA is unsupported we stop and report the blocker rather than
substituting LoRA, so a failure here exits 1 and says which check fell
over.

Checks, in order:
  1. adapters attach and are genuinely DoRA (magnitude vectors exist)
  2. exactly the intended parameters are trainable
  3. a forward pass produces a finite loss
  4. backward delivers a gradient to every trainable parameter
  5. an optimizer step actually changes their values
  6. the adapter saves, reloads, and changes inference
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from videoqa.config import PROJECT_ROOT, load_config  # noqa: E402
from videoqa.data.collate import build_training_sample, count_image_tokens  # noqa: E402
from videoqa.data.video import FrameSampler  # noqa: E402
from videoqa.evaluation.runner import load_manifest, select_subset  # noqa: E402
from videoqa.logging_utils import get_logger  # noqa: E402
from videoqa.models.dora import (  # noqa: E402
    DoraError,
    parameters_with_zero_grad,
    apply_dora,
    assert_is_dora,
    assert_only_adapters_trainable,
    changed_parameters,
    parameters_without_grad,
    snapshot_adapter,
)
from videoqa.models.internvl import InternVLRunner, peak_memory_gb, reset_peak_memory  # noqa: E402

LOGGER = get_logger("dora_smoke")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Milestone 3 DoRA smoke test")
    p.add_argument("-c", "--config", default="configs/base.yaml")
    p.add_argument("--steps", type=int, default=2,
                   help="2 by default: lora_A has zero gradient until lora_B moves off its zero init")
    p.add_argument("--batch", type=int, default=2, help="examples accumulated per step")
    p.add_argument("--lr", type=float, default=1e-4, help="high on purpose: the test is that values move")
    p.add_argument("-o", "--output", default="milestone3_dora_smoke.json")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    checks: dict[str, object] = {}

    reports_dir = config.paths.resolve("reports_dir")
    ckpt_dir = config.paths.resolve("checkpoints_dir") / "dora_smoke_adapter"

    # ---- load base -------------------------------------------------------
    LOGGER.info("loading %s", config.model.model_id)
    runner = InternVLRunner.load(config)
    model, tokenizer = runner.model, runner.tokenizer
    device, dtype = runner.device, runner.dtype
    LOGGER.info("base model on %s (%s)", device, dtype)

    # ---- check 1: adapters attach and are DoRA ---------------------------
    try:
        peft_model = apply_dora(model, config)
        n_mag = assert_is_dora(peft_model)
    except DoraError as exc:
        LOGGER.error("BLOCKER - DoRA unsupported: %s", exc)
        (reports_dir / args.output).write_text(
            json.dumps({"milestone": 3, "status": "blocked", "error": str(exc)}, indent=2)
        )
        return 1
    LOGGER.info("check 1 PASS: DoRA attached, %d magnitude vectors", n_mag)
    checks["is_dora"] = {"passed": True, "magnitude_vectors": n_mag}

    # ---- check 2: exactly the intended parameters are trainable ----------
    report = assert_only_adapters_trainable(peft_model, config)
    LOGGER.info("check 2 PASS: %d/%d trainable (%.4f%%), %d unexpected",
                report.trainable, report.total, report.percent, len(report.unexpected))
    checks["trainable"] = report.to_dict()

    # ---- build a tiny batch from the train manifest ----------------------
    manifests_dir = config.paths.resolve("manifests_dir")
    examples = select_subset(load_manifest(manifests_dir / "train.jsonl"), args.batch, seed=config.train.seed)
    sampler = FrameSampler(num_frames=config.video.num_frames, input_size=config.video.input_size,
                           backend=config.video.backend)
    samples = []
    for ex in examples:
        clip = sampler.load(PROJECT_ROOT / ex["video_path"])
        s = build_training_sample(model=model, tokenizer=tokenizer, clip=clip,
                                  question=ex["question"], answer=ex["answer"])
        samples.append(s)
        LOGGER.info("sample %s: %d tokens (%d image, %d answer)",
                    ex["example_id"], s.input_ids.shape[1], count_image_tokens(s, tokenizer),
                    s.n_answer_tokens)
    checks["batch"] = {
        "examples": [e["example_id"] for e in examples],
        "sequence_lengths": [int(s.input_ids.shape[1]) for s in samples],
        "answer_tokens": [s.n_answer_tokens for s in samples],
    }

    if config.train.gradient_checkpointing:
        peft_model.base_model.model.language_model.gradient_checkpointing_enable()
        peft_model.enable_input_require_grads()

    peft_model.train()
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    before = snapshot_adapter(peft_model)
    reset_peak_memory()

    # ---- checks 3 and 4: finite loss, gradients everywhere ---------------
    losses: list[float] = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for s in samples:
            out = peft_model(**s.to(device, dtype).as_kwargs())
            loss = out.loss / len(samples)
            if not torch.isfinite(loss):
                LOGGER.error("check 3 FAIL: loss is %s", loss.item())
                return 1
            loss.backward()
            step_loss += float(loss.detach())
        losses.append(step_loss)
        LOGGER.info("step %d loss %.4f", step + 1, step_loss)

        missing = parameters_without_grad(peft_model)
        if missing:
            LOGGER.error("check 4 FAIL: %d trainable parameters have a missing or non-finite "
                         "gradient, e.g. %s", len(missing), missing[:3])
            return 1
        zero = parameters_with_zero_grad(peft_model)
        LOGGER.info("check 4 step %d: all %d trainable tensors have finite gradients "
                    "(%d still exactly zero)", step + 1, len(trainable), len(zero))
        if step == 0:
            checks["gradients"] = {
                "passed": True,
                "tensors_with_grad": len(trainable),
                "zero_grad_on_first_step": len(zero),
                "note": ("lora_A legitimately has zero gradient while lora_B is still at its "
                         "zero init; it must be non-zero by the final step."),
            }
        if step == args.steps - 1 and zero:
            LOGGER.error("check 4 FAIL: %d tensors still have zero gradient on the final step: %s",
                         len(zero), zero[:3])
            return 1
        optimizer.step()

    checks["loss"] = {"passed": True, "values": [round(l, 6) for l in losses]}
    LOGGER.info("check 3 PASS: finite loss %s", [round(l, 4) for l in losses])

    # ---- check 5: values actually moved ----------------------------------
    moved, static = changed_parameters(before, peft_model)
    if static:
        LOGGER.error("check 5 FAIL: %d trainable tensors unchanged after the step, e.g. %s",
                     len(static), static[:3])
        return 1
    moved_mag = [n for n in moved if "lora_magnitude_vector" in n]
    LOGGER.info("check 5 PASS: all %d trainable tensors changed (%d of them DoRA magnitudes)",
                len(moved), len(moved_mag))
    checks["values_changed"] = {"passed": True, "changed": len(moved), "magnitude_changed": len(moved_mag)}

    # ---- check 6: save, reload, and confirm inference uses it ------------
    peft_model.eval()
    probe = samples[0].to(device, dtype)
    with torch.inference_mode():
        logits_adapted = peft_model(**probe.as_kwargs()).logits.float().cpu()
        with peft_model.disable_adapter():
            logits_base = peft_model(**probe.as_kwargs()).logits.float().cpu()

    delta = (logits_adapted - logits_base).abs().max().item()
    if delta == 0.0:
        LOGGER.error("check 6 FAIL: adapter changes nothing at inference")
        return 1
    LOGGER.info("adapter changes logits (max |delta| = %.6f)", delta)

    ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(ckpt_dir))
    saved_files = sorted(p.name for p in ckpt_dir.iterdir())
    LOGGER.info("adapter saved to %s (%s)", ckpt_dir, ", ".join(saved_files))

    # Genuinely reload: strip the adapter off the base, then load from disk.
    from peft import PeftModel

    base = peft_model.unload()
    reloaded = PeftModel.from_pretrained(base, str(ckpt_dir), is_trainable=False).eval()
    assert_is_dora(reloaded)
    with torch.inference_mode():
        logits_reloaded = reloaded(**probe.as_kwargs()).logits.float().cpu()

    reload_delta = (logits_reloaded - logits_adapted).abs().max().item()
    if reload_delta > 1e-3:
        LOGGER.error("check 6 FAIL: reloaded adapter gives different logits (max delta %.6f)", reload_delta)
        return 1
    LOGGER.info("check 6 PASS: reloaded adapter reproduces the adapted logits (max delta %.2e)", reload_delta)
    checks["save_reload"] = {
        "passed": True,
        "adapter_dir": str(ckpt_dir),
        "files": saved_files,
        "adapter_vs_base_max_logit_delta": round(delta, 6),
        "reloaded_vs_adapted_max_logit_delta": reload_delta,
    }

    peak = peak_memory_gb()
    LOGGER.info("peak GPU memory during training step: %.2f GB", peak)

    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / args.output
    out_path.write_text(json.dumps({
        "milestone": 3,
        "status": "passed",
        "peak_memory_gb": round(peak, 3),
        "learning_rate": args.lr,
        "steps": args.steps,
        "checks": checks,
        "config": config.to_dict(),
    }, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("ALL CHECKS PASSED - report written to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
