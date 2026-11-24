"""The training loop for run A1 (DoRA) and run A2 (plain LoRA control).

One loop, two arms
------------------
Both arms execute this identical code; the only difference is
``dora.use_dora`` in the config. That matters because the A2 question is
"is it DoRA specifically, or just fine-tuning?" -- and any difference in
data order, schedule or evaluation between the arms would answer a
different question.

Determinism
-----------
The example order is a seeded permutation, so both arms see the same
examples in the same order. Seeds are set for python, numpy and torch.

Stopping
--------
Milestone 4 requires stopping on NaN loss or collapsing validation
quality. Both are implemented: a non-finite loss aborts immediately, and
validation accuracy that falls far below the first measurement aborts
too, since continuing would only burn GPU hours.

Checkpoint selection
--------------------
On validation only. The test split is never read here.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from videoqa.config import ProjectConfig
from videoqa.data.collate import build_training_sample
from videoqa.data.video import FrameSampler, SampledClip, VideoDecodeError
from videoqa.evaluation.runner import run_evaluation
from videoqa.logging_utils import get_logger
from videoqa.models.dora import (
    adapter_variant,
    auxiliary_loss,
    changed_parameters,
    parameters_without_grad,
    snapshot_adapter,
)
from videoqa.models.internvl import InternVLRunner, peak_memory_gb, reset_peak_memory

LOGGER = get_logger("train")


class TrainingAborted(RuntimeError):
    """Raised when training stops early for a diagnosable reason."""


def set_all_seeds(seed: int) -> None:
    """Seed python, numpy and torch so both arms are comparable."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass


class ClipCache:
    """Bounded cache of decoded clips, keyed by video name.

    Training shuffles examples, so the same video reappears at
    unpredictable intervals; decoding it each time would dominate CPU
    time. The cache is bounded because the tensors are large (8 frames
    x 3 x 448 x 448) and the node has finite RAM.
    """

    def __init__(self, sampler: FrameSampler, max_items: int = 128) -> None:
        self.sampler = sampler
        self.max_items = max_items
        self._store: dict[str, SampledClip] = {}
        self.hits = 0
        self.misses = 0

    def get(self, video_name: str, path: Path) -> SampledClip:
        clip = self._store.get(video_name)
        if clip is not None:
            self.hits += 1
            return clip
        self.misses += 1
        clip = self.sampler.load(path)
        if len(self._store) >= self.max_items:
            self._store.pop(next(iter(self._store)))  # FIFO: cheap and adequate
        self._store[video_name] = clip
        return clip


@dataclass(slots=True)
class TrainingState:
    """Everything worth writing to the log at each step."""

    step: int = 0
    examples_seen: int = 0
    losses: list[float] = field(default_factory=list)
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    best_metric: float = -1.0
    best_step: int = -1
    decode_failures: int = 0
    aborted: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.step,
            "examples_seen": self.examples_seen,
            "loss_curve": [round(l, 6) for l in self.losses],
            "evaluations": self.evaluations,
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "decode_failures": self.decode_failures,
            "aborted": self.aborted,
        }


def _lr_at(step: int, total: int, base_lr: float, warmup_ratio: float, schedule: str) -> float:
    """Linear warmup then cosine decay, computed explicitly.

    Written out rather than pulled from a scheduler so the exact shape is
    visible in the logs and identical across arms.
    """
    warmup = max(1, int(total * warmup_ratio))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if schedule != "cosine":
        return base_lr
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train(
    *,
    peft_model: Any,
    tokenizer: Any,
    config: ProjectConfig,
    train_examples: Sequence[dict[str, Any]],
    val_examples: Sequence[dict[str, Any]],
    project_root: Path,
    out_dir: Path,
    reports_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    collapse_tolerance: float = 0.5,
    run_name: str | None = None,
) -> TrainingState:
    """Run one short training pass, evaluating and checkpointing on the way."""
    variant = adapter_variant(config)
    # Artifacts are named by run, not by variant: two LoRA runs (A2 and the
    # 16-frame A3) must not overwrite each other's logs.
    run_name = run_name or f"a1_{variant}"
    tc = config.train
    set_all_seeds(tc.seed)

    order = list(range(len(train_examples)))
    random.Random(tc.seed).shuffle(order)
    if tc.max_train_samples is not None:
        order = order[: tc.max_train_samples]

    accum = max(1, tc.gradient_accumulation_steps)
    total_steps = max(1, len(order) // accum)
    LOGGER.info("variant=%s examples=%d accum=%d -> %d optimizer steps",
                variant, len(order), accum, total_steps)

    sampler = FrameSampler(num_frames=config.video.num_frames,
                           input_size=config.video.input_size,
                           backend=config.video.backend)
    cache = ClipCache(sampler)

    if tc.gradient_checkpointing:
        peft_model.base_model.model.language_model.gradient_checkpointing_enable()
        peft_model.enable_input_require_grads()

    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    # Two parameter groups: the temporal adapter has its own learning rate
    # (see TemporalConfig). Each group keeps its base LR; the warmup/cosine
    # schedule is applied to both as the same multiplicative factor.
    temporal_params = [p for n, p in peft_model.named_parameters() if p.requires_grad and "temporal_" in n]
    other_params = [p for n, p in peft_model.named_parameters() if p.requires_grad and "temporal_" not in n]
    groups = []
    if other_params:
        groups.append({"params": other_params, "base_lr": tc.learning_rate})
    if temporal_params:
        groups.append({"params": temporal_params, "base_lr": config.temporal.learning_rate})
    optimizer = torch.optim.AdamW(groups, lr=tc.learning_rate, weight_decay=tc.weight_decay)

    from videoqa.models.temporal import find_temporal_adapter, save_temporal_adapter

    temporal = find_temporal_adapter(peft_model)

    def _save(path: Path) -> None:
        peft_model.save_pretrained(str(path))
        if temporal is not None:
            save_temporal_adapter(temporal, path)
    initial = snapshot_adapter(peft_model)

    # Validation generates text, which lives on the inner InternVLChatModel
    # (it owns .chat). Adapters are injected into its submodules, so this
    # object is the adapted model, not the base one.
    inner = peft_model.base_model.model
    eval_runner = InternVLRunner(inner, tokenizer, config)

    state = TrainingState()
    log_path = reports_dir / f"{run_name}_training_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", encoding="utf-8")
    reset_peak_memory()
    started = time.perf_counter()

    def _log(payload: dict[str, Any]) -> None:
        log_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        log_fh.flush()

    def _validate(step: int) -> float:
        peft_model.eval()
        preds = reports_dir / f"{run_name}_val_step{step}_predictions.jsonl"
        outcome = run_evaluation(
            runner=eval_runner, sampler=sampler, examples=val_examples,
            predictions_path=preds, project_root=project_root, log_every=10_000,
        )
        peft_model.train()
        metric = outcome.metrics["normalized"]["overall"]
        record = {
            "event": "validation", "step": step,
            "normalized": metric,
            "exact_match": outcome.metrics["official_exact_match"]["overall"],
            "formatting_gap": outcome.metrics["formatting_gap"],
            "decode_failures": outcome.decode_failures,
            "latency_mean_s": outcome.latency["mean"],
            "predictions_file": str(preds),
        }
        state.evaluations.append(record)
        _log(record)
        LOGGER.info("step %d validation: normalized %.4f  exact %.4f",
                    step, metric, outcome.metrics["official_exact_match"]["overall"])
        return metric

    try:
        peft_model.train()
        idx = 0
        for step in range(total_steps):
            lr = _lr_at(step, total_steps, tc.learning_rate, tc.warmup_ratio, tc.lr_scheduler_type)
            factor = lr / tc.learning_rate
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"] * factor

            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            step_aux = 0.0
            used = 0
            for _ in range(accum):
                if idx >= len(order):
                    break
                ex = train_examples[order[idx]]
                idx += 1
                try:
                    clip = cache.get(ex["video_name"], project_root / ex["video_path"])
                except VideoDecodeError as exc:
                    state.decode_failures += 1
                    LOGGER.warning("skipping %s: %s", ex["example_id"], exc)
                    continue
                sample = build_training_sample(
                    model=inner, tokenizer=tokenizer, clip=clip,
                    question=ex["question"], answer=ex["answer"],
                ).to(device, dtype)
                out = peft_model(**sample.as_kwargs())
                aux = auxiliary_loss(peft_model)  # 0.0 unless KaSA / MonteCLoRA
                loss = (out.loss + aux) / accum
                step_aux += float(aux.detach()) if torch.is_tensor(aux) else float(aux)
                if not torch.isfinite(loss):
                    raise TrainingAborted(f"non-finite loss at step {step}: {loss.item()}")
                loss.backward()
                step_loss += float(loss.detach())
                used += 1
                state.examples_seen += 1

            if used == 0:
                LOGGER.warning("step %d had no usable examples", step)
                continue

            missing = parameters_without_grad(peft_model)
            if missing:
                raise TrainingAborted(
                    f"{len(missing)} trainable tensors lost their gradient at step {step}: {missing[:3]}"
                )
            torch.nn.utils.clip_grad_norm_(trainable, tc.max_grad_norm)
            optimizer.step()
            state.step = step + 1
            state.losses.append(step_loss)

            if (step + 1) % tc.logging_steps == 0:
                window = state.losses[-tc.logging_steps:]
                LOGGER.info("step %d/%d loss %.4f (window mean %.4f) lr %.2e cache %d/%d",
                            step + 1, total_steps, step_loss, sum(window) / len(window),
                            lr, cache.hits, cache.hits + cache.misses)
                gate = float(temporal.temporal_gate.detach().abs().mean()) if temporal is not None else None
                _log({"event": "train", "step": step + 1, "loss": round(step_loss, 6), "temporal_gate_abs": gate,
                      "aux_loss": round(step_aux / max(1, used), 8), "lr": lr})

            is_last = (step + 1) == total_steps
            if (step + 1) % tc.eval_steps == 0 or is_last:
                metric = _validate(step + 1)
                if metric > state.best_metric:
                    state.best_metric = metric
                    state.best_step = step + 1
                    _save(out_dir / "best")
                    LOGGER.info("new best (%.4f) -> %s", metric, out_dir / "best")
                first = state.evaluations[0]["normalized"]
                if len(state.evaluations) > 1 and metric < first * collapse_tolerance:
                    raise TrainingAborted(
                        f"validation collapsed: {metric:.4f} is below {collapse_tolerance:g}x "
                        f"the first measurement ({first:.4f})"
                    )

    except TrainingAborted as exc:
        state.aborted = str(exc)
        LOGGER.error("training aborted: %s", exc)
    finally:
        wall = time.perf_counter() - started
        moved, static = changed_parameters(initial, peft_model)
        _save(out_dir / "last")
        summary = {
            "event": "summary",
            "variant": variant,
            "wall_clock_s": round(wall, 1),
            "peak_memory_gb": round(peak_memory_gb(), 3),
            "trainable_tensors_changed": len(moved),
            "trainable_tensors_unchanged": len(static),
            "clip_cache_hits": cache.hits,
            "clip_cache_misses": cache.misses,
            **state.to_dict(),
        }
        _log(summary)
        log_fh.close()
        LOGGER.info("training done in %.1fs, peak %.2f GB, best %.4f at step %d",
                    wall, peak_memory_gb(), state.best_metric, state.best_step)
    return state
