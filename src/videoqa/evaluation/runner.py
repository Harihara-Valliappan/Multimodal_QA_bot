"""The evaluation loop shared by the baseline (A0) and adapted (A1) runs.

Design notes
------------
*Incremental writes.* Predictions are appended to a JSONL file as they
are produced. A job that hits its Slurm wall clock keeps everything it
finished, and ``--resume`` skips those examples on the next attempt.
GPU hours are the scarce resource here, not disk.

*Decode failures are data.* A clip that will not decode is recorded as a
failed example with an empty prediction, not dropped. Dropping it would
silently shrink the denominator and inflate accuracy.

*The subset is deterministic.* Examples are sorted by ``example_id`` and
sampled with an explicit seed, so "the first 500 validation examples"
means the same set in every run.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from videoqa.data.video import FrameSampler, VideoDecodeError
from videoqa.evaluation.metrics import categorize_errors, normalize_answer, score_records
from videoqa.logging_utils import get_logger
from videoqa.models.internvl import InternVLRunner, peak_memory_gb, reset_peak_memory

LOGGER = get_logger("eval")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Read one split's JSONL manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path} (run scripts/build_manifests.py first)")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_subset(
    records: Sequence[dict[str, Any]], size: int | None, seed: int = 42
) -> list[dict[str, Any]]:
    """Deterministically pick ``size`` examples, or all of them.

    Sorting before sampling removes any dependence on manifest order, so
    the same seed selects the same examples even if the manifest is
    rebuilt in a different order.
    """
    ordered = sorted(records, key=lambda r: r["example_id"])
    if size is None or size >= len(ordered):
        return ordered
    rng = random.Random(seed)
    return sorted(rng.sample(ordered, size), key=lambda r: r["example_id"])


def read_predictions(path: Path) -> list[dict[str, Any]]:
    """Read a predictions file, skipping any unparseable line.

    A job killed by the wall clock can leave a half-written final line.
    Both the resume check and the end-of-run scoring read through this
    function, so a truncated line is skipped consistently in both rather
    than crashing the retry that was supposed to recover from it.
    """
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            LOGGER.warning("skipping unparseable line in %s (killed mid-write?)", path.name)
            continue
        if "example_id" in row:
            rows.append(row)
    return rows


def _already_done(path: Path) -> set[str]:
    """example_ids present in a partial predictions file."""
    return {r["example_id"] for r in read_predictions(path)}


@dataclass(slots=True)
class EvalOutcome:
    """Everything milestone 2 has to hand back."""

    records: list[dict[str, Any]]
    metrics: dict[str, Any]
    latency: dict[str, float]
    peak_memory_gb: float
    decode_failures: int
    wall_clock_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "latency_s": self.latency,
            "peak_memory_gb": round(self.peak_memory_gb, 3),
            "decode_failures": self.decode_failures,
            "wall_clock_s": round(self.wall_clock_s, 2),
            "n_examples": len(self.records),
        }


def run_evaluation(
    *,
    runner: InternVLRunner,
    sampler: FrameSampler,
    examples: Sequence[dict[str, Any]],
    predictions_path: Path,
    project_root: Path,
    resume: bool = False,
    log_every: int = 25,
    progress_hook: Callable[[int, int], None] | None = None,
) -> EvalOutcome:
    """Generate an answer for every example and score the result."""
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    done = _already_done(predictions_path) if resume else set()
    if done:
        LOGGER.info("resuming: %d predictions already on disk", len(done))
    elif predictions_path.exists():
        predictions_path.unlink()  # fresh run must not append to a stale file

    # One clip is decoded once and reused for its ~10 questions: decoding
    # dominates CPU time and the frames are identical across them.
    clip_cache: dict[str, Any] = {}
    cached_video: str | None = None

    records: list[dict[str, Any]] = []
    latencies: list[float] = []
    decode_failures = 0
    reset_peak_memory()
    started = time.perf_counter()

    with predictions_path.open("a", encoding="utf-8") as fh:
        for i, ex in enumerate(examples, 1):
            if ex["example_id"] in done:
                continue

            video_path = project_root / ex["video_path"]
            record = {
                "example_id": ex["example_id"],
                "video_name": ex["video_name"],
                "question": ex["question"],
                "answer": ex["answer"],
                "type_id": ex.get("type_id"),
                "type_name": ex.get("type_name"),
            }

            # decode (cached per video)
            if cached_video != ex["video_name"]:
                clip_cache.clear()
                try:
                    clip_cache["clip"] = sampler.load(video_path)
                    cached_video = ex["video_name"]
                except VideoDecodeError as exc:
                    clip_cache["error"] = str(exc)
                    cached_video = ex["video_name"]

            if "clip" not in clip_cache:
                decode_failures += 1
                record.update({
                    "raw_generation": "",
                    "prediction": "",
                    "prediction_normalized": "",
                    "decode_failed": True,
                    "error": clip_cache.get("error", "decode failed"),
                    "latency_s": 0.0,
                })
            else:
                clip = clip_cache["clip"]
                gen = runner.answer(clip, ex["question"])
                latencies.append(gen.latency_s)
                record.update({
                    "raw_generation": gen.text,
                    "prediction": gen.text,
                    "prediction_normalized": normalize_answer(gen.text),
                    "decode_failed": False,
                    "latency_s": round(gen.latency_s, 4),
                    "frame_indices": gen.frame_indices,
                    "prompt_tokens": gen.prompt_tokens,
                    "generated_tokens": gen.generated_tokens,
                })

            records.append(record)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()  # survive a wall-clock kill

            if progress_hook:
                progress_hook(i, len(examples))
            if i % log_every == 0:
                mean_lat = statistics.mean(latencies) if latencies else 0.0
                LOGGER.info("%d/%d  mean latency %.2fs  decode failures %d",
                            i, len(examples), mean_lat, decode_failures)

    wall = time.perf_counter() - started

    # When resuming, score everything on disk, not just this attempt's share.
    if done:
        records = read_predictions(predictions_path)
        latencies = [r["latency_s"] for r in records if not r.get("decode_failed") and r.get("latency_s")]
        decode_failures = sum(1 for r in records if r.get("decode_failed"))

    latency = {
        "mean": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "median": round(statistics.median(latencies), 4) if latencies else 0.0,
        "min": round(min(latencies), 4) if latencies else 0.0,
        "max": round(max(latencies), 4) if latencies else 0.0,
        "n_timed": len(latencies),
    }
    return EvalOutcome(
        records=records,
        metrics=score_records(records),
        latency=latency,
        peak_memory_gb=peak_memory_gb(),
        decode_failures=decode_failures,
        wall_clock_s=wall,
    )


def build_report(
    *,
    outcome: EvalOutcome,
    runner: InternVLRunner,
    split: str,
    subset_size: int | None,
    seed: int,
    config_dict: dict[str, Any],
    run_name: str,
    error_sample_limit: int = 50,
) -> dict[str, Any]:
    """Assemble the JSON report written next to the predictions."""
    return {
        "run": run_name,
        "split": split,
        "subset_size": subset_size,
        "seed": seed,
        "protocol": runner.protocol(),
        "results": outcome.to_dict(),
        "errors_sample": categorize_errors(outcome.records, limit=error_sample_limit),
        "caveat": (
            "Splits are carved from the official ActivityNet-QA TEST videos (no public "
            "mirror ships the train videos). Disjoint by video, so held-out data is unseen, "
            "but these are NOT comparable to published ActivityNet-QA test numbers."
        ),
        "config": config_dict,
    }
