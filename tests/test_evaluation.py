"""Tests for scoring and the evaluation loop.

The loop is exercised with a stub model, so the bookkeeping that matters
-- incremental writes, resume, decode failures counted not dropped -- is
verified on a login node without touching a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from videoqa.data.video import SampledClip, VideoDecodeError
from videoqa.evaluation.metrics import (
    categorize_errors,
    normalize_answer,
    official_accuracy,
    normalized_accuracy,
    score_records,
)
from videoqa.evaluation.runner import load_manifest, run_evaluation, select_subset
from videoqa.models.internvl import GenerationRecord


# --- normalization ---------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Yes.", "yes"),
    ("  NO!  ", "no"),
    ("A red car", "red car"),
    ("the dog", "dog"),
    ("two", "two"),
    ("", ""),
])
def test_normalize_answer(raw, expected) -> None:
    assert normalize_answer(raw) == expected


def test_normalization_does_not_search_inside_the_generation() -> None:
    # "not red" must NOT be credited as "red": substring matching would
    # turn a negation into a correct answer.
    assert normalize_answer("not red") != normalize_answer("red")


# --- scoring ---------------------------------------------------------------

def _rec(tid, gold, pred):
    return {"type_id": tid, "answer": gold, "prediction": pred}


def test_official_accuracy_is_exact_match() -> None:
    s = official_accuracy([_rec(3, "yes", "Yes."), _rec(3, "no", "no")])
    assert s.overall == 0.5, "'Yes.' must not match 'yes' under the official rule"


def test_normalized_accuracy_forgives_formatting_only() -> None:
    s = normalized_accuracy([_rec(3, "yes", "Yes."), _rec(3, "no", "no")])
    assert s.overall == 1.0


def test_free_pool_covers_types_three_to_eight() -> None:
    recs = [_rec(0, "a", "a"), _rec(3, "yes", "no"), _rec(8, "x", "x")]
    s = official_accuracy(recs)
    assert s.counts["motion"] == 1
    assert s.free == pytest.approx(0.5), "free pool is types 3-8 only"


def test_breakdown_omits_types_with_no_examples() -> None:
    s = official_accuracy([_rec(3, "yes", "yes")])
    assert set(s.counts) == {"yes_no"}


def test_unknown_type_id_falls_into_other() -> None:
    s = official_accuracy([_rec(99, "x", "x")])
    assert s.counts == {"other": 1}


def test_score_records_reports_the_formatting_gap() -> None:
    out = score_records([_rec(3, "yes", "Yes."), _rec(3, "no", "no")])
    assert out["official_exact_match"]["overall"] == 0.5
    assert out["normalized"]["overall"] == 1.0
    assert out["formatting_gap"] == 0.5


def test_empty_input_does_not_divide_by_zero() -> None:
    assert official_accuracy([]).overall == 0.0


def test_error_categories() -> None:
    recs = [
        {"type_id": 3, "answer": "yes", "prediction": "", "decode_failed": True},
        {"type_id": 3, "answer": "yes", "prediction": ""},
        {"type_id": 0, "answer": "running", "prediction": "the man appears to be running along the road quickly"},
        {"type_id": 3, "answer": "yes", "prediction": "maybe"},
        {"type_id": 4, "answer": "red", "prediction": "blue"},
        {"type_id": 4, "answer": "red", "prediction": "Red."},  # correct, excluded
    ]
    cats = {e["category"] for e in categorize_errors(recs)}
    assert cats == {"decode_failure", "empty_generation", "unconstrained_generation",
                    "unsupported_answer", "wrong_answer"}
    assert len(categorize_errors(recs)) == 5, "a formatting-only match is not an error"


# --- the loop --------------------------------------------------------------

class _StubRunner:
    """Answers every question with the gold answer of the first example."""

    def __init__(self, reply="yes"):
        self.reply = reply
        self.calls = 0

    def answer(self, clip, question):
        self.calls += 1
        return GenerationRecord(text=self.reply, latency_s=0.01, prompt_tokens=10,
                                generated_tokens=1, frame_indices=[0] * 8)

    def protocol(self):
        return {"stub": True}


class _StubSampler:
    """Decodes everything, except video names containing 'broken'."""

    def load(self, path):
        if "broken" in str(path):
            raise VideoDecodeError("synthetic failure")
        return SampledClip(torch.zeros(8, 3, 8, 8), [1] * 8, list(range(8)), 240)


def _examples(n, video="vidA", broken=False):
    name = "broken" if broken else video
    return [{
        "example_id": f"v_{name}_{i}",
        "video_name": name,
        "video_path": f"data/activitynet_qa/videos/{name}.mp4",
        "question": "is it red",
        "answer": "yes",
        "type_id": 3,
        "type_name": "yes_no",
    } for i in range(n)]


def test_loop_writes_one_line_per_example(tmp_path) -> None:
    out = tmp_path / "preds.jsonl"
    res = run_evaluation(runner=_StubRunner(), sampler=_StubSampler(), examples=_examples(5),
                         predictions_path=out, project_root=tmp_path)
    assert len(out.read_text().strip().splitlines()) == 5
    assert res.metrics["official_exact_match"]["overall"] == 1.0
    assert res.latency["n_timed"] == 5


def test_decode_failures_are_recorded_not_dropped(tmp_path) -> None:
    out = tmp_path / "preds.jsonl"
    res = run_evaluation(runner=_StubRunner(), sampler=_StubSampler(),
                         examples=_examples(3, broken=True), predictions_path=out,
                         project_root=tmp_path)
    assert res.decode_failures == 3
    assert len(res.records) == 3, "failed clips stay in the denominator"
    assert res.metrics["official_exact_match"]["overall"] == 0.0


def test_clip_is_decoded_once_per_video(tmp_path) -> None:
    # 10 questions about one video must not decode it 10 times.
    class CountingSampler(_StubSampler):
        def __init__(self): self.loads = 0
        def load(self, path):
            self.loads += 1
            return super().load(path)

    s = CountingSampler()
    run_evaluation(runner=_StubRunner(), sampler=s, examples=_examples(10),
                   predictions_path=tmp_path / "p.jsonl", project_root=tmp_path)
    assert s.loads == 1


def test_fresh_run_overwrites_a_stale_predictions_file(tmp_path) -> None:
    out = tmp_path / "preds.jsonl"
    out.write_text('{"example_id": "stale"}\n')
    run_evaluation(runner=_StubRunner(), sampler=_StubSampler(), examples=_examples(2),
                   predictions_path=out, project_root=tmp_path)
    assert "stale" not in out.read_text()


def test_resume_skips_completed_examples(tmp_path) -> None:
    out = tmp_path / "preds.jsonl"
    examples = _examples(6)
    run_evaluation(runner=_StubRunner(), sampler=_StubSampler(), examples=examples[:4],
                   predictions_path=out, project_root=tmp_path)
    runner = _StubRunner()
    res = run_evaluation(runner=runner, sampler=_StubSampler(), examples=examples,
                         predictions_path=out, project_root=tmp_path, resume=True)
    assert runner.calls == 2, "only the 2 unfinished examples should be generated"
    assert len(res.records) == 6, "scoring must cover everything on disk"


def test_resume_tolerates_a_truncated_final_line(tmp_path) -> None:
    out = tmp_path / "preds.jsonl"
    run_evaluation(runner=_StubRunner(), sampler=_StubSampler(), examples=_examples(3),
                   predictions_path=out, project_root=tmp_path)
    with out.open("a") as fh:
        fh.write('{"example_id": "v_vidA_3", "pred')  # killed mid-write
    res = run_evaluation(runner=_StubRunner(), sampler=_StubSampler(), examples=_examples(4),
                         predictions_path=out, project_root=tmp_path, resume=True)
    assert res.decode_failures == 0


# --- subset selection ------------------------------------------------------

def test_subset_is_deterministic_and_sorted() -> None:
    recs = [{"example_id": f"v_x_{i:03d}"} for i in range(100)]
    a = select_subset(recs, 10, seed=42)
    assert a == select_subset(recs, 10, seed=42)
    assert [r["example_id"] for r in a] == sorted(r["example_id"] for r in a)


def test_subset_ignores_manifest_order() -> None:
    recs = [{"example_id": f"v_x_{i:03d}"} for i in range(50)]
    assert select_subset(recs, 10, seed=1) == select_subset(list(reversed(recs)), 10, seed=1)


def test_subset_larger_than_pool_returns_everything() -> None:
    recs = [{"example_id": f"v_x_{i}"} for i in range(5)]
    assert len(select_subset(recs, 500, seed=1)) == 5
    assert len(select_subset(recs, None, seed=1)) == 5


def test_missing_manifest_is_a_clear_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="build_manifests"):
        load_manifest(tmp_path / "nope.jsonl")
