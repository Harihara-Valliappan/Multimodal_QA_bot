"""Unit tests for the manifest builder.

These run in under a second with no GPU and no real videos: the split
logic and the exclusion bookkeeping are what we need to trust before a
job is queued, and both can be exercised against stub files.
"""

from __future__ import annotations

import json

import pytest

from videoqa.manifest import (
    Exclusion,
    ManifestError,
    build_manifests,
    load_qa_pool,
    partition_videos,
    video_path_for,
)


def _write_pool(tmp_path, records):
    """Lay out a minimal annotations tree matching the official format."""
    ds = tmp_path / "annotations" / "dataset"
    ds.mkdir(parents=True)
    (ds / "test_q.json").write_text(
        json.dumps([
            {"question_id": r["example_id"], "video_name": r["video_name"], "question": r["question"]}
            for r in records
        ])
    )
    (ds / "test_a.json").write_text(
        json.dumps([
            {"question_id": r["example_id"], "answer": r["answer"], "type": r.get("type", 3)}
            for r in records
        ])
    )
    return tmp_path / "annotations"


def _fake_videos(tmp_path, names, *, prefix="r_", content=b"not-a-real-video"):
    vd = tmp_path / "videos"
    vd.mkdir(exist_ok=True)
    for n in names:
        (vd / f"{prefix}{n}.mp4").write_bytes(content)
    return vd


# --- splitting -------------------------------------------------------------

def test_partition_is_deterministic_for_a_seed() -> None:
    names = [f"vid{i}" for i in range(50)]
    a = partition_videos(names, train=30, val=10, seed=42)
    b = partition_videos(names, train=30, val=10, seed=42)
    assert a == b


def test_partition_changes_with_seed() -> None:
    names = [f"vid{i}" for i in range(50)]
    assert partition_videos(names, train=30, val=10, seed=1) != partition_videos(
        names, train=30, val=10, seed=2
    )


def test_partition_splits_are_disjoint_and_complete() -> None:
    names = [f"vid{i}" for i in range(50)]
    p = partition_videos(names, train=30, val=10, seed=42)
    assert len(p["train"]) == 30 and len(p["val"]) == 10 and len(p["test"]) == 10
    assert set(p["train"]) | set(p["val"]) | set(p["test"]) == set(names)
    assert not set(p["train"]) & set(p["val"])
    assert not set(p["train"]) & set(p["test"])
    assert not set(p["val"]) & set(p["test"])


def test_partition_rejects_oversized_request() -> None:
    with pytest.raises(ManifestError, match="nothing would be left"):
        partition_videos([f"vid{i}" for i in range(10)], train=9, val=1)


def test_partition_ignores_input_order() -> None:
    names = [f"vid{i}" for i in range(50)]
    assert partition_videos(names, train=30, val=10, seed=7) == partition_videos(
        list(reversed(names)), train=30, val=10, seed=7
    )


# --- path resolution -------------------------------------------------------

@pytest.mark.parametrize("prefix", ["v_", "r_", ""])
def test_video_path_handles_mirror_prefixes(tmp_path, prefix) -> None:
    vd = _fake_videos(tmp_path, ["abc123"], prefix=prefix)
    assert video_path_for(vd, "abc123") is not None


def test_video_path_returns_none_when_absent(tmp_path) -> None:
    vd = _fake_videos(tmp_path, ["abc123"])
    assert video_path_for(vd, "nope") is None


# --- joining and exclusions ------------------------------------------------

def test_load_qa_pool_joins_on_question_id_not_position(tmp_path) -> None:
    ds = tmp_path / "annotations" / "dataset"
    ds.mkdir(parents=True)
    (ds / "test_q.json").write_text(json.dumps([
        {"question_id": "v_a_1", "video_name": "a", "question": "q1"},
        {"question_id": "v_b_2", "video_name": "b", "question": "q2"},
    ]))
    # answers deliberately in the opposite order
    (ds / "test_a.json").write_text(json.dumps([
        {"question_id": "v_b_2", "answer": "second", "type": 3},
        {"question_id": "v_a_1", "answer": "first", "type": 3},
    ]))
    pool = load_qa_pool(tmp_path / "annotations")
    by_id = {r["example_id"]: r["answer"] for r in pool}
    assert by_id == {"v_a_1": "first", "v_b_2": "second"}


def test_missing_annotation_file_is_an_error(tmp_path) -> None:
    (tmp_path / "annotations" / "dataset").mkdir(parents=True)
    with pytest.raises(ManifestError, match="Missing annotation file"):
        load_qa_pool(tmp_path / "annotations")


def test_build_excludes_missing_videos_with_a_reason(tmp_path) -> None:
    records = [
        {"example_id": f"v_{v}_{i}", "video_name": v, "question": "q", "answer": "yes"}
        for v in ["a", "b", "c", "d", "ghost"]
        for i in range(2)
    ]
    ann = _write_pool(tmp_path, records)
    vd = _fake_videos(tmp_path, ["a", "b", "c", "d"])  # 'ghost' deliberately absent
    _, exclusions, stats = build_manifests(
        annotations_dir=ann, videos_dir=vd, out_dir=tmp_path / "manifests",
        train_videos=2, val_videos=1, verify_decode=False,
    )
    assert stats.videos_missing_file == 1
    assert {e.video_name for e in exclusions} == {"ghost"}
    assert {e.reason for e in exclusions} == {"video_file_missing"}


def test_build_excludes_empty_question_and_answer(tmp_path) -> None:
    records = [
        {"example_id": "v_a_1", "video_name": "a", "question": "ok", "answer": "yes"},
        {"example_id": "v_a_2", "video_name": "a", "question": "   ", "answer": "yes"},
        {"example_id": "v_b_1", "video_name": "b", "question": "ok", "answer": ""},
        {"example_id": "v_c_1", "video_name": "c", "question": "ok", "answer": "no"},
        {"example_id": "v_d_1", "video_name": "d", "question": "ok", "answer": "no"},
    ]
    ann = _write_pool(tmp_path, records)
    vd = _fake_videos(tmp_path, ["a", "b", "c", "d"])
    _, exclusions, stats = build_manifests(
        annotations_dir=ann, videos_dir=vd, out_dir=tmp_path / "manifests",
        train_videos=2, val_videos=1, verify_decode=False,
    )
    reasons = {e.example_id: e.reason for e in exclusions}
    assert reasons["v_a_2"] == "empty_question"
    assert reasons["v_b_1"] == "empty_answer"
    assert stats.exclusion_counts["empty_question"] == 1
    assert stats.exclusion_counts["empty_answer"] == 1


def test_build_writes_disjoint_splits_and_files(tmp_path) -> None:
    records = [
        {"example_id": f"v_{v}_{i}", "video_name": v, "question": "q", "answer": "yes"}
        for v in [f"vid{j}" for j in range(10)]
        for i in range(3)
    ]
    ann = _write_pool(tmp_path, records)
    vd = _fake_videos(tmp_path, [f"vid{j}" for j in range(10)])
    out = tmp_path / "manifests"
    manifests, _, stats = build_manifests(
        annotations_dir=ann, videos_dir=vd, out_dir=out,
        train_videos=6, val_videos=2, verify_decode=False,
    )
    assert stats.split_video_counts == {"train": 6, "val": 2, "test": 2}
    assert stats.split_counts == {"train": 18, "val": 6, "test": 6}

    # no video may appear in two splits
    vids = {s: {r["video_name"] for r in rows} for s, rows in manifests.items()}
    assert not vids["train"] & vids["val"] and not vids["train"] & vids["test"]

    for split in ("train", "val", "test"):
        lines = (out / f"{split}.jsonl").read_text().strip().splitlines()
        assert len(lines) == stats.split_counts[split]
        assert json.loads(lines[0])["split"] == split
    assert (out / "manifest_stats.json").is_file()
    assert (out / "exclusions.jsonl").is_file()


def test_answers_are_normalised_to_lowercase(tmp_path) -> None:
    records = [
        {"example_id": f"v_{v}_1", "video_name": v, "question": "q", "answer": "  YES  "}
        for v in ["a", "b", "c", "d"]
    ]
    ann = _write_pool(tmp_path, records)
    vd = _fake_videos(tmp_path, ["a", "b", "c", "d"])
    manifests, _, _ = build_manifests(
        annotations_dir=ann, videos_dir=vd, out_dir=tmp_path / "manifests",
        train_videos=2, val_videos=1, verify_decode=False,
    )
    assert all(r["answer"] == "yes" for rows in manifests.values() for r in rows)


def test_exclusion_serialises_for_the_report() -> None:
    e = Exclusion("v_a_1", "a", "undecodable_video", "RuntimeError: boom")
    assert e.to_dict()["reason"] == "undecodable_video"
    assert "boom" in e.to_dict()["detail"]
