"""Build deterministic train/validation/test manifests for ActivityNet-QA.

Sourcing note, recorded here because it constrains how results may be read
-------------------------------------------------------------------------
No public mirror distributes the ActivityNet-QA *training* videos. The
121 GB ``lmms-lab/ActivityNetQA`` mirror was indexed chunk by chunk (via
HTTP range reads of each zip's central directory) and contains 768 of the
800 official *test* videos plus 4,067 clips outside the benchmark, and
zero train or val videos. The only per-video source,
``VLM2Vec/ActivityNetQA``, is likewise 592 official test videos.

So the pool this module splits is drawn entirely from the official *test*
split, and we carve our own train/val/test partition out of it **by
video**. Consequences, which belong in any write-up:

* within our partition there is no leakage: a source video appears in
  exactly one split, so the held-out sets are genuinely unseen;
* our numbers are **not** comparable to published ActivityNet-QA test
  results, because our training videos are part of that official test set.

The comparison the project actually needs -- base checkpoint versus DoRA
adapter -- stays valid, since both are scored on the identical held-out
split produced here.

Determinism
-----------
Videos are sorted by id and shuffled with an explicit seed before being
dealt into splits, so the same seed reproduces the same partition on any
machine. Nothing is dropped silently: every rejected example is returned
with a machine-readable reason.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from videoqa.config import PROJECT_ROOT

# ActivityNet-QA question-type codes, as defined by the dataset authors.
# Carried through to the manifest so Milestone 5 can report per-type accuracy.
QUESTION_TYPES: dict[int, str] = {
    0: "motion",
    1: "spatial_relation",
    2: "temporal_relation",
    3: "yes_no",
    4: "color",
    5: "object",
    6: "location",
    7: "number",
    8: "other",
}


class ManifestError(RuntimeError):
    """Raised when the inputs are too broken to build a manifest from."""


@dataclass(slots=True)
class Exclusion:
    """One rejected example and why, so the report can explain every drop."""

    example_id: str
    video_name: str
    reason: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "example_id": self.example_id,
            "video_name": self.video_name,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(slots=True)
class ManifestStats:
    """Counts recorded next to the manifests as provenance."""

    pool_split: str
    qa_pairs_seen: int = 0
    videos_seen: int = 0
    videos_usable: int = 0
    videos_missing_file: int = 0
    videos_undecodable: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)
    split_video_counts: dict[str, int] = field(default_factory=dict)
    exclusion_counts: dict[str, int] = field(default_factory=dict)
    type_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pool_split": self.pool_split,
            "qa_pairs_seen": self.qa_pairs_seen,
            "videos_seen": self.videos_seen,
            "videos_usable": self.videos_usable,
            "videos_missing_file": self.videos_missing_file,
            "videos_undecodable": self.videos_undecodable,
            "split_counts": self.split_counts,
            "split_video_counts": self.split_video_counts,
            "exclusion_counts": self.exclusion_counts,
            "type_counts": self.type_counts,
        }


def load_qa_pool(annotations_dir: Path, split: str = "test") -> list[dict[str, Any]]:
    """Join the ``*_q.json`` and ``*_a.json`` files into single records.

    The two files are parallel lists keyed by ``question_id``; we join on
    that key rather than on position, so a reordered release cannot
    silently pair a question with someone else's answer.
    """
    qpath = annotations_dir / "dataset" / f"{split}_q.json"
    apath = annotations_dir / "dataset" / f"{split}_a.json"
    for p in (qpath, apath):
        if not p.is_file():
            raise ManifestError(f"Missing annotation file: {p}")

    questions = json.loads(qpath.read_text(encoding="utf-8"))
    answers = {a["question_id"]: a for a in json.loads(apath.read_text(encoding="utf-8"))}

    records: list[dict[str, Any]] = []
    for q in questions:
        qid = q["question_id"]
        a = answers.get(qid)
        records.append(
            {
                "example_id": qid,
                "video_name": q["video_name"],
                "question": (q.get("question") or "").strip(),
                "answer": (a.get("answer") if a else None) or "",
                "type_id": a.get("type") if a else None,
            }
        )
    return records


def video_path_for(videos_dir: Path, video_name: str) -> Path | None:
    """Resolve a bare ActivityNet id to a file on disk.

    Mirrors disagree on prefixes (``v_``/``r_``) and container (``.mp4``,
    ``.mkv``), so every combination is probed instead of assuming one.
    """
    for prefix in ("v_", "r_", ""):
        for ext in (".mp4", ".mkv", ".webm", ".avi", ".mov"):
            candidate = videos_dir / f"{prefix}{video_name}{ext}"
            if candidate.is_file():
                return candidate
    return None


def probe_video(path: Path, min_frames: int = 1) -> tuple[bool, str]:
    """Decode-check one clip. Returns ``(ok, detail)``.

    A path that exists is not proof of a readable video: truncated
    downloads and zero-length files both pass ``is_file()``. Milestone 1
    requires decoding a frame from every clip, so we actually pull one.
    """
    try:
        from decord import VideoReader  # imported lazily: config loads without it
    except ImportError as exc:  # pragma: no cover - environment-dependent
        return False, f"decord unavailable: {exc}"

    try:
        if path.stat().st_size == 0:
            return False, "zero-length file"
        vr = VideoReader(str(path), num_threads=1)
        n = len(vr)
        if n < min_frames:
            return False, f"only {n} frames, need >= {min_frames}"
        vr[0].asnumpy()  # force an actual decode, not just a header read
        return True, f"{n} frames"
    except Exception as exc:  # decord raises bare RuntimeError on bad files
        return False, f"{type(exc).__name__}: {exc}"


def partition_videos(
    video_names: Iterable[str],
    *,
    train: int,
    val: int,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Deal videos into splits deterministically, with no video reused.

    Sorting first makes the result independent of filesystem iteration
    order; the seeded shuffle then removes any alphabetical bias.
    Whatever remains after ``train`` and ``val`` becomes the test split.
    """
    names = sorted(set(video_names))
    if train + val >= len(names):
        raise ManifestError(
            f"Requested train={train} + val={val} from only {len(names)} usable videos; "
            "nothing would be left for the test split."
        )
    rng = random.Random(seed)
    rng.shuffle(names)
    return {
        "train": sorted(names[:train]),
        "val": sorted(names[train : train + val]),
        "test": sorted(names[train + val :]),
    }


def build_manifests(
    *,
    annotations_dir: Path,
    videos_dir: Path,
    out_dir: Path,
    pool_split: str = "test",
    train_videos: int = 400,
    val_videos: int = 100,
    seed: int = 42,
    min_frames: int = 1,
    verify_decode: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], list[Exclusion], ManifestStats]:
    """Produce the three manifests, the exclusion list and the stats block."""
    records = load_qa_pool(annotations_dir, pool_split)
    stats = ManifestStats(pool_split=pool_split, qa_pairs_seen=len(records))
    exclusions: list[Exclusion] = []

    # --- Stage 1: which videos do we actually hold, and do they decode? ---
    wanted = sorted({r["video_name"] for r in records})
    stats.videos_seen = len(wanted)
    usable: dict[str, Path] = {}
    for name in wanted:
        path = video_path_for(videos_dir, name)
        if path is None:
            stats.videos_missing_file += 1
            continue
        if verify_decode:
            ok, detail = probe_video(path, min_frames)
            if not ok:
                stats.videos_undecodable += 1
                exclusions.append(Exclusion("*", name, "undecodable_video", detail))
                continue
        usable[name] = path
    stats.videos_usable = len(usable)

    if not usable:
        raise ManifestError(f"No usable videos found under {videos_dir}")

    # --- Stage 2: split by video, so no clip spans two splits ---
    partition = partition_videos(usable, train=train_videos, val=val_videos, seed=seed)
    split_of = {v: s for s, vs in partition.items() for v in vs}
    stats.split_video_counts = {s: len(v) for s, v in partition.items()}

    # --- Stage 3: attach QA pairs, rejecting anything malformed ---
    manifests: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    seen_ids: set[str] = set()
    for r in records:
        eid, vname = r["example_id"], r["video_name"]
        if vname not in usable:
            reason = "video_file_missing" if vname not in split_of else "video_unusable"
            exclusions.append(Exclusion(eid, vname, reason))
            continue
        if eid in seen_ids:
            exclusions.append(Exclusion(eid, vname, "duplicate_example_id"))
            continue
        if not r["question"]:
            exclusions.append(Exclusion(eid, vname, "empty_question"))
            continue
        if not str(r["answer"]).strip():
            exclusions.append(Exclusion(eid, vname, "empty_answer"))
            continue
        seen_ids.add(eid)
        split = split_of[vname]
        manifests[split].append(
            {
                "example_id": eid,
                "video_name": vname,
                "video_path": _repo_relative(usable[vname]),
                "question": r["question"],
                "answer": str(r["answer"]).strip().lower(),
                "type_id": r["type_id"],
                "type_name": QUESTION_TYPES.get(r["type_id"], "unknown"),
                "split": split,
            }
        )

    stats.split_counts = {s: len(v) for s, v in manifests.items()}
    stats.exclusion_counts = dict(Counter(e.reason for e in exclusions))
    stats.type_counts = {
        s: dict(Counter(r["type_name"] for r in rows)) for s, rows in manifests.items()
    }

    # --- Stage 4: invariants the rest of the pipeline is allowed to assume ---
    _assert_disjoint(manifests)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in manifests.items():
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "exclusions.jsonl").write_text(
        "".join(json.dumps(e.to_dict(), ensure_ascii=False) + "\n" for e in exclusions),
        encoding="utf-8",
    )
    (out_dir / "manifest_stats.json").write_text(
        json.dumps(stats.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return manifests, exclusions, stats


def _repo_relative(path: Path) -> str:
    """Store paths relative to the repo root so manifests survive a move."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _assert_disjoint(manifests: dict[str, list[dict[str, Any]]]) -> None:
    """Fail loudly if a video or an example_id leaked across splits."""
    vids = {s: {r["video_name"] for r in rows} for s, rows in manifests.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = vids[a] & vids[b]
        if overlap:
            raise ManifestError(f"{len(overlap)} videos appear in both {a} and {b}: {sorted(overlap)[:5]}")
    all_ids = [r["example_id"] for rows in manifests.values() for r in rows]
    if len(all_ids) != len(set(all_ids)):
        dupes = [i for i, c in Counter(all_ids).items() if c > 1]
        raise ManifestError(f"{len(dupes)} duplicate example_ids across splits: {dupes[:5]}")
