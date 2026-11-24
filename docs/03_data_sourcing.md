# Data sourcing and the split caveat

*Investigated 2026-10-16, before any manifest was built.*

## What we wanted

The official ActivityNet-QA annotations split cleanly:

| Split | Videos | QA pairs | QA per video |
|-------|--------|----------|--------------|
| train | 3,200  | 32,000   | 10           |
| val   | 1,800  | 18,000   | 10           |
| test  | 800    | 8,000    | 10           |

Annotations are a few MB from [MILVLG/activitynet-qa](https://github.com/MILVLG/activitynet-qa)
and are already vendored under `data/activitynet_qa/annotations/`. The videos
are the problem: ActivityNet-QA ships question-answer pairs, not clips.

## What is actually available

Every candidate mirror was indexed rather than guessed at. For the zip-chunked
mirror this meant reading each archive's central directory through HTTP range
requests — about 200 KB per 4.9 GB chunk, so the survey cost almost nothing.

| Mirror | Size | Train | Val | Test | Notes |
|--------|------|-------|-----|------|-------|
| `lmms-lab/ActivityNetQA` | 121 GB | 0 | 0 | 768/800 | 28 opaque zips; also holds 4,067 clips outside the benchmark |
| `VLM2Vec/ActivityNetQA` | 13.9 GB | 0 | 0 | 609/800 | 609 individual mp4s, per-video download |
| `YimuWang/ActivityNet` | 537 GB | yes | yes | yes | full v1.3, but as 56 split-parts of a non-seekable `.tar.gz` |

**No mirror distributes the ActivityNet-QA training videos as such.** Obtaining
genuine train-split clips means streaming the 220 GB `v1-3_train_val` archive
and extracting matches on the fly, since gzip cannot be seeked.

## What we chose

The per-video mirror, with our own partition carved out of it **by video**:

```
609 videos, all carrying official annotations
    train 400 videos / val 100 / test 109
    6,090 QA pairs at exactly 10 per video
```

Rejected the streaming option: ~30 GB of transfer and bespoke extraction code,
to buy comparability this study does not depend on.

## The caveat, stated once and carried everywhere

Our training videos come from the **official test split**. Therefore:

- **Valid:** the base-vs-DoRA comparison. Our splits are disjoint by video, so
  the held-out sets are genuinely unseen by the adapter, and both models are
  scored on the identical held-out data.
- **Invalid:** any claim of an ActivityNet-QA *test-set* score. These numbers
  are not comparable to published results, and must never be tabulated
  alongside them without this note.

The caveat is enforced in three places so it cannot quietly go missing: the
module docstring in `src/videoqa/manifest.py`, the `sources.caveat` field of
`reports/milestone1_manifest_report.json`, and this document.

## Disk

The cluster home directory is quota-limited at roughly 100 GB, which is why the
121 GB mirror was never a candidate even before the split problem surfaced.

| Item | Size |
|------|------|
| InternVL3.5-8B checkpoint | 16 GB |
| 609 videos | 14 GB |
| Annotations | 14 MB |

## Built result (2026-10-16, seed 42)

| Split | Videos | QA pairs |
|-------|--------|----------|
| train | 400    | 4,000    |
| val   | 100    | 1,000    |
| test  | 109    | 1,090    |

Of the 800 videos in the official test annotations, 609 were obtainable; the
missing 191 account for all 1,910 excluded QA pairs, logged individually in
`data/manifests/exclusions.jsonl`. Every one of the 609 clips decoded
successfully, so there were no undecodable exclusions.
