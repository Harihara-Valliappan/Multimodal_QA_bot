# Milestone 2 — baseline results (run A0)

Untouched `InternVL3_5-8B` at revision `9bb6a56a`, 500 validation examples
over 100 videos, 8 uniformly sampled frames at 448px, bf16, greedy decoding.
Slurm job 42897 on one A40, 12m17s, **zero decode failures**.

## Headline

| Metric | Value |
|--------|-------|
| Exact match (official rule) | **0.192** |
| Normalized (case/punct/articles) | **0.416** |
| Formatting gap | 0.224 |
| Latency mean / median | 1.01 s / 0.97 s |
| Peak GPU memory | 17.74 GB |

17.74 GB of a 46 GB A40 leaves ample room for DoRA training.

## Per question type

| Type | n | Exact | Normalized | Gap |
|------|---|-------|------------|-----|
| motion | 47 | 0.000 | 0.085 | 0.085 |
| spatial_relation | 49 | 0.000 | 0.143 | 0.143 |
| temporal_relation | 49 | 0.000 | 0.000 | 0.000 |
| yes_no | 136 | 0.331 | 0.779 | 0.449 |
| color | 42 | 0.405 | 0.619 | 0.214 |
| object | 21 | 0.143 | 0.333 | 0.190 |
| location | 21 | 0.048 | 0.238 | 0.190 |
| number | 40 | 0.575 | 0.575 | 0.000 |
| other | 95 | 0.074 | 0.316 | 0.242 |
| **overall** | **500** | **0.192** | **0.416** | 0.224 |
| free (types 3–8) | | 0.270 | 0.555 | |

## What these numbers actually measure

**The yes/no gap is capitalization.** 0.331 → 0.779 is almost entirely the
model writing `No` where gold says `no`. Nothing about video understanding.

**The open-ended zeros are paraphrase, not failure.** `temporal_relation`
scores 0/49 on both metrics. Inspecting the generations shows the model is
frequently right and phrased differently:

| Gold | Prediction |
|------|------------|
| `snow removal` | `Shovel snow` |
| `throwing javelin` | `javelin throw` |
| `put into water tank` | `Put them in water` |
| `shave dog` | `Grooming dog` |

Exact match scores every one of these zero. Our normalization handles case,
punctuation and articles, but deliberately does **not** stem or match
substrings — `"not red"` must never be credited as `"red"`. The published
ActivityNet-QA literature uses GPT-assisted scoring precisely because string
equality collapses on open-ended types.

**Consequence for the DoRA comparison.** Fine-tuning on gold answers teaches
the dataset's *answer style* — terse noun phrases, lowercase — which will lift
exact match regardless of any improvement in video understanding. Expect a
large apparent gain that is partly or mostly formatting.

Three things keep the two effects separable in Milestone 5:
1. report strict and normalized side by side; a gain that appears only in
   strict is style-learning;
2. the per-type table — a genuine gain should move `motion`, `spatial` and
   `temporal`, not just `yes_no`;
3. raw generations are saved for every run, so any scoring rule proposed
   later can be applied without re-running the GPU.

## Known model behaviour

Three of 500 generations were the canned string *"answering does not require
reading text in the image"*, triggered by malformed questions such as *"what
is the under the person in the video"*. This is InternVL behaviour, not a
pipeline fault; `categorize_errors` labels them `unconstrained_generation`.

## Reproducing

```bash
sbatch slurm/02_baseline_eval.sbatch --smoke      # 20 examples, ~7 min
sbatch slurm/02_baseline_eval.sbatch --split val  # 500 examples, ~12 min
```

Outputs: `reports/a0_base_val_predictions.jsonl` (one row per question, with
raw generation, normalized prediction, latency and frame indices) and
`reports/a0_base_val_report.json` (protocol, metrics, error sample, full config).
