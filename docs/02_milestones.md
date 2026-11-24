# Milestones and definition of done

Each milestone must leave an artifact in `reports/`. The next milestone does not start
until that artifact exists and is valid. This is the discipline that separates an
experiment from a demo.

## Milestone 0 — Environment and compatibility audit

**Code:** `src/videoqa/env_audit.py`, `scripts/run_env_audit.py`, `slurm/00_env_audit.sbatch`

Checks performed:

| Check | FAIL condition |
|---|---|
| `python` | interpreter older than 3.10 |
| `packages` | torch / transformers / peft / accelerate missing |
| `cuda` | torch cannot be imported (no GPU is WARN, not FAIL — login nodes have none) |
| `bf16` | config asks for bf16 but the GPU cannot do it |
| `video_backend` | no video decoder importable at all |
| `peft_dora_support` | `LoraConfig` has no `use_dora`, or silently drops it |
| `disk_space` | under 100 GB free beneath `data_root` |
| `paths` | project directories cannot be created |
| `nvidia_smi` | never fails — driver view is informational |
| `checkpoint_metadata` | the checkpoint id cannot be resolved on the Hub |

**Done when:** the GPU run reports verdict PASS or WARN, the checkpoint SHA is resolved
and pasted into `configs/base.yaml`, and a single video forward pass has been shown to
work.

## Milestone 1 — Dataset manifest

Locate ActivityNet-QA annotations and videos under their license terms. Build
deterministic train/validation/test manifests keyed by `example_id`, with no source
video appearing in more than one split. Verify every path, decode a frame from every
clip, and check for duplicate ids and empty question/answer fields.

**Done when:** manifest files exist with recorded counts, and a report lists every
excluded example *with its reason*. Nothing is dropped silently.

## Milestone 2 — Baseline evaluation (run A0)

Run the untouched checkpoint on the smoke subset, then on the validation subset. Save
raw generations, normalized predictions, per-question records, accuracy, latency and
peak memory.

**Done when:** the baseline prediction file is complete and valid. **Training does not
start before this.**

## Milestone 3 — DoRA smoke test

One tiny batch: forward, loss, backward, optimizer step. Assert that the loss is finite,
that exactly the intended DoRA parameters are trainable, that they receive gradients,
and that their values actually change after the step. Save the adapter, reload it, and
confirm inference uses it.

**Done when:** all assertions pass. If DoRA is unsupported — stop and report the
blocker. Do not substitute LoRA.

## Milestone 4 — Training and validation (run A1)

One short run. Log train loss and the validation metric at each evaluation interval.
Stop on NaN loss or collapsing validation quality. Select the checkpoint on validation
only; the test split stays untouched.

**Done when:** an adapter checkpoint is saved with its exact training config, seed and
logs.

## Milestone 5 — Final comparison and report

Evaluate base and adapted models on the *same* held-out examples with identical decoding
settings. Produce the comparison table, the question-type breakdown, latency and memory,
and a fixed reproducible sample of improvements and regressions categorized by error
type.

**Done when:** the report states what improved, what did not, what the limitations are,
and what the next experiment should be.
