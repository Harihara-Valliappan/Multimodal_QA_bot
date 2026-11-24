# Project brief: decisions and rationale

Condensed from the handoff brief, with the choices that are now fixed in code.

## The question

Can DoRA fine-tuning on video-question-answer examples improve task-specific VideoQA
performance over the same pretrained checkpoint — particularly on activity-related
questions? The experiment must *test* this, not assume it. A null result, reported
honestly, is a valid outcome.

## Fixed decisions

| Decision | Choice | Why |
|---|---|---|
| Model | InternVL3.5-8B (HF format) | a single vision-language model that conditions on frames and answers questions; 8B is the largest size that trains comfortably on one 46 GB A40 |
| Training data | ActivityNet-QA | supervised QA pairs over videos — the labels match the target behaviour |
| Adaptation | DoRA (PEFT) | decomposes weight magnitude and direction while staying parameter-efficient |
| Evaluation | held-out ActivityNet-QA split | same questions, same decoding, before and after |
| Benchmark (optional) | Video-MME | generalization only; **never** trained on or used for model selection |

## Why not alternatives

**Why not train on Video-MME?** It is a benchmark. Training on its questions, or
selecting hyperparameters against its labels, destroys the only thing it is good for.
It stays as an optional out-of-domain test.

**Why not Florence-2?** Florence-2 is strong at image-level perception (detection,
captioning, OCR, grounding). This task is question answering over events across frames.
Florence-2 could later serve as a perception helper if explicit detections turn out to
improve answers.

## Model size: why 8B

The brief suggested starting at 2B. We start at **8B** instead, because the A40 has
46 GB and the only trainable parameters are DoRA adapters on the language decoder:

| Size | bf16 weights | Fits on one A40 with DoRA? | Note |
|---|---|---|---|
| 2B | ~4 GB | easily | too weak to make the comparison interesting |
| 4B | ~8 GB | easily | **documented fallback** (`configs/fallback_4b.yaml`) |
| 8B | ~16 GB | yes, with frozen vision tower, batch 1, grad checkpointing | **default** |
| 14B+ | ~28 GB+ | not with comfortable activation headroom | out of scope |

A stronger base model also makes a null result more informative: if DoRA does not help
an 8B checkpoint, that is a real finding, not an artifact of starting from something
too small to answer the questions at all.

Switch to 4B — by config, not by code edit — if the 8B run exceeds memory at the chosen
frame count, or if a training step is slow enough that one epoch will not finish in the
queue's time limit. Record which one was used; never compare an 8B adapted model to a
2B baseline.

**Why not full fine-tuning?** Cost, and attribution. A full fine-tune of every parameter
on a few thousand examples mostly measures overfitting. Freezing the vision encoder and
the projector isolates the effect of language-side adaptation, which is the variable
under study.

## Architecture and what is trained

| Component | Role | Initial policy |
|---|---|---|
| InternViT vision encoder | frames → visual features | **frozen** |
| Multimodal projector | visual features → LM token space | **frozen** (unfreezing is a later ablation) |
| Language model | combines video tokens + question → answer | **DoRA** on selected linear layers |
| Frame sampler | picks frames, preserves order | not learned; deterministic uniform |

Target modules start at `q_proj` / `v_proj` but are only committed after inspecting
`model.named_modules()` on the real checkpoint. Module names are never assumed.

## Planned runs

| Run | Setup | Question answered |
|---|---|---|
| A0 | pretrained, no tuning | how good is the original checkpoint? |
| A1 | + DoRA on the ActivityNet-QA train split | does task adaptation help? |
| A2 *(optional)* | + plain LoRA, matched settings | is it DoRA specifically, or just fine-tuning? |
| A3 *(optional)* | DoRA, different target modules / frame count | which design choice matters? |
| A4 *(optional)* | Video-MME evaluation | does the gain transfer? |

## Metrics

Accuracy under the official normalization rules, broken down by question type; latency
(mean and median, with hardware, frame count, precision, batch size); peak GPU memory
for training and inference; and a categorized sample of errors — missed action, temporal
confusion, wrong object, unsupported answer, decode failure.

## Claims that would be dishonest

- Comparing against published numbers under a different input protocol or split.
- Reporting validation performance as test performance.
- Claiming "better temporal reasoning" without a temporal-specific evaluation subset.
- Showing hand-picked qualitative wins without the matching regressions.
- Presenting DoRA as this project's invention. It is not. The contribution is the
  controlled application, the data pipeline, and the error analysis.
