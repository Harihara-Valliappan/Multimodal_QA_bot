# Milestone 3 — DoRA verification

Slurm job 42949 on one A40, 58 seconds. All six checks passed.
Report: `reports/milestone3_dora_smoke.json`.

## Results

| Check | Result |
|-------|--------|
| 1. Adapters are genuinely DoRA | 72 `lora_magnitude_vector` parameters |
| 2. Only intended parameters train | 7,852,032 / 8,536,170,496 = **0.0920%**, 0 unexpected |
| 3. Forward produces finite loss | 6.0156 → 5.3594 over two steps |
| 4. Gradients reach every parameter | 216 tensors, all finite |
| 5. Values change after the step | all 216 changed, 72 of them magnitudes |
| 6. Adapter saves, reloads, is used | reload delta **0.00e+00**; adapter vs base max logit delta 26.4 |

Peak GPU memory while training: **19.95 GB** of 46 GB. Adapter on disk: **31 MB**
(`adapter_config.json`, `adapter_model.safetensors`, `README.md`).

Trainable parameters decompose as 7,667,712 LoRA (`lora_A` + `lora_B`) plus
184,320 DoRA magnitudes, across 36 layers × {`q_proj`, `v_proj`}.

## Why "is it really DoRA?" is checked, not assumed

`LoraConfig(use_dora=True)` silently yields plain LoRA if the flag is ignored
or the targeted layer type has no DoRA implementation. The brief forbids a
silent fallback, so `assert_is_dora` looks for the thing only DoRA creates.

DoRA decomposes the update into direction and magnitude:

```
W' = m · (W + BA) / ‖W + BA‖
```

That learnable `m` appears in PEFT as `lora_magnitude_vector`. Verified on a
toy model that the marker discriminates: `use_dora=False` → 0 such parameters,
`use_dora=True` → one per target. On the real checkpoint: 72.

## Two findings worth keeping

### `task_type` must be null, not `CAUSAL_LM`

Job 42947 failed with:

```
TypeError: InternVLChatModel.forward() got an unexpected keyword argument 'inputs_embeds'
```

`PeftModelForCausalLM` injects `inputs_embeds` into `forward()`.
`InternVLChatModel` does not accept it — it takes `pixel_values` and
`image_flags` and builds embeddings itself. The generic `PeftModel` passes
through only the kwargs we supply.

`CAUSAL_LM` looks obviously right for a language model, so the reasoning is
recorded in `config.py`, both YAML configs, and a regression test. This is a
PEFT-wrapper integration issue, **not** evidence that DoRA is unsupported.

### `lora_A` has exactly zero gradient on step 1

PEFT initialises `lora_B` to zeros so the adapter starts as a no-op. Since
the update is `B @ A`, the gradient with respect to `A` is `Bᵀ · dL/dy` —
identically zero until `B` moves. Observed exactly: 72 zero-gradient tensors
on step 1, none on step 2.

An assertion of "every trainable parameter receives a non-zero gradient"
therefore fails on a perfectly healthy run. `parameters_without_grad` flags
only missing or non-finite gradients (real bugs); `parameters_with_zero_grad`
reports exact zeros, which are tolerated on step 1 and fail if they persist.
The smoke test defaults to **two** steps for this reason.

## Known benign warnings

- `torch.utils.checkpoint: None of the inputs have requires_grad=True` —
  emitted by the frozen, gradient-checkpointed vision tower. Gradients still
  reach every adapter tensor (check 5 passed), so this is informational.
- `FlashAttention2 is not installed` — `attn_implementation` is null; the
  model falls back to eager attention. A possible speedup for Milestone 4,
  not a correctness issue.

## Reproducing

```bash
sbatch slurm/03_dora_smoke.sbatch
sbatch slurm/03_dora_smoke.sbatch --steps 3 --lr 1e-4
```
