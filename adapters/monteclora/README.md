# MonteCLoRA adapter for InternVL3.5-8B

PEFT adapter (MonteCLoRA, a Bayesian LoRA variant) for
[`OpenGVLab/InternVL3_5-8B`](https://huggingface.co/OpenGVLab/InternVL3_5-8B), fine-tuned for
short-answer video question answering on ActivityNet-QA.

| | |
|---|---|
| Base model | `OpenGVLab/InternVL3_5-8B` (Apache-2.0), revision `9bb6a56a` |
| Targets | `q_proj`, `v_proj` of all 36 Qwen3 layers · r = 16, alpha = 32 |
| Trainable | 7.67M parameters (0.09% of the model) |
| Training | 2,400 QA pairs, 8 frames @ 448 px, lr 2e-5, cosine, 1 epoch, seed 42 |
| Config | [`configs/variant_monteclora.yaml`](../../configs/variant_monteclora.yaml) |
| Needs | `peft>=0.21` (MonteCLoRA support) |

Answers are terse by design (the dataset's style), for example "change tire".
