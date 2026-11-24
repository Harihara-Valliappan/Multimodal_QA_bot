# Environment: measured facts

Everything below was measured on 2026-10-15 on the NTU MSAI Slurm cluster, not assumed.
Re-run `python scripts/run_env_audit.py` to regenerate the machine-readable version
under `reports/`.

## Cluster

| Property | Value |
|---|---|
| Scheduler | Slurm 24.11 |
| Partition | `MGPU-TC2` (default, no time limit) |
| Nodes | `TC2N01-02` with 8 GPUs, `TC2N03-08` with 4 GPUs |
| Per node | 128 CPU cores, 453 GB RAM |
| GPU | NVIDIA **A40, 46 GB**, driver 570.148.08 |
| CUDA module | `module load cuda/12.8.0` |
| Home filesystem | shared NFS home (per-user quota ~100 GB) |
| Network | huggingface.co and pypi.org reachable from login and compute nodes |

Consequences for this project:

- **A40 is Ampere → bf16 works.** We use bf16 everywhere and never touch fp16 loss
  scaling.
- **No NVLink / single-GPU jobs.** InternVL3.5-8B in bf16 is ~16 GB of weights; with a
  frozen vision tower, DoRA-only optimizer state, batch size 1 and gradient
  checkpointing it fits in 46 GB with room for activations. Multi-GPU is not needed and
  would complicate the comparison. `configs/fallback_4b.yaml` exists if the measured
  peak memory or step time says otherwise — that decision is made from the milestone 3
  smoke test, not from guesswork.
- **Shared NFS home with 1.6 TB free.** Full ActivityNet video storage is not realistic
  to assume; milestone 1 plans a documented subset and records exactly how many clips
  were obtained.
- **The cluster is online**, so Hugging Face downloads work — but the checkpoint SHA is
  still pinned so results do not change under us.

## Python

| Property | Value |
|---|---|
| Cluster default | Python 3.13.5 (`/apps/anaconda3-2025.06-0`) |
| Used by this project | **Python 3.12.10** via `module load python312` |
| Environment | project-local venv at `./.venv` |

Python 3.12 is chosen because the video/ML wheel ecosystem (decord, torch cu128) has
complete coverage there; 3.13 does not yet.

## Why a venv and not conda

The existing `3d-reconstruction` project uses conda envs under `~/.conda/envs`. For this
project the environment is part of the deliverable: `requirements.txt` plus
`scripts/setup_env.sh` fully describe it, and `./.venv` cannot be accidentally shared
with or mutated by another project.

## Reproducibility record

Every milestone report written under `reports/` embeds:

- timestamp, hostname, Slurm job id, Python executable;
- versions of torch, transformers, peft, accelerate, decord and friends;
- the full resolved config used for that run.

That is what makes a number in the final table traceable back to a specific machine
state.
