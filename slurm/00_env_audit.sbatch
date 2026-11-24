#!/bin/bash
#SBATCH --job-name=vqa_audit
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out

# Milestone 0 on a real GPU node: this is the only place where the CUDA,
# bf16 and nvidia-smi checks can actually pass.
#
#   sbatch slurm/00_env_audit.sbatch
set -euo pipefail

module load cuda/12.8.0

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"   # submit from the repo root
cd "$PROJECT_DIR"
source .venv/bin/activate

export HF_HOME="$PROJECT_DIR/hf_cache"
export TOKENIZERS_PARALLELISM=false

echo "node=$(hostname)  job=${SLURM_JOB_ID}"
python scripts/run_env_audit.py -c configs/base.yaml -o milestone0_environment_audit_gpu.json
