#!/bin/bash
#SBATCH --job-name=vqa_dora_smoke
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out

# Milestone 3: prove DoRA attaches, trains and reloads on this checkpoint.
#
#   sbatch slurm/03_dora_smoke.sbatch
set -euo pipefail

module load cuda/12.8.0

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"   # submit from the repo root
cd "$PROJECT_DIR"
source .venv/bin/activate

export HF_HOME="$PROJECT_DIR/hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p logs reports checkpoints

echo "node=$(hostname)  job=${SLURM_JOB_ID}  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
python scripts/run_dora_smoke.py -c configs/base.yaml "$@"
