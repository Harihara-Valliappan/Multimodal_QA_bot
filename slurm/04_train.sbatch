#!/bin/bash
#SBATCH --job-name=vqa_train
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Milestone 4: train one adapter arm.
#
# NOTE: the 'normal' QOS caps a job at 6 hours (MaxWall 06:00:00) and
# rejects anything longer with QOSMaxWallDurationPerJobLimit -- the
# partition's own 'infinite' TIMELIMIT is misleading. Size the run to
# fit, and keep the adapter checkpoints written at every eval interval
# so a wall-clock kill still leaves something usable.
#
#   sbatch slurm/04_train.sbatch -c configs/lora_control.yaml   # A2 control
#   sbatch slurm/04_train.sbatch -c configs/base.yaml           # A1 DoRA
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
python scripts/run_training.py "$@"
