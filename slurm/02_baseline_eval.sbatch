#!/bin/bash
#SBATCH --job-name=vqa_a0_eval
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Milestone 2: baseline evaluation of the untouched checkpoint (run A0).
#
#   sbatch slurm/02_baseline_eval.sbatch --smoke        # ~20 examples first
#   sbatch slurm/02_baseline_eval.sbatch                # full val subset
#   sbatch slurm/02_baseline_eval.sbatch --resume       # continue after a kill
#
# Arguments after the script name are forwarded to run_baseline_eval.py.
set -euo pipefail

module load cuda/12.8.0

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"   # submit from the repo root
cd "$PROJECT_DIR"
source .venv/bin/activate

# Keep every download and every temp file inside the project, because the
# home directory is quota-limited at ~100 GB.
export HF_HOME="$PROJECT_DIR/hf_cache"
export HF_HUB_OFFLINE=1          # weights are already cached; fail loudly if not
export TOKENIZERS_PARALLELISM=false

mkdir -p logs reports

echo "node=$(hostname)  job=${SLURM_JOB_ID}  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
python scripts/run_baseline_eval.py -c configs/base.yaml "$@"
