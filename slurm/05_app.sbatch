#!/bin/bash
#SBATCH --job-name=vqa_app
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Live multimodal demo: camera / image / video + voice -> InternVL3.5 + MonteCLoRA -> voice.
#   sbatch slurm/05_app.sbatch
# Then follow the "CONNECT" lines this job prints into logs/vqa_app_<jobid>.out.
set -euo pipefail
module load cuda/12.8.0
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"   # submit from the repo root
cd "$PROJECT_DIR"
export HF_HOME="$PROJECT_DIR/hf_cache" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false GRADIO_ANALYTICS_ENABLED=False

# Password for the web UI: created once, readable only by you, never printed.
AUTH="$HOME/.config/videoqa-app/auth"
if [ ! -s "$AUTH" ]; then
  mkdir -p "$(dirname "$AUTH")"; umask 077
  echo "$USER:$(python3 -c 'import secrets;print(secrets.token_urlsafe(12))')" > "$AUTH"
fi
chmod 600 "$AUTH"

PORT=7860
NODE=$(hostname -s)
echo "=================================================================="
echo "CONNECT  1) on your laptop:  ssh -N -L ${PORT}:${NODE}:${PORT} ${USER}@<the host you normally ssh into>"
echo "CONNECT  2) open:           http://localhost:${PORT}"
echo "CONNECT  3) login:          user ${USER}, password: run  cat ~/.config/videoqa-app/auth  on the cluster"
echo "=================================================================="
.venv-app/bin/python scripts/run_app.py --host 0.0.0.0 --port "$PORT" --auth-file "$AUTH" "$@"
