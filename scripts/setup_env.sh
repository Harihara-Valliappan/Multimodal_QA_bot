#!/bin/bash
# Create the project-local virtual environment in ./.venv
#
# Why a venv inside the project rather than a shared conda env: the whole
# experiment is supposed to be reproducible, and a self-contained .venv
# with pinned versions is far easier to describe in a report than a conda
# env living somewhere in $HOME.
#
#   bash scripts/setup_env.sh          # full install
#   bash scripts/setup_env.sh --dev    # + pytest/ruff
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
REQ_FILE="requirements.txt"
[[ "${1:-}" == "--dev" ]] && REQ_FILE="requirements-dev.txt"

# Python 3.12 from the cluster module; the system default (3.13) has
# patchier wheel coverage for the video/ML stack.
module load python312 2>/dev/null || echo "WARN: module python312 unavailable, using $(command -v python3)"
PY=$(command -v python3.12 || command -v python3)
echo "Using interpreter: $PY ($("$PY" -V))"

if [[ ! -d "$VENV_DIR" ]]; then
    "$PY" -m venv "$VENV_DIR"
    echo "Created venv at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# CUDA 12.8 wheels, matching the A40 driver on MGPU-TC2.
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.8.0" "torchvision==0.23.0"

python -m pip install -r "$PROJECT_DIR/$REQ_FILE"
python -m pip install -e "$PROJECT_DIR"

echo
echo "Done. Activate with:  source $VENV_DIR/bin/activate"
python -c "import torch, transformers, peft; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)"
