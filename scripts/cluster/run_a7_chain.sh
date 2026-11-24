#!/bin/bash
# A7: waits for A6, selects the LoRA LR on val, then probe -> train -> VAL eval
# for LoRA+temporal and temporal-only. Test is deliberately NOT run here.
set -uo pipefail
cd "$(dirname "$0")/../.."
until grep -q "CHAIN DONE" logs/chain_a6.log; do sleep 120; done
.venv/bin/python -I scripts/cluster/select_lr_and_write_a7.py || { echo "LR selection failed"; exit 1; }
T=slurm/04_train.sbatch; V=slurm/02_baseline_eval.sbatch
exec scripts/cluster/submit_chain_mixed.sh \
  "$T|-c configs/a7_lora_temporal.yaml --max-train-samples 32 --val-size 10 --run-name probe_lora_temporal" \
  "$T|-c configs/a7_temporal_only.yaml --max-train-samples 32 --val-size 10 --run-name probe_temporal_only" \
  "$T|-c configs/a7_lora_temporal.yaml --val-size 100 --run-name a7_lora_temporal" \
  "$T|-c configs/a7_temporal_only.yaml --val-size 100 --run-name a7_temporal_only" \
  "$V|-c configs/a7_lora_temporal.yaml --split val --adapter checkpoints/a7_lora_temporal/best --run-name a7_lora_temporal_val" \
  "$V|-c configs/a7_temporal_only.yaml --split val --adapter checkpoints/a7_temporal_only/best --run-name a7_temporal_only_val"
