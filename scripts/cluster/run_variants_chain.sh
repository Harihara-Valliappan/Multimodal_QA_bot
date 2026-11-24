#!/bin/bash
# Train each A4 variant, then test-evaluate it. Ordered so that, under the
# 2-jobs-per-user limit, every eval is submitted only after its training
# job has finished (at most one job is ever pending).
set -uo pipefail
SQ=/cm/local/apps/slurm/current/bin/squeue
SB=/cm/local/apps/slurm/current/bin/sbatch
cd "$(dirname "$0")/../.."
T=slurm/04_train.sbatch; V=slurm/02_baseline_eval.sbatch
E="--split test --limit 1090"
steps=(
 "$T|-c configs/variant_kasa.yaml --val-size 100 --run-name a4_kasa"
 "$T|-c configs/variant_monteclora.yaml --val-size 100 --run-name a4_monteclora"
 "$V|$E --adapter checkpoints/a4_kasa/best --run-name a4_kasa_test"
 "$T|-c configs/variant_bdlora.yaml --val-size 100 --run-name a4_bdlora"
 "$V|$E --adapter checkpoints/a4_monteclora/best --run-name a4_monteclora_test"
 "$T|-c configs/variant_velora.yaml --val-size 100 --run-name a4_velora"
 "$V|$E --adapter checkpoints/a4_bdlora/best --run-name a4_bdlora_test"
 "$V|$E --adapter checkpoints/a4_velora/best --run-name a4_velora_test"
)
for s in "${steps[@]}"; do
  script="${s%%|*}"; args="${s#*|}"
  while [ "$($SQ -u "$USER" -h | wc -l)" -ge 2 ]; do sleep 60; done
  echo "$(date '+%F %T') submitting $script $args"
  # shellcheck disable=SC2086
  $SB "$script" $args
done
while [ "$($SQ -u "$USER" -h | wc -l)" -gt 0 ]; do sleep 60; done
echo "$(date '+%F %T') CHAIN DONE"
