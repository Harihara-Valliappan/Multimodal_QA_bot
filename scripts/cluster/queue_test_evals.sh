#!/bin/bash
# Submit the remaining test-split evaluations as Slurm slots free up.
#
# The 'normal' QOS allows 2 jobs per user (1 running + 1 queued), so all
# three test evals cannot be submitted at once and sbatch --dependency
# does not help: a pending dependent job still counts against the limit.
set -uo pipefail
SQ=/cm/local/apps/slurm/current/bin/squeue
SB=/cm/local/apps/slurm/current/bin/sbatch
cd "$(dirname "$0")/../.."

wait_for_slot() {
  while [ "$($SQ -u "$USER" -h | wc -l)" -ge 2 ]; do sleep 60; done
}

for arm in "checkpoints/a1_lora/best:a2_lora_test" "checkpoints/a1_dora/best:a1_dora_test"; do
  adapter="${arm%%:*}"; name="${arm##*:}"
  wait_for_slot
  echo "submitting $name"
  $SB slurm/02_baseline_eval.sbatch --split test --limit 1090 --adapter "$adapter" --run-name "$name"
done

# wait for everything to drain
while [ "$($SQ -u "$USER" -h | wc -l)" -gt 0 ]; do sleep 60; done
echo "ALL TEST EVALS DONE"
