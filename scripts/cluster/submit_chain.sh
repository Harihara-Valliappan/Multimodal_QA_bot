#!/bin/bash
# Submit sbatch jobs one after another within the QOS 2-job-per-user limit.
#   scripts/cluster/submit_chain.sh <sbatch-script> "<args for job 1>" "<args for job 2>" ...
# A pending job counts against the limit, so --dependency cannot queue more.
set -uo pipefail
SQ=/cm/local/apps/slurm/current/bin/squeue
SB=/cm/local/apps/slurm/current/bin/sbatch
cd "$(dirname "$0")/../.."
script="$1"; shift
for args in "$@"; do
  while [ "$($SQ -u "$USER" -h | wc -l)" -ge 2 ]; do sleep 60; done
  echo "$(date '+%F %T') submitting: $args"
  # shellcheck disable=SC2086
  $SB "$script" $args
done
while [ "$($SQ -u "$USER" -h | wc -l)" -gt 0 ]; do sleep 60; done
echo "$(date '+%F %T') CHAIN DONE"
