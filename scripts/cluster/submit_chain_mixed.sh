#!/bin/bash
# Like submit_chain.sh, but each argument is "<sbatch-script>|<args>".
set -uo pipefail
SQ=/cm/local/apps/slurm/current/bin/squeue
SB=/cm/local/apps/slurm/current/bin/sbatch
cd "$(dirname "$0")/../.."
for s in "$@"; do
  script="${s%%|*}"; args="${s#*|}"
  while [ "$($SQ -u "$USER" -h | wc -l)" -ge 2 ]; do sleep 60; done
  echo "$(date '+%F %T') submitting $script $args"
  # shellcheck disable=SC2086
  $SB "$script" $args
done
while [ "$($SQ -u "$USER" -h | wc -l)" -gt 0 ]; do sleep 60; done
echo "$(date '+%F %T') CHAIN DONE"
