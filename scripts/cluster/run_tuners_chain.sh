#!/bin/bash
# A5: SHiRA and PEANuT. Waits for the A4 variant chain, then probe -> train -> test-eval.
set -uo pipefail
cd "$(dirname "$0")/../.."
until grep -q "CHAIN DONE" logs/chain_variants.log; do sleep 120; done
exec scripts/cluster/submit_chain_mixed.sh \
  "slurm/04_train.sbatch|-c configs/tuner_shira.yaml --max-train-samples 32 --val-size 10 --run-name probe_shira" \
  "slurm/04_train.sbatch|-c configs/tuner_peanut.yaml --max-train-samples 32 --val-size 10 --run-name probe_peanut" \
  "slurm/04_train.sbatch|-c configs/tuner_shira.yaml --val-size 100 --run-name a5_shira" \
  "slurm/04_train.sbatch|-c configs/tuner_peanut.yaml --val-size 100 --run-name a5_peanut" \
  "slurm/02_baseline_eval.sbatch|--split test --limit 1090 --adapter checkpoints/a5_shira/best --run-name a5_shira_test" \
  "slurm/02_baseline_eval.sbatch|--split test --limit 1090 --adapter checkpoints/a5_peanut/best --run-name a5_peanut_test"
