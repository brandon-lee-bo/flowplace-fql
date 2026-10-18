#!/bin/bash
#SBATCH --job-name=table2_preservation_watch
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=10:00:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/table2_preservation_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Use Slurm}"
RUNNER=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/analysis/watch_table2_preservation.py
test "$(sha256sum "$RUNNER" | cut -d ' ' -f1)" = "$1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec python3 "$RUNNER"
