#!/bin/bash
#SBATCH --job-name=table2_capvalidated_archive
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/table2_capvalidated_archive_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec python3 "$ROOT/backend/scripts/archive_table2_capvalidated.py" "$@" --threads "$SLURM_CPUS_PER_TASK"
