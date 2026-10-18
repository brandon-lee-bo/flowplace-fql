#!/bin/bash
#SBATCH --job-name=table2_capacity_audit
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/table2_capacity_audit_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec python3 "$ROOT/analysis/run_table2_capacity_audit.py" "$@"
