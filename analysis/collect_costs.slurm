#!/bin/bash
#SBATCH --job-name=fql_scoped_cost_accounting
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --exclude=enterprise,natural-selection
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/logs/scoped_cost_accounting_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec .venv/bin/python -u analysis/collect_costs.py "$@"
