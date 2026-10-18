#!/bin/bash
#SBATCH --job-name=qstudy_congestion_support
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --exclude=natural-selection,enterprise,blue-space
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/logs/qstudy_congestion_support_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec .venv/bin/python -u -m analysis.qstudy_congestion_budget_support --output "/scratch/bohanli/flowplace_graph/qstudy_congestion_budget_support_v1/job${SLURM_JOB_ID}"
