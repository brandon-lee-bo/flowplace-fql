#!/bin/bash
#SBATCH --job-name=flowplace-innovus-retry
#SBATCH --partition=eda
#SBATCH --qos=eda
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:12:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/iccad/logs/innovus_retry_%j.log
set -euo pipefail
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TMPDIR="/scratch/bohanli/flowplace_iccad/innovus_retry_${SLURM_JOB_ID}/tmp"
mkdir -p "$TMPDIR"
cd "$ROOT"
exec /usr/bin/python3 iccad/scripts/retry_innovus.py
