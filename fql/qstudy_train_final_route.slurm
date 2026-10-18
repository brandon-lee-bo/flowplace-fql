#!/bin/bash
#SBATCH --job-name=qstudy_train_drt
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --exclude=natural-selection,enterprise,blue-space
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=02:30:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/logs/qstudy_train_drt_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
unset SLURM_RESTART_COUNT
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="/scratch/bohanli/flowplace_graph/train_drt_host/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
exec .venv/bin/python -u -m fql.qstudy_train_final_route "$@"
