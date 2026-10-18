#!/bin/bash
#SBATCH --job-name=qstudy_flow_branch
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --exclude=natural-selection,enterprise
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/logs/qstudy_flow_branch_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TMPDIR="/scratch/bohanli/flowplace_graph/qstudy_branch_tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
exec .venv/bin/python -u -m fql.qstudy_branch_launch "$@"
