#!/bin/bash
#SBATCH --job-name=qstudy_critic_matrix
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --exclude=natural-selection,enterprise
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@90
#SBATCH --requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/logs/qstudy_matrix_%A_%a.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled MPLBACKEND=Agg
export TMPDIR="/scratch/bohanli/flowplace_graph/qstudy_tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
exec .venv/bin/python -u -m fql.qstudy_batch run "$@"
