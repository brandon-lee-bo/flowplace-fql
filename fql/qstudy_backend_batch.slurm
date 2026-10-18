#!/bin/bash
#SBATCH --job-name=qstudy_physical_labels
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --exclude=natural-selection,enterprise
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/logs/qstudy_physical_%A_%a.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="/scratch/bohanli/flowplace_graph/qstudy_backend/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
exec .venv/bin/python -u -m fql.qstudy_controller "$@"
