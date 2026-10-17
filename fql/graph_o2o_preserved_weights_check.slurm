#!/bin/bash
#SBATCH --job-name=fql_preserved_weights_reload
#SBATCH --partition=gpu-scavenger
#SBATCH --qos=gpu-scavenger
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --exclude=natural-selection,enterprise
#SBATCH --requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/logs/graph_o2o_preserved_weights_check_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR="/scratch/bohanli/flowplace_graph/preserved_weights_check_job/${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}"
mkdir -p "$TMPDIR"
exec .venv/bin/python -u -m fql.graph_o2o_preserved_weights_check "$@"
