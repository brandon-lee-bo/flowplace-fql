#!/bin/bash
#SBATCH --job-name=qstudy_graph_inference_cost
#SBATCH --partition=gpu-scavenger
#SBATCH --qos=gpu-scavenger
#SBATCH --gres=gpu:nvidia_l20:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --exclude=natural-selection,enterprise,blue-space
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/logs/qstudy_graph_inference_cost_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR="/scratch/bohanli/flowplace_graph/qstudy_graph_inference_cost/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
exec .venv/bin/python -u -m analysis.qstudy_graph_inference_cost "$@"
