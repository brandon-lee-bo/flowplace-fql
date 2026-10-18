#!/bin/bash
# Explicit first measured zero-Q BC continuation only; no default study chosen.
#SBATCH --job-name=qstudy_scalar_bc_train
#SBATCH --partition=gpu-scavenger
#SBATCH --qos=gpu-scavenger
#SBATCH --gres=gpu:nvidia_l20:1
#SBATCH --exclude=natural-selection,enterprise,blue-space
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/logs/qstudy_scalar_bc_train_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 WANDB_MODE=disabled MPLBACKEND=Agg
exec .venv/bin/python -u -m fql.qstudy_scalar_online_bc_train "$@"
