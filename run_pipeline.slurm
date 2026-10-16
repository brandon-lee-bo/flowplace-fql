#!/bin/bash
#SBATCH --job-name=flowplace-fql
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/logs/pipeline_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec .venv/bin/python "$@"
