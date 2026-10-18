#!/bin/bash
#SBATCH --job-name=fql_fresh_flow_pilot
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/logs/fresh_flow_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLBACKEND=Agg WANDB_MODE=disabled
exec .venv/bin/python -m fql.run_fresh_flow --output /fact_data/bohanli/flowplace_fql/fresh_flow_pilot_${SLURM_JOB_ID} "$@"
