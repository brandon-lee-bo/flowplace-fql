#!/bin/bash
#SBATCH --job-name=flowplace-iccad-generate
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/iccad/logs/generate_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONHASHSEED=0
export MPLBACKEND=Agg
exec .venv/bin/python iccad/scripts/generate_iccad.py \
    --output "iccad/assets/generated_superblue18_job${SLURM_JOB_ID}" "$@"
