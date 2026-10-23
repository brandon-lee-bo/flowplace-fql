#!/bin/bash
#SBATCH --job-name=versions_whole
#SBATCH --partition=dev
#SBATCH --gres=gpu:nvidia_l20:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/logs/versions_whole_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 WANDB_MODE=disabled MPLBACKEND=Agg
export TMPDIR="/scratch/bohanli/flowplace_graph/${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor" TORCH_HOME=/scratch/bohanli/torch
mkdir -p "$TMPDIR"
exec .venv/bin/python -u -m fql.versions_whole_train "$@"
