#!/bin/bash
#SBATCH --job-name=euler-bc-learnability
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/baseline/logs/euler_bc_%j.out
#SBATCH --error=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/baseline/logs/euler_bc_%j.err
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
.venv/bin/python -u baseline/diagnose_euler_bc.py \
  --dataset-dir datasets/bp_fe_macro_deterministic_v1 \
  --hard-training-dir runs/bp_fe_macro_deterministic_v1 \
  --euler-diagnostic baseline/artifacts/euler_sampler_train16 \
  --output baseline/artifacts/euler_bc_learnability_v2
