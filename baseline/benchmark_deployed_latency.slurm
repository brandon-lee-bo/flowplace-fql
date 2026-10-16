#!/bin/bash
#SBATCH --job-name=placement-deploy-latency
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/baseline/logs/deployed_latency_%j.out
#SBATCH --error=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/baseline/logs/deployed_latency_%j.err
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
.venv/bin/python -u baseline/benchmark_deployed_latency.py \
  --dataset-dir datasets/bp_fe_macro_deterministic_v1 \
  --training-dir runs/bp_fe_macro_deterministic_v1 \
  --output baseline/artifacts/deployed_latency_v1
