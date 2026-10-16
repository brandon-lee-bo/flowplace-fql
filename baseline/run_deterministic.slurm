#!/bin/bash
#SBATCH --job-name=flowplace-det-baseline
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/baseline/artifacts/deterministic-baseline-%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python baseline/run_pretrained.py --designs bp_fe --seeds 300 --deterministic --output baseline/artifacts/pretrained_deterministic
.venv/bin/python baseline/validate_adapter.py --deterministic --compiled-reference baseline/artifacts/pretrained_deterministic/bp_fe/seed_300
.venv/bin/python baseline/run_official_stream.py --seed 300 --deterministic --output baseline/artifacts/official_stream_deterministic_seed300
