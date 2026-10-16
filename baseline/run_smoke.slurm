#!/bin/bash
#SBATCH --job-name=flowplace-smoke
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/baseline/artifacts/smoke-%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg
.venv/bin/python baseline/run_pretrained.py --designs bp_fe --seeds 300 --output baseline/artifacts/pretrained_original
.venv/bin/python baseline/validate_adapter.py --compiled-reference baseline/artifacts/pretrained_original/bp_fe/seed_300
