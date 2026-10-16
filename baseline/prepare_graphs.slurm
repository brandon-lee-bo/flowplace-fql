#!/bin/bash
#SBATCH --job-name=flowplace-graphs
#SBATCH --partition=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/baseline/artifacts/graphs-%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS=1
.venv/bin/python baseline/prepare_graphs.py
