#!/bin/bash
#SBATCH --job-name=qstudy_scalar_advance
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:10:00
#SBATCH --requeue
#SBATCH --exclude=natural-selection,enterprise,blue-space
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec .venv/bin/python -u tools/qstudy_scalar_control_advance.py "$@"
