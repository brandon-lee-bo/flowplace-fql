#!/bin/bash
#SBATCH --job-name=qstudy_scalar_route_contracts
#SBATCH --partition=dev
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:02:00
#SBATCH --no-requeue
#SBATCH --exclude=natural-selection,enterprise,blue-space
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/logs/qstudy_scalar_route_contracts_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1
export TMPDIR="/scratch/bohanli/flowplace_graph/scalar_route_contracts/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
exec .venv/bin/python tools/test_qstudy_scalar_online_control_route.py
