#!/bin/bash
#SBATCH --job-name=table2_native_geometry
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/table2_native_geometry_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Use Slurm}"
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec python3 "$ROOT/analysis/probe_table2_native_macro_geometry.py" "$@"
