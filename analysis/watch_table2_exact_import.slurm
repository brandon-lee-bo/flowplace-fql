#!/bin/bash
#SBATCH --job-name=table2_exact_background
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/table2_exact_background_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export FLOWPLACE_TABLE2_OPENROAD_EXE=/fact_data/bohanli/flowplace_fql_backend/tools/openroad_20241214/probe_1787995/extracted/usr/bin/openroad
python3 "$ROOT/analysis/watch_table2_exact_import.py" &
child=$!
trap 'kill -TERM "$child" 2>/dev/null || true; wait "$child" || true; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
trap 'kill -TERM "$child" 2>/dev/null || true; wait "$child" || true; exit 143' TERM INT
wait "$child"
