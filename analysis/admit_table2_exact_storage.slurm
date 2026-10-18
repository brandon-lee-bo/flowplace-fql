#!/bin/bash
#SBATCH --job-name=table2_storage_admission
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=12:00:00
#SBATCH --exclude=quantum,infinite-frontier
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/table2_storage_admission_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
test "$(sha256sum "$ROOT/analysis/admit_table2_exact_storage.py" | cut -d ' ' -f 1)" = "$1"
python3 "$ROOT/analysis/admit_table2_exact_storage.py" &
child=$!
trap 'kill -TERM "$child" 2>/dev/null || true; wait "$child" || true; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
trap 'kill -TERM "$child" 2>/dev/null || true; wait "$child" || true; exit 143' TERM INT
wait "$child"
