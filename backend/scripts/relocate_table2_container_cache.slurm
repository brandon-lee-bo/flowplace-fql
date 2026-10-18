#!/bin/bash
#SBATCH --job-name=table2_cache_relocate
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/table2_cache_relocate_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${1:?Runner SHA256}"
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
RUNNER="$ROOT/backend/scripts/relocate_table2_container_cache.py"
[[ $(sha256sum "$RUNNER" | cut -d ' ' -f 1) == "$1" ]] || exit 2
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec python3 "$RUNNER"
