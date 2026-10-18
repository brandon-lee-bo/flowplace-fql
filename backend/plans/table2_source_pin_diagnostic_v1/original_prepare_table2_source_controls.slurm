#!/bin/bash
#SBATCH --job-name=table2_prepare_source_controls
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --exclude=quantum,infinite-frontier
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/table2_prepare_source_controls_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
test "$(sha256sum "$ROOT/analysis/prepare_table2_source_controls.py" | cut -d ' ' -f 1)" = "$1"
exec python3 "$ROOT/analysis/prepare_table2_source_controls.py"
