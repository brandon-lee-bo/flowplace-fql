#!/bin/bash
#SBATCH --job-name=table2_exact_archive_task
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --array=0-5%1
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/table2_exact_archive_task_%A_%a.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${SLURM_ARRAY_TASK_ID:?Requires a task array}"
: "${1:?Frozen tasks.jsonl absolute path}"
: "${2:?Tasks SHA256}"
: "${3:?Archive helper SHA256}"
: "${4:?Archive task driver SHA256}"
[[ ${5:-} == commit-verified-artifacts ]] || { echo 'Explicit commit-verified-artifacts argument required' >&2; exit 2; }
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec python3 "$ROOT/backend/scripts/archive_table2_exact_task.py" \
  --tasks "$1" --tasks-sha256 "$2" --helper-sha256 "$3" --driver-sha256 "$4" \
  --commit-verified-artifacts
