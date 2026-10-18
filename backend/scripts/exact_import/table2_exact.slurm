#!/bin/bash
#SBATCH --job-name=table2_exact_import
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:15:00
#SBATCH --array=0-5%1
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/table2_exact_%A_%a.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${SLURM_ARRAY_TASK_ID:?Requires an array task}"
: "${1:?Pass the frozen tasks.jsonl absolute path}"
: "${2:?Pass its SHA256}"
BASE=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend
IMAGE=${FLOWPLACE_TABLE2_TOOL_IMAGE:-/fact_data/bohanli/flowplace_fql_backend/containers/orfs_26Q4-151-g8b5e62fbb.sif}
OPENROAD_OVERRIDE=${FLOWPLACE_TABLE2_OPENROAD_EXE:-}
if [[ -n "$OPENROAD_OVERRIDE" ]]; then
  test -x "$OPENROAD_OVERRIDE"
fi
TASKS=$(realpath "$1")
TASKS_SHA=$2
TASKS_PARENT=$(dirname "$TASKS")
cd "$BASE"
export TMPDIR="/scratch/bohanli/flowplace_table2/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export SINGULARITY_TMPDIR="$TMPDIR"
export APPTAINER_TMPDIR="$TMPDIR"
export NUM_CORES="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Bind the actual data/scratch roots so both result symlinks and frozen DEFs
# remain accessible. Task manifest files may live in another user data folder.
singularity exec --cleanenv \
 --bind /fact_home/bohanli/RL_Placement,/fact_data/bohanli,/scratch/bohanli,"$TASKS_PARENT" \
 --env "NUM_CORES=$NUM_CORES,OMP_NUM_THREADS=$NUM_CORES,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,SLURM_JOB_ID=$SLURM_JOB_ID,SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID,SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID,TMPDIR=$TMPDIR,FLOWPLACE_TOOL_IMAGE=$IMAGE,FLOWPLACE_OPENROAD_OVERRIDE=$OPENROAD_OVERRIDE" \
 "$IMAGE" bash -c 'set -euo pipefail; source /OpenROAD-flow-scripts/env.sh; unset FLOW_HOME DESIGN_HOME PLATFORM_HOME WORK_HOME SCRIPTS_DIR UTILS_DIR; export OPENROAD_EXE=${FLOWPLACE_OPENROAD_OVERRIDE:-$(command -v openroad)}; export YOSYS_EXE=$(command -v yosys); export KLAYOUT_CMD=$(command -v klayout); exec python3 scripts/exact_import/run_table2.py --tasks "$1" --tasks-sha256 "$2"' bash "$TASKS" "$TASKS_SHA"
