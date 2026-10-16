#!/bin/bash
#SBATCH --job-name=flowplace_drt20
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/drt20_%A_%a.log
#SBATCH --array=0-1%1
#SBATCH --no-requeue
set -euo pipefail
sources=(released_bp_fe_03 generated_bp_fe_seed300_01)
source_run="${sources[$SLURM_ARRAY_TASK_ID]}"
set -- --source-run "$source_run" --run-id "diagnostic_drt20_${source_run}_job${SLURM_ARRAY_JOB_ID}" --end-iter 20
: "${SLURM_JOB_ID:?Submit using sbatch}"
BASE=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend
IMAGE=/fact_data/bohanli/flowplace_fql_backend/containers/orfs_26Q4-151-g8b5e62fbb.sif
cd "$BASE"
export NUM_CORES="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
singularity exec --cleanenv --bind /fact_home/bohanli/RL_Placement,/fact_data/bohanli/flowplace_fql_backend \
 --env "NUM_CORES=$NUM_CORES,OMP_NUM_THREADS=$NUM_CORES,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,SLURM_JOB_ID=$SLURM_JOB_ID" \
 "$IMAGE" bash -c 'set -euo pipefail; source /OpenROAD-flow-scripts/env.sh; unset FLOW_HOME DESIGN_HOME PLATFORM_HOME WORK_HOME SCRIPTS_DIR UTILS_DIR; export OPENROAD_EXE=$(command -v openroad); export YOSYS_EXE=$(command -v yosys); export KLAYOUT_CMD=$(command -v klayout); exec python3 scripts/run_deep_drt.py "$@"' bash "$@"
