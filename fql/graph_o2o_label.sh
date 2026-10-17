#!/bin/bash
# Same physical runner/image as the offline labels. Add scratch TMPDIR only.
set -euo pipefail
: "${SLURM_JOB_ID:?Use Slurm}"
BASE=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend
IMAGE=/fact_data/bohanli/flowplace_fql_backend/containers/orfs_26Q4-151-g8b5e62fbb.sif
cd "$BASE"
export NUM_CORES="$SLURM_CPUS_PER_TASK" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="/scratch/bohanli/flowplace_graph/grt_tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
exec singularity exec --cleanenv --bind /fact_home/bohanli/RL_Placement,/fact_data/bohanli,/scratch/bohanli \
 --env "NUM_CORES=$NUM_CORES,OMP_NUM_THREADS=$NUM_CORES,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,SLURM_JOB_ID=$SLURM_JOB_ID,TMPDIR=$TMPDIR" \
 "$IMAGE" bash -c 'set -euo pipefail; source /OpenROAD-flow-scripts/env.sh; unset FLOW_HOME DESIGN_HOME PLATFORM_HOME WORK_HOME SCRIPTS_DIR UTILS_DIR; export OPENROAD_EXE=$(command -v openroad); export YOSYS_EXE=$(command -v yosys); export KLAYOUT_CMD=$(command -v klayout); exec python3 scripts/run_grt_label.py "$@"' bash "$@"
