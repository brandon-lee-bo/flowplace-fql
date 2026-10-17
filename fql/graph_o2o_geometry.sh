#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Use Slurm}"
PROJECT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
IMAGE=/fact_data/bohanli/flowplace_fql_backend/containers/orfs_26Q4-151-g8b5e62fbb.sif
export TMPDIR="/scratch/bohanli/flowplace_graph/grt_tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
exec singularity exec --cleanenv --bind /fact_home/bohanli/RL_Placement,/fact_data/bohanli,/scratch/bohanli \
 --env "GO2O_AUDIT_ODB=$1,TMPDIR=$TMPDIR,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1" \
 "$IMAGE" bash -c 'set -euo pipefail; source /OpenROAD-flow-scripts/env.sh; exec openroad -no_init -exit "$1"' bash "$PROJECT/fql/graph_o2o_geometry.tcl"
