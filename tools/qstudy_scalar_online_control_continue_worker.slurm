#!/bin/bash
# Stage-specific resources/signals are pinned by the independent addendum.
#SBATCH --requeue
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR="/scratch/bohanli/flowplace_graph/scalar_control_host/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
exec .venv/bin/python -u tools/qstudy_scalar_online_control_continue.py task "$@"
