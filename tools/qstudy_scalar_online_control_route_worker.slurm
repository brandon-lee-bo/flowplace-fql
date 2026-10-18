#!/bin/bash
# Explicit resources come from the independent route manifest.
#SBATCH --no-requeue
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS=1
export TMPDIR="/scratch/bohanli/flowplace_graph/scalar_route_host/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
exec .venv/bin/python -u tools/qstudy_scalar_online_control_route.py task "$@"
