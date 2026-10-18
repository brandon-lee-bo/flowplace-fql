#!/bin/bash
# CPU-only compile/replay of one CPU legalization operator, not placement.
#SBATCH --job-name=table2-abacus-floor-div
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/table2_abacus_patch_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch}"
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
PARENT=/fact_data/bohanli/flowplace_fql_backend/dreamplace_gpu_build/job_1787994
cd "$ROOT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PARENT/install:$ROOT/.venv/lib/python3.9/site-packages:$ROOT/iccad/assets/dmp_env/lib/python3.9/site-packages:/usr/lib64/python3.9/site-packages:/usr/lib/python3.9/site-packages"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TMPDIR="/scratch/bohanli/flowplace_table2/abacus_patch_tmp_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
ulimit -c 0
exec "$PARENT/env/bin/python" backend/scripts/build_dreamplace_abacus_patch.py
