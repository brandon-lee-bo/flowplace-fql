#!/bin/bash
#SBATCH --job-name=fql_preserve_native_evidence
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --exclude=natural-selection,enterprise
#SBATCH --requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/logs/graph_o2o_preserve_native_evidence_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="/scratch/bohanli/flowplace_graph/native_evidence_preservation_job/${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}"
mkdir -p "$TMPDIR"
exec .venv/bin/python -u -m fql.graph_o2o_preserve_native_evidence "$@"
