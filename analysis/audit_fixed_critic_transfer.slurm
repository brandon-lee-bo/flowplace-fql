#!/bin/bash
#SBATCH --job-name=fql_fixed_Q_deduplicated_audit
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --exclude=natural-selection,enterprise
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/logs/fixed_critic_transfer_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="/scratch/bohanli/flowplace_graph/fixed_critic_transfer/${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}"
mkdir -p "$TMPDIR"
exec .venv/bin/python -u -m analysis.audit_fixed_critic_transfer "$@"
