#!/bin/bash
#SBATCH --job-name=flowplace_table2
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-4%2
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/table2_%A_%a.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend
names=(ariane133 ariane136 bp bp_be swerv_wrapper)
design=${names[$SLURM_ARRAY_TASK_ID]}
run_id=released_${design}_job${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
if [[ ${SLURM_RESTART_COUNT:-0} -gt 0 ]]; then
  run_id=${run_id}_restart${SLURM_RESTART_COUNT}
fi
bash scripts/evaluate.slurm --design "$design" --run-id "$run_id"
python scripts/compare_paper.py "results/$run_id"
