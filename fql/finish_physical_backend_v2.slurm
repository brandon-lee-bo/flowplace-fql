#!/bin/bash
# Same audit/aggregation as v1, with explicit preserved-preemption recovery.
#SBATCH --job-name=fql_physical_summary
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/logs/physical_summary_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${1:?task manifest required}" "${2:?new result directory required}" "${3:?frozen retry plan required}"
mkdir -p "$2"
.venv/bin/python -m fql.freeze_physical_retry --plan "$3" --output "$2/infrastructure_retry_map.json"
.venv/bin/python backend/scripts/audit_grt_tasks.py "$1" --output "$2/raw_audit.json" --infrastructure-retry-map "$2/infrastructure_retry_map.json"
.venv/bin/python backend/scripts/classify_failure_batch.py "$1" --output "$2/outcome_classification.json" --infrastructure-retry-map "$2/infrastructure_retry_map.json"
.venv/bin/python -m fql.aggregate_backend_candidates --tasks "$1" --output "$2/aggregate" --allow-exploratory-physical-failures --infrastructure-retry-map "$2/infrastructure_retry_map.json"
