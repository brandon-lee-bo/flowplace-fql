#!/bin/bash
#SBATCH --job-name=grt_parser_recovery
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/grt_parser_recovery_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend
for row in 0000 0001 0003; do
 run="$PWD/results/bp_fe_grt_v1_row$row"
 export FQL_REPARSE_ODB="$run/orfs_work/results/nangate45/bp_fe/eval/5_1_grt.odb"
 export FQL_REPARSE_REPORT_DIR="$run/orfs_work/reports/nangate45/bp_fe/eval"
 singularity exec --cleanenv --bind /fact_home/bohanli/RL_Placement --env "FQL_REPARSE_ODB=$FQL_REPARSE_ODB,FQL_REPARSE_REPORT_DIR=$FQL_REPARSE_REPORT_DIR" /fact_data/bohanli/flowplace_fql_backend/containers/orfs_26Q4-151-g8b5e62fbb.sif /OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad -exit scripts/recover_grt_markers.tcl
 python3 scripts/grt_metrics.py "$run"
done
