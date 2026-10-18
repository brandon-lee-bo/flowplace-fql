#!/bin/bash
#SBATCH --job-name=flowplace-innovus-check
#SBATCH --partition=eda
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/iccad/logs/innovus_probe_%j.log
set -euo pipefail
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/iccad
RUN="$ROOT/assets/innovus_probe_${SLURM_JOB_ID}"
mkdir "$RUN"
cd "$RUN"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
INNOVUS=/fact_data/softwares/cadence/INNOVUS251/bin/innovus
export CDS_LIC_FILE="${CDS_LIC_FILE:-/fact_data/softwares/cadence/license/license.dat}"
export CDS_LIC_ONLY="${CDS_LIC_ONLY:-1}"
hostname
date -Is
readlink -f "$INNOVUS"
sha256sum "$(readlink -f "$INNOVUS")"
timeout 30 "$INNOVUS" -version > version.log 2>&1 || cat version.log
cat version.log
cat > probe.tcl <<'TCL'
puts "FLOWPLACE_INNOVUS_TCL_STARTED"
foreach cmd {earlyGlobalRoute setEarlyGlobalRouteMode reportCongestion timeDesign report_timing init_design defIn read_def read_netlist read_physical read_libs} {
    puts "COMMAND $cmd AVAILABLE [expr {[llength [info commands $cmd]] > 0}]"
}
puts "FLOWPLACE_INNOVUS_TCL_COMPLETED"
exit
TCL
timeout 180 "$INNOVUS" -no_gui -batch -files probe.tcl -log probe
grep -q FLOWPLACE_INNOVUS_TCL_COMPLETED probe.log
