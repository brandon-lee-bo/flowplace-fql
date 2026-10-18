#!/bin/bash
#SBATCH --job-name=flowplace-iccad-inputs
#SBATCH --partition=eda
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/iccad/logs/inputs_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p iccad/assets/benchmarks
sha256sum iccad/assets/iccad2015.ot.tar.gz iccad/assets/tech.lef backend/releases/flowplace_DEFS/iccad2015/*.def > "iccad/assets/input_hashes_${SLURM_JOB_ID}.txt"
tar -tzf iccad/assets/iccad2015.ot.tar.gz > iccad/assets/benchmark_archive_listing.txt
tar -xzf iccad/assets/iccad2015.ot.tar.gz --no-same-owner -C iccad/assets/benchmarks
find iccad/assets/benchmarks -maxdepth 3 -type f > iccad/assets/benchmark_files.txt
cat iccad/assets/benchmark_files.txt
