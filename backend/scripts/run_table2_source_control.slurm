#!/bin/bash
#SBATCH --job-name=table2_fixed_author_source_control
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --exclude=quantum,infinite-frontier
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/table2_source_control_%A_%a.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
: "${SLURM_ARRAY_TASK_ID:?Array job required}"
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
cd "$ROOT"
python3 - "$1" "$2" "$3" "$4" <<'PY'
import hashlib,json,os,pathlib,sys
assert hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest()==sys.argv[2]
assert hashlib.sha256(pathlib.Path(sys.argv[3]).read_bytes()).hexdigest()==sys.argv[4]
for name,digest in json.loads(pathlib.Path(sys.argv[3]).read_text()).items():
 assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==digest,name
task=json.loads(pathlib.Path(sys.argv[1]).read_text().splitlines()[int(os.environ['SLURM_ARRAY_TASK_ID'])])
free=int(os.getxattr('/fact_data/bohanli','ceph.quota.max_bytes'))-int(os.getxattr('/fact_data/bohanli','ceph.dir.rbytes'))
# Primary and compatibility cohorts have finished before this array is submitted.
# Reserve 8 GiB plus 1 GiB archival-copy margin, above conservatively observed bulk.
needed=(9 if task['design']=='bp' else 2)*1024**3+9*1024**3
assert free>=needed,('Insufficient real quota; no evaluation started',free,needed)
PY
mapfile -t fields < <(python3 - "$1" <<'PY'
import json,os,pathlib,sys
t=json.loads(pathlib.Path(sys.argv[1]).read_text().splitlines()[int(os.environ['SLURM_ARRAY_TASK_ID'])])
print(t['openroad_exe']);print(t['run_id'])
PY
)
export FLOWPLACE_TABLE2_OPENROAD_EXE="${fields[0]}"
run_id="${fields[1]}"
bash "$ROOT/backend/scripts/table2_fresh.slurm" "$1" "$2"
RUN="/fact_data/bohanli/flowplace_fql_backend/table2_results/$run_id"
ARCHIVE="/fact_data/bohanli/flowplace_fql_backend/table2_archives/$run_id/source_control_v1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 backend/scripts/compare_table2_method.py "$RUN"
python3 backend/scripts/archive_table2_staged.py create --run "$RUN" --archive-dir "$ARCHIVE" --threads 2
python3 backend/scripts/archive_table2_staged.py commit --archive-dir "$ARCHIVE" \
    --scratch-dir "/scratch/bohanli/flowplace_table2/artifact_cache/$run_id/source_control_v1" \
    --confirm-remove-bulk-backup
