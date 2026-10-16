#!/bin/bash
#SBATCH --job-name=flowplace_ppa_batch
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-0%2
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/ppa_batch_%A_%a.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend
: "${1:?Pass absolute JSONL task-manifest path}"
python - "$1" <<'PY'
import json,os,subprocess,sys
from pathlib import Path
path=Path(sys.argv[1]).resolve();rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
index=int(os.environ['SLURM_ARRAY_TASK_ID'])
if not 0 <= index < len(rows):raise SystemExit(f'Task {index} outside manifest of {len(rows)} rows')
r=rows[index]
for k in ['design','def_path','run_id']:
 if not isinstance(r.get(k),str) or not r[k]:raise SystemExit(f'Missing or invalid {k}')
if Path(r['run_id']).name!=r['run_id']:raise SystemExit('run_id must be a single path component')
print(json.dumps({'task_manifest':str(path),'task_index':index,'row':r},indent=2),flush=True)
command=['bash','scripts/evaluate.slurm','--design',r['design'],'--def-path',str(Path(r['def_path']).resolve()),'--run-id',r['run_id']]
raise SystemExit(subprocess.run(command).returncode)
PY
