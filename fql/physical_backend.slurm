#!/bin/bash
#SBATCH --job-name=fql_physical_grt
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --array=0-11%2
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/fql_physical_grt_%A_%a.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
: "${1:?absolute frozen task manifest required}"
.venv/bin/python - "$1" <<'PY'
import hashlib,json,os,pathlib,subprocess,sys
path=pathlib.Path(sys.argv[1]).resolve()
report=json.loads((path.parent.parent/'report.json').read_text())
if report['status']!='completed' or hashlib.sha256(path.read_bytes()).hexdigest()!=report['tasks_sha256']:raise ValueError('Task manifest not bound')
rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
if len(rows)!=12:raise ValueError('Frozen twelve-case budget differs')
row=rows[int(os.environ['SLURM_ARRAY_TASK_ID'])]
if hashlib.sha256(pathlib.Path(row['def_path']).read_bytes()).hexdigest()!=row['def_sha256']:raise ValueError('DEF hash mismatch')
command=['bash','backend/scripts/label_grt.slurm','--design',row['design'],'--def-path',row['def_path'],'--run-id',row['run_id'],'--storage','bulk']
print(json.dumps({'task':row,'tasks_sha256':report['tasks_sha256']}),flush=True)
raise SystemExit(subprocess.run(command).returncode)
PY
