#!/bin/bash
#SBATCH --job-name=t2_drc_cap_reconcile
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/t2_drc_cap_reconcile_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 - <<'CHECK'
import sys,json,hashlib,os
from pathlib import Path
sys.path.insert(0,'analysis')
import table2_drc_cap as m
assert m.digest(Path(m.__file__))=='a0b377454028e6b369e7372220fde02dd0adfbd03ba487fbc1a34ce9ce8707e6'
p=Path('backend/results/table2_source_era_dreamplace_v1_swerv_wrapper').resolve()
manifest=m.read_json(p/'manifest.json');metrics,raw,audit=m.verify_metrics(p,manifest)
proof=audit['posthoc_drc_cap_evidence']
assert raw['DRC']==24340 and proof['native_type_counts']['Short']==21329 and proof['exported_type_counts']['Short']==10000
# Recheck an ordinary passing row through the same metric verifier.
q=Path('backend/results/table2_source_era_flowplace_v1_swerv_wrapper').resolve()
m.verify_metrics(q,m.read_json(q/'manifest.json'))
out=Path('/fact_data/bohanli/flowplace_fql_backend/table2_reports/drc_cap_reconciliation_swerv_v1');out.mkdir(exist_ok=False)
(out/'summary.json').write_text(json.dumps({'success':True,'posthoc_only':True,'original_slurm_and_manifest_stay_failed':True,'runner_sha256':'a0b377454028e6b369e7372220fde02dd0adfbd03ba487fbc1a34ce9ce8707e6','slurm_job_id':os.environ['SLURM_JOB_ID'],'manifest_sha256':m.digest(p/'manifest.json'),'fullflow_audit_sha256':m.digest(p/'fullflow_audit.json'),'metrics':metrics,'raw':raw,'proof':proof},indent=2)+'\n')
print(json.dumps(proof))
CHECK
