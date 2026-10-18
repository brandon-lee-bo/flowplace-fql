#!/bin/bash
#SBATCH --job-name=t2exact_native_check
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/t2exact_native_check_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
python3 - <<'CHECK'
import importlib.util, json, hashlib, os
from pathlib import Path
p=Path('analysis/audit_table2_exact_import.py')
assert hashlib.sha256(p.read_bytes()).hexdigest() == '5f5ec388c03e9c513ee4a6656a82e3fc7bd05110db299b156a46cc5be7017403'
spec=importlib.util.spec_from_file_location('exact_auditor',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rows=[]
for method in ('flowplace','dreamplace'):
 run=(Path('backend/results')/('t2_exactimport_v1_'+method+'_swerv_wrapper')).resolve()
 manifest=json.loads((run/'manifest.json').read_text());units,actual=m.macros(Path(manifest['executed_def_path']))
 native=m.native_macroplacement(run,units,actual);geometry=m.geometry_check(run,manifest,units,actual)
 assert native['final_matches_native'] and geometry['legal_no_halo']
 # A one-DBU deliberate perturbation must fail exact matching.
 name=next(iter(actual));actual[name]['xy_dbu'][0]+=1
 negative=m.native_macroplacement(run,units,actual)
 assert not negative['final_matches_native'] and len(negative['disagreements'])==1
 rows.append({'run':str(run),'count':len(actual),'native':native,'geometry':geometry,'one_dbu_negative_control_detected':True})
out=Path('/fact_data/bohanli/flowplace_fql_backend/table2_reports/exact_native_restoration_check_v1');out.mkdir(exist_ok=False)
(out/'summary.json').write_text(json.dumps({'job':os.environ['SLURM_JOB_ID'],'source_sha256':'5f5ec388c03e9c513ee4a6656a82e3fc7bd05110db299b156a46cc5be7017403','scope':'Post-import OpenDB log versus input DEF; final routed DEF remains untested until each physical run completes.','rows':rows,'success':True},indent=2)+'\n')
print('PASS: both 28-macro native restorations exact; one-DBU negative controls detected')
CHECK
