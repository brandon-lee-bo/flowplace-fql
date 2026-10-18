#!/usr/bin/env python3
"""Preserve only frozen cohort artifacts after their run has terminated; never change an experiment."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
import table2_drc_cap as cap
BASE=ROOT/'backend/plans/table2_preservation_v1'
RUNS=Path('/fact_data/bohanli/flowplace_fql_backend/table2_results')
ARCHIVES=Path('/fact_data/bohanli/flowplace_fql_backend/table2_archives')
CACHE='/scratch/bohanli/flowplace_table2/artifact_cache'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def submit(cmd):return subprocess.check_output(['sbatch','--parsable','--mem=20G','--exclude=enterprise']+cmd,cwd=ROOT,text=True).strip().split(';')[0]
def main():
 assert os.environ.get('SLURM_JOB_ID');os.chdir(ROOT);BASE.mkdir(exist_ok=True)
 with (BASE/'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  path=BASE/'state.json'
  state=json.loads(path.read_text()) if path.exists() else {'status':'watching','submissions':{},'errors':[]}
  state.setdefault('controller_history',[]).append({'slurm_job_id':os.environ['SLURM_JOB_ID'],'time':time.time(),'protocol':'explicit_capacity_preservation_v1'})
  state.update(status='watching',slurm_job_id=os.environ['SLURM_JOB_ID']);save(path,state)
  hashes=json.loads((BASE/'source_sha256_capacity.json').read_text())
  for p,d in hashes.items():assert sha(ROOT/p)==d,p
  while True:
   terminal=True
   try:
    for cohort in ['table2_background_v1','table2_exact_import_v1']:
     batch=json.loads((ROOT/'backend/plans'/cohort/'state.json').read_text());terminal &= batch['status']=='audit_submitted'
     candidates=[]
     for item in batch['selected'].values():
      run=RUNS/item['task']['run_id'];mp=run/'manifest.json';ap=run/'fullflow_audit.json'
      if mp.is_file() and ap.is_file():
       m=json.loads(mp.read_text());a=json.loads(ap.read_text())
       if cap.is_cap_audit_failure(m,a):candidates.append((run,'cap',None))
     for bind in batch['infrastructure_retries']:
      assert sha(bind['path'])==bind['sha256']
      b=json.loads(Path(bind['path']).read_text());candidates.append((RUNS/b['old_run_id'],'runtime' if b.get('reason_kind') == 'runtime_capacity_no_final_metrics' else 'interrupted',bind))
     for run,kind,bind in candidates:
      key=run.name
      if key in state['submissions']:continue
      tree=run/'orfs_work/results'
      if not tree.is_dir() or tree.is_symlink():continue
      if any((ARCHIVES/key).glob('*/archive_manifest.json')) or any((ARCHIVES/key).glob('*/source_inventory.json')):continue
      dest=ARCHIVES/key/('preservation_watcher_'+kind+'_v1')
      helper='archive_table2_capvalidated' if kind=='cap' else ('archive_table2_runtime_interrupted' if kind=='runtime' else 'archive_interrupted_table2_staged')
      extra=[] if kind=='cap' else ['--retry-binding',bind['path'],'--retry-binding-sha256',bind['sha256']]
      cmd=['backend/scripts/'+helper+'.slurm','create','--run',str(run),'--archive-dir',str(dest)]+extra
      a=submit(cmd)
      cmd2=['--dependency=afterok:'+a,'backend/scripts/'+helper+'.slurm','commit','--archive-dir',str(dest),'--scratch-dir',CACHE+'/'+key+'/preservation_watcher_'+kind+'_v1','--confirm-remove-bulk-backup']+extra
      b=submit(cmd2)
      state['submissions'][key]={'classification':kind,'create':a,'commit':b,'commands':[cmd,cmd2],'time':time.time()};save(path,state)
    state['last_checked_unix']=time.time();save(path,state)
    if terminal:
     state['status']='all_cohorts_terminal_preservation_submitted';save(path,state);return
   except Exception as exc:
    state['errors'].append({'time':time.time(),'error':repr(exc)});save(path,state);print(repr(exc),flush=True)
   time.sleep(45)
if __name__=='__main__':main()
