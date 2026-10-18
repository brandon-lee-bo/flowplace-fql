#!/bin/bash
#SBATCH --job-name=table2_pinned_submodule_recovery
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --exclude=quantum,infinite-frontier
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/openroad_pinned_submodules_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 GIT_TERMINAL_PROMPT=0
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
python3 - "$SLURM_JOB_ID" <<'PY'
import hashlib,json,os,pathlib,shutil,subprocess,sys,time
job=sys.argv[1]
task=pathlib.Path('/scratch/bohanli/flowplace_table2/openroad_cd519_source_v1')
source=task/'source'
out=pathlib.Path('/fact_data/bohanli/flowplace_fql_backend/tools/openroad_cd519_source_v1')/('submodules_'+job)
out.mkdir(exist_ok=False)
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()=='cd519bb5e861ec4d5d00931c85650c179a864e34'
saved=task/('interrupted_clone_'+job)
saved.mkdir(exist_ok=False)
record={'schema_version':1,'job':job,'started_unix':time.time(),'commands':[], 'preserved_paths':[],
 'policy':'Pinned Git commits, HTTP/1.1 partial clone and sparse checkout excluding tests/docs/examples only; no algorithm edits.'}
def save():
 (out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
def run(label,command,timeout=600):
 with (out/(label+'.log')).open('w') as log:
  r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 record['commands'].append({'label':label,'command':command,'returncode':r.returncode,'time':time.time()})
 save()
 assert r.returncode==0,label
git=['git','-c','http.version=HTTP/1.1','-c','http.lowSpeedLimit=1024','-c','http.lowSpeedTime=60']
modules=[('sta','src/sta','module/OpenSTA','9afc487f07c02e009ca0906c5ab4c5d62fbcbc79','OpenSTA'),
         ('abc','third-party/abc','src/abc','ef5389d31526003c2ebd7e6d6d6fe3848a20f0a2','abc')]
try:
 for name,path,module,pin,repo in modules:
  for category,old in [('worktree',source/path),('gitdir',source/'.git/modules'/module)]:
   if old.exists():
    dest=saved/(name+'_'+category)
    old.rename(dest)
    record['preserved_paths'].append({'old':str(old),'preserved':str(dest)})
  work=source/path
  run(name+'_clone',git+['clone','--filter=blob:none','--depth=1','--no-checkout',
      'https://github.com/The-OpenROAD-Project/'+repo+'.git',str(work)])
  run(name+'_sparse_flag',git+['-C',str(work),'config','core.sparseCheckout','true'])
  run(name+'_sparse_mode',git+['-C',str(work),'config','core.sparseCheckoutCone','false'])
  (work/'.git/info/sparse-checkout').write_text('/*\n!/test/\n!/doc/\n!/examples/\n')
  run(name+'_fetch_pin',git+['-C',str(work),'fetch','--filter=blob:none','--depth=1','--no-tags','origin',pin])
  run(name+'_checkout_pin',git+['-C',str(work),'checkout','--detach','FETCH_HEAD'])
  actual=subprocess.check_output(['git','-C',str(work),'rev-parse','HEAD'],text=True).strip()
  assert actual==pin
  run(name+'_clean',git+['-C',str(work),'diff','--exit-code'])
  run(name+'_tree',git+['-C',str(work),'ls-tree','-r','HEAD'])
 run('absorb',git+['-C',str(source),'submodule','absorbgitdirs','src/sta','third-party/abc'])
 run('final_status',git+['-C',str(source),'submodule','status','--recursive'])
 status=(out/'final_status.log').read_text()
 assert not any(line.startswith(('-', '+', 'U')) for line in status.splitlines())
 record.update(status='completed',completed_unix=time.time())
 save()
except Exception as exc:
 record.update(status='failed',error=repr(exc),failed_unix=time.time())
 save()
 raise
print(json.dumps(record,indent=2))
PY
