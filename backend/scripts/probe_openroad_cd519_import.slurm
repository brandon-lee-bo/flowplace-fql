#!/bin/bash
#SBATCH --job-name=table2_historical_tool_import
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --exclude=quantum,infinite-frontier
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/openroad_cd519_import_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
python3 - "$1" <<'PY'
import hashlib,json,os,pathlib,subprocess,sys,time
root=pathlib.Path('/fact_home/bohanli/RL_Placement/experiments/flowplace_fql')
base=pathlib.Path('/fact_data/bohanli/flowplace_fql_backend/tools/openroad_cd519_source_v1')
assert sys.argv[1].isdecimal()
candidates=[]
for p in (base/('build_'+sys.argv[1])).glob('attempt_*/manifest.json'):
 if (p.parent/'exit_code.txt').read_text().strip()=='0':candidates.append(p)
assert len(candidates)==1
build_path=candidates[0];build=json.loads(build_path.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
binary=pathlib.Path(build['binary']);assert sha(binary)==build['binary_sha256']
assert build['revision']=='cd519bb5e861ec4d5d00931c85650c179a864e34'
out=base/('import_'+os.environ['SLURM_JOB_ID']);out.mkdir(exist_ok=False)
platform=root/'backend/thirdparty/OpenROAD-flow-scripts/platforms/nangate45/lef'
lefs=[platform/n for n in ['NangateOpenCellLibrary.tech.lef','NangateOpenCellLibrary.macro.mod.lef',
 'fakeram45_512x64.lef','fakeram45_64x7.lef','fakeram45_64x96.lef']]
design=root/'backend/releases/flowplace_DEFS/openroad/bp_fe.def'
tcl='set_thread_count 1\n'+''.join('read_lef {'+str(p)+'}\n' for p in lefs)
tcl+='read_def {'+str(design)+'}\n'
tcl+='set block [ord::get_db_block]\nset count 0\nforeach inst [$block getInsts] { if {[[$inst getMaster] isBlock]} { incr count } }\n'
tcl+='if {$count != 11} { error "Expected 11 bp_fe macros, saw $count" }\n'
tcl+='puts "PINNED_IMPORT_OK macros=$count instances=[llength [$block getInsts]] dbu=[$block getDbUnitsPerMicron]"\n'
script=out/'probe.tcl';script.write_text(tcl)
image='/fact_data/bohanli/flowplace_fql_backend/containers/orfs_26Q4-151-g8b5e62fbb.sif'
command=['singularity','exec','--cleanenv','--bind',str(root)+',/fact_data/bohanli,/scratch/bohanli',
 '--env','OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1',image,str(binary),'-exit',str(script)]
with (out/'import.log').open('w') as stream:
 result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
log=(out/'import.log').read_text()
record={'time':time.time(),'job':os.environ['SLURM_JOB_ID'],'build_manifest':str(build_path),
 'build_manifest_sha256':sha(build_path),'binary_sha256':sha(binary),'command':command,
 'returncode':result.returncode,'passed':result.returncode==0 and 'PINNED_IMPORT_OK macros=11' in log,
 'input_sha256':{str(p):sha(p) for p in lefs+[design,script]},
 'scope':'CLI/LEF/DEF import smoke only; no placement, routing, PPA or legality claim.'}
(out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
assert record['passed'],record
print(json.dumps(record,indent=2))
PY
