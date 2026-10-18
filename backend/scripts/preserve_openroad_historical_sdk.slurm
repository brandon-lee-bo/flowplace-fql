#!/bin/bash
#SBATCH --job-name=table2_preserve_historical_sdk
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --exclude=quantum,infinite-frontier
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/openroad_historical_sdk_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
python3 - <<'PY'
import hashlib,json,os,pathlib,shutil,stat,tarfile,time
task=pathlib.Path('/scratch/bohanli/flowplace_table2/openroad_cd519_source_v1')
base=pathlib.Path('/fact_data/bohanli/flowplace_fql_backend/tools/openroad_cd519_source_v1')
sdk=task/'dependencies/or-tools-9.10.4067'
assert sdk.is_dir()
out=base/'runtime_sdk_v1'
out.mkdir(exist_ok=False)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
 return h.hexdigest()
def inventory():
 rows={}
 for directory,dirs,files in os.walk(sdk,followlinks=False):
  for name in dirs+files:
   p=pathlib.Path(directory)/name
   if p.is_symlink():rows[p.relative_to(sdk).as_posix()]={'kind':'symlink','target':os.readlink(p)}
   elif p.is_file():rows[p.relative_to(sdk).as_posix()]={'kind':'file','size':p.stat().st_size,'sha256':sha(p)}
 return rows
expected=inventory()
archive=task/'dependencies/or-tools-9.10.4067-runtime.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz',compresslevel=3) as stream:
 stream.add(sdk,arcname='or-tools-9.10.4067',recursive=True)
seen={}
with tarfile.open(archive,'r:gz') as stream:
 for member in stream:
  if member.isdir():continue
  relative=pathlib.PurePosixPath(member.name).relative_to('or-tools-9.10.4067').as_posix()
  assert relative not in seen
  if member.issym():seen[relative]={'kind':'symlink','target':member.linkname}
  elif member.isfile():
   h=hashlib.sha256();f=stream.extractfile(member)
   for block in iter(lambda:f.read(1024**2),b''):h.update(block)
   seen[relative]={'kind':'file','size':member.size,'sha256':h.hexdigest()}
  elif member.islnk():
   h=hashlib.sha256();f=stream.extractfile(member);size=0
   for block in iter(lambda:f.read(1024**2),b''):h.update(block);size+=len(block)
   seen[relative]={'kind':'file','size':size,'sha256':h.hexdigest()}
  else:raise AssertionError(('Unexpected SDK member',member.name,member.type))
assert seen==expected,'Archive must reproduce every SDK file and symbolic link'
assert inventory()==expected,'SDK changed during preservation'
free=int(os.getxattr('/fact_data/bohanli','ceph.quota.max_bytes'))-int(os.getxattr('/fact_data/bohanli','ceph.dir.rbytes'))
assert free>=8*1024**3+archive.stat().st_size,'Keep the durable storage reserve'
dest=out/archive.name
with archive.open('rb') as src,dest.open('xb') as dst:
 shutil.copyfileobj(src,dst,1024**2);dst.flush();os.fsync(dst.fileno())
assert sha(dest)==sha(archive)
(out/'inventory.json').write_text(json.dumps(expected,indent=2)+'\n')
manifest={'time':time.time(),'job':os.environ['SLURM_JOB_ID'],'sdk_version':'9.10.4067',
 'source':str(sdk),'archive':str(dest),'archive_sha256':sha(dest),'member_count':len(expected),
 'inventory_sha256':sha(out/'inventory.json'),'full_member_bytes_verified':True,
 'restore':'If scratch SDK is missing, extract this archive into '+str(task/'dependencies')+'. The published binary uses this recorded runtime path and the frozen SIF.',
 'scope':'Dependency preservation only. No layout, binary, source, or evaluation change.'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
PY
