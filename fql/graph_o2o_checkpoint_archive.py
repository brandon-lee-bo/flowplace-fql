"""Losslessly archive closed training intermediates, keeping final checkpoints.

Only completed pilot/control phases are eligible. A durable archive is fully
decoded and checked before any source is replaced by a verified scratch link.
No checkpoint tensor, training record or physical result is rewritten.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

from .graph_o2o_archive import atomic_json, sha


def verify(archive, expected):
    process = subprocess.Popen(['zstd','-dc','--long=28',str(archive)],stdout=subprocess.PIPE)
    seen = {}
    try:
        with tarfile.open(fileobj=process.stdout,mode='r|') as stream:
            for item in stream:
                if not item.isfile() or item.name not in expected or item.name in seen:
                    raise ValueError('Unexpected training archive member')
                digest=hashlib.sha256()
                source=stream.extractfile(item)
                for block in iter(lambda:source.read(1024*1024),b''):
                    digest.update(block)
                seen[item.name]=dict(bytes=item.size,sha256=digest.hexdigest())
        if process.wait() or seen!=expected:
            raise ValueError('Decoded training archive differs from original files')
    finally:
        if process.poll() is None:process.kill()
        process.wait()


def run(args):
    experiment=Path(args.experiment).resolve()
    allowed=Path('/fact_data/bohanli/flowplace_fql')
    if experiment not in [allowed/'graph_o2o_v1',allowed/'graph_o2o_offline_continue_v1']:
        raise ValueError('Explicit scope is the current graph pilot and matched control only')
    control=experiment.name=='graph_o2o_offline_continue_v1'
    rounds=sorted({int(n) for n in args.rounds.split(',')})
    archive_dir=experiment/'training_archives'/('seed_'+str(args.seed)+'_rounds_'+'_'.join(map(str,rounds)))
    archive_dir.mkdir(parents=True,exist_ok=True)
    manifest_path=archive_dir/'manifest.json'
    archive=archive_dir/'intermediates.tar.zst'
    cache=Path('/scratch/bohanli/flowplace_graph/training_archives')/(
        ('offline_continue_' if control else '')+archive_dir.name)
    cache.mkdir(parents=True,exist_ok=True)
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if sha(archive)!=manifest['archive_sha256']:
            raise ValueError('Changed durable training archive')
        expected=manifest['files']
        verify(archive,expected)
    else:
        paths=[];bindings={}
        shutil.copyfile(__file__,archive_dir/('archive_source_job_'+os.environ['SLURM_JOB_ID']+'.py'))
        shutil.copyfile(Path(__file__).with_suffix('.slurm'),archive_dir/'archive_wrapper.slurm')
        for number in rounds:
            folder=experiment/(f'training_stage_{number}' if control else f'seed_{args.seed}/round_{number}')
            metadata=['completed.json'] if control else ['training_status.json','requests/completed.json','bc/completed.json','fql/completed.json']
            for name in metadata:
                path=folder/name;record=json.loads(path.read_text())
                if name!='requests/completed.json' and record.get('status')!='completed':
                    raise ValueError('Only closed completed training phases are eligible')
                bindings[str(path.relative_to(experiment))]=sha(path)
            locations=[folder] if control else [folder/'bc',folder/'fql']
            for location in locations:
                completed=json.loads((location/'completed.json').read_text())
                if sha(location/'checkpoint.pt')!=completed['checkpoint_sha256']:
                    raise ValueError('Final checkpoint changed')
            # Final checkpoint.pt stays a real durable file for all consumers.
            paths += [path for location in ([folder] if control else [folder,*locations]) for path in location.glob('*.pt')
                      if path.name!='checkpoint.pt']
        if any(path.is_symlink() or not path.is_file() for path in paths):
            raise ValueError('Unexpected or previously archived source')
        paths.sort(key=lambda p:(p.name,str(p)))
        expected={str(path.relative_to(experiment)):dict(bytes=path.stat().st_size,sha256=sha(path)) for path in paths}
        names=cache/'files.null';names.write_bytes(b''.join(str(path.relative_to(experiment)).encode()+b'\0' for path in paths))
        staged=cache/('archive_job_'+os.environ['SLURM_JOB_ID']+'.tar.zst')
        subprocess.run(['tar','-I','zstd -T2 -3 --long=28','-cf',str(staged),'-C',str(experiment),
                        '--null','-T',str(names)],check=True)
        verify(staged,expected)
        quota=Path('/fact_data/bohanli')
        remaining=int(os.getxattr(quota,'ceph.quota.max_bytes'))-int(os.getxattr(quota,'ceph.dir.rbytes'))
        if remaining < staged.stat().st_size+8*1024**3:
            raise RuntimeError('Archive copy would breach the existing 8GiB reserve')
        temporary=archive.with_suffix('.pending_'+os.environ['SLURM_JOB_ID'])
        shutil.copyfile(staged,temporary)
        with temporary.open('rb') as handle:os.fsync(handle.fileno())
        if sha(temporary)!=sha(staged):raise ValueError('Durable archive copy changed bytes')
        temporary.replace(archive)
        manifest=dict(status='durable_verified_before_cache_links',config_sha256=sha(experiment/'config.json'),
            seed=args.seed,rounds=rounds,files=expected,closed_phase_bindings=bindings,
            archive_source_sha256=sha(__file__),
            archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
            source_bytes=sum(item['bytes'] for item in expected.values()),
            slurm_job_id=os.environ['SLURM_JOB_ID'],
            restoration='zstd -dc --long=28 intermediates.tar.zst | tar -xf - -C NEW_EMPTY_DIRECTORY; verify manifest file hashes',
            final_checkpoint_files_unchanged=True)
        atomic_json(manifest_path,manifest)
        staged.unlink()
    if sha(experiment/'config.json')!=manifest['config_sha256']:
        raise ValueError('Configuration changed during archival')
    for relative,digest in manifest['closed_phase_bindings'].items():
        if sha(experiment/relative)!=digest:raise ValueError('Closed training phase changed')
    for name,record in expected.items():
        path=experiment/name;target=cache/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():raise ValueError('Source/cache missing; restore verified durable archive first')
        if sha(path)!=record['sha256']:raise ValueError('Source checkpoint changed during archival')
        if not target.exists():shutil.copyfile(path,target)
        if target.stat().st_size!=record['bytes'] or sha(target)!=record['sha256']:
            raise ValueError('Scratch checkpoint differs from archived bytes')
        if not path.is_symlink():
            temporary=path.with_name(path.name+'.verified_link_'+os.environ['SLURM_JOB_ID'])
            temporary.symlink_to(target)
            temporary.replace(path)
        elif path.resolve()!=target:
            raise ValueError('Unexpected checkpoint link destination')
    manifest['status']='durable_archive_and_verified_scratch_links'
    manifest['reclaimed_bytes']=manifest['source_bytes']-manifest['archive_bytes']
    atomic_json(manifest_path,manifest)
    print(json.dumps({key:manifest[key] for key in
        ['status','archive_bytes','source_bytes','reclaimed_bytes','final_checkpoint_files_unchanged']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--rounds',required=True)
    args=parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:raise RuntimeError('Use Slurm')
    run(args)
