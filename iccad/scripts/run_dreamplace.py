"""Run pinned DREAMPlace on a released ICCAD DEF, with explicit provenance.

This is a reconstructed backend configuration, not the unpublished FlowPlace
ICCAD evaluation script. Its OpenTimer timing is not an Innovus PPA result.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from audit_inputs import BASE, ROOT, inspect_def, lef_blocks, sha, compare_fixed

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--design',default='superblue18')
    p.add_argument('--run-id',required=True)
    p.add_argument('--def-path',type=Path)
    a=p.parse_args()
    run=BASE/'assets/runs'/a.run_id
    run.mkdir(parents=True,exist_ok=False)
    d=a.design; source=BASE/'assets/DREAMPlace-4.1.0'
    installed=BASE/'assets/dmp_install_cpu'
    bench=BASE/'assets/benchmarks/iccad2015.ot'/d
    def_path=(a.def_path or ROOT/'backend/releases/flowplace_DEFS/iccad2015'/f'{d}.def').resolve()
    config_path=source/'test/iccad2015.ot'/f'{d}.json'
    cfg=json.loads(config_path.read_text()); original=dict(cfg)
    files={'def_input':def_path,'lef_input':bench/f'{d}.lef',
           'verilog_input':bench/f'{d}.v','early_lib_input':bench/f'{d}_Early.lib',
           'late_lib_input':bench/f'{d}_Late.lib','sdc_input':bench/f'{d}.sdc'}
    for key,value in files.items():
        assert value.is_file(),value
        cfg[key]=str(value)
    cfg.update(gpu=0,num_threads=int(os.environ['SLURM_CPUS_PER_TASK']),result_dir=str(run/'placement'))
    changes={k:dict(before=original.get(k),after=v) for k,v in cfg.items() if v != original.get(k)}
    (run/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    shutil.copy2(__file__,run/'run_dreamplace.py')
    shutil.copy2(Path(__file__).with_name('audit_inputs.py'),run/'audit_inputs.py')
    executed_files={str(f.relative_to(installed)):sha(f) for f in sorted((installed/'dreamplace').rglob('*'))
                    if f.is_file() and f.suffix in {'.py','.so'}}
    (run/'installation_file_hashes.json').write_text(json.dumps(executed_files,indent=2)+'\n')
    (run/'source.patch').write_text(subprocess.check_output(['git','-C',str(source),'diff','--submodule=diff'],text=True))
    metadata=dict(design=d,slurm_job=os.environ.get('SLURM_JOB_ID'),stage='DREAMPlace_only',
        configuration_origin='DREAMPlace 4.1.0 test/iccad2015.ot; FlowPlace config unpublished',
        implementation='CPU',config_changes=changes,input_sha256={k:sha(v) for k,v in files.items()},
        source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        config_sha256=sha(run/'config.json'),installation_hash_manifest_sha256=sha(run/'installation_file_hashes.json'),
        source_patch_sha256=sha(run/'source.patch'),complete=False,innovus_evaluated=False)
    (run/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    try:
        blocks=lef_blocks(files['lef_input']); before=inspect_def(def_path,blocks)
        (run/'geometry_before.json').write_text(json.dumps(before,indent=2)+'\n')
        if before['overlap_count'] or before['out_of_die']:
            raise RuntimeError('Input FIXED macros overlap or leave the die; refuse to silently repair')
        assert before['macro_count']==512, before['macro_count']
        cmd=[sys.executable,str(installed/'dreamplace/Placer.py'),str(run/'config.json')]
        metadata['command']=cmd
        start=time.monotonic()
        with (run/'dreamplace.log').open('w') as f:
            r=subprocess.run(cmd,cwd=installed,stdout=f,stderr=subprocess.STDOUT)
        metadata.update(returncode=r.returncode,elapsed_seconds=time.monotonic()-start)
        if r.returncode: raise RuntimeError(f'DREAMPlace failed with code {r.returncode}')
        output=run/'placement'/d/f'{d}.gp.def'
        assert output.is_file(),output
        after=inspect_def(output,blocks,set(before['macros']))
        (run/'geometry_after.json').write_text(json.dumps(after,indent=2)+'\n')
        metadata['fixed_macro_check']=compare_fixed(before,after)
        if before['component_identity_multiset_digest']!=after['component_identity_multiset_digest']:
            raise RuntimeError('DREAMPlace changed the component inventory')
        metadata.update(complete=True,output_def=str(output),output_sha256=sha(output),fixed_macros_preserved=True)
    except Exception as e:
        metadata['error']=repr(e)
        raise
    finally:
        (run/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__': main()
