"""Host Slurm wrapper: preserve a full-route measurement and archive intermediates."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from .graph_o2o_archive import archive_run, atomic_json, sha
from .graph_o2o_backend import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-run',type=Path)
    source.add_argument('--source-outcome',type=Path)
    p.add_argument('--run-id',required=True)
    p.add_argument('--method',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--end-iter',type=int,default=20)
    p.add_argument('--storage',choices=['bulk','scratch'],default='bulk')
    args=p.parse_args()
    job=os.environ['SLURM_JOB_ID']
    args.output.mkdir(parents=True,exist_ok=True)
    source_binding={}
    if args.source_outcome is not None:
        outcome=json.loads(args.source_outcome.read_text())
        source_binding=dict(source_outcome=str(args.source_outcome.resolve()),
                            source_outcome_sha256=sha(args.source_outcome))
        if outcome['status']!='success':
            if outcome['status'] not in ['physical_failure','geometry_failure']:
                raise ValueError('Unknown upstream GRT outcome; no fictitious final PPA')
            result=dict(status='upstream_physical_failure',**source_binding,
                        failure_class=outcome['failure_class'],metrics=None,slurm_job_id=job)
            existing=args.output/'outcome.json'
            if existing.exists():
                old=json.loads(existing.read_text())
                if any(old.get(key)!=result[key]for key in
                       ['status','source_outcome','source_outcome_sha256','failure_class','metrics']):
                    raise ValueError('Different upstream-failure record already exists')
                return
            atomic_json(existing,result);return
        args.source_run=Path(outcome['physical_run'])
        if (sha(args.source_run/'manifest.json')!=outcome['physical_manifest_sha256'] or
                sha(args.source_run/'metrics.json')!=outcome['metrics_sha256'] or
                not outcome['backend_macro_legal'] or
                sha(outcome['backend_macro_geometry_path'])!=outcome['backend_macro_geometry_sha256']):
            raise ValueError('Changed or geometrically invalid source GRT evidence')
    spec=dict(source_run=str(args.source_run.resolve()),source_manifest_sha256=sha(args.source_run/'manifest.json'),
              method=args.method,route_end_iteration=args.end_iter,run_id=args.run_id)
    spec.update(source_binding)
    if args.storage != 'bulk':
        spec['storage'] = args.storage
    spec_path=args.output/'spec.json'
    if spec_path.exists():
        if json.loads(spec_path.read_text())!=spec:
            raise ValueError('Existing full-route specification differs')
    else:
        atomic_json(spec_path,spec)
    if (args.output/'outcome.json').exists():
        result=json.loads((args.output/'outcome.json').read_text())
        if result['spec_sha256']!=sha(spec_path):
            raise ValueError('Completed full-route outcome binding changed')
        print(json.dumps(dict(status='already_completed',outcome=str(args.output/'outcome.json'))));return
    run=None
    for old_binding in sorted(args.output.glob('attempt_*/binding.json')):
        binding=json.loads(old_binding.read_text())
        if binding['spec_sha256']!=sha(spec_path):
            raise ValueError('Prior attempt belongs to a different physical specification')
        candidate=Path(binding['physical_run'])
        if (candidate/'manifest.json').exists():
            old=json.loads((candidate/'manifest.json').read_text())
            if old.get('success') and old.get('time_end'):
                run=candidate;break  # Recover reporting/storage without repeating completed routing.
    if run is None:
        attempt=args.output/('attempt_'+job);attempt.mkdir(exist_ok=False)
        snapshot=attempt/'source';snapshot.mkdir()
        sources={}
        for name in ['graph_o2o_full_route.py', 'graph_o2o_full_route.slurm',
                     'graph_o2o_finish.py', 'graph_o2o_finish.sh', 'graph_o2o_archive.py']:
            original=Path(__file__).with_name(name)
            shutil.copyfile(original,snapshot/name)
            sources[name]=sha(snapshot/name)
        run_id=args.run_id+'_a'+job
        run=ROOT/'backend/results'/run_id
        atomic_json(attempt/'binding.json',dict(spec_sha256=sha(spec_path),physical_run=str(run),
                    slurm_job_id=job,started_unix=time.time(),host_runner_sha256=sha(__file__),
                    source_sha256=sources))
        with (attempt/'runner.log').open('w') as log:
            process=subprocess.run(['bash',str(ROOT/'fql/graph_o2o_finish.sh'),
                '--source-run',str(args.source_run),'--run-id',run_id,'--method',args.method,
                '--storage',args.storage,
                '--end-iter',str(args.end_iter)],stdout=log,stderr=subprocess.STDOUT)
        atomic_json(attempt/'exit.json',dict(returncode=process.returncode,finished_unix=time.time()))
        if process.returncode:
            raise RuntimeError('Full route failed; retain all evidence and classify before retry')
    manifest=json.loads((run/'manifest.json').read_text())
    metrics=json.loads((run/'metrics.json').read_text())
    if not manifest['success'] or sha(run/'metrics.json')!=manifest['canonical_metrics_sha256']:
        raise ValueError('Final PPA not established')
    archive=archive_run(run)
    result=dict(status='completed',spec=spec,spec_sha256=sha(spec_path),
                physical_run=str(run),physical_manifest_sha256=sha(run/'manifest.json'),
                metrics=metrics,metrics_sha256=sha(run/'metrics.json'),
                route_drc_clean=manifest['route_drc_clean'],
                final_macros_match_audited_grt=manifest['final_macros_match_audited_grt'],
                archive=archive,slurm_job_id=job)
    atomic_json(args.output/'outcome.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
