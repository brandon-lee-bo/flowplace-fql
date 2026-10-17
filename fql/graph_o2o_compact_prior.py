"""Reclaim quota from twelve completed prior FQL GRT runs, without losing data.

This does not run, modify or resume historical Table 2. Every archived file is
decoded and hash-verified by graph_o2o_archive before original-byte eviction.
"""
import json
import os
from pathlib import Path
import shutil
import time
from .graph_o2o_archive import ROOT, archive_run, atomic_json, sha


def headroom():
    root=Path('/fact_data/bohanli')
    return int(os.getxattr(root,'ceph.quota.max_bytes'))-int(os.getxattr(root,'ceph.dir.rbytes'))


def main():
    job=os.environ['SLURM_JOB_ID']
    experiment=Path('/fact_data/bohanli/flowplace_fql/graph_o2o_v1')
    plan_path=experiment/'prior_grt_storage_plan.json'
    if plan_path.exists():
        plan=json.loads(plan_path.read_text())
    else:
        rows=[];skipped=[]
        for seed in [0,1,2]:
            for method in ['bc_continued','fql_physical_alpha_1']:
                for index in [192,193]:
                    run=ROOT/f'fqlphys_1788587_s{seed}_{method}_i{index}'
                    manifest=json.loads((run/'manifest.json').read_text())
                    if not manifest.get('time_end'):
                        skipped.append(dict(run=str(run),manifest_sha256=sha(run/'manifest.json'),
                                            reason='Interrupted original remains untouched; use its completed scheduler recovery'))
                        run=run.with_name(run.name+'_preempt_r1')
                        manifest=json.loads((run/'manifest.json').read_text())
                    if not manifest.get('time_end') or not manifest.get('success'):
                        raise ValueError('Expected completed prior measurement, never compact a live/preempted run')
                    rows.append(dict(run=str(run),manifest_sha256=sha(run/'manifest.json'),
                                     metrics_sha256=sha(run/'metrics.json')))
        plan=dict(created_unix=time.time(),rows=rows,skipped=skipped,scope='Twelve completed prior FQL GRT measurements; no Table2 tasks',
                  policy='Lossless verified durable archive, original physical manifests/metrics unchanged')
        atomic_json(plan_path,plan)
    folder=experiment/('storage_compaction_job_'+job);folder.mkdir(exist_ok=False)
    for name in ['graph_o2o_archive.py','graph_o2o_compact_prior.py']:
        shutil.copyfile(Path(__file__).with_name(name),folder/name)
    report=dict(slurm_job_id=job,plan_sha256=sha(plan_path),headroom_before_bytes=headroom(),results=[])
    for row in plan['rows']:
        run=Path(row['run'])
        if sha(run/'manifest.json')!=row['manifest_sha256'] or sha(run/'metrics.json')!=row['metrics_sha256']:
            raise ValueError('Prior measurement changed since storage plan')
        result=archive_run(run,prior_manifest_sha256=row['manifest_sha256'])
        if sha(run/'manifest.json')!=row['manifest_sha256'] or sha(run/'metrics.json')!=row['metrics_sha256']:
            raise ValueError('Storage operation changed physical evidence')
        report['results'].append(dict(run=str(run),archive=result))
        report['headroom_now_bytes']=headroom()
        atomic_json(folder/'report.json',report)
        print(json.dumps(dict(run=run.name,saved_bytes=result['source_bytes']-result['archive_bytes'],
                              headroom_bytes=report['headroom_now_bytes'])),flush=True)
    report['completed']=True;report['finished_unix']=time.time()
    atomic_json(folder/'report.json',report)


if __name__=='__main__':
    main()
