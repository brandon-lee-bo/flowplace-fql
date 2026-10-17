"""Evaluate the predeclared best-measured-layout control at equal label budget."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import time

from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import ROOT, load_requests
from .graph_o2o_collect import audit_outcome
from .graph_o2o_controller import LIVE, RESTARTABLE, states, submit_unique


def select(experiment, plan):
    labels=ROOT/'datasets/bp_fe_grt_v1/backend_labels.jsonl'
    if sha(labels)!=plan['offline_labels_sha256'] or sha(experiment/'config.json')!=plan['config_sha256']:
        raise ValueError('Reference plan data/configuration changed')
    cfg=json.loads((experiment/'config.json').read_text())
    manifest=json.loads((ROOT/'datasets/bp_fe_grt_v1/manifest.json').read_text())
    offline=[]
    for line in labels.read_text().splitlines():
        row=json.loads(line)
        if row['critic_split']!=0:
            continue
        run=Path(row['manifest_path']).parent
        if sha(run/'manifest.json')!=row['manifest_sha256'] or sha(run/'metrics.json')!=row['metrics_sha256']:
            raise ValueError('Offline reference source changed')
        offline.append(dict(key=f"offline_{row['row_index']:04d}",status='success',rWL_um=row['rWL_um'],
                            source_run=str(run),physical_manifest_sha256=row['manifest_sha256']))
    if len(offline)!=plan['offline_TRAIN_candidates']:
        raise ValueError('Incorrect offline reference denominator')
    online={arm:[] for arm in plan['arms']}
    for number in plan['online_collection_rounds']:
        requests,_=load_requests(experiment,plan['seed'],number)
        for request in requests:
            if request['arm'] not in online or request['role']!='collect':
                continue
            row=audit_outcome(request,cfg,manifest['reward_scale_train_only'])
            online[row['arm']].append(dict(key=f"online_r{number}_i{row['index']:04d}",status=row['status'],
                rWL_um=(row['metrics'] or {}).get('rWL_um'),source_run=row.get('physical_run'),
                physical_manifest_sha256=row.get('physical_manifest_sha256'),
                outcome_path=row['outcome_path'],outcome_sha256=row['outcome_sha256']))
    choices={}
    for arm in plan['arms']:
        if len(online[arm])!=plan['online_candidates_per_arm']:
            raise ValueError('Incorrect online reference denominator')
        rows=offline+online[arm]
        good=[r for r in rows if r['status']=='success']
        choices[arm]=dict(requested=len(rows),successful=len(good),failed=len(rows)-len(good),
                          selected=min(good,key=lambda r:(r['rWL_um'],r['key'])),candidates=rows)
    return dict(plan_sha256=sha(experiment/'training_replay_reference_plan.json'),choices=choices,
                note='Selection uses TRAIN and collected GRT labels only; final PPA is subsequently measured, not a selection metric')


def main(args):
    exp=Path(args.experiment).resolve()
    plan=json.loads((exp/'training_replay_reference_plan.json').read_text())
    chosen=select(exp,plan)
    folder=exp/'best_replay_reference';folder.mkdir(exist_ok=True)
    snapshot=folder/('source_job_'+os.environ['SLURM_JOB_ID']);snapshot.mkdir(exist_ok=False)
    shutil.copyfile(__file__,snapshot/Path(__file__).name)
    path=folder/'selection.json'
    if path.exists():
        if json.loads(path.read_text())!=chosen:
            raise ValueError('A previously frozen reference selection changed')
    else:
        atomic_json(path,chosen)
    journal_path=folder/'journal.json'
    journal=json.loads(journal_path.read_text()) if journal_path.exists() else dict(
        selection_sha256=sha(path),arms={})
    if journal['selection_sha256']!=sha(path):
        raise ValueError('Reference selection differs from job journal')
    save=lambda:atomic_json(journal_path,journal)
    fixed=json.loads((exp/'full_route/best_train/outcome.json').read_text())
    outcomes={}
    for arm,choice in chosen['choices'].items():
        selected=choice['selected'];source=Path(selected['source_run']).resolve()
        if (Path(fixed['spec']['source_run']).resolve()==source and
            fixed['spec']['source_manifest_sha256']==selected['physical_manifest_sha256'] and
            fixed['spec']['route_end_iteration']==plan['detailed_route_end_iteration']):
            run=Path(fixed['physical_run'])
            if sha(run/'manifest.json')!=fixed['physical_manifest_sha256'] or sha(run/'metrics.json')!=fixed['metrics_sha256']:
                raise ValueError('Fixed full-route reference changed')
            outcomes[arm]=dict(reused_identical_fixed_reference=True,outcome=fixed)
            continue
        entry=journal['arms'].setdefault(arm,dict(jobs=[]));save()
        output=folder/arm
        while True:
            if not entry['jobs']:
                name='go2opool_'+chosen['plan_sha256'][:8]+'_'+arm+'_0'
                job=submit_unique(name,[],[str(ROOT/'fql/graph_o2o_full_route.slurm'),
                    '--source-run',str(source),'--run-id','go2o_drt_v1_best_replay_'+arm,
                    '--method','best_measured_replay_'+arm,'--output',str(output),
                    '--end-iter',str(plan['detailed_route_end_iteration'])])
                entry['jobs'].append(job);save()
            observed=states(entry['jobs'][-1]);entry['last_states']=observed;save()
            if any(s in LIVE or s=='ACCOUNTING_PENDING' for s in observed):
                time.sleep(15);continue
            if all(s=='COMPLETED' for s in observed):
                result=json.loads((output/'outcome.json').read_text())
                if result['status']!='completed':
                    raise ValueError('Completed reference job lacks a full physical outcome')
                outcomes[arm]=dict(reused_identical_fixed_reference=False,outcome=result)
                break
            if all(s in RESTARTABLE or s=='COMPLETED' for s in observed):
                name='go2opool_'+chosen['plan_sha256'][:8]+'_'+arm+'_'+str(len(entry['jobs']))
                job=submit_unique(name,[],[str(ROOT/'fql/graph_o2o_full_route.slurm'),
                    '--source-run',str(source),'--run-id','go2o_drt_v1_best_replay_'+arm,
                    '--method','best_measured_replay_'+arm,'--output',str(output),
                    '--end-iter',str(plan['detailed_route_end_iteration'])])
                entry['jobs'].append(job);save();continue
            raise RuntimeError('Inspect failed reference evidence before repeating any routing: '+str(observed))
    report=dict(status='completed',selection_sha256=sha(path),choices=chosen['choices'],outcomes=outcomes,
                independent_layouts=len({r['outcome']['physical_run'] for r in outcomes.values()}),
                note='One selected reference per arm; do not present reused layouts as independent training seeds')
    atomic_json(folder/'report.json',report)
    journal['completed']=True;save()
    print(json.dumps(dict(status='completed',selected={a:c['selected']['key'] for a,c in chosen['choices'].items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True)
    args=parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use Slurm')
    main(args)
