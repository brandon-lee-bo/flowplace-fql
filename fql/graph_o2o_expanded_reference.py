"""Evaluate best-measured expanded TRAIN/replay controls at the same label budget."""
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
    from .graph_o2o_expanded_report import offline_candidates, choice
    cfg=json.loads((experiment/'config.json').read_text())
    offline,binding,labels=offline_candidates(experiment,cfg)
    if (sha(experiment/'config.json')!=plan['config_sha256'] or
            sha(experiment/'expanded_training_binding.json')!=plan['expanded_training_binding_sha256'] or
            sha(labels)!=plan['offline_labels_sha256'] or
            len(offline)!=plan['offline_TRAIN_candidates']):
        raise ValueError('Expanded best-measured reference context changed')
    online={arm:[]for arm in plan['arms']}
    for number in plan['online_collection_rounds']:
        requests,_=load_requests(experiment,plan['seed'],number)
        for request in requests:
            if request['arm']not in online or request['role']!='collect':continue
            row=audit_outcome(request,cfg,binding['reward_scale'])
            online[row['arm']].append(dict(key=f"online_r{number}_{row['index']:04d}",status=row['status'],
                rWL_um=(row['metrics']or{}).get('rWL_um'),source_run=row.get('physical_run'),
                physical_manifest_sha256=row.get('physical_manifest_sha256'),
                metrics_sha256=row.get('metrics_sha256'),outcome_path=row['outcome_path'],
                outcome_sha256=row['outcome_sha256']))
    if any(len(rows)!=plan['online_candidates_per_arm']for rows in online.values()):
        raise ValueError('Expanded online reference denominator differs')
    choices={arm:choice(offline+online[arm])for arm in plan['arms']}
    return dict(plan_sha256=sha(experiment/'training_replay_reference_plan.json'),choices=choices,
        note='Selection over384 fixed TRAIN and16 own used online labels uses only GRT; final PPA subsequently measured')


def main(args):
    exp=Path(args.experiment).resolve()
    plan=json.loads((exp/'training_replay_reference_plan.json').read_text())
    chosen=select(exp,plan)
    folder=exp/'best_replay_reference';folder.mkdir(exist_ok=True)
    restart=int(os.environ.get('SLURM_RESTART_COUNT','0'))
    source_key=os.environ['SLURM_JOB_ID']+(f'_restart_{restart}'if restart else'')
    snapshot=folder/('source_job_'+source_key);snapshot.mkdir(exist_ok=False)
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
    fixed=json.loads((exp/'full_route/best_offline384/outcome.json').read_text())
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
                    '--source-run',str(source),'--run-id',plan['run_id_prefix']+'_best_replay_'+arm,
                    '--method','best_measured_replay_'+arm,'--output',str(output),
                    '--end-iter',str(plan['detailed_route_end_iteration']),'--storage','scratch'])
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
                    '--source-run',str(source),'--run-id',plan['run_id_prefix']+'_best_replay_'+arm,
                    '--method','best_measured_replay_'+arm,'--output',str(output),
                    '--end-iter',str(plan['detailed_route_end_iteration']),'--storage','scratch'])
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
