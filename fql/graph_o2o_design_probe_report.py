"""Account for all fixed new-design import probes and their actual geometry."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import statistics

from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import ROOT
from .graph_o2o_collect import METRICS


def run(args):
    plan_path=Path(args.plan).resolve();plan=json.loads(plan_path.read_text())
    declared=Path(plan['design_plan']);design=json.loads(declared.read_text())
    if sha(declared)!=plan['design_plan_sha256']:
        raise ValueError('The fixed new-design source plan changed')
    audit_path=Path(design['audit_output'])/'evidence.json'
    evidence=json.loads(audit_path.read_text())
    root=Path(plan['output']);outcomes=[];bindings=[];common_flow=None
    for index,family in enumerate(plan['TRAIN_families']):
        path=root/f'family_{family:04d}/outcome.json'
        out=json.loads(path.read_text());spec=out['spec']
        probe=evidence['import_probes'][index]
        if (spec['plan_sha256']!=sha(plan_path)or spec['source_family']!=family or
                spec['behavior_audit_sha256']!=sha(audit_path)or
                spec['probe_sha256']!=probe['sha256']or sha(probe['path'])!=probe['sha256']or
                spec['design']!=design['design']or spec['threads']!=8 or
                spec['reward']is not None or spec['reward_scale']is not None):
            raise ValueError('Physical probe does not bind its fixed TRAIN action and evaluator')
        if out['status']=='success':
            physical=Path(out['physical_run'])
            manifest=json.loads((physical/'manifest.json').read_text())
            if (sha(physical/'manifest.json')!=out['physical_manifest_sha256']or
                    sha(physical/'metrics.json')!=out['metrics_sha256']or
                    json.loads((physical/'metrics.json').read_text())!=out['metrics']or
                    not out['backend_macro_legal']or
                    sha(out['backend_geometry_path'])!=out['backend_geometry_sha256']or
                    manifest['design']!=design['design']or int(manifest['num_cores'])!=8):
                raise ValueError('Measured probe PPA or actual macro geometry changed')
            flow={key:manifest[key]for key in ['physical_flow_revision','physical_flow_components']}
            if common_flow is None:common_flow=flow
            if flow!=common_flow:raise ValueError('Different physical flows in fixed import probes')
        elif out['status']not in ['geometry_failure','physical_failure']or out['metrics']is not None:
            raise ValueError('Unresolved/unsupported probe outcome; do not omit it')
        outcomes.append(out);bindings.append(dict(path=str(path),sha256=sha(path)))
    good=[out for out in outcomes if out['status']=='success']
    means={key:statistics.mean(out['metrics'][key]for out in good)for key in METRICS}if len(good)==len(outcomes)else None
    snapshot=root/('report_source_job_'+os.environ['SLURM_JOB_ID']+'.py')
    shutil.copyfile(__file__,snapshot)
    result=dict(status='all_fixed_probes_accounted',plan_sha256=sha(plan_path),
        design=design['design'],requested=len(outcomes),successful=len(good),
        failures=[dict(family=out['spec']['source_family'],status=out['status'],failure_class=out['failure_class'])
                  for out in outcomes if out['status']!='success'],
        mean=means,bindings=bindings,common_physical_flow=common_flow,
        backend_geometry_legal=sum(out.get('backend_macro_legal',False)for out in outcomes),
        stage='global_route_only',detailed_route_DRC=None,RL_training=False,
        real_reward_scale=None,all_outcomes_retained=True,heldout_PPA_used=False,
        source_sha256=sha(snapshot),slurm_job_id=os.environ['SLURM_JOB_ID'],goal_completed=False)
    atomic_json(root/'summary.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True)
    args=p.parse_args()
    if 'SLURM_JOB_ID'not in os.environ:raise RuntimeError('Use Slurm for physical-evidence aggregation')
    run(args)
