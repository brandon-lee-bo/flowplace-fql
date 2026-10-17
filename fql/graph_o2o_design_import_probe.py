"""Check actual macro import and GRT on fixed TRAIN probes of a new design.

No proxy becomes a backend reward here. Measured PPA is retained separately;
the new design's RL reward scale will be frozen from its TRAIN labels later.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .graph_o2o_archive import archive_run, atomic_json, sha
from .graph_o2o_backend import ROOT, REFERENCE, verify_physical
from .graph_o2o_controller import states, RESTARTABLE
from .graph_o2o_geometry import audit


def run(args):
    plan_path=Path(args.plan).resolve();plan=json.loads(plan_path.read_text())
    design_plan_path=Path(plan['design_plan']);design_plan=json.loads(design_plan_path.read_text())
    if sha(design_plan_path)!=plan['design_plan_sha256']or sha(REFERENCE)!=plan['reference_manifest_sha256']:
        raise ValueError('Import probe context or physical evaluator changed')
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Import-probe implementation changed before execution: '+name)
    evidence_path=Path(design_plan['audit_output'])/'evidence.json'
    evidence=json.loads(evidence_path.read_text())
    if (evidence['status']!='behavior_data_audited_no_real_rewards_yet'or
            evidence['plan_sha256']!=plan['design_plan_sha256']or
            evidence['design']!=design_plan['design']):
        raise ValueError('The new design lacks its complete behavior/import audit')
    index=args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    expected=design_plan['import_probe_TRAIN_families']
    if expected!=plan['TRAIN_families']or len(evidence['import_probes'])!=len(expected):
        raise ValueError('Fixed TRAIN probe denominator changed')
    item=evidence['import_probes'][index];probe_path=Path(item['path'])
    row=json.loads(probe_path.read_text())
    if (sha(probe_path)!=item['sha256']or row!=item['record']or
            row['source_family']!=expected[index]or row['real_backend_reward']is not None):
        raise ValueError('Fixed probe coordinates or source changed')
    output=Path(plan['output'])/f"family_{expected[index]:04d}"
    output.mkdir(parents=True,exist_ok=True)
    spec=dict(plan_sha256=sha(plan_path),behavior_audit_sha256=sha(evidence_path),
        probe_path=str(probe_path),probe_sha256=sha(probe_path),design=design_plan['design'],
        source_family=expected[index],source_row_index=row['source_row_index'],
        stage='global_route_only',not_final_table2=True,storage='scratch',threads=8,
        TRAIN_only=True,policy_feedback=False,reward=None,reward_scale=None,
        note='Real PPA/import feasibility probe; no RL training or final detailed-route DRC claim')
    outcome_path=output/'outcome.json'
    if outcome_path.exists():
        out=json.loads(outcome_path.read_text())
        if out['spec']!=spec:raise ValueError('Existing probe outcome changed context')
        if out.get('physical_run'):
            physical=Path(out['physical_run'])
            if sha(physical/'manifest.json')!=out['physical_manifest_sha256']:
                raise ValueError('Existing physical probe evidence changed')
        print(json.dumps(dict(status='already_measured',outcome=str(outcome_path))),flush=True)
        return
    if not row['geometry']['legal']or not row.get('rounded_DEF_legal'):
        atomic_json(outcome_path,dict(spec=spec,status='geometry_failure',metrics=None,
            failure_class='normalized_geometry'if not row['geometry']['legal']else'rounded_DEF_geometry',
            slurm_job_id=os.environ['SLURM_JOB_ID']))
        return
    if sha(row['def_path'])!=row['def_sha256']:
        raise ValueError('The exported new-design DEF changed')
    attempts=output/'attempts';attempts.mkdir(exist_ok=True)
    measured=None
    restart=int(os.environ.get('SLURM_RESTART_COUNT','0'))
    source_key=os.environ['SLURM_JOB_ID']+(f'_restart_{restart}'if restart else'')
    for path in sorted(attempts.glob('*/attempt.json')):
        previous=json.loads(path.read_text());physical=Path(previous['physical_run'])
        if previous['spec']!=spec:raise ValueError('Interrupted attempt belongs to another probe')
        manifest_path=physical/'manifest.json'
        if manifest_path.exists():
            manifest=json.loads(manifest_path.read_text())
            if manifest.get('time_end'):
                measured=verify_physical(physical,row,bool(manifest['success']))
                break
        if previous['job_id']==os.environ['SLURM_JOB_ID']and previous['restart_count']<restart:
            continue  # This process is the scheduler's actual requeued allocation.
        observed=states(previous['job_id'])
        if not all(state in RESTARTABLE for state in observed):
            raise RuntimeError('Do not duplicate a live, unresolved or ordinary failed physical attempt: '+str(observed))
    if measured is None:
        physical_id=f"go2o_{design_plan['design']}_import_f{expected[index]:04d}_a{source_key}"
        physical=ROOT/'backend/results'/physical_id
        attempt=attempts/source_key;attempt.mkdir(exist_ok=False)
        for name in ['graph_o2o_design_import_probe.py','graph_o2o_design_import_probe.slurm']:
            shutil.copyfile(Path(__file__).with_name(name),attempt/name)
        atomic_json(attempt/'attempt.json',dict(spec=spec,physical_run=str(physical),
            job_id=os.environ['SLURM_JOB_ID'],restart_count=restart,started_unix=time.time()))
        with(attempt/'runner.log').open('w')as log:
            process=subprocess.run(['bash',str(ROOT/'fql/graph_o2o_label.sh'),
                '--design',design_plan['design'],'--def-path',row['def_path'],
                '--run-id',physical_id,'--storage','scratch'],stdout=log,stderr=subprocess.STDOUT)
        atomic_json(attempt/'process_exit.json',dict(returncode=process.returncode,ended_unix=time.time()))
        if not(physical/'manifest.json').exists():
            raise RuntimeError('Physical wrapper did not create a manifest; no PPA/reward assigned')
        manifest=json.loads((physical/'manifest.json').read_text())
        measured=verify_physical(physical,row,bool(manifest['success']))
        if measured['status']=='success'and process.returncode:
            raise RuntimeError('A wrapper failure contradicts successful physical evidence')
    manifest=json.loads((Path(measured['physical_run'])/'manifest.json').read_text())
    if manifest['design']!=design_plan['design']:
        raise ValueError('Physical runner used the wrong netlist')
    if measured['status']=='success':
        geometry=audit(measured['physical_run'])
        path=Path(measured['physical_run'])/'backend_macro_audit.json'
        measured.update(backend_geometry_path=str(path),backend_geometry_sha256=sha(path),
            backend_macro_legal=geometry['legal_no_halo'])
        if not geometry['legal_no_halo']:
            measured.update(status='physical_failure',failure_class='backend_macro_overlap_or_bounds',
                observed_metrics=measured['metrics'],metrics=None)
    atomic_json(output/'measurement.json',dict(spec=spec,**measured))
    archive=archive_run(measured['physical_run'])
    atomic_json(outcome_path,dict(spec=spec,**measured,archive=archive,
        slurm_job_id=os.environ['SLURM_JOB_ID'],finished_unix=time.time()))
    print(json.dumps(dict(status=measured['status'],outcome=str(outcome_path),
        design=design_plan['design'],real_backend_reward=None,metrics=measured['metrics'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True)
    p.add_argument('--index',type=int);args=p.parse_args()
    if 'SLURM_JOB_ID'not in os.environ or int(os.environ['SLURM_CPUS_PER_TASK'])!=8:
        raise RuntimeError('Use an eight-thread Slurm physical evaluation')
    run(args)
