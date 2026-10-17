"""Fixed, resumable real-label collection for a second circuit.

All requested TRAIN/validation actions are accounted for. Known physical
failures retain their utility and null PPA; unknown errors are not labels.
The four exact TRAIN probes are reused, not relegalized or remeasured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

import numpy as np

from .graph_o2o_archive import archive_run, atomic_json, sha
from .graph_o2o_backend import ROOT
from .graph_o2o_controller import states, LIVE, RESTARTABLE
from .graph_o2o_design_verify import context, verify
from .graph_o2o_geometry import audit


def inputs(path):
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    for name, digest in plan['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Fixed label implementation changed: ' + name)
    original_path = Path(plan['design_plan'])
    original = json.loads(original_path.read_text())
    if sha(original_path) != plan['design_plan_sha256']:
        raise ValueError('Original second-design selection changed')
    for name, digest in original['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Original behavior generator/exporter changed: ' + name)
    if sha(Path(original['audit_output']) / 'evidence.json') != plan['behavior_audit_sha256']:
        raise ValueError('Original behavior data audit changed')
    if plan['selection'] != original['future_initial_real_label_selection']:
        raise ValueError('Labels crossed the selection declared before PPA existed')
    source_path = Path(original['dataset']) / 'dataset.npz'
    if sha(source_path) != plan['behavior_dataset_sha256']:
        raise ValueError('Frozen behavior coordinates changed')
    verification_path = Path(plan['verification_plan'])
    if sha(verification_path) != plan['verification_plan_sha256']:
        raise ValueError('Design-specific evaluator binding changed')
    verification = json.loads(verification_path.read_text())
    context(verification)
    if verification['design'] != original['design'] or plan['requested'] != 80:
        raise ValueError('Wrong design or label denominator')
    with np.load(source_path, allow_pickle=False) as a:
        arrays = {key: a[key].copy() for key in ['raw_action', 'action', 'noise', 'split',
            'critic_split', 'critic_parent_index', 'variant', 'legal']}
    pairs = [(family, variant) for family in
        plan['selection']['TRAIN_families'] + plan['selection']['validation_families']
        for variant in plan['selection']['variants']]
    selected = []
    for family, variant in pairs:
        found = np.flatnonzero((arrays['critic_parent_index'] == family) & (arrays['variant'] == variant))
        if len(found) != 1:
            raise ValueError('Missing or repeated fixed family/variant')
        row = int(found[0])
        split = 0 if family in plan['selection']['TRAIN_families'] else 1
        if arrays['split'][family] != split or arrays['critic_split'][row] != split:
            raise ValueError('TRAIN/validation family boundary changed')
        selected.append((family, variant, row, split))
    if len(selected) != 80 or len({row for _, _, row, _ in selected}) != 80:
        raise ValueError('Fixed label denominator changed')
    return path, plan, original, verification, arrays, selected


def requests(plan_path, plan, selected, arrays=None):
    folder = Path(plan['output']) / 'requests'
    marker = json.loads((folder / 'completed.json').read_text())
    task_path = folder / 'tasks.jsonl'
    if marker['label_plan_sha256'] != sha(plan_path) or marker['tasks_sha256'] != sha(task_path):
        raise ValueError('Fixed label manifest changed')
    rows = [json.loads(line) for line in task_path.read_text().splitlines() if line.strip()]
    if len(rows) != 80 or marker['count'] != 80:
        raise ValueError('Some requested actions disappeared')
    for index, (row, item) in enumerate(zip(rows, selected)):
        family, variant, source, split = item
        if (row['index'] != index or row['source_family'] != family or row['variant'] != variant or
                row['source_row_index'] != source or row['critic_split'] != split or
                row['label_plan_sha256'] != sha(plan_path)):
            raise ValueError('A requested action crossed its fixed source identity')
        local = Path(row['sample_dir']) / 'request.json'
        if json.loads(local.read_text()) != row:
            raise ValueError('Request differs from its immutable manifest')
        with np.load(Path(row['sample_dir']) / 'actions.npz', allow_pickle=False) as a:
            for key, inline in [('raw_action', 'raw_action'), ('action', 'executed_action')]:
                if not np.array_equal(a[key], np.asarray(row[inline], dtype=np.float32)):
                    raise ValueError('Raw/executed action sidecar differs')
                if arrays is not None and not np.array_equal(a[key].reshape(-1), arrays[key][source]):
                    raise ValueError('Measured action differs from the preselected source dataset row')
        action_sha = hashlib.sha256(np.asarray(row['executed_action'], dtype='<f4').tobytes()).hexdigest()
        if action_sha != row['action_sha256']:
            raise ValueError('Executed action identity differs')
        if row.get('def_path') and sha(row['def_path']) != row['def_sha256']:
            raise ValueError('Frozen exported DEF changed')
    return rows, task_path


def export(plan_path, plan, original, verification, arrays, selected):
    import torch
    from analysis.placement_metrics import MacroHPWL, legality
    from baseline.api import RUNTIME, load_problem, save_sample
    from backend.scripts.export_macro_def import convert

    destination = Path(plan['output']) / 'requests'
    if (destination / 'completed.json').exists():
        requests(plan_path, plan, selected, arrays)
        return
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    if sha(RUNTIME / 'benchmark_cache' / f"{original['design']}.pt") != original['cache_sha256']:
        raise ValueError('New-design macro identities or cache geometry changed')
    problem = load_problem(original['design'], 'cpu')
    metric = MacroHPWL(problem.benchmark_info)
    data_audit = json.loads((Path(original['audit_output']) / 'evidence.json').read_text())
    probes = {item['record']['source_family']: item for item in data_audit['import_probes']}
    probe_plan = json.loads(Path(verification['original_probe_plan']).read_text())
    rows = []
    for index, (family, variant, source, split) in enumerate(selected):
        folder = destination / f'row_{index:04d}'
        path = folder / 'request.json'
        if path.exists():
            old = json.loads(path.read_text())
            if old['source_row_index'] != source or old['label_plan_sha256'] != sha(plan_path):
                raise ValueError('Interrupted export belongs to a different plan')
            rows.append(old)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        raw = arrays['raw_action'][source].reshape(-1, 2)
        action = arrays['action'][source].reshape(-1, 2)
        noise = arrays['noise'][family].reshape(-1, 2)
        geometry = legality(action, metric.sizes)
        if geometry['legal'] != bool(arrays['legal'][source]):
            raise ValueError('Independent source geometry check disagrees')
        row = dict(index=index, source_family=family, variant=variant, source_row_index=source,
            critic_split=split, design=original['design'], terminal_mask=0,
            raw_action=raw.tolist(), executed_action=action.tolist(), geometry=geometry,
            action_sha256=hashlib.sha256(np.asarray(action, dtype='<f4').tobytes()).hexdigest(),
            sample_dir=str(folder), label_plan_sha256=sha(plan_path),
            collection_kind='fixed_offline_behavior_not_online_feedback',
            behavior_dataset_sha256=plan['behavior_dataset_sha256'],
            training_eligible=split == 0, proxy_reward_used=False)
        np.savez_compressed(folder / 'actions.npz', raw_action=raw, action=action, noise=noise)
        if variant == 0 and family in probes:
            item = probes[family]
            probe = item['record']
            if (probe['raw_action'] != row['raw_action'] or probe['executed_action'] != row['executed_action'] or
                    probe['source_row_index'] != source or sha(item['path']) != item['sha256']):
                raise ValueError('Attempt to reuse a probe with different source/action')
            measured_path = Path(probe_plan['output']) / f'family_{family:04d}' / 'outcome.json'
            measured = json.loads(measured_path.read_text())
            if measured['spec']['probe_sha256'] != item['sha256']:
                raise ValueError('Probe score is not bound to this exact action')
            row.update(def_path=probe['def_path'], def_sha256=probe['def_sha256'],
                rounded_geometry=probe['rounded_DEF_legal'], reuse_probe_outcome=str(measured_path),
                reuse_probe_outcome_sha256=sha(measured_path))
        elif geometry['legal']:
            save_sample(folder, torch.from_numpy(action), torch.from_numpy(noise), problem, row)
            def_path = folder / f"{original['design']}.def"
            # Export is deterministic. Reuse only a previously completed, bound export.
            if def_path.exists():
                export_audit = json.loads(def_path.with_suffix('.audit.json').read_text())
                if export_audit['output_sha256'] != sha(def_path):
                    raise ValueError('Interrupted DEF/export audit mismatch')
            else:
                export_audit = convert(original['design'], folder / 'macro_physical.json', def_path)
            row.update(def_path=str(def_path), def_sha256=sha(def_path),
                       rounded_geometry=export_audit['legal_no_halo'])
        if not geometry['legal']:
            row['pre_backend_failure'] = 'normalized_geometry'
        elif not row.get('rounded_geometry'):
            row['pre_backend_failure'] = 'rounded_DEF_geometry'
        atomic_json(path, row)
        rows.append(row)
    task_path = destination / 'tasks.jsonl'
    task_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    atomic_json(destination / 'completed.json', dict(count=len(rows), tasks_sha256=sha(task_path),
        label_plan_sha256=sha(plan_path), slurm_job_id=os.environ['SLURM_JOB_ID']))
    requests(plan_path, plan, selected, arrays)
    print(json.dumps(dict(status='all_80_fixed_actions_exported', reused_exact_probes=4,
                         new_physical_requests=76)), flush=True)


def checked_outcome(row, verification):
    path = Path(row['sample_dir']) / 'outcome.json'
    out = json.loads(path.read_text())
    if (out['request_sha256'] != sha(Path(row['sample_dir']) / 'request.json') or
            out['action_sha256'] != row['action_sha256'] or out['label_plan_sha256'] != row['label_plan_sha256'] or
            out['raw_action'] != row['raw_action'] or out['executed_action'] != row['executed_action'] or
            out['reward'] is not None):
        raise ValueError('Physical label changed fixed request/action or assigned a premature reward')
    if out.get('physical_run'):
        verified = verify(out['physical_run'], row, verification)
        for key in ['physical_manifest_sha256', 'physical_flow_revision']:
            if verified[key] != out[key]:
                raise ValueError('Measured physical evidence changed: ' + key)
        if verified.get('metrics_sha256') != out.get('metrics_sha256'):
            raise ValueError('Physical PPA file changed')
        if out['status'] == 'success' and verified['metrics'] != out['metrics']:
            raise ValueError('Reported PPA differs from reparsed physical logs')
        if 'backend_geometry_path' in out:
            if sha(out['backend_geometry_path']) != out['backend_geometry_sha256']:
                raise ValueError('Actual macro geometry evidence changed')
            geometry = json.loads(Path(out['backend_geometry_path']).read_text())
            if geometry['legal_no_halo'] != out['backend_macro_legal']:
                raise ValueError('Actual macro geometry status changed')
        if out['status'] == 'physical_failure':
            supported = ((verified['status'] == 'physical_failure' and
                          out['failure_class'] == verified['failure_class']) or
                         (verified['status'] == 'success' and not out.get('backend_macro_legal', True) and
                          out['failure_class'] == 'backend_macro_overlap_or_bounds'))
            if not supported:
                raise ValueError('Declared physical failure lacks matching independent evidence')
    elif out['status'] == 'geometry_failure' and out.get('failure_class') != row.get('pre_backend_failure'):
        raise ValueError('Geometry failure differs from the independent request audit')
    if out['status'] == 'success':
        if out['metrics'] is None or not out.get('backend_macro_legal'):
            raise ValueError('Successful physical label lacks legal geometry/PPA')
    elif out['status'] not in ['physical_failure', 'geometry_failure'] or out['metrics'] is not None:
        raise ValueError('Unresolved failures cannot become real labels')
    return out


def measure(plan_path, plan, original, verification, arrays, selected, index):
    rows, task_path = requests(plan_path, plan, selected, arrays)
    row = rows[index]
    output = Path(row['sample_dir'])
    if (output / 'outcome.json').exists():
        checked_outcome(row, verification)
        return
    base = dict(request_sha256=sha(output / 'request.json'), label_plan_sha256=sha(plan_path),
        action_sha256=row['action_sha256'], raw_action=row['raw_action'], executed_action=row['executed_action'],
        index=index, source_family=row['source_family'], variant=row['variant'], critic_split=row['critic_split'],
        terminal_mask=0, reward=None, reward_scale=None, policy_feedback=False,
        verification_plan_sha256=plan['verification_plan_sha256'], stage='global_route_only',
        not_final_table2=True, detailed_route_DRC=None, slurm_job_id=os.environ['SLURM_JOB_ID'])
    if row.get('pre_backend_failure'):
        atomic_json(output / 'outcome.json', dict(base, status='geometry_failure', metrics=None,
            failure_class=row['pre_backend_failure'], finished_unix=time.time()))
        return
    measured = None
    reused = None
    if 'reuse_probe_outcome' in row:
        path = Path(row['reuse_probe_outcome'])
        if sha(path) != row['reuse_probe_outcome_sha256']:
            raise ValueError('Exact reused probe outcome changed')
        probe = json.loads(path.read_text())
        measured = verify(probe['physical_run'], row, verification)
        reused = str(path)
    attempts = output / 'attempts'
    attempts.mkdir(exist_ok=True)
    restart = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
    key = os.environ['SLURM_JOB_ID'] + (f'_restart_{restart}' if restart else '')
    if measured is None:
        for path in sorted(attempts.glob('*/attempt.json')):
            previous = json.loads(path.read_text())
            if previous['request_sha256'] != base['request_sha256']:
                raise ValueError('Historical label attempt changed action')
            physical = Path(previous['physical_run'])
            manifest_path = physical / 'manifest.json'
            if manifest_path.exists() and json.loads(manifest_path.read_text()).get('time_end'):
                measured = verify(physical, row, verification)
                reused = str(path)
                break
            if previous['job_id'] == os.environ['SLURM_JOB_ID'] and previous['restart_count'] < restart:
                continue
            observed = states(previous['job_id'])
            if any(state in LIVE for state in observed) or not all(state in RESTARTABLE for state in observed):
                raise RuntimeError('Do not duplicate unresolved label attempt: ' + str(observed))
    if measured is None:
        name = f"go2o_{original['design']}_initial_i{index:03d}_a{key}"
        physical = ROOT / 'backend/results' / name
        attempt = attempts / key
        attempt.mkdir(exist_ok=False)
        atomic_json(attempt / 'attempt.json', dict(request_sha256=base['request_sha256'],
            physical_run=str(physical), job_id=os.environ['SLURM_JOB_ID'], restart_count=restart,
            source_sha256=plan['source_sha256'], started_unix=time.time()))
        with (attempt / 'runner.log').open('w') as log:
            process = subprocess.run(['bash', str(ROOT / 'fql/graph_o2o_label.sh'),
                '--design', original['design'], '--def-path', row['def_path'], '--run-id', name,
                '--storage', 'scratch'], stdout=log, stderr=subprocess.STDOUT)
        atomic_json(attempt / 'process_exit.json', dict(returncode=process.returncode, ended_unix=time.time()))
        measured = verify(physical, row, verification)
        if process.returncode and measured['status'] == 'success':
            raise RuntimeError('Wrapper exit contradicts successful physical manifest')
    if measured['status'] == 'success':
        geometry = audit(measured['physical_run'])
        geometry_path = Path(measured['physical_run']) / 'backend_macro_audit.json'
        measured.update(backend_geometry_path=str(geometry_path), backend_geometry_sha256=sha(geometry_path),
                        backend_macro_legal=geometry['legal_no_halo'])
        if not geometry['legal_no_halo']:
            measured.update(status='physical_failure', failure_class='backend_macro_overlap_or_bounds',
                            observed_metrics=measured['metrics'], metrics=None)
    atomic_json(output / 'measurement.json', dict(base, **measured, reused_completed_measurement=reused))
    archive = archive_run(measured['physical_run'])
    atomic_json(output / 'outcome.json', dict(base, **measured, archive=archive,
        reused_completed_measurement=reused, finished_unix=time.time()))
    checked_outcome(row, verification)
    print(json.dumps(dict(index=index, status=measured['status'], reused=reused, reward=None)), flush=True)


def report(plan_path, plan, original, verification, arrays, selected):
    rows, task_path = requests(plan_path, plan, selected, arrays)
    outcomes = [checked_outcome(row, verification) for row in rows]
    successful_train = [out['metrics']['rWL_um'] for out in outcomes
                        if out['critic_split'] == 0 and out['status'] == 'success']
    if not successful_train:
        raise ValueError('No successful TRAIN labels exist to define a physical reward scale')
    scale = statistics.mean(successful_train)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid TRAIN-only physical reward scale')
    rewards = [-out['metrics']['rWL_um'] / scale if out['status'] == 'success'
               else plan['physical_failure_reward'] for out in outcomes]
    labels = []
    for row, out, reward in zip(rows, outcomes, rewards):
        path = Path(row['sample_dir']) / 'outcome.json'
        labels.append(dict(source_row_index=row['source_row_index'], source_family=row['source_family'],
            variant=row['variant'], critic_split=row['critic_split'], raw_action=row['raw_action'],
            executed_action=row['executed_action'], action_sha256=row['action_sha256'], status=out['status'],
            failure_class=out.get('failure_class'), metrics=out['metrics'], reward=reward, terminal_mask=0,
            outcome_path=str(path), outcome_sha256=sha(path), request_sha256=out['request_sha256'],
            physical_run=out.get('physical_run'), physical_manifest_sha256=out.get('physical_manifest_sha256')))
    dataset = {key: np.asarray(values) for key, values in dict(
        raw_action=[np.asarray(row['raw_action'], dtype=np.float32).reshape(-1) for row in rows],
        action=[np.asarray(row['executed_action'], dtype=np.float32).reshape(-1) for row in rows],
        reward=np.asarray(rewards, dtype=np.float32),
        backend_rwl_um=[out['metrics']['rWL_um'] if out['status'] == 'success' else np.nan for out in outcomes],
        backend_success=[out['status'] == 'success' for out in outcomes],
        source_row_index=[row['source_row_index'] for row in rows],
        critic_split=[row['critic_split'] for row in rows],
        critic_group_id=[row['source_family'] for row in rows],
        critic_parent_index=[row['source_family'] for row in rows],
        legal=[row['geometry']['legal'] for row in rows]).items()}
    destination = Path(plan['output']) / 'dataset'
    destination.mkdir(exist_ok=True)
    binding = dict(label_plan_sha256=sha(plan_path), tasks_sha256=sha(task_path),
        requested=80, TRAIN_requested=64, validation_requested=16,
        successful=sum(out['status'] == 'success' for out in outcomes),
        reward_scale_train_only=scale, reward_scale_source='Successful TRAIN GRT rWL mean, frozen before any policy training; failed TRAIN rows retained with declared utility',
        reward_stage='global_route_only', design=original['design'], physical_failure_reward=plan['physical_failure_reward'],
        behavior_dataset_sha256=plan['behavior_dataset_sha256'], test_labels_collected=False,
        all_requested_accounted=True, proxy_rewards_used=False,
        outcomes=[dict(path=item['outcome_path'], sha256=item['outcome_sha256']) for item in labels])
    manifest_path = destination / 'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if any(old[key] != value for key, value in binding.items()):
            raise ValueError('Previously frozen real reward dataset changed')
        if sha(destination / 'dataset.npz') != old['dataset_sha256']:
            raise ValueError('Previously frozen real reward arrays changed')
        return
    pending = destination / 'dataset.pending.npz'
    np.savez_compressed(pending, **dataset)
    pending.replace(destination / 'dataset.npz')
    atomic_json(destination / 'backend_labels.json', labels)
    atomic_json(manifest_path, dict(binding, dataset_sha256=sha(destination / 'dataset.npz'),
        backend_labels_sha256=sha(destination / 'backend_labels.json'), slurm_job_id=os.environ['SLURM_JOB_ID']))
    print(json.dumps(binding), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--stage', required=True, choices=['export', 'measure', 'report'])
    p.add_argument('--index', type=int)
    args = p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run exports, backend work and data audits through Slurm')
    values = inputs(args.plan)
    if args.stage == 'measure':
        if int(os.environ['SLURM_CPUS_PER_TASK']) != 8:
            raise RuntimeError('The fixed physical evaluator requires eight threads')
        index = args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
        measure(*values, index)
    else:
        (export if args.stage == 'export' else report)(*values)
