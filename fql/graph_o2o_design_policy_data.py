"""Bind second-design policy training to all fixed real backend labels.

Known physical failures are retained as utility labels, never numerical PPA.
The initial TRAIN-only reward scale stays fixed throughout online learning.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import ROOT
from .graph_o2o_design_labels import inputs, requests, checked_outcome
from .graph_o2o_design_verify import context as verify_context


def context(experiment):
    experiment = Path(experiment).resolve()
    cfg = json.loads((experiment / 'config.json').read_text())
    implementation = json.loads((experiment / 'implementation_plan.json').read_text())
    if implementation['config_sha256'] != sha(experiment / 'config.json'):
        raise ValueError('The predeclared policy configuration changed')
    for name, digest in implementation['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('The frozen design-policy implementation changed: ' + name)
    declared = cfg['initial_labels']
    plan_path = Path(declared['plan'])
    if sha(plan_path) != declared['plan_sha256']:
        raise ValueError('The fixed initial real-label plan changed')
    plan = json.loads(plan_path.read_text())
    design_path = Path(plan['design_plan'])
    design = json.loads(design_path.read_text())
    if sha(design_path) != plan['design_plan_sha256'] or design['design'] != cfg['design']:
        raise ValueError('Policy and measured netlist identities differ')
    for name, digest in plan['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('The original label implementation changed: ' + name)
    verification_path = Path(plan['verification_plan'])
    if sha(verification_path) != plan['verification_plan_sha256']:
        raise ValueError('Design-specific physical verification changed')
    verification = json.loads(verification_path.read_text())
    verify_context(verification)
    if verification['design'] != cfg['design']:
        raise ValueError('Wrong physical design context')
    return experiment, cfg, plan_path, plan, design, verification


def validate_arrays(arrays, manifest, labels):
    """Independently verify labels/arrays, including retained failure rows."""
    if (manifest['reward_stage'] != 'global_route_only' or manifest['requested'] != 80 or
            manifest['TRAIN_requested'] != 64 or manifest['validation_requested'] != 16 or
            not manifest['all_requested_accounted'] or manifest['test_labels_collected'] or
            manifest['proxy_rewards_used'] or len(labels) != 80 or len(manifest['outcomes']) != 80):
        raise ValueError('Real-label scope or denominator changed')
    split = arrays['critic_split']
    groups = arrays['critic_group_id']
    train = np.flatnonzero(split == 0)
    valid = np.flatnonzero(split == 1)
    if len(train) != 64 or len(valid) != 16 or set(groups[train]) & set(groups[valid]):
        raise ValueError('TRAIN/validation size or family separation changed')
    if set(groups[train]) != set(range(32)) or set(groups[valid]) != set(range(192, 200)):
        raise ValueError('Family selection differs from the pre-PPA declaration')
    if len(set(arrays['source_row_index'].tolist())) != 80:
        raise ValueError('A fixed source label is repeated')
    for key in ['raw_action', 'action', 'reward']:
        if len(arrays[key]) != 80 or not np.isfinite(arrays[key]).all():
            raise ValueError('Missing/nonfinite real action or utility: ' + key)
    successful = arrays['backend_success']
    if successful.dtype != np.bool_:
        raise ValueError('Physical-success mask must be boolean')
    rwl = arrays['backend_rwl_um']
    if (not np.isfinite(rwl[successful]).all() or (rwl[successful] <= 0).any() or
            not np.isnan(rwl[~successful]).all()):
        raise ValueError('Physical PPA must exist only for successful measurements')
    selected = train[successful[train]]
    scale = float(manifest['reward_scale_train_only'])
    if not len(selected) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Cannot determine a real TRAIN reward scale')
    if not np.isclose(rwl[selected].mean(), scale, rtol=1e-12, atol=1e-9):
        raise ValueError('Reward scale is not the successful TRAIN mean')
    expected = np.full(80, manifest['physical_failure_reward'], dtype=np.float64)
    expected[successful] = -rwl[successful] / scale
    np.testing.assert_allclose(arrays['reward'], expected, rtol=1e-6, atol=1e-7)
    if manifest['successful'] != int(successful.sum()):
        raise ValueError('Successful denominator differs from stored arrays')
    for index, label in enumerate(labels):
        if (label['terminal_mask'] != 0 or label['critic_split'] != int(split[index]) or
                label['source_row_index'] != int(arrays['source_row_index'][index]) or
                label['source_family'] != int(groups[index]) or
                (label['status'] == 'success') != bool(successful[index])):
            raise ValueError('A numerical label crossed its source/terminal identity')
        for key, field in [('raw_action', 'raw_action'), ('action', 'executed_action')]:
            np.testing.assert_array_equal(arrays[key][index].reshape(-1),
                                          np.asarray(label[field], dtype=np.float32).reshape(-1))
        if not np.isclose(label['reward'], expected[index], rtol=1e-12, atol=1e-12):
            raise ValueError('Label reward differs from its measured outcome')
        if successful[index]:
            if label['metrics']['rWL_um'] != rwl[index] or not bool(arrays['legal'][index]):
                raise ValueError('Successful physical label has invalid geometry or PPA')
        elif label['status'] not in ['physical_failure', 'geometry_failure'] or label['metrics'] is not None:
            raise ValueError('Unresolved error was turned into a physical label')
    data = {}
    for name, indices in [('train', train), ('validation', valid)]:
        data[name] = {key: value[indices].copy() for key, value in arrays.items()}
        data[name]['group_id'] = groups[indices].copy()
        data[name]['original_local_rows'] = indices
    return data


def load_dataset(experiment, freeze=False):
    experiment, cfg, plan_path, plan, design, verification = context(experiment)
    dataset = Path(plan['output']) / 'dataset'
    manifest_path = dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    npz = dataset / 'dataset.npz'
    label_path = dataset / 'backend_labels.json'
    if (manifest['design'] != cfg['design'] or manifest['label_plan_sha256'] != sha(plan_path) or
            sha(npz) != manifest['dataset_sha256'] or sha(label_path) != manifest['backend_labels_sha256'] or
            manifest['behavior_dataset_sha256'] != plan['behavior_dataset_sha256'] or
            manifest['physical_failure_reward'] != cfg['physical_failure_reward']):
        raise ValueError('Frozen initial RL dataset changed')
    values = inputs(plan_path)
    requested, task_path = requests(values[0], values[1], values[5], values[4])
    labels = json.loads(label_path.read_text())
    with np.load(npz, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    data = validate_arrays(arrays, manifest, labels)
    if len(requested) != len(labels) or manifest['tasks_sha256'] != sha(task_path):
        raise ValueError('Real-label requests differ from the frozen dataset')
    outcomes = []
    for request, label, declared in zip(requested, labels, manifest['outcomes']):
        for field in ['source_row_index', 'source_family', 'variant', 'critic_split']:
            if label[field] != request[field]:
                raise ValueError('A frozen dataset row changed its preselected identity: ' + field)
        path = Path(request['sample_dir']) / 'outcome.json'
        if (label['outcome_path'] != str(path) or declared['path'] != str(path) or
                sha(path) != label['outcome_sha256'] or declared['sha256'] != sha(path)):
            raise ValueError('A dataset label lost its measured outcome binding')
        out = checked_outcome(request, verification)
        for field in ['status', 'raw_action', 'executed_action', 'action_sha256', 'metrics',
                      'physical_run', 'physical_manifest_sha256', 'failure_class', 'request_sha256']:
            if label.get(field) != out.get(field):
                raise ValueError('Dataset label differs from its independently verified outcome: ' + field)
        outcomes.append(dict(path=str(path), sha256=sha(path), status=out['status']))
    behavior_path = Path(design['dataset']) / 'dataset.npz'
    if sha(behavior_path) != plan['behavior_dataset_sha256']:
        raise ValueError('Additional TRAIN-only behavior targets changed')
    binding = dict(version=1, design=cfg['design'], config_sha256=sha(experiment / 'config.json'),
        implementation_plan_sha256=sha(experiment / 'implementation_plan.json'),
        label_plan_path=str(plan_path), label_plan_sha256=sha(plan_path),
        dataset_path=str(npz), dataset_sha256=sha(npz), manifest_path=str(manifest_path),
        manifest_sha256=sha(manifest_path), backend_labels_path=str(label_path),
        backend_labels_sha256=sha(label_path), behavior_dataset_path=str(behavior_path),
        behavior_dataset_sha256=sha(behavior_path), requested=80, TRAIN_requested=64,
        validation_requested=16, reward_scale=manifest['reward_scale_train_only'],
        physical_failure_reward=cfg['physical_failure_reward'], terminal_mask=0,
        heldout_used=False, outcomes=outcomes)
    binding_path = experiment / 'dataset_binding.json'
    if binding_path.exists():
        if json.loads(binding_path.read_text()) != binding:
            raise ValueError('Policy training data changed since the first audit')
    elif freeze:
        atomic_json(binding_path, binding)
    else:
        raise ValueError('Freeze and audit the complete real-label dataset before policy training')
    return data, manifest, binding


def measured_context(experiment):
    """Small per-worker check after the full dataset audit has completed."""
    experiment, cfg, plan_path, plan, design, verification = context(experiment)
    path = experiment / 'dataset_binding.json'
    binding = json.loads(path.read_text())
    if (binding['config_sha256'] != sha(experiment / 'config.json') or
            binding['implementation_plan_sha256'] != sha(experiment / 'implementation_plan.json') or
            binding['label_plan_sha256'] != sha(plan_path) or binding['design'] != cfg['design']):
        raise ValueError('Measured policy context changed')
    for field, digest in [('dataset_path', 'dataset_sha256'), ('manifest_path', 'manifest_sha256'),
                          ('backend_labels_path', 'backend_labels_sha256')]:
        if sha(binding[field]) != binding[digest]:
            raise ValueError('Frozen policy reward data changed: ' + field)
    return cfg, binding, verification
