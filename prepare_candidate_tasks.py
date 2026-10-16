"""Freeze a paired cross-fidelity backend comparison before routed outcomes.

The input methods were selected on macro-only validation. This experiment
measures whether that proxy transfers; it does not select models on GRT results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from analysis.placement_metrics import MacroHPWL, legality
from baseline.api import load_problem, save_sample

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', required=True)
    p.add_argument('--training-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--count', type=int, default=3)
    p.add_argument('--prefix', default='bp_fe_proxy_transfer_v1')
    args = p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run numerical validation and exports through Slurm')
    if args.count < 1 or Path(args.prefix).name != args.prefix:
        raise ValueError('Invalid sample count or run ID prefix')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    ds, training, out = map(lambda x: Path(x).resolve(),
                            [args.dataset_dir, args.training_dir, args.output])
    out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((ds / 'manifest.json').read_text())
    if meta.get('reward_stage') == 'global_route_only':
        raise ValueError('This exporter is specifically a macro-proxy transfer experiment')
    design = meta['args']['design']
    selection_path = training / 'selection.json'
    selection = json.loads(selection_path.read_text())
    if Path(selection['training_dir']).resolve() != training:
        raise ValueError('Selection belongs to a different training run')
    if sha(training / 'measured_validation_summary.json') != selection['source_sha256']:
        raise ValueError('Validation results changed after selection')
    methods = ['teacher_original', 'teacher_relegalized'] + selection['methods']
    with np.load(ds / 'dataset.npz') as f:
        noise, split = f['noise'], f['split']
    indices = np.flatnonzero(split == 2)[:args.count]
    if len(indices) != args.count:
        raise ValueError('Not enough heldout families')
    plan = {
        'args': vars(args), 'design': design,
        'dataset_sha256': sha(ds / 'dataset.npz'),
        'selection_sha256': sha(selection_path), 'selection_path': str(selection_path),
        'methods': methods, 'indices': indices.tolist(),
        'task_ids': [f'{args.prefix}_m{m:02d}_i{i:04d}'
                     for m in range(len(methods)) for i in indices],
        'selection': 'All macro-validation-selected methods and seeds; first fixed heldout indices',
        'phase': 'exploratory',
        'purpose': 'Exploratory transfer from macro-only HPWL to actual global routing',
        'backend_stage': 'global_route_only', 'not_final_table2': True,
        'no_backend_based_model_selection': True,
        'slurm_job_id': os.environ['SLURM_JOB_ID'],
    }
    (out / 'selection_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    problem = load_problem(design, 'cpu')
    metric = MacroHPWL(problem.benchmark_info)
    tasks = []
    for m, method in enumerate(methods):
        measured = training / 'measured_heldout' / method
        source_path = measured / 'evaluated.npz'
        records = {r['index']: r for r in json.loads((measured / 'records.json').read_text())}
        with np.load(source_path) as f:
            source_indices, actions, source_split = f['indices'], f['action'], f['split']
        for i in indices:
            loc = np.flatnonzero(source_indices == i)
            if len(loc) != 1 or source_split[loc[0]] != 2:
                raise ValueError(f'Missing or wrongly split candidate {method}/{i}')
            row = int(loc[0]); x = actions[row]
            record = records[int(i)]
            geom = legality(x, metric.sizes)
            if not geom['legal'] or not record['legal']:
                raise ValueError(f'Illegal requested candidate {method}/{i}; preserve failed plan')
            if not np.isclose(metric(x), record['macro_hpwl'], rtol=1e-7, atol=1e-4):
                raise ValueError('Executed geometry differs from measured proxy record')
            run_id = f'{args.prefix}_m{m:02d}_i{i:04d}'
            folder = out / run_id
            save_sample(folder, torch.as_tensor(x),
                        torch.as_tensor(noise[i].reshape(x.shape)), problem,
                        {'method': method, 'index': int(i), 'split': 2,
                         'selection': 'Fixed before any candidate backend measurements'})
            def_path = folder / f'{design}.def'
            subprocess.run([sys.executable, str(ROOT / 'backend/scripts/export_macro_def.py'),
                            '--design', design, '--coordinates', str(folder / 'macro_physical.json'),
                            '--output', str(def_path)], check=True, stdout=subprocess.DEVNULL)
            audit_path = def_path.with_suffix('.audit.json')
            first = method.split('/')[0]
            seed = int(first[5:]) if first.startswith('seed_') and first[5:].isdigit() else None
            tasks.append({
                'task_id': run_id, 'run_id': run_id, 'method': method,
                'index': int(i), 'split': 2, 'training_seed': seed, 'design': design,
                'algorithm': method.split('/')[-1],
                'def_path': str(def_path), 'def_sha256': sha(def_path),
                'def_audit_path': str(audit_path), 'def_audit_sha256': sha(audit_path),
                'dataset_sha256': plan['dataset_sha256'],
                'action_path': str(source_path), 'action_file_sha256': sha(source_path),
                'action_key': 'action', 'action_row': row,
                'normalized_action_flat': x.reshape(-1).tolist(),
                'action_sha256': hashlib.sha256(np.ascontiguousarray(x, dtype='<f4').tobytes()).hexdigest(),
                'macro_hpwl': record['macro_hpwl'], 'legal': True,
                'predicted_q': record['predicted_q'], 'q_reward_stage': 'macro_only_hpwl',
                'metrics_path': str(ROOT / 'backend/results' / run_id / 'metrics.json'),
                'source_record': record,
            })
    (out / 'tasks.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    print(json.dumps({'count': len(tasks), 'plan': str(out / 'selection_plan.json'),
                      'tasks': str(out / 'tasks.jsonl')}, indent=2))


if __name__ == '__main__':
    main()
