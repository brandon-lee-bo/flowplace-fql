"""Declare and dispatch the fixed critic matrix; scheduler changes are separate."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from .graph_o2o_archive import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/fact_data/bohanli/flowplace_fql/qstudy_v1')
VARIANTS = ['legacy', 'multi_mse', 'multi_huber', 'huber_rank', 'huber_local', 'huber_residual', 'huber_large']


def prepare(output, datasets, steps):
    output.mkdir(parents=True, exist_ok=False)
    sources = output/'source_snapshot'
    sources.mkdir()
    names = ['fql/qstudy_screen.py', 'fql/qstudy_models.py', 'fql/qstudy_data.py',
             'fql/graph_o2o_models.py', 'fql/physical_ridge.py', 'fql/analytic.py',
             'fql/qstudy_batch.py', 'fql/qstudy_batch.slurm', 'docs/qstudy_plan.md']
    hashes = {}
    for name in names:
        dest = sources/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
        hashes[name] = sha(dest)
    tasks = []
    for dataset in datasets:
        path = BASE/'data_audit'/dataset/'dataset.npz'
        if not path.exists():
            raise ValueError('Audit must complete before the matrix is frozen')
        for seed in [0, 1, 2]:
            for variant in VARIANTS:
                tasks.append(dict(dataset=str(path), variant=variant, seed=seed, steps=steps,
                    folds=3, output=str(output/dataset/variant/f'seed_{seed}'), dataset_sha256=sha(path)))
    path = output/'tasks.jsonl'
    path.write_text(''.join(json.dumps(t)+'\n' for t in tasks))
    atomic_json(output/'plan.json', dict(tasks_sha256=sha(path), source_sha256=hashes,
        count=len(tasks), datasets=datasets, seeds=[0, 1, 2], variants=VARIANTS,
        steps=steps, folds=3, protocol='docs/qstudy_plan.md', status='declared_before_results',
        model_selection='grouped cross-validation first; historical validation secondary'))
    print(json.dumps(dict(task_file=str(path), count=len(tasks))))


def run(path, index):
    plan = json.loads(path.with_name('plan.json').read_text())
    if sha(path) != plan['tasks_sha256']:
        raise ValueError('Declared matrix changed')
    for name, digest in plan['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Frozen screening source changed: '+name)
    tasks = [json.loads(line) for line in path.read_text().splitlines()]
    task = tasks[index]
    if sha(task['dataset']) != task['dataset_sha256']:
        raise ValueError('Audited numerical input changed')
    from .qstudy_screen import main
    sys.argv = ['qstudy_screen']
    for key in ['dataset', 'output', 'variant', 'seed', 'steps', 'folds']:
        sys.argv += ['--'+key, str(task[key])]
    main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--datasets', nargs='+', default=['bp_fe384', 'bp_be64'])
    p.add_argument('--steps', type=int, default=2000)
    p = sub.add_parser('run')
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.output, args.datasets, args.steps)
    else:
        if 'SLURM_JOB_ID' not in os.environ:
            raise RuntimeError('Run the matrix under Slurm')
        run(args.matrix, args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID']))


if __name__ == '__main__':
    main()
