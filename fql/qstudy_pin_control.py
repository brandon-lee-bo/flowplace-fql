"""Test explicit per-hypernet pin geometry without enlarging the critic trunk.

The existing 17 global geometry features collapse net identity into total HPWL.
This small-data control appends each macro hypernet's normalized x/y spans and
their squares. It uses only static public netlist/pin data and raw proposals.
It cannot reconstruct the absent standard-cell timing graph, and never assumes
a derivative of hard legalization. No actor or backend jobs are launched here.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import signal

import numpy as np
import torch

from .graph_o2o_archive import atomic_json, sha
from .qstudy_data import METRIC_NAMES, load_dataset
from .qstudy_models import MultiMetricCritic, measured_costs
from . import qstudy_screen as screen

ROOT = Path(__file__).resolve().parents[1]


class PinFeatureCritic(MultiMetricCritic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec.update(schema='qstudy_pin_feature_critic_v1',
            added_features='per-hypernet normalized x/y spans and squared spans',
            added_feature_count=4*self.extractor.hpwl.num_nets,
            topology_scope='macro hypernets only; no standard-cell timing graph')

    def pin_spans(self, actions):
        x = self._coordinates(actions)
        h = self.extractor.hpwl
        if h.num_nets == 0:
            return x.new_zeros((len(x), 0, 2))
        # Offsets are center-relative cache units; normalized canvas is [-1,1].
        pins = x[:, h.pin_nodes] + 2*h.pin_offsets.to(x.dtype)/h.canvas.to(x.dtype)
        index = h.pin_nets[None, :, None].expand(len(x), -1, 2)
        shape = (len(x), h.num_nets, 2)
        low = x.new_full(shape, float('inf')).scatter_reduce(1, index, pins,
            reduce='amin', include_self=True)
        high = x.new_full(shape, -float('inf')).scatter_reduce(1, index, pins,
            reduce='amax', include_self=True)
        return high-low

    def features(self, actions):
        spans = self.pin_spans(actions)
        return torch.cat((super().features(actions), spans.flatten(1),
                          spans.square().flatten(1)), dim=1)


def make_model(cache, x, costs, mask, movable, variant, families, seed):
    if variant != 'pin_huber':
        raise ValueError('This fixed control implements only pin_huber')
    model = PinFeatureCritic(cache, x, costs, movable, screen.METRICS,
        train_mask=mask, train_ids=[f'family{int(f)}_row{i}' for i,f in enumerate(families)],
        train_splits=['train']*len(x), width=128, depth=2, seed=seed)
    return model, model.normalization_metadata()


def run(args):
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run all critic fitting under Slurm')
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    torch.use_deterministic_algorithms(True)
    for sig in [signal.SIGTERM, signal.SIGUSR1]:
        signal.signal(sig, screen._stop)
    data, manifest = load_dataset(args.dataset)
    cache_path = ROOT/'baseline/runtime/benchmark_cache'/(manifest['design']+'.pt')
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    from baseline.api import load_problem
    movable = ~load_problem(manifest['design'], 'cpu').cond.is_ports.bool()
    arrays = {}
    for split in ['train', 'validation']:
        source = data[split]
        costs, mask = measured_costs(torch.tensor(source['metrics'][:, :5], dtype=torch.float32),
            METRIC_NAMES[:5], torch.tensor(source['metric_mask'][:, :5]))
        keep = mask[:, 0] & (costs[:, 0] > 0)
        arrays[split] = dict(x=torch.tensor(source['raw_action'], dtype=torch.float32).reshape(
            len(source['raw_action']), -1, 2)[keep], costs=costs[keep], mask=mask[keep],
            families=torch.tensor(source['family'], dtype=torch.long)[keep])
    files = ['fql/qstudy_pin_control.py', 'fql/qstudy_pin_control.slurm',
        'fql/qstudy_screen.py', 'fql/qstudy_models.py', 'fql/qstudy_data.py',
        'fql/physical_ridge.py', 'fql/analytic.py']
    binding = dict(dataset_sha256=sha(args.dataset),
        manifest_sha256=sha(args.dataset.with_name('manifest.json')),
        cache_sha256=sha(cache_path), source_sha256={n:sha(ROOT/n) for n in files},
        config={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        fold_seed=41307, heldout_used=False, backend_calls=0,
        hypothesis='Per-net geometry regularizes connectivity inference from small data',
        matched_control='existing multi_huber; same trunk, optimizer, batches and folds; extra input parameters disclosed')
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'binding.json'
    if path.exists() and json.loads(path.read_text()) != binding:
        raise ValueError('Changed pin-feature control source/data/configuration')
    atomic_json(path, binding)
    for name in files:
        dest = args.output/'source_snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and sha(dest) != binding['source_sha256'][name]:
            raise ValueError('Frozen source snapshot changed')
        if not dest.exists():
            shutil.copyfile(ROOT/name, dest)
    context = {**arrays['train'], 'cache':cache, 'movable':movable, 'binding':binding}
    groups = np.unique(context['families'].numpy())
    np.random.default_rng(41307).shuffle(groups)
    results = []
    # Explicit process-local dependency injection reuses the exact frozen fit
    # loop; it changes no source file or running baseline experiment.
    previous = screen.make_model
    screen.VARIANTS['pin_huber'] = dict(regression='huber')
    screen.make_model = make_model
    try:
        for fold, held in enumerate(np.array_split(groups, args.folds)):
            target = np.flatnonzero(np.isin(context['families'], held))
            train = np.flatnonzero(~np.isin(context['families'], held))
            results.append(screen.fit(args, context, 'cv_'+str(fold), train, target, arrays['train']))
        results.append(screen.fit(args, context, 'refit', np.arange(len(context['x'])),
            np.arange(len(arrays['validation']['x'])), arrays['validation']))
    finally:
        screen.make_model = previous
        del screen.VARIANTS['pin_huber']
    atomic_json(args.output/'completed.json', dict(status='completed', binding=binding, outcomes=results))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--folds', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    args = p.parse_args()
    if args.seed < 0 or args.steps < 1 or args.folds < 2:
        p.error('Invalid fitting budget/seed')
    args.variant = 'pin_huber'
    run(args)


if __name__ == '__main__':
    main()
