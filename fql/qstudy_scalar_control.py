"""Direct scalar timing-utility control; no backend calls or actor advancement.

The comparator is the frozen multi_huber screen on exactly the same family
folds, rows, initialization seeds and minibatch RNG. Both predictions are
expressed in U=sum_k w_k*cost_k/mean_TRAIN(cost_k), never normalized by mean(U)
again. Historical validation remains previously inspected tuning data.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from .graph_o2o_archive import atomic_json, sha
from .qstudy_data import METRIC_NAMES, load_dataset
from .qstudy_models import MultiMetricCritic, measured_costs
from .qstudy_screen import atomic_torch, diagnostics


ROOT = Path(__file__).resolve().parents[1]
METRICS = tuple(METRIC_NAMES[:5])
WEIGHTS = (1., .15, .35, 0., 0.)
FOLD_SEED = 41307
STOP = False
SOURCE_NAMES = (
    'fql/qstudy_scalar_control.py', 'fql/qstudy_scalar_control.slurm',
    'fql/qstudy_models.py', 'fql/qstudy_data.py', 'fql/qstudy_screen.py',
    'fql/graph_o2o_models.py', 'fql/physical_ridge.py', 'fql/analytic.py',
    'fql/graph_o2o_archive.py', 'baseline/api.py',
    'upstream/flowplace/benchmark_utils/openroad_utils.py',
)


def _stop(*_):
    global STOP
    STOP = True


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def fit_utility_scales(costs, mask):
    """Same masked TRAIN mean-magnitude scales as the frozen multihead model."""
    valid = mask.bool() & torch.isfinite(costs)
    if bool(((costs < 0) & valid).any()):
        raise ValueError('Utility requires nonnegative converted costs')
    counts = valid.sum(0)
    means = torch.where(valid, costs.abs(), torch.zeros_like(costs)).sum(0) / counts.clamp_min(1)
    return torch.where(means > 1e-8, means, torch.ones_like(means))


def utility(costs, scales, mask=None, weights=WEIGHTS):
    """Return cost U and validity; zero-weight missing metrics cannot create NaN."""
    scales = torch.as_tensor(scales, dtype=costs.dtype, device=costs.device)
    weights = torch.as_tensor(weights, dtype=costs.dtype, device=costs.device)
    if costs.ndim != 2 or scales.shape != (costs.shape[1],) or weights.shape != scales.shape:
        raise ValueError('Utility costs/scales/weights shapes differ')
    if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
        raise ValueError('Utility scales must be finite and positive')
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or not bool((weights > 0).any()):
        raise ValueError('Utility weights must be finite, nonnegative and active')
    valid = torch.isfinite(costs)
    if mask is not None:
        if mask.shape != costs.shape:
            raise ValueError('Utility metric mask shape differs')
        valid &= mask.bool()
    active = weights > 0
    observed = valid[:, active].all(1)
    safe = torch.where(valid, costs, torch.zeros_like(costs))
    result = (safe * weights / scales).sum(1)
    return result.masked_fill(~observed, float('nan')), observed


def scalar_reward(model, actions, uncertainty_beta=0.):
    """Q in identical utility units to the multihead weighted reward.

Deliberately bypass model.reward(), which would divide by mean_TRAIN(U).
This optional ensemble penalty is a heuristic, not calibrated confidence/CQL.
    """
    if uncertainty_beta < 0:
        raise ValueError('Uncertainty penalty must be nonnegative')
    predictions = model(actions).squeeze(-1)
    return -predictions.mean(1) - uncertainty_beta * predictions.std(1, unbiased=False)


def family_folds(families, folds=3):
    groups = np.unique(np.asarray(families))
    if folds < 2 or len(groups) < folds:
        raise ValueError('Grouped CV needs at least one family per held-out fold')
    np.random.default_rng(FOLD_SEED).shuffle(groups)
    result = []
    for fold, held in enumerate(np.array_split(groups, folds)):
        test = np.flatnonzero(np.isin(families, held))
        train = np.flatnonzero(~np.isin(families, held))
        result.append(dict(name=f'cv_{fold}', train_indices=train.tolist(),
                           target_indices=test.tolist(), target_families=held.tolist()))
    return result


def _indices_identity(context, target_data, train_indices, target_indices):
    return dict(train_indices=np.asarray(train_indices).tolist(),
                target_indices=np.asarray(target_indices).tolist(),
                train_source_ids=[context['source_ids'][i] for i in train_indices],
                target_source_ids=[target_data['source_ids'][i] for i in target_indices])


def load_comparison(args, context, name, train_indices, target_indices, target_data, scales):
    """Bind and verify frozen multi_huber artifacts before fitting the control."""
    folder = Path(args.screen_root)/f'seed_{args.seed}'/name
    done_path, pred_path, checkpoint = [folder/n for n in ['completed.json', 'predictions.npz', 'resume.pt']]
    done = json.loads(done_path.read_text())
    old = done['binding']
    if done['variant'] != 'multi_huber' or done['seed'] != args.seed or done['steps'] != args.steps:
        raise ValueError('Comparator variant/seed/training budget differs')
    for key in ['dataset_sha256', 'manifest_sha256', 'cache_sha256']:
        if old[key] != context['binding'][key]:
            raise ValueError('Comparator input binding differs: '+key)
    for key in ['seed', 'steps', 'folds', 'batch_size', 'lr', 'weight_decay']:
        if old['config'][key] != getattr(args, key):
            raise ValueError('Comparator optimizer/sampling configuration differs: '+key)
    if old.get('fold_seed') != FOLD_SEED or old.get('heldout_used') or old.get('backend_calls') != 0:
        raise ValueError('Comparator fold/provenance contract differs')
    for path, digest in old['source_sha256'].items():
        if sha(ROOT/path) != digest or context['binding']['source_sha256'].get(path) != digest:
            raise ValueError('Frozen comparator source changed: '+path)
    if sha(checkpoint) != done['checkpoint_sha256']:
        raise ValueError('Comparator checkpoint changed after completion')
    with np.load(pred_path, allow_pickle=False) as saved:
        if (not np.array_equal(saved['train_indices'], train_indices) or
                not np.array_equal(saved['target_indices'], target_indices)):
            raise ValueError('Comparator family fold or target row identity differs')
        expected = target_data['costs'][target_indices].cpu().numpy()
        if (saved['prediction'].shape != expected.shape or
                not np.array_equal(saved['truth'], expected, equal_nan=True) or
                not np.array_equal(saved['family'], target_data['families'][target_indices].cpu().numpy())):
            raise ValueError('Comparator target costs/families differ')
        predicted = torch.tensor(saved['prediction'].copy(), dtype=target_data['costs'].dtype)
    if not bool(torch.isfinite(predicted).all()):
        raise ValueError('Comparator predictions must be finite')
    if not np.allclose(done['normalization']['objective_scale'], scales.cpu().numpy(), rtol=1e-6, atol=1e-8):
        raise ValueError('Comparator TRAIN utility scales differ')
    values, _ = utility(predicted, scales)
    binding = dict(folder=str(folder.resolve()), completed_sha256=sha(done_path),
                   predictions_sha256=sha(pred_path), checkpoint_sha256=sha(checkpoint))
    return values.cpu().numpy(), binding


def _completed(folder, expected_binding):
    done = json.loads((folder/'completed.json').read_text())
    if done['binding'] != expected_binding:
        raise ValueError('Completed scalar fit binding changed')
    for key, filename in [('checkpoint_sha256', 'resume.pt'), ('predictions_sha256', 'predictions.npz')]:
        if sha(folder/filename) != done[key]:
            raise ValueError('Completed scalar artifact changed: '+filename)
    return done


def fit(args, context, name, train_indices, target_indices, target_data, intercept=None):
    """One scalar fit; intercept is only a test hook before each actual update."""
    train_indices, target_indices = np.asarray(train_indices), np.asarray(target_indices)
    x, costs, mask = [context[k][train_indices] for k in ['x', 'costs', 'mask']]
    tx, tc, tm, tf = [target_data[k][target_indices] for k in ['x', 'costs', 'mask', 'families']]
    scales = fit_utility_scales(costs, mask)
    target, valid = utility(costs, scales, mask)
    truth, target_valid = utility(tc, scales, tm)
    if not bool(valid.all()) or not bool(target_valid.all()):
        raise ValueError('Every matched row must have all active timing utility labels')
    if name.startswith('cv_') and set(context['families'][train_indices].tolist()) & set(tf.tolist()):
        raise ValueError('A placement family crosses TRAIN and CV target')
    comparison, comparison_binding = load_comparison(args, context, name, train_indices, target_indices, target_data, scales)
    identity = _indices_identity(context, target_data, train_indices, target_indices)
    binding = dict(run=context['binding'], fit=name, comparison=comparison_binding, **identity,
                   weights=list(WEIGHTS), utility_scales=scales.cpu().tolist(),
                   q_units='-sum(weight*cost/TRAIN_mean_cost); no second mean(U) normalization')
    folder = args.output/name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'completed.json').exists():
        return _completed(folder, binding)
    torch.manual_seed(args.seed)
    model = MultiMetricCritic(context['cache'], x, target[:, None], context['movable'], ['weighted_utility'],
        train_ids=identity['train_source_ids'], train_splits=['train']*len(x), width=128, depth=2,
        ensemble_size=2, hpwl_residual=False, seed=args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device='cpu').manual_seed(810000+args.seed)
    checkpoint = folder/'resume.pt'
    begin = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['binding'] != binding:
            raise ValueError('Scalar resume input/config/source binding changed')
        if saved['normalization'] != model.normalization_metadata():
            raise ValueError('Scalar resume TRAIN normalization changed')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng'])
        generator.set_state(saved['batch_rng'])
        begin = saved['step']
    if not 0 <= begin <= args.steps:
        raise ValueError('Scalar checkpoint step outside declared budget')
    started = time.monotonic()
    for step in range(begin, args.steps):
        batch = torch.randint(len(x), (min(args.batch_size, len(x)),), generator=generator)
        if intercept is not None:
            intercept(x[batch])
        loss, parts = model.loss(x[batch], target[batch, None], regression='huber')
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite scalar critic loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if (step+1) % 100 == 0 or STOP or step+1 == args.steps:
            atomic_torch(checkpoint, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                rng=torch.get_rng_state(), batch_rng=generator.get_state(), step=step+1,
                binding=binding, normalization=model.normalization_metadata(), spec=model.spec))
            record = dict(step=step+1, loss=float(loss), grad=float(gradient),
                          parts={k: float(v) for k,v in parts.items()}, slurm_job_id=os.environ['SLURM_JOB_ID'])
            with (folder/'learning_curve.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            print(json.dumps(dict(fit=name, **record)), flush=True)
        if STOP:
            raise SystemExit(85)
    model.eval()
    with torch.no_grad():
        predicted = -scalar_reward(model, tx).cpu().numpy()
        train_prediction = -scalar_reward(model, x).cpu().numpy()
    truth = truth.cpu().numpy()
    scalar_metrics = diagnostics(predicted, truth, tf.cpu().numpy())
    scalar_metrics['train_mae'] = float(np.abs(train_prediction-target.cpu().numpy()).mean())
    multi_metrics = diagnostics(comparison, truth, tf.cpu().numpy())
    prediction_path = folder/'predictions.npz'
    temporary = prediction_path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, scalar_utility=predicted, multihead_utility=comparison, truth=truth,
        family=tf.cpu().numpy(), utility_scales=scales.cpu().numpy(), weights=np.asarray(WEIGHTS),
        train_indices=train_indices, target_indices=target_indices,
        train_source_ids=np.asarray(identity['train_source_ids']), target_source_ids=np.asarray(identity['target_source_ids']))
    temporary.replace(prediction_path)
    result = dict(status='completed', binding=binding, seed=args.seed, steps=args.steps,
        primary_target='timing_utility', weights=list(WEIGHTS),
        scope='train_family_cv' if name.startswith('cv_') else 'historical_validation_previously_inspected',
        scalar=scalar_metrics, multihead=multi_metrics,
        deltas_scalar_minus_multihead={k:(scalar_metrics[k]-multi_metrics[k] if
            scalar_metrics.get(k) is not None and multi_metrics.get(k) is not None else None)
            for k in ['mae','spearman','kendall','family_pair_accuracy','top_regret']},
        train_rows=len(x), target_rows=len(tx), normalization=model.normalization_metadata(),
        checkpoint_sha256=sha(checkpoint), predictions_sha256=sha(prediction_path),
        elapsed_this_attempt_s=time.monotonic()-started, slurm_job_id=os.environ['SLURM_JOB_ID'],
        actor_advanced=False, backend_calls=0, heldout_used=False)
    atomic_json(folder/'completed.json', result)
    return result


def summarize(outcomes):
    """Average fold ranks; never pool U ranks across different fold scalings."""
    folds = [r for r in outcomes if r['scope']=='train_family_cv']
    result = dict(cv_aggregation='within_fold_metrics_then_equal_fold_mean', folds=len(folds),
                  warning='Each fold uses its own TRAIN cost scales; no pooled cross-fold utility rank',
                  primary_target='timing_utility', weights=list(WEIGHTS), actor_advanced=False)
    for arm in ['scalar', 'multihead']:
        summary = {}
        for key in ['mae','spearman','kendall','family_pair_accuracy','top_regret']:
            values = [r[arm][key] for r in folds if r[arm].get(key) is not None]
            summary[key] = dict(n=len(values), mean=float(np.mean(values)) if values else None,
                                sd=float(np.std(values,ddof=1)) if len(values)>1 else None)
        count = sum(r[arm]['family_pairs'] for r in folds)
        summary['family_pair_accuracy_pooled_within_fold'] = (
            sum(r[arm]['family_pair_accuracy']*r[arm]['family_pairs'] for r in folds
                if r[arm]['family_pair_accuracy'] is not None)/count if count else None)
        result[arm] = summary
    result['historical_validation'] = next((r for r in outcomes if r['scope']!='train_family_cv'),None)
    result['deltas_scalar_minus_multihead'] = {
        key:(result['scalar'][key]['mean']-result['multihead'][key]['mean'] if
             result['scalar'][key]['mean'] is not None and result['multihead'][key]['mean'] is not None else None)
        for key in ['mae','spearman','kendall','family_pair_accuracy','top_regret']}
    return result


def run(args):
    # Re-entering a completed invocation must not append seed_N twice.
    args = argparse.Namespace(**vars(args))
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run scalar critic fitting through Slurm')
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0 or args.weight_decay < 0 or args.seed < 0:
        raise ValueError('Invalid scalar training budget or optimizer')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',1)))
    torch.use_deterministic_algorithms(True)
    data, manifest = load_dataset(args.dataset)
    cache_path = ROOT/'baseline/runtime/benchmark_cache'/(manifest['design']+'.pt')
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    from baseline.api import load_problem
    movable = ~load_problem(manifest['design'],'cpu').cond.is_ports.bool()
    arrays = {}
    for split in ['train', 'validation']:
        source = data[split]
        costs, mask = measured_costs(torch.tensor(source['metrics'][:,:5],dtype=torch.float32),
                                    METRICS, torch.tensor(source['metric_mask'][:,:5]))
        keep = mask[:,0] & (costs[:,0]>0)
        indices = np.flatnonzero(keep.numpy())
        arrays[split] = dict(x=torch.tensor(source['raw_action'],dtype=torch.float32).reshape(len(costs),-1,2)[keep],
            costs=costs[keep], mask=mask[keep], families=torch.tensor(source['family'],dtype=torch.long)[keep],
            source_ids=[f"{split}_row{i}_family{source['family'][i]}_source{source['source_row'][i]}_variant{source['variant'][i]}"
                        for i in indices])
    plans = family_folds(arrays['train']['families'].numpy(),args.folds)
    plans.append(dict(name='refit', train_indices=list(range(len(arrays['train']['x']))),
                      target_indices=list(range(len(arrays['validation']['x'])))))
    args.output = args.output.resolve()/f'seed_{args.seed}'
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source_hashes = {name:sha(ROOT/name) for name in SOURCE_NAMES}
        binding = dict(schema='qstudy_scalar_timing_control_v1',
            dataset_sha256=sha(args.dataset), manifest_sha256=sha(args.dataset.with_name('manifest.json')),
            cache_sha256=sha(cache_path), source_sha256=source_hashes,
            config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            threads=torch.get_num_threads(), torch_version=torch.__version__, numpy_version=np.__version__,
            primary_target='timing_utility', metric_names=list(METRICS), weights=list(WEIGHTS),
            folds=plans, fold_plan_sha256=json_sha(plans), fold_seed=FOLD_SEED,
            train_source_ids=arrays['train']['source_ids'], validation_source_ids=arrays['validation']['source_ids'],
            train_source_ids_sha256=json_sha(arrays['train']['source_ids']),
            validation_status='historical tuning validation; previously inspected, not a fresh final test',
            backend_calls=0, heldout_used=False, actor_advanced=False)
        old = args.output/'binding.json'
        if old.exists() and json.loads(old.read_text()) != binding:
            raise ValueError('Scalar run input/source/fold plan changed')
        snapshots=args.output/'source_snapshot'
        for name,digest in source_hashes.items():
            destination=snapshots/name
            if destination.exists():
                if sha(destination)!=digest:
                    raise ValueError('Scalar source snapshot changed')
            else:
                destination.parent.mkdir(parents=True,exist_ok=True)
                temporary=destination.with_name(destination.name+'.tmp')
                shutil.copyfile(ROOT/name,temporary)
                temporary.replace(destination)
        atomic_json(old,binding)
        context={**arrays['train'], 'cache':cache,'movable':movable,'binding':binding}
        outcomes=[]
        for plan in plans:
            target_data=arrays['validation'] if plan['name']=='refit' else arrays['train']
            outcomes.append(fit(args,context,plan['name'],plan['train_indices'],plan['target_indices'],target_data))
        report=dict(status='completed',seed=args.seed,binding=binding,outcomes=outcomes,**summarize(outcomes))
        atomic_json(args.output/'summary.json',report)
        atomic_json(args.output/'completed.json',dict(status='completed',binding=binding,
                    summary_sha256=sha(args.output/'summary.json'),actor_advanced=False))
        print(json.dumps(dict(status='completed',seed=args.seed,output=str(args.output))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--screen-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=None)
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--folds',type=int,default=3)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--weight-decay',type=float,default=1e-4)
    args=parser.parse_args()
    if args.seed is None:
        if 'SLURM_ARRAY_TASK_ID' not in os.environ:
            parser.error('--seed or SLURM_ARRAY_TASK_ID is required')
        args.seed=int(os.environ['SLURM_ARRAY_TASK_ID'])
    for sig in [signal.SIGTERM,signal.SIGUSR1]:
        signal.signal(sig,_stop)
    run(args)


if __name__=='__main__':
    main()
