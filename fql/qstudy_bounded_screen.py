"""Prospective, fixed-budget scalar bounded-utility critic screen (no actors/EDA).

Three objectives share physical TRAIN reference scales within each family fold.
The model regresses B directly; Q=-mean(B) has NO second mean(B) normalization.
CV utilities from different folds are never pooled. Validation is historical
development data, and gradient diagnostics do not validate actor directions.
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
from .qstudy_bounded_utility import derive_cost_utility, make_spec
from .qstudy_data import METRIC_NAMES, load_dataset
from .qstudy_models import MultiMetricCritic
from .qstudy_report import diagnostics
from .qstudy_scalar_control import family_folds
from .qstudy_screen import atomic_torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path('/fact_data/bohanli/flowplace_fql/qstudy_v1/data_audit/bp_fe384/dataset.npz')
OBJECTIVES = dict(wl=(1., 0., 0., 0., 0.), timing=(1., .15, .35, 0., 0.),
                  congestion=(1., .15, .35, .1, .1))
CONFIG = dict(steps=2000, folds=3, fold_seed=41307, batch_size=64, lr=3e-4,
              weight_decay=1e-4, width=128, depth=2, ensemble_size=2,
              minibatch_seed_offset=810000, regression='huber', beta=0.,
              hpwl_residual=False, target='bounded_cost', feature_input='raw_coordinates_plus_17')
SOURCE_NAMES = ('fql/qstudy_bounded_screen.py', 'fql/qstudy_bounded_screen.slurm',
    'fql/qstudy_bounded_utility.py', 'fql/qstudy_models.py', 'fql/qstudy_data.py',
    'fql/qstudy_report.py', 'fql/qstudy_scalar_control.py', 'fql/qstudy_screen.py',
    'fql/graph_o2o_models.py', 'fql/physical_ridge.py', 'fql/analytic.py',
    'fql/graph_o2o_archive.py', 'baseline/api.py',
    'upstream/flowplace/benchmark_utils/openroad_utils.py')
STOP = False


def _stop(*_):
    global STOP
    STOP = True


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def arrays(source, split):
    """Retain every original row, native unit, missing label and logical status."""
    n = len(source['family'])
    return dict(x=torch.as_tensor(source['raw_action'].copy(), dtype=torch.float32).reshape(n, -1, 2),
        executed=np.asarray(source['executed_action']).copy(), native=np.asarray(source['metrics']).copy(),
        mask=np.asarray(source['metric_mask'], dtype=bool).copy(),
        statuses=np.asarray(source['status'], dtype=str).copy(),
        families=np.asarray(source['family']).copy(),
        source_ids=[f"{split}_row{i}_family{source['family'][i]}_source{source['source_row'][i]}_variant{source['variant'][i]}"
                    for i in range(n)])


def fit_scales(native, statuses, mask):
    """Masked success-only fold-TRAIN mean native costs; failures have no PPA.

These physical scales are shared by all objectives. Missing/all-zero columns
use 1 as an explicit fallback; absent active metrics still invalidate labels.
    """
    values = torch.as_tensor(native[:, :5], dtype=torch.float64).clone()
    observed = torch.as_tensor(mask[:, :5], dtype=torch.bool) & torch.isfinite(values)
    observed &= torch.as_tensor(np.asarray(statuses) == 'success')[:, None]
    values[:, 1:3] = (-values[:, 1:3]).clamp_min(0)
    if bool(((values < 0) & observed).any()):
        raise ValueError('Negative observed physical TRAIN cost')
    counts = observed.sum(0)
    means = torch.where(observed, values, torch.zeros_like(values)).sum(0) / counts.clamp_min(1)
    scales = torch.where(means > 1e-8, means, torch.ones_like(means))
    return scales.tolist(), dict(observed_counts=counts.tolist(),
        fallback_to_one=(means <= 1e-8).tolist(), policy='success-only masked TRAIN mean cost; zero/absent mean -> 1')


def specification(context, train_indices, objective):
    ids = [context['source_ids'][i] for i in train_indices]
    scales, scale_audit = fit_scales(context['native'][train_indices],
        context['statuses'][train_indices], context['mask'][train_indices])
    reference = dict(split='train', dataset_sha256=context['binding']['dataset_sha256'],
                     train_rows_sha256=json_sha(ids))
    return make_spec(OBJECTIVES[objective], scales, reference), scale_audit


def labels(data, indices, spec):
    return derive_cost_utility(torch.as_tensor(data['native'][indices], dtype=torch.float64),
        data['statuses'][indices].tolist(), spec,
        torch.as_tensor(data['mask'][indices], dtype=torch.bool))


def scalar_reward(model, actions):
    """Terminal Q in fixed bounded utility units; beta=0, no Bellman bootstrap."""
    return -model(actions).mean((1, 2))


def input_gradient_audit(model, actions, movable):
    if not len(actions):
        return dict(rows=0, finite=True, nonzero_rows=0, rms_mean=None, rms_maximum=None)
    probe = actions.detach().clone().requires_grad_(True)
    gradient, = torch.autograd.grad(scalar_reward(model, probe).sum(), probe)
    rms = gradient[:, movable].square().mean((1, 2)).sqrt()
    finite = bool(torch.isfinite(gradient).all())
    if not finite:
        raise ValueError('Nonfinite critic input gradients')
    return dict(rows=len(probe), finite=finite, nonzero_rows=int((rms > 0).sum()),
                rms_mean=float(rms.mean()), rms_maximum=float(rms.max()),
                interpretation='raw-input differentiability only; not legalizer/backend gradient accuracy')


def _completed(folder, binding):
    done = json.loads((folder/'completed.json').read_text())
    if done['binding'] != binding:
        raise ValueError('Completed bounded fit binding changed')
    for key, filename in [('checkpoint_sha256', 'resume.pt'), ('predictions_sha256', 'predictions.npz')]:
        if sha(folder/filename) != done[key]:
            raise ValueError('Completed bounded artifact changed: '+filename)
    return done


def fit(args, context, name, train_indices, target_indices, target_data, intercept=None):
    """One bounded scalar fit. Test hook intercept runs before a real update."""
    train_indices, target_indices = np.asarray(train_indices, dtype=int), np.asarray(target_indices, dtype=int)
    if len(set(train_indices.tolist())) != len(train_indices) or len(set(target_indices.tolist())) != len(target_indices):
        raise ValueError('Repeated logical fit indices')
    if name.startswith('cv_') and (set(context['families'][train_indices]) & set(target_data['families'][target_indices])):
        raise ValueError('A placement family crosses TRAIN and CV target')
    spec, scale_audit = specification(context, train_indices, args.objective)
    train_labels, target_labels = labels(context, train_indices, spec), labels(target_data, target_indices, spec)
    valid_train = train_labels['valid'].cpu().numpy()
    admitted = train_indices[valid_train]
    if len(admitted) < 2:
        raise ValueError('Need at least two valid TRAIN utility labels')
    x = context['x'][admitted]
    target = train_labels['cost'][train_labels['valid']].to(dtype=x.dtype)
    tx = target_data['x'][target_indices]
    identity = dict(train_indices=train_indices.tolist(), admitted_train_indices=admitted.tolist(),
        target_indices=target_indices.tolist(), train_source_ids=[context['source_ids'][i] for i in train_indices],
        target_source_ids=[target_data['source_ids'][i] for i in target_indices])
    binding = dict(run=context['binding'], fit=name, objective=args.objective, utility_spec=spec,
        utility_scale_audit=scale_audit, config={k:getattr(args,k) for k in ['seed','steps','batch_size','lr','weight_decay']},
        **identity, q_units='-mean predicted B; no mean(B) renormalization', beta=0.)
    folder = args.output/name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'completed.json').exists():
        return _completed(folder, binding)
    torch.manual_seed(args.seed)
    model = MultiMetricCritic(context['cache'], x, target[:, None], context['movable'], ['bounded_cost'],
        train_ids=[context['source_ids'][i] for i in admitted], train_splits=['train']*len(x),
        width=128, depth=2, ensemble_size=2, hpwl_residual=False, seed=args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device='cpu').manual_seed(810000+args.seed)
    checkpoint, begin = folder/'resume.pt', 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['binding'] != binding or saved['utility_spec'] != spec:
            raise ValueError('Bounded resume input/config/source/spec binding changed')
        if saved['normalization'] != model.normalization_metadata():
            raise ValueError('Bounded resume TRAIN normalization changed')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng'])
        generator.set_state(saved['batch_rng'])
        begin = saved['step']
    if not 0 <= begin <= args.steps:
        raise ValueError('Checkpoint step outside declared budget')
    started = time.monotonic()
    for step in range(begin, args.steps):
        batch = torch.randint(len(x), (min(args.batch_size, len(x)),), generator=generator)
        if intercept is not None:
            intercept(x[batch])
        loss, parts = model.loss(x[batch], target[batch, None], regression='huber')
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite bounded critic loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if (step+1) % 100 == 0 or STOP or step+1 == args.steps:
            atomic_torch(checkpoint, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                rng=torch.get_rng_state(), batch_rng=generator.get_state(), step=step+1,
                binding=binding, utility_spec=spec, normalization=model.normalization_metadata(), spec=model.spec,
                actor_contract=dict(q_units=binding['q_units'], beta=0., native_reward_unchanged=True)))
            record = dict(step=step+1, loss=float(loss), grad=float(gradient),
                parts={k:float(v) for k,v in parts.items()}, slurm_job_id=os.environ.get('SLURM_JOB_ID'))
            with (folder/'learning_curve.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            print(json.dumps(dict(fit=name, objective=args.objective, **record)), flush=True)
        if STOP:
            raise SystemExit(85)
    model.eval()
    with torch.no_grad():
        prediction = -scalar_reward(model, tx).cpu().numpy()
        train_prediction = -scalar_reward(model, x).cpu().numpy()
        train_prediction_all = -scalar_reward(model, context['x'][train_indices]).cpu().numpy()
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite bounded critic predictions')
    truth = target_labels['cost'].cpu().numpy()
    valid = target_labels['valid'].cpu().numpy()
    constant = float(target.mean())
    families = target_data['families'][target_indices]
    target_success = target_labels['success'].cpu().numpy() & valid
    result = dict(status='completed', binding=binding, seed=args.seed, steps=args.steps,
        scope='train_family_cv' if name.startswith('cv_') else 'historical_validation_previously_inspected',
        critic=diagnostics(prediction, truth, families),
        constant_train_B=constant, constant=diagnostics(np.full(len(tx), constant), truth, families),
        train_mae=float(np.abs(train_prediction-target.cpu().numpy()).mean()),
        train_rows=len(train_indices), admitted_train_rows=len(admitted), excluded_train_rows=int((~valid_train).sum()),
        target_rows=len(tx), valid_target_rows=int(valid.sum()), invalid_target_rows=int((~valid).sum()),
        target_status_counts={s:int((target_data['statuses'][target_indices]==s).sum())
                              for s in np.unique(target_data['statuses'][target_indices])},
        prediction_range=dict(minimum=float(prediction.min()) if len(tx) else None,
            maximum=float(prediction.max()) if len(tx) else None,
            success_below_zero=int(((prediction < 0)&target_success).sum()),
            success_at_or_above_two=int(((prediction >= 2)&target_success).sum()),
            any_below_zero=int((prediction < 0).sum()), any_above_four=int((prediction > 4).sum()),
            warning='Bounded labels do not constrain model outputs or guarantee physical success'),
        input_gradients=input_gradient_audit(model,tx,context['movable']),
        normalization=model.normalization_metadata(), utility_spec=spec, utility_scale_audit=scale_audit,
        elapsed_this_attempt_s=time.monotonic()-started, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        actor_advanced=False, backend_calls=0, heldout_used=False)
    prediction_path = folder/'predictions.npz'
    temporary = prediction_path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, prediction=prediction, truth=truth, utility_truth=-truth, valid=valid,
        native_metrics=target_data['native'][target_indices], native_metric_mask=target_data['mask'][target_indices],
        status=target_data['statuses'][target_indices], family=families,
        raw_action=tx.cpu().numpy(), executed_action=target_data['executed'][target_indices],
        train_prediction=train_prediction, train_truth=target.cpu().numpy(),
        train_prediction_all=train_prediction_all, train_truth_all=train_labels['cost'].cpu().numpy(),
        train_native_metrics=context['native'][train_indices], train_native_metric_mask=context['mask'][train_indices],
        train_status=context['statuses'][train_indices], train_valid=valid_train,
        constant_prediction=np.full(len(tx),constant), train_indices=train_indices,
        admitted_train_indices=admitted, target_indices=target_indices,
        train_source_ids=np.asarray(identity['train_source_ids']), target_source_ids=np.asarray(identity['target_source_ids']))
    temporary.replace(prediction_path)
    result.update(checkpoint_sha256=sha(checkpoint), predictions_sha256=sha(prediction_path))
    atomic_json(folder/'completed.json',result)
    return result


def summarize(outcomes):
    """CV-fold means are descriptive, not independent-seed uncertainty."""
    folds = [r for r in outcomes if r['scope']=='train_family_cv']
    result = dict(cv_folds=len(folds), cv_aggregation='equal_fold_mean_of_within_fold_metrics',
        warning='Do not pool bounded labels/predictions across fold-specific physical scales; fold SD is not seed uncertainty',
        backend_calls=0, actor_advanced=False)
    for arm in ['critic','constant']:
        result[arm] = {key:dict(n=len(values), mean=float(np.mean(values)) if values else None)
            for key in ['mae','spearman','kendall','family_pair_accuracy','top_regret']
            for values in [[r[arm][key] for r in folds if r[arm].get(key) is not None]]}
        count = sum(r[arm]['family_pairs'] for r in folds)
        result[arm]['within_fold_pair_accuracy'] = (sum(r[arm]['family_pair_accuracy']*r[arm]['family_pairs']
            for r in folds if r[arm]['family_pair_accuracy'] is not None)/count if count else None)
        result[arm]['within_fold_pairs'] = count
    result['historical_validation'] = next((r for r in outcomes if r['scope']!='train_family_cv'),None)
    return result


def load_inputs(dataset):
    data, manifest = load_dataset(dataset)
    cache_path = ROOT/'baseline/runtime/benchmark_cache'/(manifest['design']+'.pt')
    input_binding = dict(dataset_path=str(dataset.resolve()), dataset_sha256=sha(dataset),
        manifest_sha256=sha(dataset.with_name('manifest.json')), rows_sha256=sha(dataset.with_name('rows.jsonl')),
        cache_path=str(cache_path), cache_sha256=sha(cache_path))
    return {split:arrays(data[split],split) for split in ['train','validation']}, manifest, input_binding


def prepare(args):
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Prepare numerical/data plan through Slurm')
    data, manifest, inputs = load_inputs(args.dataset.resolve())
    if manifest['design']!='bp_fe' or len(data['train']['x'])!=384 or len(data['validation']['x'])!=16:
        raise ValueError('This prospective stage is fixed to audited bp_fe384 TRAIN /16 historical validation')
    if not (data['train']['statuses']=='success').all() or not np.isfinite(data['train']['native']).all():
        raise ValueError('Initial declared 384 TRAIN rows must remain finite native successes')
    if set(data['train']['families']) & set(data['validation']['families']):
        raise ValueError('TRAIN/historical validation family collision')
    folds = family_folds(data['train']['families'],3)
    folds.append(dict(name='refit',train_indices=list(range(384)),target_indices=list(range(16))))
    tasks = [dict(index=i,objective=objective,seed=seed) for i,(objective,seed) in
             enumerate((objective,seed) for objective in OBJECTIVES for seed in range(3))]
    source = {name:sha(ROOT/name) for name in SOURCE_NAMES}
    plan = dict(schema='qstudy_bounded_scalar_screen_v1', output=str(args.output.resolve()),
        design='bp_fe',inputs=inputs,source_sha256=source,config=CONFIG,objectives={k:list(v) for k,v in OBJECTIVES.items()},
        tasks=tasks,folds=folds,fold_plan_sha256=json_sha(folds),metric_names=list(METRIC_NAMES),
        train_source_ids=data['train']['source_ids'],validation_source_ids=data['validation']['source_ids'],
        train_source_ids_sha256=json_sha(data['train']['source_ids']),
        scope='prospective development critic screen; historical validation was previously inspected',
        q_units='-mean predicted B; no mean(B) renormalization', native_reward_unchanged=True,
        backend_calls=0, actor_advanced=False, heldout_used=False)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'plan.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=args.output/'plan.json'
        if path.exists():
            if json.loads(path.read_text())!=plan:
                raise ValueError('Immutable bounded screen plan already differs')
        else:
            atomic_json(path,plan)
        for name,digest in source.items():
            dest=args.output/'source_snapshot'/name
            if dest.exists():
                if sha(dest)!=digest:
                    raise ValueError('Immutable source snapshot differs: '+name)
            else:
                dest.parent.mkdir(parents=True,exist_ok=True)
                temporary=dest.with_name(dest.name+'.tmp')
                shutil.copyfile(ROOT/name,temporary)
                temporary.replace(dest)
    print(json.dumps(dict(plan=str(path),plan_sha256=sha(path),tasks=len(tasks))),flush=True)


def run(args):
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run bounded critic training through Slurm')
    plan = json.loads(args.plan.read_text())
    if plan['config']!=CONFIG or plan['objectives']!={k:list(v) for k,v in OBJECTIVES.items()}:
        raise ValueError('Bounded stage declared configuration differs')
    expected_tasks=[dict(index=i,objective=o,seed=s) for i,(o,s) in
                    enumerate((o,s) for o in OBJECTIVES for s in range(3))]
    if plan['tasks']!=expected_tasks or not 0<=args.index<9:
        raise ValueError('Bounded stage must contain the exact nine declared tasks')
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest or sha(Path(plan['output'])/'source_snapshot'/name)!=digest:
            raise ValueError('Bounded stage source changed: '+name)
    data,manifest,inputs = load_inputs(Path(plan['inputs']['dataset_path']))
    if inputs!=plan['inputs'] or manifest['design']!=plan['design']:
        raise ValueError('Bounded input/cache binding changed')
    expected_folds=family_folds(data['train']['families'],3)
    expected_folds.append(dict(name='refit',train_indices=list(range(len(data['train']['x']))),
                               target_indices=list(range(len(data['validation']['x'])))))
    if (plan['folds']!=expected_folds or json_sha(expected_folds)!=plan['fold_plan_sha256'] or
        plan['train_source_ids']!=data['train']['source_ids'] or plan['validation_source_ids']!=data['validation']['source_ids']):
        raise ValueError('Bounded split/source identity changed')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',1)))
    torch.use_deterministic_algorithms(True)
    from baseline.api import load_problem
    cache=torch.load(inputs['cache_path'],map_location='cpu',weights_only=False)
    movable=~load_problem(plan['design'],'cpu').cond.is_ports.bool()
    task=plan['tasks'][args.index]
    fit_args=argparse.Namespace(**CONFIG,**task,output=Path(plan['output'])/task['objective']/f"seed_{task['seed']}")
    fit_args.output.mkdir(parents=True,exist_ok=True)
    binding=dict(schema=plan['schema'],**inputs,plan_sha256=sha(args.plan),task=task,
        source_sha256=plan['source_sha256'],fold_plan_sha256=plan['fold_plan_sha256'],
        threads=torch.get_num_threads(),torch_version=torch.__version__,numpy_version=np.__version__)
    context={**data['train'],'cache':cache,'movable':movable,'binding':binding}
    with (fit_args.output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        outcomes=[]
        for fold in plan['folds']:
            target_data=data['validation'] if fold['name']=='refit' else data['train']
            outcomes.append(fit(fit_args,context,fold['name'],fold['train_indices'],fold['target_indices'],target_data))
        report=dict(status='completed',binding=binding,objective=task['objective'],seed=task['seed'],
                    outcomes=outcomes,**summarize(outcomes))
        atomic_json(fit_args.output/'summary.json',report)
        atomic_json(fit_args.output/'completed.json',dict(status='completed',binding=binding,
            summary_sha256=sha(fit_args.output/'summary.json'),backend_calls=0,actor_advanced=False))
    print(json.dumps(dict(status='completed',task=task,output=str(fit_args.output))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare')
    p.add_argument('--dataset',type=Path,default=DEFAULT_DATA)
    p.add_argument('--output',type=Path,required=True)
    p=commands.add_parser('run')
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--index',type=int,default=None)
    args=parser.parse_args()
    for sig in [signal.SIGTERM,signal.SIGUSR1]:
        signal.signal(sig,_stop)
    if args.command=='run' and args.index is None:
        if 'SLURM_ARRAY_TASK_ID' not in os.environ:
            parser.error('--index or SLURM_ARRAY_TASK_ID is required')
        args.index=int(os.environ['SLURM_ARRAY_TASK_ID'])
    (prepare if args.command=='prepare' else run)(args)


if __name__=='__main__':
    main()
