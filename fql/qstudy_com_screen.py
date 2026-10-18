"""One prospective local COM-inspired bounded-WL critic contrast, CPU only.

The frozen supervised fit bytecode is reused in an isolated namespace. Only its
critic class and local STOP flag differ. No old globals, Bellman targets, actor,
legalizer, native rewards or physical measurements are changed. This bounded,
fixed-coefficient Huber adaptation is not an official COMs reproduction and has
no certified lower-bound guarantee.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
from types import FunctionType

import numpy as np
import torch

from . import qstudy_bounded_screen as supervised
from . import qstudy_bounded_screen_report as control_report
from .graph_o2o_archive import atomic_json, sha
from .qstudy_models import MultiMetricCritic

ROOT = supervised.ROOT
DEFAULT_CONTROL = Path('/fact_data/bohanli/flowplace_fql/qstudy_v1/bounded_screen_v1/plan.json')
LOSS = dict(version='local_com_bounded_wl_v1', coefficient=.1, ascent_steps=3,
    ascent_rms_step=.01, radius_rms=.02, box=[-1., 1.],
    candidate_score='negative_ensemble_mean_TRAIN_standardized_B',
    eligibility='all_movable_raw_centers_inside_actor_box; full_supervised_batch_retained',
    candidate_selection='highest_Q_of_anchor_and_three_visited; first_tie',
    candidate_gradient='detached_every_step_and_after_selection; no_higher_order',
    outer_penalty='mean(z_B(data)-z_B(detached_adversary)); both_terms_differentiated',
    entropy=0., dual_coefficient_update=False, native_Q='-ensemble_mean_predicted_B')
CONFIG = dict(supervised.CONFIG, loss=LOSS)
SOURCE_NAMES = tuple(dict.fromkeys((*supervised.SOURCE_NAMES,
    'fql/qstudy_bounded_screen_report.py', 'fql/qstudy_com_screen.py', 'fql/qstudy_com_screen.slurm')))
OFFICIAL = dict(paper='https://proceedings.mlr.press/v139/trabucco21a.html',
    code='https://github.com/brandontrabucco/design-baselines/blob/master/design_baselines/coms_cleaned/trainers.py',
    distinction='official MSE/dual alpha/unbounded particle ascent; this fixed local Huber contrast differs')
STOP = False
_ACTIVE_NAMESPACES = []


def _stop(*_):
    global STOP
    STOP = True
    for namespace in _ACTIVE_NAMESPACES:
        namespace['STOP'] = True


def eligible_centers(actions, movable):
    if actions.ndim != 3 or actions.shape[-1] != 2 or not torch.isfinite(actions).all():
        raise ValueError('COM anchors must be finite [batch,nodes,2] raw coordinates')
    if movable.shape != (actions.shape[1],) or movable.dtype != torch.bool or not bool(movable.any()):
        raise ValueError('COM requires a nonempty Boolean movable mask in cache order')
    return (actions[:, movable].abs() <= 1).all((1, 2))


def adversaries(model, actions, movable, config=LOSS):
    """Feasible box/radius truncation, not a claimed exact joint projection.

Only already-in-box anchors are eligible. Scaling a feasible displacement back
towards such an anchor preserves box feasibility. Out-of-box raw anchors are
never silently clamped; their supervised observations remain admitted.
"""
    anchors = actions.detach()
    eligible = eligible_centers(anchors, movable)
    best_all = anchors.clone()
    audit = dict(batch_rows=len(anchors), eligible_rows=int(eligible.sum()),
        skipped_outside_center_box=int((~eligible).sum()), radius_rms_max=0.,
        fixed_max_abs_change=0., box_excess_max=0., visited_box_clip_rows=0,
        selected_step_mean=0., inner_score_gain_mean=0., inner_gradient_finite=True)
    if not bool(eligible.any()):
        return best_all, eligible, audit
    anchor = anchors[eligible]
    point = anchor.clone()
    best = point.clone()
    with torch.no_grad():
        best_score = -model.standardized(point).mean((1, 2))
    if not torch.isfinite(best_score).all():
        raise ValueError('Nonfinite anchor COM score')
    initial_score = best_score.clone()
    chosen = torch.zeros(len(point), dtype=torch.long, device=point.device)
    for step in range(config['ascent_steps']):
        leaf = point.detach().requires_grad_(True)
        score = -model.standardized(leaf).mean((1, 2))
        gradient = (torch.autograd.grad(score.sum(), leaf, create_graph=False,
            allow_unused=True)[0] if score.requires_grad else None)
        gradient = torch.zeros_like(leaf) if gradient is None else gradient
        if not torch.isfinite(gradient).all():
            raise ValueError('Nonfinite inner COM input gradient')
        gradient = torch.where(movable[None, :, None], gradient, 0.)
        norm = gradient[:, movable].square().mean((1, 2)).sqrt()
        direction = gradient / norm.clamp_min(1e-12)[:, None, None]
        proposal = point.detach() + config['ascent_rms_step'] * direction
        audit['visited_box_clip_rows'] += int((proposal[:, movable].abs() > 1).any((1, 2)).sum())
        delta = torch.where(movable[None, :, None], proposal.clamp(-1, 1)-anchor, 0.)
        radius = delta[:, movable].square().mean((1, 2)).sqrt()
        scale = (config['radius_rms']/radius.clamp_min(1e-12)).clamp(max=1.)
        point = (anchor + delta*scale[:, None, None]).detach()
        with torch.no_grad():
            score = -model.standardized(point).mean((1, 2))
            if not torch.isfinite(score).all():
                raise ValueError('Nonfinite visited COM score')
            better = score > best_score
            best = torch.where(better[:, None, None], point, best)
            best_score = torch.where(better, score, best_score)
            chosen = torch.where(better, chosen.new_full(chosen.shape, step+1), chosen)
    best_all[eligible] = best.detach()
    rms = (best-anchor)[:, movable].square().mean((1, 2)).sqrt()
    fixed_change = ((best-anchor)[:, ~movable].abs().max()
                    if bool((~movable).any()) else best.new_zeros(()))
    excess = (best[:, movable].abs()-1).clamp_min(0).max()
    audit.update(radius_rms_max=float(rms.max()), fixed_max_abs_change=float(fixed_change),
        box_excess_max=float(excess), selected_step_mean=float(chosen.float().mean()),
        inner_score_gain_mean=float((best_score-initial_score).mean()))
    if (not torch.isfinite(best_all).all() or float(rms.max()) > config['radius_rms']+1e-6
            or float(fixed_change) != 0 or float(excess) > 1e-7):
        raise ValueError('COM candidate violated finite/radius/fixed-node/box bounds')
    return best_all.detach(), eligible, audit


def _gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return sum((g.detach().square().sum() for g in gradients if g is not None), loss.new_zeros(())).sqrt()


def critic_class(config):
    """A per-fit subclass, with no mutations to the frozen critic or fit globals."""
    config = copy.deepcopy(config)

    class ConservativeCritic(MultiMetricCritic):
        def loss(self, actions, costs, **kwargs):
            reg, parts = super().loss(actions, costs, **kwargs)
            # Exact beta-zero numerical path: no candidate computation/RNG or
            # extra autograd. Used only by software equivalence tests.
            if config['coefficient'] == 0:
                return reg, parts
            candidates, eligible, audit = adversaries(self, actions, self.movable, config)
            if bool(eligible.any()):
                penalty = (self.standardized(actions[eligible])-
                           self.standardized(candidates[eligible])).mean()
            else:
                penalty = reg*0
            weighted = config['coefficient']*penalty
            parameters = tuple(p for p in self.parameters() if p.requires_grad)
            reg_norm = _gradient_norm(reg, parameters)
            penalty_norm = _gradient_norm(weighted, parameters)
            if not torch.isfinite(penalty) or not torch.isfinite(reg_norm+penalty_norm):
                raise ValueError('Nonfinite COM outer loss/parameter gradient')
            details = dict(parts, com_penalty=penalty.detach(), com_weighted_penalty=weighted.detach(),
                com_regression_parameter_grad_norm=reg_norm,
                com_weighted_parameter_grad_norm=penalty_norm,
                com_gradient_ratio=penalty_norm/reg_norm.clamp_min(1e-12), **{'com_'+k:v for k,v in audit.items()})
            return reg+weighted, details

    return ConservativeCritic


def fit(args, context, name, train_indices, target_indices, target_data, intercept=None, *, loss_config=LOSS):
    config = copy.deepcopy(loss_config)
    if args.objective != 'wl' or config.keys() != LOSS.keys():
        raise ValueError('This contrast supports only the declared bounded WL loss')
    # Test beta=0 is the only alternative; production plans require LOSS exactly.
    if config != LOSS and config != dict(LOSS, coefficient=0.):
        raise ValueError('Undeclared COM hyperparameter change')
    context = dict(context, binding=dict(context['binding'], com_loss=config, com_official_reference=OFFICIAL))
    namespace = dict(supervised.fit.__globals__, MultiMetricCritic=critic_class(config), STOP=STOP)
    private = FunctionType(supervised.fit.__code__, namespace, supervised.fit.__name__,
                           supervised.fit.__defaults__, supervised.fit.__closure__)
    _ACTIVE_NAMESPACES.append(namespace)
    try:
        result = private(args, context, name, train_indices, target_indices, target_data, intercept=intercept)
    finally:
        _ACTIVE_NAMESPACES[:] = [ns for ns in _ACTIVE_NAMESPACES if ns is not namespace]
    admitted = result['binding']['admitted_train_indices']
    eligible = eligible_centers(context['x'][admitted], context['movable'])
    audit = dict(loss=config, official_reference=OFFICIAL,
        supervised_admitted_rows=len(admitted), eligible_COM_rows=int(eligible.sum()),
        skipped_COM_outside_center_box=int((~eligible).sum()),
        label_or_family_based_eligibility=False, missing_infrastructure_labels_not_synthesized=True,
        whole_layout_terminal_bandit=True, Bellman_bootstrap=False,
        native_B_units_unchanged=True, learned_critic_no_certified_lower_bound=True,
        gradient_pressure='Measured from both outer terms; logged every100 updates/final by frozen fit',
        no_actor_or_backend_direction_accuracy_claim=True)
    if 'com_audit' in result and result['com_audit'] != audit:
        raise ValueError('Completed COM loss/coverage audit changed')
    result = dict(result, com_audit=audit)
    atomic_json(args.output/name/'completed.json', result)
    return result


def _folds(data):
    folds = supervised.family_folds(data['train']['families'], 3)
    return folds+[dict(name='refit', train_indices=list(range(len(data['train']['x']))),
                       target_indices=list(range(len(data['validation']['x']))))]


def checked_control(path, inputs, folds):
    """Bind all 12 already-completed WL controls; never fit or infer a model."""
    path = Path(path).resolve()
    plan, data = control_report.checked_plan(path)
    if plan['inputs'] != inputs or plan['folds'] != folds:
        raise ValueError('Supervised control inputs/folds differ from COM')
    artifacts = []
    for task in (t for t in plan['tasks'] if t['objective'] == 'wl'):
        for fold in folds:
            folder = Path(plan['output'])/'wl'/f"seed_{task['seed']}"/fold['name']
            done = json.loads((folder/'completed.json').read_text())
            control_report.verify_fit(folder, done, task, fold, plan, sha(path), data)
            artifacts.append(dict(seed=task['seed'], fit=fold['name'],
                files={str(folder/name):sha(folder/name)
                    for name in ('completed.json', 'resume.pt', 'predictions.npz')}))
    if len(artifacts) != 12:
        raise ValueError('Need all 12 immutable completed supervised WL fits')
    return dict(plan_path=str(path), plan_sha256=sha(path), artifacts=artifacts,
        control_training_config=plan['config'], prediction_or_optimization_calls=0)


def prepare(args):
    data, manifest, inputs = supervised.load_inputs(args.dataset.resolve())
    train, val = data['train'], data['validation']
    if (manifest['design'] != 'bp_fe' or len(train['x']) != 384 or len(val['x']) != 16
            or not (train['statuses'] == 'success').all() or not np.isfinite(train['native']).all()
            or set(train['families']) & set(val['families'])):
        raise ValueError('COM is fixed to finite native bp_fe384 TRAIN /16 historical development')
    folds = _folds(data)
    control = checked_control(args.control_plan, inputs, folds)
    sources = {name:sha(ROOT/name) for name in SOURCE_NAMES}
    plan = dict(schema='qstudy_local_com_bounded_wl_v1', output=str(args.output.resolve()),
        design='bp_fe', inputs=inputs, config=CONFIG, source_sha256=sources,
        objectives=dict(wl=list(supervised.OBJECTIVES['wl'])),
        tasks=[dict(index=s, objective='wl', seed=s) for s in range(3)],
        folds=folds, fold_plan_sha256=supervised.json_sha(folds),
        train_source_ids=train['source_ids'], validation_source_ids=val['source_ids'],
        maximum_new_fits=12, supervised_control=control,
        control_note='beta-zero equivalence established by tiny software fit; no2000-step control rerun',
        historical_validation_previously_inspected=True, prospective_new_native_calls=0,
        actor_advanced=False, selected_winner=None, official_reference=OFFICIAL)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'plan.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        path = args.output/'plan.json'
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError('Immutable COM plan changed')
        atomic_json(path, plan)
        for name, digest in sources.items():
            dest = args.output/'source_snapshot'/name
            if dest.exists():
                if sha(dest) != digest:
                    raise ValueError('Immutable COM source snapshot changed')
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/name, dest)
    print(json.dumps(dict(plan=str(path), sha256=sha(path), new_fits=12)), flush=True)


def run(args):
    if sha(args.plan) != args.plan_sha256:
        raise ValueError('COM plan SHA changed')
    plan = json.loads(args.plan.read_text())
    if (plan['schema'] != 'qstudy_local_com_bounded_wl_v1' or plan['config'] != CONFIG
            or plan['tasks'] != [dict(index=s, objective='wl', seed=s) for s in range(3)]
            or plan['maximum_new_fits'] != 12 or not 0 <= args.index < 3
            or set(plan['source_sha256']) != set(SOURCE_NAMES)):
        raise ValueError('Undeclared COM stage/configuration/budget')
    for name, digest in plan['source_sha256'].items():
        if sha(ROOT/name) != digest or sha(Path(plan['output'])/'source_snapshot'/name) != digest:
            raise ValueError('COM source changed: '+name)
    data, manifest, inputs = supervised.load_inputs(Path(plan['inputs']['dataset_path']))
    if (inputs != plan['inputs'] or manifest['design'] != plan['design'] or _folds(data) != plan['folds']
            or supervised.json_sha(plan['folds']) != plan['fold_plan_sha256']
            or plan['train_source_ids'] != data['train']['source_ids']
            or plan['validation_source_ids'] != data['validation']['source_ids']):
        raise ValueError('COM input/family/source identity changed')
    if checked_control(plan['supervised_control']['plan_path'], inputs, plan['folds']) != plan['supervised_control']:
        raise ValueError('Immutable supervised control binding changed')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    torch.use_deterministic_algorithms(True)
    from baseline.api import load_problem
    cache = torch.load(inputs['cache_path'], map_location='cpu', weights_only=False)
    movable = ~load_problem(plan['design'], 'cpu').cond.is_ports.bool()
    task = plan['tasks'][args.index]
    fit_args = argparse.Namespace(**supervised.CONFIG, **task,
        output=Path(plan['output'])/'wl'/f"seed_{task['seed']}")
    fit_args.output.mkdir(parents=True, exist_ok=True)
    binding = dict(schema=plan['schema'], **inputs, plan_sha256=sha(args.plan), task=task,
        source_sha256=plan['source_sha256'], fold_plan_sha256=plan['fold_plan_sha256'],
        supervised_control=plan['supervised_control'],
        threads=torch.get_num_threads(), torch_version=str(torch.__version__), numpy_version=np.__version__)
    context = dict(data['train'], cache=cache, movable=movable, binding=binding)
    with (fit_args.output/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        outcomes = [fit(fit_args, context, p['name'], p['train_indices'], p['target_indices'],
            data['validation'] if p['name'] == 'refit' else data['train']) for p in plan['folds']]
        result = dict(status='completed', binding=binding, seed=task['seed'], objective='wl',
            outcomes=outcomes, **supervised.summarize(outcomes))
        atomic_json(fit_args.output/'summary.json', result)
        atomic_json(fit_args.output/'completed.json', dict(status='completed', binding=binding,
            summary_sha256=sha(fit_args.output/'summary.json'), backend_calls=0, actor_advanced=False))
    print(json.dumps(dict(status='completed', output=str(fit_args.output))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--dataset', type=Path, default=supervised.DEFAULT_DATA)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--control-plan', type=Path, default=DEFAULT_CONTROL)
    p = commands.add_parser('run')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--index', type=int)
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('COM preparation/fits/tests require bounded CPU Slurm')
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, _stop)
    if args.command == 'run' and args.index is None:
        if 'SLURM_ARRAY_TASK_ID' not in os.environ:
            parser.error('--index or SLURM_ARRAY_TASK_ID is required')
        args.index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    (prepare if args.command == 'prepare' else run)(args)


if __name__ == '__main__':
    main()
