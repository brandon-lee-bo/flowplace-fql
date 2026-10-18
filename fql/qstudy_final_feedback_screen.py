"""Six fixed scalar critics using audited, newly collected TRAIN-only DRT.

Four predeclared family folds and a full-TRAIN in-sample refit share the exact
frozen bounded.fit training bytecode. Neither historical16 nor old policy DRT
is used. This module never generates layouts or invokes physical tools.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
from types import FunctionType

import numpy as np
import torch

from analysis.placement_metrics import MacroHPWL
from analysis.qstudy_rule_audit import RULES, rules, structure
from analysis import qstudy_final_power_audit as power_audit
from . import qstudy_bounded_screen as bounded
from . import qstudy_train_final_route as collector
from .qstudy_data import METRIC_NAMES, action_sha, load_dataset
from .qstudy_models import MultiMetricCritic


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'qstudy_final_feedback_scalar_screen_v1'
SPEC_VERSION = 'qstudy_final_feedback_bounded_utility_v1'
PROTOCOL = 'docs/qstudy_final_feedback_protocol.md'
PROTOCOL_SHA = 'ecb148af55e0e491f4016a7e14ed900cc4e295c087263aa1c194c0f738b28e20'
EXPERT_NAMES = ('edge_clearance', 'cluster_bbox_fraction', 'dataflow_distance', 'narrow_channel_risk')
COST_NAMES = ('grt_wl', 'drt_wl', 'drt_tns_violation', 'drt_power', 'drt_area', *EXPERT_NAMES)
OBJECTIVES = dict(grt_wl=(1., 0., 0., 0., 0., 0., 0., 0., 0.),
    drt_wl=(0., 1., 0., 0., 0., 0., 0., 0., 0.),
    drt_timing=(0., .75, .25, 0., 0., 0., 0., 0., 0.),
    drt_power=(0., .75, 0., .25, 0., 0., 0., 0., 0.),
    drt_multi_ppa=(0., .5, .25, .15, .1, 0., 0., 0., 0.),
    drt_expert=(0., .8, 0., 0., 0., .05, .05, .05, .05))
CONFIG = dict(bounded.CONFIG, folds=4)
Q_UNITS = '-mean predicted B; no mean(B) renormalization'
SOURCE_NAMES = tuple(dict.fromkeys(('fql/qstudy_final_feedback_screen.py',
    'fql/qstudy_final_feedback_screen.slurm', PROTOCOL, 'analysis/qstudy_rule_audit.py',
    'analysis/placement_metrics.py', *power_audit.SOURCE_NAMES, *bounded.SOURCE_NAMES, *collector.SOURCE_NAMES)))
STOP = False
_ACTIVE_FIT_GLOBALS = []
read, bind, sha, json_sha = collector.read, collector.bind, collector.sha, bounded.json_sha
scalar_reward = bounded.scalar_reward
ACTOR_CONTRACT = dict(q_units=Q_UNITS, beta=0., native_reward_unchanged=True)


def _slurm():
    collector.slurm()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))


def _stop(*_):
    global STOP
    STOP = True
    for namespace in _ACTIVE_FIT_GLOBALS:
        namespace['STOP'] = True


def rule_definition():
    return dict(coordinates='saved executed_action in original cache node order',
        rules={name:RULES[name] for name in EXPERT_NAMES},
        combination='equal mean of four individually fold-TRAIN-success-mean-normalized costs',
        narrow_channel_tau='2/127 of each [0,1] core axis',
        excluded=['I/O', 'corner', 'STA criticality', 'new action mask'])


def tasks():
    return [dict(index=i, objective=o, seed=s) for i, (o, s) in
            enumerate((o, s) for o in OBJECTIVES for s in range(3))]


def folds_for(context):
    assigned = np.asarray(context['fold'], dtype=int)
    family = np.asarray(context['families'])
    if set(assigned.tolist()) != set(range(4)):
        raise ValueError('Exactly four inherited TRAIN family folds are required')
    for f in np.unique(family):
        if len(set(assigned[family == f].tolist())) != 1:
            raise ValueError('A TRAIN family crosses inherited folds')
    folds = [dict(name=f'cv_{k}', train_indices=np.flatnonzero(assigned != k).tolist(),
                  target_indices=np.flatnonzero(assigned == k).tolist()) for k in range(4)]
    all_rows = list(range(len(family)))
    return folds + [dict(name='refit', train_indices=all_rows, target_indices=all_rows)]


def checked_collection(plan_path, plan_sha256, report_path, report_sha256):
    """Reinspect every final outcome; a report alone cannot admit a label."""
    plan = collector.checked_plan(plan_path, plan_sha256)
    report_binding = bind(report_path, report_sha256)
    report = read(report_path)
    records = []
    for task in plan['tasks']:
        row = collector.inspect_result(plan, task)
        if row['state'] != 'completed':
            raise ValueError('TRAIN64 collection is not completely resolved and audited')
        row['attempts'] = collector.attempt_evidence(task['output'])
        records.append(row)
    if (len(records) != 64 or report.get('version') != collector.VERSION or
        report.get('status') != 'complete' or report.get('requested') != 64 or
        report.get('counts') != {'completed':64} or report.get('records') != records or
        report.get('plan') != bind(plan_path, plan_sha256) or
        report.get('source_sha256') != collector.sources() or
        report.get('family_folds') != plan['selection']['selected_families'] or
        report.get('source_split') != 'original384TRAIN_only' or
        report.get('policy_evaluation_or_old_DRT_used') is not False or
        report.get('DRC_zero') != sum(r['DRC_zero'] for r in records)):
        raise ValueError('Complete TRAIN64 report/provenance/outcome binding differs')
    # Recheck original GRT inputs, not only their final-route continuation.
    for task in plan['tasks']:
        if collector.native_source(task['provenance']) != task['native']:
            raise ValueError('Original measured TRAIN GRT source changed')
    return plan, records, dict(collector_plan=bind(plan_path, plan_sha256), collector_report=report_binding,
        outcome_bindings=[r['outcome'] for r in records], records_sha256=json_sha(records))


def add_costs(context, cache):
    geometry, _ = structure(cache)
    hpwl = MacroHPWL(cache)
    n = len(context['x'])
    executed = context['executed'].reshape(n, len(geometry['names']), 2)
    scores = [rules(layout, geometry, hpwl) for layout in executed]
    final = context['final_native']
    values = np.column_stack((context['native'][:, 0], final[:, 0],
        np.maximum(-final[:, 2], 0.), final[:, 3], final[:, 4],
        *[np.asarray([np.nan if row[name] is None else row[name] for row in scores])
          for name in EXPERT_NAMES])).astype(np.float64)
    context.update(cost_values=values, cost_mask=np.isfinite(values))
    return context


def load_inputs(plan_path, plan_sha256, report_path, report_sha256, power_audit_path, power_audit_sha256):
    plan, records, inputs = checked_collection(plan_path, plan_sha256, report_path, report_sha256)
    power = power_audit.checked_report(power_audit_path,power_audit_sha256,
                                       inputs['collector_plan'],inputs['collector_report'])
    inputs.update(power_audit=power['audit_binding'],power_audit_completion=power['completion_binding'])
    data, manifest = load_dataset(Path(plan['dataset']['path']))
    train = data['train']  # Historical validation arrays never enter this context.
    key_to_index = {(int(f), int(v)):i for i, (f, v) in enumerate(zip(train['family'], train['variant']))}
    indices = [key_to_index[(t['family'], t['variant'])] for t in plan['tasks']]
    selected = {key:np.asarray(train[key])[indices] for key in
                ('raw_action', 'executed_action', 'metrics', 'metric_mask', 'family', 'variant')}
    for i, task in enumerate(plan['tasks']):
        row = task['provenance']
        if (action_sha(selected['raw_action'][i]) != row['raw_action_sha256'] or
            action_sha(selected['executed_action'][i]) != row['action_sha256'] or
            any(row['metrics'][name] != selected['metrics'][i,k] for k,name in enumerate(METRIC_NAMES))):
            raise ValueError('Selected raw/executed/native TRAIN correspondence changed')
    final = np.asarray([[r['metrics'][name] for name in collector.FINAL_METRICS] for r in records], dtype=np.float64)
    if (not np.isfinite(final).all() or (final[:,[0,3,4,5]] < 0).any() or
        not np.equal(final[:,5], np.floor(final[:,5])).all()):
        raise ValueError('Completed final native labels are missing, negative or malformed')
    context = dict(x=torch.as_tensor(selected['raw_action'].copy(), dtype=torch.float32).reshape(64,-1,2),
        executed=selected['executed_action'].copy(), native=selected['metrics'].copy(),
        mask=selected['metric_mask'].copy(), final_native=final, final_mask=np.isfinite(final),
        statuses=np.where(final[:,5] == 0, 'success', 'physical_failure'),
        native_statuses=np.full(64, 'completed'), families=selected['family'].copy(),
        fold=np.asarray([t['fold'] for t in plan['tasks']]),
        source_ids=[f"TRAIN_DRT64_{t['key']}_source{t['provenance']['source_row']}" for t in plan['tasks']])
    if (manifest['design'] != 'bp_fe' or len(set(context['families'])) != 32 or
        any(len(set(context['families'][context['fold']==k])) != 8 for k in range(4))):
        raise ValueError('Exact selected32families/fourfold64 denominator changed')
    cache_path = ROOT/'baseline/runtime/benchmark_cache/bp_fe.pt'
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    add_costs(context, cache)
    inputs.update(dataset_path=plan['dataset']['path'], dataset_sha256=plan['dataset']['sha256'],
        cache=bind(cache_path), dataset_manifest=plan['dataset_manifest'], dataset_rows=plan['rows'],
        raw_action_sha256=[action_sha(x) for x in selected['raw_action']],
        executed_action_sha256=[action_sha(x) for x in context['executed']],
        rule_definition_sha256=json_sha(rule_definition()), context_sha256=json_sha(context_evidence(context)))
    context.update(cache=cache, binding=dict(dataset_sha256=inputs['dataset_sha256']))
    return context, inputs


def context_evidence(context):
    """Independent final6, original GRT7, rules and feasibility are all retained."""
    result = {name:np.asarray(context[name]).tolist() for name in
        ('native','mask','final_native','final_mask','cost_values','cost_mask','executed','statuses',
         'native_statuses','families','fold')}
    result.update(raw_action=context['x'].tolist(), source_ids=context['source_ids'],
        native_metric_names=list(METRIC_NAMES), final_metric_names=list(collector.FINAL_METRICS),
        cost_names=list(COST_NAMES), metric_stage_note='GRT7 and final6 are separate native measurements',
        power_note='fixed-tool default-activity estimate; not measured workload energy')
    return result


def fit_scales(values, statuses, mask):
    values = np.asarray(values, dtype=np.float64)
    observed = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if values.ndim != 2 or values.shape[1] != len(COST_NAMES) or observed.shape != values.shape:
        raise ValueError('Expected rows by nine base/rule costs')
    observed &= (np.asarray(statuses) == 'success')[:,None]
    if ((values < 0) & observed).any():
        raise ValueError('Observed native/rule costs must be nonnegative')
    counts = observed.sum(0)
    means = np.where(observed, values, 0.).sum(0) / np.maximum(counts, 1)
    spans = np.asarray([np.ptp(values[observed[:,k],k]) if counts[k] else 0. for k in range(len(COST_NAMES))])
    bad = (counts < 2) | (means <= 1e-8) | (spans <= 1e-12)
    return np.where(means > 1e-8, means, 1.).tolist(), dict(observed_counts=counts.tolist(),
        means=means.tolist(), train_span=spans.tolist(), inadmissible=bad.tolist(),
        policy='fit TRAIN feasible-success means; active missing/constant/zero term refuses whole study')


def validate_spec(spec):
    keys = {'version','objective','cost_names','weights','scales','reference',
            'rule_definition_sha256','success_cap','failure_cost','feasibility'}
    if not isinstance(spec, dict) or set(spec) != keys:
        raise ValueError('Final-feedback utility specification keys differ')
    objective = spec['objective']
    if (spec['version'] != SPEC_VERSION or objective not in OBJECTIVES or
        spec['weights'] != list(OBJECTIVES[objective]) or spec['cost_names'] != list(COST_NAMES) or
        spec['rule_definition_sha256'] != json_sha(rule_definition()) or
        spec['success_cap'] != 2. or spec['failure_cost'] != 4. or
        spec['feasibility'] != 'completed_final_DRC0; final_DRCpositive=physical_failure_for_all_objectives'):
        raise ValueError('Fixed final-feedback objective/feasibility differs')
    scales = np.asarray(spec['scales'], dtype=float)
    ref = spec['reference']
    if scales.shape != (9,) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Nine positive finite TRAIN scales required')
    if (set(ref) != {'split','dataset_sha256','train_rows_sha256'} or ref['split'] != 'train' or
        any(not isinstance(ref[k],str) or len(ref[k]) != 64 or
            any(c not in '0123456789abcdef' for c in ref[k]) for k in ('dataset_sha256','train_rows_sha256'))):
        raise ValueError('Explicit TRAIN-only reference identity required')
    return spec


def specification(context, train_indices, objective):
    if objective not in OBJECTIVES:
        raise ValueError('Undeclared final-feedback objective')
    indices = np.asarray(train_indices, dtype=int)
    if indices.ndim != 1 or len(set(indices.tolist())) != len(indices):
        raise ValueError('Distinct TRAIN scale indices required')
    scales, audit = fit_scales(context['cost_values'][indices], context['statuses'][indices], context['cost_mask'][indices])
    active = np.asarray(OBJECTIVES[objective]) > 0
    success = context['statuses'][indices] == 'success'
    missing = (~context['cost_mask'][indices] | ~np.isfinite(context['cost_values'][indices]))[success]
    if (missing[:,active].any() or any(a and b for a,b in zip(active,audit['inadmissible']))):
        raise ValueError('Required final-feedback term unavailable, constant or zero; refuse whole protocol')
    spec = dict(version=SPEC_VERSION, objective=objective, cost_names=list(COST_NAMES),
        weights=list(OBJECTIVES[objective]), scales=scales,
        reference=dict(split='train', dataset_sha256=context['binding']['dataset_sha256'],
            train_rows_sha256=json_sha([context['source_ids'][i] for i in indices])),
        rule_definition_sha256=json_sha(rule_definition()), success_cap=2., failure_cost=4.,
        feasibility='completed_final_DRC0; final_DRCpositive=physical_failure_for_all_objectives')
    return validate_spec(spec), audit


def labels(context, indices, spec):
    spec = validate_spec(spec)
    indices = np.asarray(indices, dtype=int)
    values = torch.as_tensor(context['cost_values'][indices], dtype=torch.float64)
    observed = torch.as_tensor(context['cost_mask'][indices], dtype=torch.bool) & torch.isfinite(values)
    weights = torch.tensor(spec['weights'],dtype=values.dtype)
    scales = torch.tensor(spec['scales'],dtype=values.dtype)
    active = weights > 0
    statuses = context['statuses'][indices]
    success = torch.as_tensor(statuses == 'success') & (observed[:,active] & (values[:,active] >= 0)).all(1)
    failure = torch.tensor([s in ('physical_failure','geometry_failure') for s in statuses])
    cost = torch.full((len(indices),), float('nan'), dtype=torch.float64)
    composite = cost.clone()
    if bool(success.any()):
        c = (values[success][:,active]/scales[active]*weights[active]).sum(1)
        if not bool(torch.isfinite(c).all()):
            raise ValueError('Nonfinite normalized final-feedback cost')
        composite[success], cost[success] = c, 2*c/(1+c)
    cost[failure] = 4.
    return dict(cost=cost, utility=-cost, composite_cost=composite, valid=success|failure,
                success=success, failure=failure)


def admit(context, folds):
    if any(s not in ('success','physical_failure','geometry_failure') for s in context['statuses']):
        raise ValueError('Unresolved infrastructure/tool/missing status blocks whole study')
    for fold in folds:
        for objective in OBJECTIVES:
            spec,_ = specification(context, fold['train_indices'], objective)
            if not bool(labels(context, fold['train_indices'], spec)['valid'].all()):
                raise ValueError('No objective may silently omit TRAIN rows')


def fit(args, context, name, train_indices, target_indices, target_data, intercept=None):
    """Exact old optimization; private output adapters add final-native evidence."""
    spec,_ = specification(context, train_indices, args.objective)
    target = labels(target_data, target_indices, spec)
    train = labels(context, train_indices, spec)
    scope = 'train_family_cv' if name.startswith('cv_') else 'train_in_sample_refit'
    additions = {}
    for prefix, data, indices, truth in (('',target_data,target_indices,target), ('train_',context,train_indices,train)):
        for key in ('final_native','final_mask','cost_values','cost_mask','native_statuses'):
            additions[prefix+key] = np.asarray(data[key])[indices]
        additions[prefix+'composite_cost'] = truth['composite_cost'].numpy()
    class NumpyOutputs:
        def __getattr__(self, key):
            return getattr(np, key)
        def savez_compressed(self, path, **values):
            np.savez_compressed(path, **values, **additions)
    def write_completed(path, value):
        if Path(path).name == 'completed.json':
            value.update(scope=scope, final_metric_names=list(collector.FINAL_METRICS), cost_names=list(COST_NAMES),
                historical_validation_used=False, full_refit_is_in_sample=scope == 'train_in_sample_refit')
        bounded.atomic_json(path, value)
    namespace = dict(bounded.fit.__globals__, specification=specification, labels=labels,
        STOP=STOP, np=NumpyOutputs(), atomic_json=write_completed)
    private = FunctionType(bounded.fit.__code__, namespace, bounded.fit.__name__, bounded.fit.__defaults__)
    private.__kwdefaults__ = bounded.fit.__kwdefaults__
    def observe(batch):
        if intercept is not None:
            intercept(batch)
        namespace['STOP'] = STOP
    _ACTIVE_FIT_GLOBALS.append(namespace)
    try:
        result = private(args,context,name,train_indices,target_indices,target_data,intercept=observe)
    finally:
        _ACTIVE_FIT_GLOBALS.remove(namespace)
    if (result['scope'] != scope or result.get('historical_validation_used') is not False or
        result.get('full_refit_is_in_sample') != (scope == 'train_in_sample_refit')):
        raise ValueError('Completed fit scope cannot be relabeled as heldout validation')
    with np.load(args.output/name/'predictions.npz', allow_pickle=False) as saved:
        for key,value in additions.items():
            numeric = np.asarray(value).dtype.kind in 'fc'
            if not np.array_equal(saved[key],value,equal_nan=numeric):
                raise ValueError('Saved final-feedback label evidence changed: '+key)
    return result


def source_binding():
    bind(ROOT/PROTOCOL, PROTOCOL_SHA)
    return {name:sha(ROOT/name) for name in SOURCE_NAMES}


def prepare(args):
    _slurm()
    context, inputs = load_inputs(args.collector_plan,args.collector_plan_sha256,
        args.collector_report,args.collector_report_sha256,args.power_audit,args.power_audit_sha256)
    folds = folds_for(context)
    admit(context,folds)
    output = Path(args.output).resolve()
    if output.is_relative_to(Path(read(args.collector_plan)['output'])):
        raise ValueError('Keep new critic outputs outside immutable collector evidence')
    plan = dict(schema=SCHEMA, output=str(output), design='bp_fe', inputs=inputs,
        source_sha256=source_binding(),config=CONFIG,objectives={k:list(v) for k,v in OBJECTIVES.items()},
        tasks=tasks(),folds=folds,fold_plan_sha256=json_sha(folds),cost_names=list(COST_NAMES),
        native_metric_names=list(METRIC_NAMES),final_metric_names=list(collector.FINAL_METRICS),
        rule_definition=rule_definition(),train_source_ids=context['source_ids'],q_units=Q_UNITS,
        requested_train_rows=64,source_families=32,full_refit_scope='train_in_sample_refit',
        historical_validation_used=False,old_policy_DRT_used=False,backend_calls=0,actor_advanced=False,
        training_engine='frozen bounded.fit exact bytecode in private namespace; label and output adapters only')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'plan.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name,digest in plan['source_sha256'].items():
            dest=output/'source_snapshot'/name
            if dest.exists():
                bind(dest,digest)
            else:
                dest.parent.mkdir(parents=True,exist_ok=True)
                temporary=dest.with_name(dest.name+'.tmp')
                shutil.copyfile(ROOT/name,temporary)
                bind(temporary,digest)
                temporary.replace(dest)
        collector.freeze(output/'input_evidence.json',context_evidence(context))
        plan['input_evidence'] = bind(output/'input_evidence.json')
        collector.freeze(output/'plan.json',plan)
    print(json.dumps(dict(plan=str(output/'plan.json'),plan_sha256=sha(output/'plan.json'),tasks=18,fits=90)),flush=True)


def checked_plan(path, expected_sha256):
    _slurm()
    if (not isinstance(expected_sha256,str) or len(expected_sha256)!=64 or
        any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise ValueError('An externally pinned screen plan SHA is required')
    bind(path,expected_sha256)
    plan=read(path)
    if (plan['schema'] != SCHEMA or plan['config'] != CONFIG or plan['tasks'] != tasks() or
        plan['objectives'] != {k:list(v) for k,v in OBJECTIVES.items()} or
        plan['source_sha256'] != source_binding() or plan['q_units'] != Q_UNITS or
        plan['historical_validation_used'] is not False or plan['old_policy_DRT_used'] is not False or
        plan['full_refit_scope'] != 'train_in_sample_refit' or plan['backend_calls'] != 0 or
        plan['actor_advanced'] is not False or plan['requested_train_rows'] != 64 or plan['source_families'] != 32 or
        Path(path).resolve() != Path(plan['output'])/'plan.json'):
        raise ValueError('Fixed final-feedback screen declaration/source contract changed')
    for name,digest in plan['source_sha256'].items():
        bind(Path(plan['output'])/'source_snapshot'/name,digest)
    inputs=plan['inputs']
    context,actual=load_inputs(inputs['collector_plan']['path'],inputs['collector_plan']['sha256'],
        inputs['collector_report']['path'],inputs['collector_report']['sha256'],
        inputs['power_audit']['path'],inputs['power_audit']['sha256'])
    folds=folds_for(context)
    admit(context,folds)
    if (actual != inputs or plan['folds'] != folds or plan['fold_plan_sha256'] != json_sha(folds) or
        plan['train_source_ids'] != context['source_ids'] or plan['cost_names'] != list(COST_NAMES) or
        plan['native_metric_names'] != list(METRIC_NAMES) or plan['final_metric_names'] != list(collector.FINAL_METRICS) or
        plan['rule_definition'] != rule_definition()):
        raise ValueError('Measured TRAIN64 labels/coordinates/folds/rules changed')
    bind(plan['input_evidence']['path'],plan['input_evidence']['sha256'])
    if read(plan['input_evidence']['path']) != context_evidence(context):
        raise ValueError('Independent native GRT/final evidence changed')
    from baseline.api import load_problem
    context['movable']=~load_problem('bp_fe','cpu').cond.is_ports.bool()
    return plan,dict(train=context),context


def checked_model(context,spec,seed,indices,checkpoint,done):
    """Reconstruct normalization from fit TRAIN only, then strictly load."""
    bind(checkpoint,done['checkpoint_sha256'])
    target=labels(context,indices,spec)
    if not bool(target['valid'].all()):
        raise ValueError('Invalid declared final-feedback TRAIN target')
    model=MultiMetricCritic(context['cache'],context['x'][indices],target['cost'].float()[:,None],
        context['movable'],['bounded_cost'],train_ids=[context['source_ids'][i] for i in indices],
        train_splits=['train']*len(indices),width=128,depth=2,ensemble_size=2,hpwl_residual=False,seed=seed)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if (done['steps'] != CONFIG['steps'] or saved['step'] != CONFIG['steps'] or
        saved['binding'] != done['binding'] or saved['utility_spec'] != spec or done['utility_spec'] != spec or
        saved['normalization'] != model.normalization_metadata() or done['normalization'] != model.normalization_metadata() or
        saved['spec'] != model.spec or saved['actor_contract'] != ACTOR_CONTRACT or
        saved['model'].get('_extra_state') != model.get_extra_state()):
        raise ValueError('Final-feedback checkpoint step/spec/normalization identity differs')
    for name,value in model.named_buffers():
        actual=saved['model'].get(name)
        if not isinstance(actual,torch.Tensor) or actual.dtype != value.dtype or not torch.equal(actual,value):
            raise ValueError('Final-feedback TRAIN/cache buffer changed: '+name)
    model.load_state_dict(saved['model'],strict=True)
    if any(not bool(v.isfinite().all()) for v in model.state_dict().values()
           if isinstance(v,torch.Tensor) and v.is_floating_point()):
        raise ValueError('Nonfinite final-feedback checkpoint')
    return model


def run_binding(plan,plan_path,task):
    return dict(schema=SCHEMA,**plan['inputs'],plan_sha256=sha(plan_path),task=task,
        source_sha256=plan['source_sha256'],fold_plan_sha256=plan['fold_plan_sha256'],
        input_evidence=plan['input_evidence'],threads=torch.get_num_threads(),
        torch_version=torch.__version__,numpy_version=np.__version__)


def checked_refit(plan_path,objective,seed,expected_plan_sha256,loaded=None):
    """Verify all five fixed fits; return full64 in-sample refit, never best-CV."""
    bind(plan_path,expected_plan_sha256)
    plan,data,context=checked_plan(plan_path,expected_plan_sha256) if loaded is None else loaded
    task=next((t for t in plan['tasks'] if t['objective']==objective and t['seed']==seed),None)
    if task is None:
        raise ValueError('Undeclared final-feedback refit task')
    folder=Path(plan['output'])/objective/f'seed_{seed}'
    done,summary=read(folder/'completed.json'),read(folder/'summary.json')
    run=done['binding']
    expected_run=run_binding(plan,plan_path,task)
    expected_run['threads']=run.get('threads')
    if (run != expected_run or not isinstance(run['threads'],int) or run['threads']<1 or
        done['status'] != 'completed' or done['summary_sha256'] != sha(folder/'summary.json') or
        done['backend_calls'] != 0 or done['actor_advanced'] is not False or done['historical_validation_used'] is not False or
        summary['status'] != 'completed' or summary['binding'] != run or summary['objective'] != objective or
        summary['seed'] != seed or len(summary['outcomes']) != 5):
        raise ValueError('Final-feedback task is not a bound five-fit completion')
    evidence={str(plan_path):sha(plan_path),str(folder/'completed.json'):sha(folder/'completed.json'),
              str(folder/'summary.json'):sha(folder/'summary.json')}
    selected=None
    for j,fold in enumerate(plan['folds']):
        fp=folder/fold['name']
        completed=read(fp/'completed.json')
        spec,audit=specification(context,fold['train_indices'],objective)
        target=labels(context,fold['target_indices'],spec)
        train=labels(context,fold['train_indices'],spec)
        identity=dict(train_indices=fold['train_indices'],admitted_train_indices=fold['train_indices'],
            target_indices=fold['target_indices'],train_source_ids=[context['source_ids'][i] for i in fold['train_indices']],
            target_source_ids=[context['source_ids'][i] for i in fold['target_indices']])
        expected=dict(run=run,fit=fold['name'],objective=objective,utility_spec=spec,utility_scale_audit=audit,
            config={k:dict(seed=seed,**CONFIG)[k] for k in ('seed','steps','batch_size','lr','weight_decay')},
            **identity,q_units=Q_UNITS,beta=0.)
        scope='train_family_cv' if fold['name'].startswith('cv_') else 'train_in_sample_refit'
        if (completed != summary['outcomes'][j] or completed['binding'] != expected or
            completed['status'] != 'completed' or completed['scope'] != scope or
            completed['historical_validation_used'] is not False or
            completed['full_refit_is_in_sample'] != (scope=='train_in_sample_refit') or
            completed['backend_calls'] != 0 or completed['actor_advanced'] is not False or
            completed['heldout_used'] is not False):
            raise ValueError('Final-feedback fit identity/scope differs')
        for filename,key in (('resume.pt','checkpoint_sha256'),('predictions.npz','predictions_sha256')):
            evidence[str(fp/filename)]=bind(fp/filename,completed[key])['sha256']
        evidence[str(fp/'completed.json')]=sha(fp/'completed.json')
        with np.load(fp/'predictions.npz',allow_pickle=False) as saved:
            expected_arrays=dict(train_indices=fold['train_indices'],target_indices=fold['target_indices'],
                admitted_train_indices=fold['train_indices'],truth=target['cost'].numpy(),utility_truth=target['utility'].numpy(),
                valid=target['valid'].numpy(),family=context['families'][fold['target_indices']],
                raw_action=context['x'][fold['target_indices']].numpy(),executed_action=context['executed'][fold['target_indices']],
                native_metrics=context['native'][fold['target_indices']],native_metric_mask=context['mask'][fold['target_indices']],
                status=context['statuses'][fold['target_indices']],train_source_ids=identity['train_source_ids'],
                target_source_ids=identity['target_source_ids'],train_truth=train['cost'].float().numpy(),
                train_truth_all=train['cost'].numpy(),train_valid=train['valid'].numpy())
            for prefix,indices,truth in (('',fold['target_indices'],target),('train_',fold['train_indices'],train)):
                for key in ('final_native','final_mask','cost_values','cost_mask','native_statuses'):
                    expected_arrays[prefix+key]=context[key][indices]
                expected_arrays[prefix+'composite_cost']=truth['composite_cost'].numpy()
            for key,value in expected_arrays.items():
                numeric=np.asarray(value).dtype.kind in 'fc'
                if not np.array_equal(saved[key],value,equal_nan=numeric):
                    raise ValueError('Saved final-feedback prediction/label provenance differs: '+key)
            if not np.isfinite(saved['prediction']).all():
                raise ValueError('Nonfinite final-feedback predictions')
        model=checked_model(context,spec,seed,fold['train_indices'],fp/'resume.pt',completed)
        if fold['name']=='refit':
            selected=dict(model=model,spec=spec,scale_audit=audit,context=context,
                checkpoint=bind(fp/'resume.pt'),completion=bind(fp/'completed.json'),
                normalization=model.normalization_metadata(),done=completed)
    if selected is None:
        raise ValueError('Missing full64 in-sample refit')
    return dict(**selected,plan=plan,data=data,evidence=evidence)


def summarize(outcomes):
    cv=[r for r in outcomes if r['scope']=='train_family_cv']
    result=dict(cv_folds=len(cv),cv_aggregation='equal fold mean; fold scales are not pooled',
        full_train_in_sample_refit=next((r for r in outcomes if r['scope']=='train_in_sample_refit'),None),
        warning='In-sample refit is not validation; raw Q gradients do not validate physical directions',
        backend_calls=0,actor_advanced=False,historical_validation_used=False)
    for arm in ('critic','constant'):
        result[arm]={key:dict(n=len(values),mean=float(np.mean(values)) if values else None)
            for key in ('mae','spearman','kendall','family_pair_accuracy','top_regret')
            for values in [[r[arm][key] for r in cv if r[arm].get(key) is not None]]}
    return result


def run(args):
    _slurm()
    torch.use_deterministic_algorithms(True)
    plan,_,context=checked_plan(args.plan,args.plan_sha256)
    if not 0<=args.index<len(tasks()):
        raise ValueError('Task index outside eighteen predeclared tasks')
    task=plan['tasks'][args.index]
    fit_args=argparse.Namespace(**CONFIG,**task,
        output=Path(plan['output'])/task['objective']/f"seed_{task['seed']}")
    fit_args.output.mkdir(parents=True,exist_ok=True)
    context['binding']=run_binding(plan,args.plan,task)
    with (fit_args.output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        outcomes=[fit(fit_args,context,f['name'],f['train_indices'],f['target_indices'],context) for f in plan['folds']]
        bounded.atomic_json(fit_args.output/'summary.json',dict(status='completed',binding=context['binding'],
            objective=task['objective'],seed=task['seed'],outcomes=outcomes,**summarize(outcomes)))
        bounded.atomic_json(fit_args.output/'completed.json',dict(status='completed',binding=context['binding'],
            summary_sha256=sha(fit_args.output/'summary.json'),backend_calls=0,actor_advanced=False,
            historical_validation_used=False))
    print(json.dumps(dict(status='completed',task=task,output=str(fit_args.output))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='stage',required=True)
    p=sub.add_parser('prepare')
    p.add_argument('--collector-plan',type=Path,required=True)
    p.add_argument('--collector-plan-sha256',required=True)
    p.add_argument('--collector-report',type=Path,required=True)
    p.add_argument('--collector-report-sha256',required=True)
    p.add_argument('--power-audit',type=Path,required=True)
    p.add_argument('--power-audit-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('run')
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True)
    p.add_argument('--index',type=int)
    args=parser.parse_args()
    for sig in (signal.SIGUSR1,signal.SIGTERM):
        signal.signal(sig,_stop)
    if args.stage=='run' and args.index is None:
        if 'SLURM_ARRAY_TASK_ID' not in os.environ:
            parser.error('run requires --index or SLURM_ARRAY_TASK_ID')
        args.index=int(os.environ['SLURM_ARRAY_TASK_ID'])
    (prepare if args.stage=='prepare' else run)(args)


if __name__=='__main__':
    main()
