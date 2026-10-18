"""Read-only paired local-COM versus frozen supervised bounded-WL diagnostics.

Only retained arrays/checkpoints are inspected: no model forward, new layout,
training, replay or physical tool. COM is a local fixed-coefficient adaptation,
not an official COMs reproduction or a certified lower bound. No actor/PPA
improvement, winner selection or promotion follows from this report.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle

import numpy as np
import torch

from . import qstudy_com_screen as com
from . import qstudy_bounded_screen as screen
from . import qstudy_bounded_screen_report as original
from .graph_o2o_archive import atomic_json, sha
from .qstudy_bounded_utility import assert_common_scale, validate_spec
from .qstudy_data import METRIC_NAMES
from .qstudy_models import _masked_stats
from .qstudy_report import clean, csv_text, describe, diagnostics

ROOT=Path(__file__).resolve().parents[1]
STATISTICS=original.STATISTICS
SCOPES=original.SCOPES
read_json=original.read_json
equal_array=original.equal_array
native_summary=original.native_summary
DEPENDENCIES=('fql/qstudy_com_report.py','fql/qstudy_com_screen.py',
    'fql/qstudy_bounded_screen.py','fql/qstudy_bounded_screen_report.py',
    'fql/qstudy_bounded_utility.py','fql/qstudy_report.py','fql/qstudy_models.py')


def checked_plan(path,expected_sha=None):
    if expected_sha is not None and sha(path)!=expected_sha:
        raise ValueError('COM plan SHA changed')
    plan=read_json(path)
    if (plan['schema']!='qstudy_local_com_bounded_wl_v1' or plan['config']!=com.CONFIG or
        plan['tasks']!=[dict(index=s,objective='wl',seed=s) for s in range(3)] or
        plan['maximum_new_fits']!=12 or plan['design']!='bp_fe' or
        plan['objectives']!={'wl':list(screen.OBJECTIVES['wl'])} or
        plan['official_reference']!=com.OFFICIAL or plan['actor_advanced'] or
        plan['selected_winner'] is not None or plan['prospective_new_native_calls']!=0 or
        not plan['historical_validation_previously_inspected'] or
        set(plan['source_sha256'])!=set(com.SOURCE_NAMES)):
        raise ValueError('Undeclared COM plan/loss/reference/budget')
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest or sha(Path(plan['output'])/'source_snapshot'/name)!=digest:
            raise ValueError('Frozen COM source changed: '+name)
    data,manifest,inputs=screen.load_inputs(Path(plan['inputs']['dataset_path']))
    if (inputs!=plan['inputs'] or manifest['design']!='bp_fe' or
        len(data['train']['x'])!=384 or len(data['validation']['x'])!=16 or
        not (data['train']['statuses']=='success').all() or not np.isfinite(data['train']['native']).all() or
        set(data['train']['families'])&set(data['validation']['families']) or
        com._folds(data)!=plan['folds'] or screen.json_sha(plan['folds'])!=plan['fold_plan_sha256'] or
        data['train']['source_ids']!=plan['train_source_ids'] or
        data['validation']['source_ids']!=plan['validation_source_ids']):
        raise ValueError('COM dataset/cache/TRAIN/family/source identity differs')
    controls=com.checked_control(plan['supervised_control']['plan_path'],inputs,plan['folds'])
    if controls!=plan['supervised_control']:
        raise ValueError('Immutable supervised control inventory changed')
    old_plan,old_data=original.checked_plan(Path(controls['plan_path']))
    return plan,data,old_plan,old_data


def curve_diagnostics(folder,steps):
    path=folder/'learning_curve.jsonl'
    expected=set(range(100,steps+1,100))|{steps}
    if not path.exists():
        return dict(status='missing',expected_steps=sorted(expected),missing_steps=sorted(expected),
                    observed_steps=[],coverage=0.,sha256=None,records=[])
    content=path.read_bytes()
    rows=[]
    for line in content.splitlines(keepends=True):
        try:
            row=json.loads(line)
        except (ValueError,UnicodeError):
            if not line.endswith(b'\n') and content.endswith(line):
                # Interrupted trailing write is evidence of a logging gap, not
                # a fabricated record. A committed/interior corruption fails.
                break
            raise ValueError('Corrupt committed COM learning-curve record')
        step=row['step'];parts=row['parts']
        if not isinstance(step,int) or not 1<=step<=steps or (rows and step<=rows[-1]['step']):
            raise ValueError('Nonmonotone/invalid COM learning-curve step')
        required=['regression','com_penalty','com_weighted_penalty','com_regression_parameter_grad_norm',
            'com_weighted_parameter_grad_norm','com_gradient_ratio','com_batch_rows','com_eligible_rows',
            'com_skipped_outside_center_box','com_radius_rms_max','com_fixed_max_abs_change',
            'com_box_excess_max','com_visited_box_clip_rows','com_selected_step_mean',
            'com_inner_score_gain_mean','com_inner_gradient_finite']
        if any(k not in parts or not np.isfinite(parts[k]) for k in required) or any(
                not np.isfinite(row[k]) for k in ['loss','grad']):
            raise ValueError('Missing/nonfinite COM gradient/coverage diagnostic')
        close=lambda a,b:np.isclose(a,b,rtol=3e-5,atol=1e-7)
        rn,pn=parts['com_regression_parameter_grad_norm'],parts['com_weighted_parameter_grad_norm']
        if (not close(row['loss'],parts['regression']+parts['com_weighted_penalty']) or
            not close(parts['com_weighted_penalty'],com.LOSS['coefficient']*parts['com_penalty']) or
            not close(parts['com_gradient_ratio'],pn/max(rn,1e-12)) or rn<0 or pn<0 or row['grad']<0):
            raise ValueError('COM outer-loss sign/weight/gradient pressure identity differs')
        count,eligible,skipped=[parts['com_'+k] for k in ['batch_rows','eligible_rows','skipped_outside_center_box']]
        if (any(v!=int(v) for v in [count,eligible,skipped]) or not 0<count<=64 or
            min(eligible,skipped)<0 or eligible+skipped!=count or
            not 0<=parts['com_radius_rms_max']<=com.LOSS['radius_rms']+1e-6 or
            parts['com_fixed_max_abs_change']!=0 or not 0<=parts['com_box_excess_max']<=1e-7 or
            not 0<=parts['com_selected_step_mean']<=com.LOSS['ascent_steps'] or
            not 0<=parts['com_visited_box_clip_rows']<=eligible*com.LOSS['ascent_steps'] or
            parts['com_inner_score_gain_mean']< -1e-7 or parts['com_inner_gradient_finite']!=1):
            raise ValueError('COM radius/box/fixed-node/eligibility diagnostic differs')
        rows.append(row)
    observed={r['step'] for r in rows}
    keys=['com_penalty','com_weighted_penalty','com_regression_parameter_grad_norm',
          'com_weighted_parameter_grad_norm','com_gradient_ratio','com_eligible_rows',
          'com_skipped_outside_center_box','com_radius_rms_max','com_inner_score_gain_mean']
    return dict(status='complete' if expected<=observed else 'partial',sha256=sha(path),
        expected_steps=sorted(expected),observed_steps=sorted(observed),missing_steps=sorted(expected-observed),
        coverage=len(expected&observed)/len(expected),records=rows,
        sampled_diagnostics={k:describe([r['parts'][k] for r in rows]) for k in keys},
        interpretation='Logged batch diagnostics only, not every update; parameter norms measured before outer gradient clipping')


def verify_fit(folder,done,task,fold,plan,plan_sha,data,control_folder,control_done):
    """Verify saved checkpoint/labels/units without evaluating the neural model."""
    train=np.asarray(fold['train_indices'],dtype=int)
    target=np.asarray(fold['target_indices'],dtype=int)
    target_data=data['validation'] if fold['name']=='refit' else data['train']
    context={**data['train'],'binding':plan['inputs']}
    spec,scale_audit=screen.specification(context,train,task['objective'])
    labels=screen.labels(context,train,spec)
    target_labels=screen.labels(target_data,target,spec)
    valid_train=labels['valid'].numpy()
    admitted=train[valid_train]
    binding=done['binding']
    run=binding['run']
    expected_run=dict(schema=plan['schema'],**plan['inputs'],plan_sha256=plan_sha,task=task,
        source_sha256=plan['source_sha256'],fold_plan_sha256=plan['fold_plan_sha256'],
        supervised_control=plan['supervised_control'],com_loss=com.LOSS,com_official_reference=com.OFFICIAL)
    if any(run.get(k)!=v for k,v in expected_run.items()):
        raise ValueError('Fit run/plan/source binding differs')
    if set(run)!=set(expected_run)|{'threads','torch_version','numpy_version'} or run['threads']<1:
        raise ValueError('Undeclared run binding/environment keys')
    identity=dict(train_indices=train.tolist(),admitted_train_indices=admitted.tolist(),target_indices=target.tolist(),
        train_source_ids=[context['source_ids'][i] for i in train],
        target_source_ids=[target_data['source_ids'][i] for i in target])
    expected_binding=dict(run=run,fit=fold['name'],objective=task['objective'],utility_spec=spec,
        utility_scale_audit=scale_audit,config={k:task['seed'] if k=='seed' else plan['config'][k]
            for k in ['seed','steps','batch_size','lr','weight_decay']},**identity,
        q_units='-mean predicted B; no mean(B) renormalization',beta=0.)
    if binding!=expected_binding or validate_spec(done['utility_spec'])!=spec or done['utility_scale_audit']!=scale_audit:
        raise ValueError('Fit identities/TRAIN reference/objective spec differ')
    scope=SCOPES[1] if fold['name']=='refit' else SCOPES[0]
    if (done['status']!='completed' or done['seed']!=task['seed'] or done['steps']!=plan['config']['steps'] or
        done['scope']!=scope or done['backend_calls']!=0 or done['actor_advanced'] or done['heldout_used']):
        raise ValueError('Fit budget/scope/completion differs')
    checkpoint,prediction_path=folder/'resume.pt',folder/'predictions.npz'
    if sha(checkpoint)!=done['checkpoint_sha256'] or sha(prediction_path)!=done['predictions_sha256']:
        raise ValueError('Completed checkpoint/prediction SHA changed')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if (saved['step']!=plan['config']['steps'] or saved['binding']!=binding or saved['utility_spec']!=spec or
        saved['normalization']!=done['normalization'] or
        saved['actor_contract']!=dict(q_units=binding['q_units'],beta=0.,native_reward_unchanged=True)):
        raise ValueError('Checkpoint budget/spec/normalization contract differs')
    for key,expected in dict(width=128,depth=2,ensemble_size=2,hpwl_residual=False,metric_names=['bounded_cost']).items():
        if saved['spec'].get(key)!=expected:
            raise ValueError('Checkpoint scalar architecture differs: '+key)
    train_truth=labels['cost'][labels['valid']].float()
    mean,std,magnitude,count=_masked_stats(train_truth[:,None],torch.ones((len(train_truth),1),dtype=torch.bool))
    metadata=done['normalization']
    for key,expected in [('target_mean',mean),('target_std',std),('objective_scale',magnitude),('train_observed_count',count)]:
        equal_array(metadata[key],expected.numpy(),key)
        equal_array(saved['model'][key],expected.numpy(),key+' checkpoint buffer')
    # Same inputs/fold and no residual: ALL TRAIN-fitted buffers must match
    # the independently validated original control, without a model forward.
    original=torch.load(control_folder/'resume.pt',map_location='cpu',weights_only=False)
    if saved['spec']!=original['spec'] or metadata!=control_done['normalization']:
        raise ValueError('Paired control feature/target normalization/spec differs')
    for key in ['feature_mean','feature_std','target_mean','target_std','objective_scale',
                'train_observed_count','baseline_slope','baseline_intercept','hpwl_mean','hpwl_std','movable']:
        equal_array(saved['model'][key],original['model'][key],key+' paired control buffer')
        if key in metadata:
            equal_array(metadata[key],saved['model'][key],key+' metadata buffer')
    for key in ['rng','batch_rng']:
        if saved[key].dtype!=torch.uint8 or saved[key].ndim!=1 or not saved[key].numel():
            raise ValueError('Checkpoint RNG state missing or invalid')
    if not saved['optimizer']['state'] or any(int(v['step'])!=plan['config']['steps']
            for v in saved['optimizer']['state'].values()):
        raise ValueError('Checkpoint optimizer update count differs')
    for value in saved['model'].values():
        if isinstance(value,torch.Tensor) and not torch.isfinite(value).all():
            raise ValueError('Nonfinite trained model buffer/parameter')
    eligible=com.eligible_centers(context['x'][admitted],saved['model']['movable'])
    expected_audit=dict(loss=com.LOSS,official_reference=com.OFFICIAL,
        supervised_admitted_rows=len(admitted),eligible_COM_rows=int(eligible.sum()),
        skipped_COM_outside_center_box=int((~eligible).sum()),
        label_or_family_based_eligibility=False,missing_infrastructure_labels_not_synthesized=True,
        whole_layout_terminal_bandit=True,Bellman_bootstrap=False,native_B_units_unchanged=True,
        learned_critic_no_certified_lower_bound=True,
        gradient_pressure='Measured from both outer terms; logged every100 updates/final by frozen fit',
        no_actor_or_backend_direction_accuracy_claim=True)
    if done.get('com_audit')!=expected_audit:
        raise ValueError('COM loss/reference/TRAIN eligibility audit differs')
    truth=target_labels['cost'].numpy()
    expected_arrays=dict(train_indices=train,admitted_train_indices=admitted,target_indices=target,
        train_source_ids=np.asarray(identity['train_source_ids']),target_source_ids=np.asarray(identity['target_source_ids']),
        truth=truth,utility_truth=-truth,valid=target_labels['valid'].numpy(),
        native_metrics=target_data['native'][target],native_metric_mask=target_data['mask'][target],
        status=target_data['statuses'][target],family=target_data['families'][target],
        raw_action=target_data['x'][target].numpy(),executed_action=target_data['executed'][target],
        train_native_metrics=context['native'][train],train_native_metric_mask=context['mask'][train],
        train_status=context['statuses'][train],train_valid=valid_train,train_truth=train_truth.numpy(),
        train_truth_all=labels['cost'].numpy())
    with np.load(prediction_path,allow_pickle=False) as source:
        for key,value in expected_arrays.items():
            equal_array(source[key],value,key)
        arrays={key:source[key].copy() for key in source.files}
    for key,n in [('prediction',len(target)),('train_prediction',len(admitted)),('train_prediction_all',len(train))]:
        if arrays[key].shape!=(n,) or not np.isfinite(arrays[key]).all():
            raise ValueError('Invalid saved prediction shape/values: '+key)
    if not np.allclose(arrays['train_prediction_all'][valid_train],arrays['train_prediction'],rtol=1e-6,atol=1e-6):
        raise ValueError('TRAIN admitted prediction mapping differs')
    constant=float(train_truth.mean())
    equal_array(arrays['constant_prediction'],np.full(len(target),constant),'TRAIN constant B')
    if done['constant_train_B']!=constant:
        raise ValueError('TRAIN constant B differs')
    metrics=diagnostics(arrays['prediction'],truth,arrays['family'])
    baseline=diagnostics(arrays['constant_prediction'],truth,arrays['family'])
    if metrics!=done['critic'] or baseline!=done['constant']:
        raise ValueError('Stored bounded metrics differ from saved predictions')
    valid=target_labels['valid'].numpy()
    counts=dict(train_rows=len(train),admitted_train_rows=len(admitted),excluded_train_rows=int((~valid_train).sum()),
                target_rows=len(target),valid_target_rows=int(valid.sum()),invalid_target_rows=int((~valid).sum()))
    if any(done.get(k)!=v for k,v in counts.items()):
        raise ValueError('Stored target/TRAIN validity accounting differs')
    success=target_labels['success'].numpy()&valid
    p=arrays['prediction']
    ranges=dict(minimum=float(p.min()),maximum=float(p.max()),
        success_below_zero=int(((p<0)&success).sum()),success_at_or_above_two=int(((p>=2)&success).sum()),
        any_below_zero=int((p<0).sum()),any_above_four=int((p>4).sum()))
    if any(done['prediction_range'].get(k)!=v for k,v in ranges.items()):
        raise ValueError('Saved prediction range accounting differs')
    gradient=done['input_gradients']
    if (not gradient['finite'] or gradient['rows']!=len(target) or
        not 0<=gradient['nonzero_rows']<=len(target) or
        any(not np.isfinite(gradient[k]) or gradient[k]<0 for k in ['rms_mean','rms_maximum'])):
        raise ValueError('Invalid saved raw-input gradient diagnostic')
    best=int(np.argmin(p))
    selected=dict(target_local_index=best,target_source_id=identity['target_source_ids'][best],
        status=str(arrays['status'][best]),prediction_B=float(p[best]),truth_B=float(truth[best]) if valid[best] else None,
        native_metrics={name:float(arrays['native_metrics'][best,i]) if arrays['native_metric_mask'][best,i] else None
                        for i,name in enumerate(METRIC_NAMES)},
        rule='argmin prediction among ALL fixed target proposals; no native-label filtering; ties first index')
    return dict(objective=task['objective'],seed=task['seed'],fit=fold['name'],scope=scope,
        critic=metrics,constant=baseline,constant_train_B=constant,utility_spec=spec,
        native=native_summary(arrays['native_metrics'],arrays['native_metric_mask'],arrays['status']),
        validity=dict(rows=len(target),valid=int(valid.sum()),invalid=int((~valid).sum()),
            confirmed_failure=int(target_labels['confirmed_failure'].sum()),
            train_rows=len(train),admitted_train_rows=len(admitted),excluded_train_rows=int((~valid_train).sum())),
        out_of_range=ranges,input_gradients=gradient,selected_candidate=selected,
        com_audit=expected_audit,learning_curve=curve_diagnostics(folder,plan['config']['steps']),
        bindings=dict(completed_sha256=sha(folder/'completed.json'),checkpoint_sha256=sha(checkpoint),
                      predictions_sha256=sha(prediction_path),environment={k:run[k] for k in ['threads','torch_version','numpy_version']}))


def paired_rows(fits,controls):
    """One matched fold pair; no rescaling or concatenating fold-specific B."""
    pairs=[]
    for row in fits:
        key=(row['seed'],row['fit'])
        old=next((r for r in controls if (r['seed'],r['fit'])==key),None)
        if old is None:
            continue
        assert_common_scale(row['utility_spec'],old['utility_spec'])
        if row['bindings']['environment']!=old['bindings']['environment']:
            raise ValueError('COM/control paired execution environments differ')
        if row['native']!=old['native'] or row['validity']!=old['validity']:
            raise ValueError('COM/control fixed candidate-pool labels/accounting differ')
        pairs.append(dict(seed=row['seed'],fit=row['fit'],scope=row['scope'],
            com=row,supervised=old,
            delta_com_minus_supervised={k:(row['critic'][k]-old['critic'][k]
                if row['critic'][k] is not None and old['critic'][k] is not None else None) for k in STATISTICS},
            same_fold_native_targets_and_TRAIN_scale=True))
    return pairs


def paired_summaries(pairs):
    seeds=[]
    for seed in range(3):
        for scope in SCOPES:
            rows=[r for r in pairs if r['seed']==seed and r['scope']==scope]
            expected={'cv_0','cv_1','cv_2'} if scope==SCOPES[0] else {'refit'}
            if {r['fit'] for r in rows}!=expected:
                continue
            summary=dict(seed=seed,scope=scope,folds=len(rows),
                aggregation='equal_mean_of_within_fold_metrics; native fold B never pooled',
                delta_com_minus_supervised={k:(float(np.mean(values)) if len(values)==len(rows) else None)
                    for k in STATISTICS for values in [[r['delta_com_minus_supervised'][k] for r in rows
                                                       if r['delta_com_minus_supervised'][k] is not None]]})
            for arm in ['com','supervised']:
                summary[arm]={k:(float(np.mean(values)) if len(values)==len(rows) else None)
                    for k in STATISTICS for values in [[r[arm]['critic'][k] for r in rows if r[arm]['critic'][k] is not None]]}
                summary[arm]['out_of_range']={k:sum(r[arm]['out_of_range'][k] for r in rows)
                    for k in ['success_below_zero','success_at_or_above_two','any_below_zero','any_above_four']}
                summary[arm]['raw_input_gradient_rms_equal_fold_mean']=float(np.mean(
                    [r[arm]['input_gradients']['rms_mean'] for r in rows]))
            summary['logged_COM_pressure']=dict(
                fits_with_complete_log=sum(r['com']['learning_curve']['status']=='complete' for r in rows),
                expected_fit_logs=len(rows),
                log_coverage_equal_fold_mean=float(np.mean([r['com']['learning_curve']['coverage'] for r in rows])),
                interpretation='Per-100/final sampled batch pressure; not all-update pressure or backend gradient accuracy')
            seeds.append(summary)
    aggregate=[]
    for scope in SCOPES:
        rows=[r for r in seeds if r['scope']==scope]
        complete={r['seed'] for r in rows}=={0,1,2}
        for statistic in STATISTICS:
            values=[r['delta_com_minus_supervised'][statistic] for r in rows]
            all_defined=len(values)==3 and all(v is not None for v in values)
            aggregate.append(dict(scope=scope,statistic=statistic,planned_seeds=[0,1,2],
                observed_seeds=[r['seed'] for r in rows],aggregate_eligible=complete and all_defined,
                descriptive_partial=describe(values),
                paired_three_seed_delta=describe(values) if complete and all_defined else None,
                interpretation='COM minus supervised; MAE/top-regret lower, rho/pair higher; sample SD across training seeds'))
    return seeds,aggregate


def collect(plan_path,expected_sha=None):
    plan,data,old_plan,old_data=checked_plan(plan_path,expected_sha)
    plan_sha=sha(plan_path)
    controls=[];control_map={}
    for task in (r for r in old_plan['tasks'] if r['objective']=='wl'):
        for fold in old_plan['folds']:
            folder=Path(old_plan['output'])/'wl'/f"seed_{task['seed']}"/fold['name']
            done=read_json(folder/'completed.json')
            row=original.verify_fit(folder,done,task,fold,old_plan,
                plan['supervised_control']['plan_sha256'],old_data)
            controls.append(row)
            control_map[(task['seed'],fold['name'])]=(folder,done)
    fits=[];tasks=[]
    errors=(ValueError,KeyError,TypeError,FileNotFoundError,RuntimeError,OSError,EOFError,pickle.UnpicklingError)
    for task in plan['tasks']:
        folder=Path(plan['output'])/'wl'/f"seed_{task['seed']}"
        rows=[];records=[]
        status=dict(**task,status='partial',completed_fits=[],missing_fits=[],error=None)
        try:
            for fold in plan['folds']:
                path=folder/fold['name']/'completed.json'
                if not path.exists():
                    status['missing_fits'].append(fold['name'])
                    continue
                before=sha(path);done=read_json(path)
                row=verify_fit(path.parent,done,task,fold,plan,plan_sha,data,
                    *control_map[(task['seed'],fold['name'])])
                if sha(path)!=before:
                    raise ValueError('COM completion mutated during report')
                rows.append(row);records.append(done)
                status['completed_fits'].append(fold['name'])
            marker=folder/'completed.json'
            if marker.exists():
                completion=read_json(marker);summary=read_json(folder/'summary.json')
                base_run=dict(records[0]['binding']['run']) if records else {}
                # Explicit two-level COM contract: fit run has loss/reference;
                # top task summary has exactly the frozen base run binding.
                expected_fit_run=dict(summary['binding'],com_loss=com.LOSS,com_official_reference=com.OFFICIAL)
                if (len(rows)!=4 or completion['status']!='completed' or summary['status']!='completed' or
                    completion['binding']!=summary['binding'] or
                    sha(folder/'summary.json')!=completion['summary_sha256'] or
                    completion['backend_calls']!=0 or completion['actor_advanced'] or
                    summary['seed']!=task['seed'] or summary['objective']!='wl' or
                    summary['outcomes']!=records or base_run!=expected_fit_run or
                    any(r['binding']['run']!=expected_fit_run for r in records)):
                    raise ValueError('COM task marker/base binding/retained evidence differs')
                recomputed=screen.summarize(records)
                if any(summary.get(k)!=v for k,v in recomputed.items()):
                    raise ValueError('COM task summary differs from completed folds')
                status.update(status='completed',completed_sha256=sha(marker),summary_sha256=sha(folder/'summary.json'))
            fits.extend(rows)
        except errors as exc:
            status.update(status='integrity_error',error=str(exc),completed_fits=[])
        tasks.append(status)
    pairs=paired_rows(fits,controls)
    paired_seeds,paired_multi=paired_summaries(pairs)
    seed_rows=original.seed_summaries(fits)
    multi=[r for r in original.aggregate(seed_rows) if r['objective']=='wl']
    for row in multi:
        row['aggregate_eligible']=set(row.get('seeds') or row.get('verified_replica_seeds') or [])=={0,1,2}
    # Recheck immutable inventories after reading; do not silently accept a
    # concurrently replaced original control or new completion.
    for item in plan['supervised_control']['artifacts']:
        for path,digest in item['files'].items():
            if sha(Path(path))!=digest:
                raise ValueError('Original supervised control changed while reporting')
    if sha(plan_path)!=plan_sha:
        raise ValueError('COM plan mutated during report')
    return dict(schema='qstudy_local_com_bounded_wl_report_v1',
        status='integrity_error' if any(r['status']=='integrity_error' for r in tasks) else
            'completed' if all(r['status']=='completed' for r in tasks) else 'partial',
        plan_path=str(Path(plan_path).resolve()),plan_sha256=plan_sha,
        source_sha256=plan['source_sha256'],report_dependency_sha256={n:sha(ROOT/n) for n in DEPENDENCIES},
        control_inventory=plan['supervised_control'],expected_tasks=3,expected_new_fits=12,
        completed_tasks=sum(r['status']=='completed' for r in tasks),verified_new_fits=len(fits),
        verified_original_controls=len(controls),integrity_errors=sum(r['status']=='integrity_error' for r in tasks),
        tasks=tasks,fits=fits,controls=controls,paired_folds=pairs,seed_summaries=seed_rows,multi_seed=multi,
        paired_seed_deltas=paired_seeds,paired_multi_seed=paired_multi,
        conventions=dict(cv='Same-fold diagnostics, then equal fold mean per seed, then three independent seeds; no B pooling',
            historical_validation='Previously inspected16 development rows, separate from grouped CV; not fresh confirmation',
            native_seven='All seven fixed-pool labels retained/audited; only bounded WL is predicted; no policy/PPA gain',
            gradients='Raw critic input gradients are not physical direction accuracy; saved COM parameter pressure is sampled',
            regularization='Local fixed-coefficient standardized Huber adaptation; not official COMs reproduction or certified bound',
            failures='Missing/infrastructure never synthesized; explicit validity denominators',
            uncertainty='Sample SD across independent training seeds; no row/fold/ensemble-as-seed uncertainty'),
        model_forward_calls=0,training_updates=0,layout_generation_calls=0,backend_calls=0,
        actor_advanced=False,selected_winner=None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run checkpoint/array reporting through bounded CPU Slurm')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',1)))
    result=collect(args.plan,args.plan_sha256)
    output=args.output.resolve()
    for protected in [Path(read_json(args.plan)['output']).resolve(),
                      Path(read_json(args.plan)['inputs']['dataset_path']).resolve().parent,
                      Path(read_json(result['control_inventory']['plan_path'])['output']).resolve()]:
        if output==protected or output in protected.parents or protected in output.parents:
            raise ValueError('Reports must be separate from training/control/dataset directories')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    snapshot=output/(stamp+'_job'+os.environ['SLURM_JOB_ID'])
    snapshot.mkdir(parents=True,exist_ok=False)
    result.update(slurm_job_id=os.environ['SLURM_JOB_ID'],snapshot=str(snapshot))
    (snapshot/'reporter_source.py').write_bytes(Path(__file__).read_bytes())
    (snapshot/'plan.json').write_bytes(args.plan.read_bytes())
    atomic_json(snapshot/'report.json',clean(result))
    for name in ['tasks','fits','paired_folds','seed_summaries','multi_seed','paired_seed_deltas','paired_multi_seed']:
        (snapshot/(name+'.csv')).write_text(csv_text(result[name]))
    atomic_json(snapshot/'completed.json',dict(status='completed',report_status=result['status'],
        files={p.name:sha(p) for p in snapshot.iterdir() if p.is_file()},plan_sha256=result['plan_sha256'],
        actor_advanced=False,backend_calls=0,model_forward_calls=0))
    print(json.dumps(dict(status=result['status'],snapshot=str(snapshot),verified_new_fits=result['verified_new_fits'],
        integrity_errors=result['integrity_errors'])),flush=True)


if __name__=='__main__':
    main()
