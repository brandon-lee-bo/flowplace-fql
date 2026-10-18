"""Read-only, immutable partial/full reports of the nine bounded critic tasks.

No fitting, model inference, actor promotion or physical calls. CV diagnostics
are computed within each fold, then averaged per seed; differently scaled fold
B values are never concatenated. Cross-objective deltas describe different
prediction targets, not a common-utility policy improvement.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
import json
import os
from pathlib import Path
import pickle

import numpy as np
import torch

from .graph_o2o_archive import atomic_json, sha
from .qstudy_bounded_utility import assert_common_scale, validate_spec
from . import qstudy_bounded_screen as screen
from .qstudy_data import METRIC_NAMES
from .qstudy_models import _masked_stats
from .qstudy_report import clean, csv_text, describe, diagnostics


ROOT=Path(__file__).resolve().parents[1]
STATISTICS=('mae','spearman','kendall','family_pair_accuracy','top_regret')
SCOPES=('train_family_cv','historical_validation_previously_inspected')


def read_json(path):
    return json.loads(Path(path).read_text())


def equal_array(actual,expected,name):
    actual,expected=np.asarray(actual),np.asarray(expected)
    same=(np.array_equal(actual,expected,equal_nan=True) if actual.dtype.kind in 'fc' or expected.dtype.kind in 'fc'
          else np.array_equal(actual,expected))
    if not same:
        raise ValueError('Prediction identity/native label mismatch: '+name)


def checked_plan(path):
    plan=read_json(path)
    tasks=[dict(index=i,objective=o,seed=s) for i,(o,s) in
           enumerate((o,s) for o in screen.OBJECTIVES for s in range(3))]
    if (plan['schema']!='qstudy_bounded_scalar_screen_v1' or plan['tasks']!=tasks or
        plan['config']!=screen.CONFIG or plan['objectives']!={k:list(v) for k,v in screen.OBJECTIVES.items()} or
        plan['metric_names']!=list(METRIC_NAMES) or plan['design']!='bp_fe'):
        raise ValueError('Bounded nine-task plan/configuration differs')
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest or sha(Path(plan['output'])/'source_snapshot'/name)!=digest:
            raise ValueError('Frozen screen source changed: '+name)
    data,manifest,inputs=screen.load_inputs(Path(plan['inputs']['dataset_path']))
    if inputs!=plan['inputs'] or manifest['design']!='bp_fe' or len(data['train']['x'])!=384 or len(data['validation']['x'])!=16:
        raise ValueError('Bounded dataset/cache binding differs')
    folds=screen.family_folds(data['train']['families'],3)
    folds.append(dict(name='refit',train_indices=list(range(384)),target_indices=list(range(16))))
    if (plan['folds']!=folds or plan['fold_plan_sha256']!=screen.json_sha(folds) or
        plan['train_source_ids']!=data['train']['source_ids'] or plan['validation_source_ids']!=data['validation']['source_ids'] or
        plan['train_source_ids_sha256']!=screen.json_sha(data['train']['source_ids'])):
        raise ValueError('Bounded common family folds/source identities differ')
    if set(data['train']['families'])&set(data['validation']['families']):
        raise ValueError('TRAIN/historical-validation family leakage')
    return plan,data


def native_summary(native,mask,statuses):
    success=np.asarray(statuses)=='success'
    return dict(rows=len(statuses),status_counts={s:int((statuses==s).sum()) for s in np.unique(statuses)},
        metrics={name:dict(observed=int((success&mask[:,i]).sum()),missing=int((~mask[:,i]).sum()),
            success_mean=float(np.mean(native[success&mask[:,i],i])) if (success&mask[:,i]).any() else None)
            for i,name in enumerate(METRIC_NAMES)},
        interpretation='fixed supervised candidate-pool labels, not policy-generated improvement')


def verify_fit(folder,done,task,fold,plan,plan_sha,data):
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
        source_sha256=plan['source_sha256'],fold_plan_sha256=plan['fold_plan_sha256'])
    if any(run.get(k)!=v for k,v in expected_run.items()):
        raise ValueError('Fit run/plan/source binding differs')
    if set(run)!=set(expected_run)|{'threads','torch_version','numpy_version'} or run['threads']<1:
        raise ValueError('Undeclared run binding/environment keys')
    identity=dict(train_indices=train.tolist(),admitted_train_indices=admitted.tolist(),target_indices=target.tolist(),
        train_source_ids=[context['source_ids'][i] for i in train],
        target_source_ids=[target_data['source_ids'][i] for i in target])
    expected_binding=dict(run=run,fit=fold['name'],objective=task['objective'],utility_spec=spec,
        utility_scale_audit=scale_audit,config={k:task['seed'] if k=='seed' else screen.CONFIG[k]
            for k in ['seed','steps','batch_size','lr','weight_decay']},**identity,
        q_units='-mean predicted B; no mean(B) renormalization',beta=0.)
    if binding!=expected_binding or validate_spec(done['utility_spec'])!=spec or done['utility_scale_audit']!=scale_audit:
        raise ValueError('Fit identities/TRAIN reference/objective spec differ')
    scope=SCOPES[1] if fold['name']=='refit' else SCOPES[0]
    if (done['status']!='completed' or done['seed']!=task['seed'] or done['steps']!=screen.CONFIG['steps'] or
        done['scope']!=scope or done['backend_calls']!=0 or done['actor_advanced'] or done['heldout_used']):
        raise ValueError('Fit budget/scope/completion differs')
    checkpoint,prediction_path=folder/'resume.pt',folder/'predictions.npz'
    if sha(checkpoint)!=done['checkpoint_sha256'] or sha(prediction_path)!=done['predictions_sha256']:
        raise ValueError('Completed checkpoint/prediction SHA changed')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if (saved['step']!=screen.CONFIG['steps'] or saved['binding']!=binding or saved['utility_spec']!=spec or
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
        bindings=dict(completed_sha256=sha(folder/'completed.json'),checkpoint_sha256=sha(checkpoint),
                      predictions_sha256=sha(prediction_path),environment={k:run[k] for k in ['threads','torch_version','numpy_version']}))


def seed_summaries(fits):
    rows=[]
    for objective in screen.OBJECTIVES:
        for seed in range(3):
            matched=[r for r in fits if r['objective']==objective and r['seed']==seed]
            for scope in SCOPES:
                selected=[r for r in matched if r['scope']==scope]
                expected={'cv_0','cv_1','cv_2'} if scope==SCOPES[0] else {'refit'}
                if {r['fit'] for r in selected}!=expected:
                    continue
                row=dict(objective=objective,seed=seed,scope=scope,folds=len(selected),
                    aggregation='equal_mean_of_within_fold_metrics' if scope==SCOPES[0] else 'one full-TRAIN refit on historical16')
                for arm in ['critic','constant']:
                    row[arm]={key:(float(np.mean(values)) if len(values)==len(selected) else None)
                        for key in STATISTICS for values in [[r[arm][key] for r in selected if r[arm][key] is not None]]}
                    count=sum(r[arm]['family_pairs'] for r in selected)
                    row[arm]['within_fold_pair_accuracy']=(sum(r[arm]['family_pair_accuracy']*r[arm]['family_pairs']
                        for r in selected if r[arm]['family_pair_accuracy'] is not None)/count if count else None)
                    row[arm]['within_fold_pairs']=count
                row['validity']={k:sum(r['validity'][k] for r in selected)
                    for k in ['rows','valid','invalid','confirmed_failure']}
                row['out_of_range']={k:sum(r['out_of_range'][k] for r in selected)
                    for k in ['success_below_zero','success_at_or_above_two','any_below_zero','any_above_four']}
                row['input_gradients']=dict(finite=all(r['input_gradients']['finite'] for r in selected),
                    rows=sum(r['input_gradients']['rows'] for r in selected),
                    nonzero_rows=sum(r['input_gradients']['nonzero_rows'] for r in selected),
                    equal_fold_rms_mean=float(np.mean([r['input_gradients']['rms_mean'] for r in selected])),
                    maximum=max(r['input_gradients']['rms_maximum'] for r in selected))
                row['native_fixed_pool_metrics']={metric:dict(
                    observed=sum(r['native']['metrics'][metric]['observed'] for r in selected),
                    missing=sum(r['native']['metrics'][metric]['missing'] for r in selected),
                    success_mean=(sum(r['native']['metrics'][metric]['success_mean']*r['native']['metrics'][metric]['observed']
                        for r in selected if r['native']['metrics'][metric]['success_mean'] is not None)/n if n else None))
                    for metric in METRIC_NAMES
                    for n in [sum(r['native']['metrics'][metric]['observed'] for r in selected)]}
                row['delta_critic_minus_constant']={key:(row['critic'][key]-row['constant'][key]
                    if row['critic'][key] is not None and row['constant'][key] is not None else None) for key in STATISTICS}
                rows.append(row)
    return rows


def aggregate(seeds):
    rows=[]
    for objective in screen.OBJECTIVES:
        for scope in SCOPES:
            selected=[r for r in seeds if r['objective']==objective and r['scope']==scope]
            for arm in ['critic','constant']:
                for key in (*STATISTICS,'within_fold_pair_accuracy'):
                    valid=[r for r in selected if r[arm][key] is not None]
                    deterministic=arm=='constant'
                    values=[r[arm][key] for r in valid]
                    if deterministic and values:
                        if any(v!=values[0] for v in values[1:]):
                            raise ValueError('Deterministic TRAIN-mean comparator differs across seeds')
                        values=values[:1]
                    rows.append(dict(objective=objective,scope=scope,arm=arm,statistic=key,
                        seeds=[] if deterministic else [r['seed'] for r in valid],planned_seeds=[0,1,2],
                        verified_replica_seeds=[r['seed'] for r in valid] if deterministic else None,
                        independent_training_runs=0 if deterministic else len(valid),**describe(values),
                        uncertainty_unit='one deterministic comparator, no training-seed SD' if deterministic else
                            'independent critic-training seeds on one fixed dataset/fold partition; sample SD'))
    return rows


def objective_deltas(seeds):
    pairs=[]
    for first,second in combinations(screen.OBJECTIVES,2):
        for scope in SCOPES:
            for seed in range(3):
                a=next((r for r in seeds if (r['objective'],r['scope'],r['seed'])==(first,scope,seed)),None)
                b=next((r for r in seeds if (r['objective'],r['scope'],r['seed'])==(second,scope,seed)),None)
                if a is None or b is None:
                    continue
                for statistic in STATISTICS:
                    av,bv=a['critic'][statistic],b['critic'][statistic]
                    if av is not None and bv is not None:
                        pairs.append(dict(first=first,second=second,scope=scope,seed=seed,statistic=statistic,
                            first_value=av,second_value=bv,delta_second_minus_first=bv-av,
                            target_definition_differs=True,
                            interpretation='prediction-difficulty difference across objectives; not common-utility policy gain'))
    groups=defaultdict(list)
    for row in pairs:
        groups[tuple(row[k] for k in ['first','second','scope','statistic'])].append(row)
    summary=[dict(zip(['first','second','scope','statistic'],key),**describe([r['delta_second_minus_first'] for r in values]),
                 seeds=[r['seed'] for r in values],planned_seeds=[0,1,2],target_definition_differs=True)
             for key,values in sorted(groups.items())]
    return pairs,summary


def collect(plan_path):
    plan,data=checked_plan(plan_path)
    plan_sha=sha(plan_path)
    fits,tasks=[],[]
    for task in plan['tasks']:
        folder=Path(plan['output'])/task['objective']/f"seed_{task['seed']}"
        task_fits=[]
        status=dict(**task,status='partial',completed_fits=[],missing_fits=[],error=None)
        try:
            done_records=[]
            for fold in plan['folds']:
                path=folder/fold['name']/'completed.json'
                if not path.exists():
                    status['missing_fits'].append(fold['name'])
                    continue
                before=sha(path)
                done=read_json(path)
                row=verify_fit(path.parent,done,task,fold,plan,plan_sha,data)
                if sha(path)!=before:
                    raise ValueError('Fit completion mutated during read')
                task_fits.append(row)
                done_records.append(done)
                status['completed_fits'].append(fold['name'])
            marker=folder/'completed.json'
            if marker.exists():
                completion=read_json(marker)
                summary=read_json(folder/'summary.json')
                if (len(task_fits)!=4 or completion['status']!='completed' or
                    sha(folder/'summary.json')!=completion['summary_sha256'] or
                    summary['outcomes']!=done_records or completion['binding']!=summary['binding'] or
                    any(r['binding']['run']!=summary['binding'] for r in done_records)):
                    raise ValueError('Task completion/retained fit evidence differs')
                status.update(status='completed',completed_sha256=sha(marker),summary_sha256=sha(folder/'summary.json'))
            fits.extend(task_fits)
        except (ValueError,KeyError,FileNotFoundError,RuntimeError,OSError,EOFError,pickle.UnpicklingError) as exc:
            # Invalidate ALL evidence from a corrupt task, including earlier fits.
            status.update(status='integrity_error',error=str(exc),completed_fits=[])
        tasks.append(status)
    # Exact physical scales/references and environments must match paired folds.
    for fold in plan['folds']:
        same=[r for r in fits if r['fit']==fold['name']]
        if same:
            for row in same[1:]:
                assert_common_scale(same[0]['utility_spec'],row['utility_spec'])
                if row['bindings']['environment']!=same[0]['bindings']['environment']:
                    raise ValueError('Paired fold run environments differ')
    seeds=seed_summaries(fits)
    pairs,paired=objective_deltas(seeds)
    return dict(schema='qstudy_bounded_scalar_report_v1',
        status='integrity_error' if any(r['status']=='integrity_error' for r in tasks) else
               'completed' if all(r['status']=='completed' for r in tasks) else 'partial',
        plan_path=str(plan_path.resolve()),plan_sha256=plan_sha,
        expected_tasks=9,completed_tasks=sum(r['status']=='completed' for r in tasks),
        expected_fits=36,verified_fits=len(fits),integrity_errors=sum(r['status']=='integrity_error' for r in tasks),
        tasks=tasks,fits=fits,seed_summaries=seeds,multi_seed=aggregate(seeds),
        paired_objective_seed_deltas=pairs,paired_objective_summary=paired,
        conventions=dict(cv='Within-fold diagnostics then equal fold mean per seed; NEVER pool fold B predictions/truth',
            historical_validation='16 previously inspected development rows; not fresh confirmation',
            native_metrics='Native seven metrics are unchanged fixed-pool labels; not final DRT/Table2 and not policy improvement',
            pair_ties='True within-family ties excluded; predicted ties count incorrect; constant ranking undefined',
            gradients='Saved finite raw-input gradients only; no gradient/backend direction accuracy established',
            objectives='Cross-objective B diagnostics have different target definitions; no automatic winner',
            failures='Confirmed physical/geometry failure B4; missing/infrastructure has no label, never fabricated numeric PPA',
            uncertainty='Sample SD across critic seeds, not fold/row/ensemble uncertainty or calibrated confidence'),
        backend_calls=0,actor_advanced=False,model_inference_calls=0,heldout_used=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run numerical aggregation through Slurm')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',1)))
    result=collect(args.plan)
    output=args.output.resolve()
    for protected in [Path(read_json(args.plan)['output']).resolve(),
                      Path(read_json(args.plan)['inputs']['dataset_path']).resolve().parent]:
        if output==protected or output in protected.parents or protected in output.parents:
            raise ValueError('Read-only report outputs must be separate from fits/datasets')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    snapshot=args.output.resolve()/(stamp+'_job'+os.environ['SLURM_JOB_ID'])
    snapshot.mkdir(parents=True,exist_ok=False)
    result.update(reporter_sha256=sha(Path(__file__)),slurm_job_id=os.environ['SLURM_JOB_ID'],snapshot=str(snapshot))
    (snapshot/'reporter_source.py').write_bytes(Path(__file__).read_bytes())
    (snapshot/'plan.json').write_bytes(args.plan.read_bytes())
    atomic_json(snapshot/'report.json',clean(result))
    for filename,key in [('seed_summaries.csv','seed_summaries'),('multi_seed.csv','multi_seed'),
                         ('paired_objective_seed_deltas.csv','paired_objective_seed_deltas'),
                         ('paired_objective_summary.csv','paired_objective_summary'),('tasks.csv','tasks'),('fits.csv','fits')]:
        (snapshot/filename).write_text(csv_text(result[key]))
    files={p.name:sha(p) for p in snapshot.iterdir() if p.is_file()}
    atomic_json(snapshot/'completed.json',dict(status='completed',report_status=result['status'],files=files,
        plan_sha256=result['plan_sha256'],actor_advanced=False,backend_calls=0))
    atomic_json(args.output/'latest.json',dict(snapshot=str(snapshot),report_sha256=sha(snapshot/'report.json'),
        completed_sha256=sha(snapshot/'completed.json'),status=result['status']))
    print(json.dumps(dict(status=result['status'],snapshot=str(snapshot),verified_fits=result['verified_fits'],
                         completed_tasks=result['completed_tasks'],integrity_errors=result['integrity_errors'])),flush=True)


if __name__=='__main__':
    main()
