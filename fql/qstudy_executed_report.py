"""Saved-prediction paired report for the frozen executed-input diagnostic.

No model/checkpoint is loaded, no predictions are generated, and no actor or
physical flow runs. Recomputed TRAIN statistics only verify saved normalizers.
All five heads are positive costs, including max(-WNS,0) and max(-TNS,0).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import qstudy_executed_control as control
from . import qstudy_report as reporting

ROOT = Path(__file__).resolve().parents[1]
METRICS = reporting.METRICS
STATISTICS = ('mae','normalized_mae','spearman','kendall','family_pair_accuracy','top_regret')


def expected_binding(path, plan, task):
    return dict(dataset_sha256=task['dataset_sha256'], manifest_sha256=task['manifest_sha256'],
        cache_sha256=task['cache_sha256'], config=dict(output=task['output'],dataset=task['dataset'],
        variant=control.VARIANT,seed=task['seed'],**task['config']), source_sha256=plan['source_sha256'],
        fold_seed=control.FOLD_SEED,heldout_used=False,backend_calls=0,
        input_representation=control.INPUT,plan_sha256=reporting.sha(path),task=task)


def load_plan(path):
    path=Path(path).resolve()
    plan=reporting.read_json(path)
    if (plan['task_count']!=6 or plan['datasets']!=list(control.DATASETS) or plan['seeds']!=[0,1,2]
            or plan['variant']!=control.VARIANT or plan['config']!=control.CONFIG
            or plan['input_representation']!=control.INPUT):
        raise ValueError('Unsupported diagnostic plan')
    control.check_sources(plan['source_sha256'])
    for name,digest in plan['source_sha256'].items():
        if reporting.sha(path.parent/'source_snapshot'/name)!=digest:
            raise ValueError('Declared source snapshot changed: '+name)
    tasks_path=path.with_name('tasks.json')
    if reporting.sha(tasks_path)!=plan['tasks_sha256']:
        raise ValueError('Declared tasks changed')
    tasks=reporting.read_json(tasks_path)
    expected=[(dataset,seed) for dataset in control.DATASETS for seed in (0,1,2)]
    if len(tasks)!=6 or [(Path(t['dataset']).parent.name,t['seed']) for t in tasks]!=expected:
        raise ValueError('Six task identities differ')
    for task,(dataset,seed) in zip(tasks,expected):
        if (Path(task['output'])!=path.parent/dataset/f'seed_{seed}' or
                Path(task['dataset'])!=Path(plan['data_root'])/dataset/'dataset.npz' or
                Path(task['raw_output'])!=Path(plan['raw_matrix'])/dataset/control.VARIANT/f'seed_{seed}' or
                task['config']!=control.CONFIG or task['variant']!=control.VARIANT or
                task['input_representation']!=control.INPUT):
            raise ValueError('Task paths/config/input differ')
    return plan,tasks


def paired_fit(task, fold, arrays, binding):
    """Admit a fit only after its wrapper inventory and source labels validate."""
    folder=Path(task['output'])/fold['name']
    completed,verified=folder/'completed.json',folder/'verified.json'
    status=dict(dataset=Path(task['dataset']).parent.name,seed=task['seed'],fit=fold['name'])
    if not completed.exists() or not verified.exists():
        if verified.exists():
            raise ValueError('Verified inventory without completion')
        return dict(**status,status='missing_or_awaiting_inventory',
                    missing=[n for n in ('completed.json','verified.json') if not (folder/n).exists()]),[],None
    done=reporting.read_json(completed)
    observed=control.verify_prediction(folder,done,binding,fold,arrays)
    hashes={control.array_sha(row) for row in arrays['train']['x'][fold['train_indices']]}
    observed.update(raw_comparison=task['raw_comparison']['fits'][fold['name']],
        executed_target_rows_also_in_train=sum(control.array_sha(row) in hashes
            for row in arrays[fold['split']]['x'][fold['target_indices']]))
    if reporting.read_json(verified)!=observed:
        raise ValueError('Executed verified SHA/collision inventory changed')
    old_folder=Path(task['raw_output'])/fold['name']
    old=reporting.read_json(old_folder/'completed.json')
    raw_norm,exe_norm=old['normalization'],done['normalization']
    for key in ('target_mean','target_std','objective_scale','train_observed_count'):
        if raw_norm[key]!=exe_norm[key]:
            raise ValueError('Paired target normalization differs: '+key)
    with np.load(old_folder/'predictions.npz',allow_pickle=False) as p:
        raw=p['prediction'].copy()
    with np.load(folder/'predictions.npz',allow_pickle=False) as p:
        executed,truth,family=[p[k].copy() for k in ('prediction','truth','family')]
    rows=[]
    for k,metric in enumerate(METRICS):
        row=dict(**status,metric=metric,scope='historical_refit_development' if fold['name']=='refit' else 'grouped_train_cv')
        for arm,prediction in (('raw',raw),('executed',executed)):
            metrics=reporting.diagnostics(prediction[:,k],truth[:,k],family)
            metrics['normalized_mae']=(metrics['mae']/raw_norm['objective_scale'][k]
                if metrics['mae'] is not None else None)
            row[arm]=metrics
        row['delta_executed_minus_raw']={key:row['executed'][key]-row['raw'][key]
            if row['raw'].get(key) is not None and row['executed'].get(key) is not None else None
            for key in STATISTICS}
        rows.append(row)
    status.update(status='verified',executed_target_rows_also_in_train=observed['executed_target_rows_also_in_train'],
        artifact_sha256=dict(**observed,verified_sha256=reporting.sha(verified)),
        representation_effects=dict(feature_mean_changed=raw_norm['feature_mean']!=exe_norm['feature_mean'],
            feature_std_changed=raw_norm['feature_std']!=exe_norm['feature_std'],
            hpwl_prior_slope_changed=raw_norm['baseline_slope']!=exe_norm['baseline_slope'],
            hpwl_prior_intercept_changed=raw_norm['baseline_intercept']!=exe_norm['baseline_intercept']),
        target_normalization_identical=True)
    return status,rows,dict(name=fold['name'],**observed)


def aggregate(fit_rows,statuses):
    """Equal fold metrics inside a seed, then independent seed means/SD."""
    seeds=[]
    for dataset in control.DATASETS:
        for seed in (0,1,2):
            for scope,names in [('grouped_train_cv',{'cv_0','cv_1','cv_2'}),
                                ('historical_refit_development',{'refit'})]:
                states=[s for s in statuses if s['dataset']==dataset and s['seed']==seed and s['fit'] in names]
                complete=len(states)==len(names) and all(s['status']=='verified' and
                    s.get('admitted_for_metrics',True) for s in states)
                for metric in METRICS:
                    rows=[r for r in fit_rows if r['dataset']==dataset and r['seed']==seed and
                          r['scope']==scope and r['metric']==metric]
                    value=dict(dataset=dataset,seed=seed,scope=scope,metric=metric,
                        complete=complete,required_folds=len(names),available_folds=len(rows))
                    for arm in ('raw','executed','delta_executed_minus_raw'):
                        value[arm]={}
                        for statistic in STATISTICS:
                            values=[r[arm][statistic] for r in rows if r[arm].get(statistic) is not None]
                            value[arm][statistic]=dict(defined_folds=len(values),
                                mean=float(np.mean(values)) if complete and values else None)
                    seeds.append(value)
    summary=[]
    for dataset in control.DATASETS:
        for scope in ('grouped_train_cv','historical_refit_development'):
            for metric in METRICS:
                rows=[r for r in seeds if r['dataset']==dataset and r['scope']==scope and r['metric']==metric]
                for statistic in STATISTICS:
                    value=dict(dataset=dataset,scope=scope,metric=metric,statistic=statistic,
                        required_independent_seeds=3,complete_seeds=[r['seed'] for r in rows if r['complete']])
                    for arm in ('raw','executed','delta_executed_minus_raw'):
                        values=[r[arm][statistic]['mean'] for r in rows if r[arm][statistic]['mean'] is not None]
                        value[arm]=dict(defined_seeds=len(values),mean=float(np.mean(values)) if len(values)==3 else None,
                            sd=float(np.std(values,ddof=1)) if len(values)==3 else None)
                    summary.append(value)
    return seeds,summary


def build(plan_path):
    plan_path=Path(plan_path).resolve()
    plan,tasks=load_plan(plan_path)
    statuses,fit_rows,task_states,errors=[],[],[],[]
    for task in tasks:
        identity=dict(dataset=Path(task['dataset']).parent.name,seed=task['seed'])
        local_status,local_rows,inventories=[],[],[]
        try:
            current,arrays=control.make_task(task['dataset'],task['seed'],task['output'],
                plan['raw_matrix'],plan['source_sha256'],task['config'])
            if current!=task:
                raise ValueError('Declared dataset/raw comparator/folds changed')
            binding=expected_binding(plan_path,plan,task)
            folder=Path(task['output'])
            if (folder/'binding.json').exists() and reporting.read_json(folder/'binding.json')!=binding:
                raise ValueError('Executed task binding changed')
            for fold in task['folds']:
                try:
                    status,rows,inventory=paired_fit(task,fold,arrays,binding)
                    local_status.append(status)
                    local_rows.extend(rows)
                    if inventory is not None:inventories.append(inventory)
                except (OSError,ValueError,KeyError,TypeError) as error:
                    local_status.append(dict(**identity,fit=fold['name'],status='invalid',error=str(error)))
                    errors.append(dict(**identity,fit=fold['name'],error=str(error)))
            final_path=folder/'completed.json'
            task_complete=False
            if final_path.exists():
                final=reporting.read_json(final_path)
                if (final['status']!='completed' or final['binding']!=binding or len(inventories)!=4 or
                        final['fits']!=inventories or final['backend_calls']!=0 or final['actor_advanced']):
                    raise ValueError('Executed four-fit task completion differs')
                task_complete=True
            task_states.append(dict(**identity,status='completed' if task_complete else 'incomplete',
                declared_failure_accounting=task['accounting'],raw_comparison=task['raw_comparison'],
                executed_completion_sha256=reporting.sha(final_path) if final_path.exists() else None))
        except (OSError,ValueError,KeyError,TypeError) as error:
            errors.append(dict(**identity,fit=None,error=str(error)))
            local_rows=[]
            local_status=[dict(**identity,fit=f['name'],status='invalid',error=str(error)) for f in task['folds']]
            task_states.append(dict(**identity,status='invalid',error=str(error)))
        invalid=any(s['status']=='invalid' for s in local_status)
        for status in local_status:status['admitted_for_metrics']=not invalid
        if invalid:
            local_rows=[]  # No task-level aggregate from partially corrupted artifacts.
        statuses.extend(local_status)
        fit_rows.extend(local_rows)
    seed_rows,summary=aggregate(fit_rows,statuses)
    complete=sum(t['status']=='completed' for t in task_states)
    return dict(schema='qstudy_executed_paired_report_v1',
        status='invalid_artifacts' if errors else 'complete' if complete==6 else 'incomplete',
        plan=dict(path=str(plan_path),sha256=reporting.sha(plan_path),tasks_sha256=plan['tasks_sha256']),
        reporter_sha256=reporting.sha(__file__),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        report_dependency_sha256={name:reporting.sha(ROOT/name) for name in
            ('fql/qstudy_report.py','fql/qstudy_executed_control.py')},
        tasks_complete=complete,tasks_required=6,task_status=task_states,fit_status=statuses,
        integrity_errors=errors,fold_metrics=fit_rows,seed_metrics=seed_rows,three_seed_summary=summary,
        primary_metrics=['WNS_ns','TNS_ns'],metrics=list(METRICS),
        definitions=dict(WNS_ns='positive violation max(-native_WNS_ns,0)',
            TNS_ns='positive violation max(-native_TNS_ns,0)',all_costs='smaller is better'),
        aggregation='equal within-seed fold scores, then three independent seed means and sample SD; no cross-fold rank pooling',
        historical_refit='Previously inspected development data, separately reported; not fresh test',
        caveats=['Coordinate-derived features, TRAIN feature normalization and TRAIN-fitted smooth-HPWL prior all change',
            'Original rows/family folds retained, including repeated executed coordinates; collision rows disclosed per fit',
            'This is a prediction comparison, not legalizer differentiability, actor-gradient validity or causal proof',
            'Undefined ranks/pair scores stay null; incomplete seed aggregates are withheld',
            'Five cost heads predicted; power and area remain source-only metrics'],
        layout_generation_calls=0,model_forward_calls=0,critic_refits=0,backend_calls=0,
        dataset_mutations=0,actor_advanced=False,winner_selected=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    control.require_slurm()
    plan,tasks=load_plan(args.plan)
    output=args.output.resolve()
    protected=[args.plan.parent.resolve(),Path(plan['raw_matrix']).resolve(),Path(plan['data_root']).resolve()]
    if any(output==x or x in output.parents or output in x.parents for x in protected):
        raise ValueError('Reports must be separate from source artifacts')
    result=build(args.plan)
    output.mkdir(parents=True,exist_ok=False)
    reporting.atomic_text(output/'report.json',reporting.json_text(result))
    for key in ('fold_metrics','seed_metrics','three_seed_summary'):
        reporting.atomic_text(output/(key+'.csv'),reporting.csv_text(result[key]))
    print(json.dumps(dict(status=result['status'],tasks_complete=result['tasks_complete'],
        integrity_errors=len(result['integrity_errors']),output=str(output))),flush=True)


if __name__=='__main__':main()
