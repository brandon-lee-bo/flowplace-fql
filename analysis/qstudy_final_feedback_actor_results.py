"""Saved-only TRAIN64 final-feedback actor comparison:144 GRT and72 DRT.

Eight GRT noises and four final-route noises have separate denominators.
Critic calibration and all six preference utilities require the corresponding
audited final result. No fitting, inference, EDA, replay or automatic promotion.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import torch

from analysis.placement_metrics import MacroHPWL
from analysis.qstudy_rule_audit import rules, structure
from analysis import qstudy_rule_reward_actor_results as shared
from fql import qstudy_backend as backend
from fql import qstudy_final_feedback_actor as actor
from fql import qstudy_final_feedback_screen as screen
from fql import qstudy_actor_report as native
from fql import qstudy_flow_baseline_report as baseline
from fql import qstudy_final_route as final
from fql.graph_o2o_backend import load_requests
from fql.graph_o2o_archive import atomic_json
from fql.qstudy_data import bind, sha
from fql.qstudy_report import csv_text


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0,1,2)
GRT_INDICES = tuple(range(8))
DRT_INDICES = tuple(range(4))
CONTROLS = ('grt_wl','drt_wl')
FINAL_ADMITTED = {'completed','upstream_physical_failure'}
SOURCE_NAMES = ('analysis/qstudy_final_feedback_actor_results.py',
    'analysis/qstudy_final_feedback_actor_results.slurm',
    'analysis/qstudy_rule_reward_actor_results.py','analysis/qstudy_rule_reward_results.py',
    'fql/qstudy_final_feedback_actor.py','fql/qstudy_final_feedback_screen.py',
    'fql/qstudy_final_feedback_route.py','fql/qstudy_final_route.py',
    'fql/qstudy_actor_report.py','fql/qstudy_flow_baseline_report.py',
    'fql/qstudy_backend.py','fql/graph_o2o_backend.py','fql/graph_o2o_final_evidence.py',
    'analysis/qstudy_final_power_audit.py','analysis/qstudy_rule_audit.py',
    'docs/qstudy_final_feedback_protocol.md')
read,eligible_stats = shared.read,shared.eligible_stats


def route_api():
    # The separately owned adapter is metadata-only; its plan validation fixes
    # all72 requests. Lazy import permits isolated accounting software fixtures.
    from fql import qstudy_final_feedback_route
    return qstudy_final_feedback_route


def checked_training(folder,cfg,seed,refit,common):
    """Reuse the strict saved1000-step/frozen-state checker, never a forward."""
    view=dict(cfg,rule_behavior=cfg['final_feedback_behavior'],rule_common=cfg['final_feedback_common'])
    training=shared.checked_training(folder,view,seed,refit,common)
    if training.get('completed'):
        training['pairing']['critic_source_inputs']=cfg['final_feedback_critic_inputs']
        training['critic_training_rows']=64
        training['behavior_targets']=256
        training['physical_context_rows']=384
    return training


def checked_published_config(cfg,preparation,inputs,root,refit,seed):
    """Bind separate final64 labels, original384 physical context and256 BC."""
    if (cfg['final_feedback_plan']!=preparation['plan'] or
        cfg['final_feedback_behavior']!=preparation['behavior'] or
        cfg['final_feedback_common']!=preparation['common'] or
        cfg['final_feedback_sources']!=preparation['source_sha256'] or
        cfg['final_feedback_critic_inputs']!=preparation['critic_inputs'] or
        cfg['final_feedback_critic_inputs']['inputs']!=inputs or
        cfg['final_feedback_evaluation']!=actor.FINAL_EVALUATION or
        preparation['evaluation']!=actor.FINAL_EVALUATION or
        Path(cfg['final_feedback_preparation_root']).resolve()!=Path(root).resolve() or
        cfg['dataset_path']!=inputs['dataset_path'] or cfg['dataset_sha256']!=inputs['dataset_sha256'] or
        cfg['cache_sha256']!=inputs['cache']['sha256'] or
        cfg['final_feedback_utility_spec']!=refit['spec'] or
        cfg['critic_checkpoint_sha256'][str(seed)]!=refit['checkpoint']['sha256'] or
        Path(cfg['critic_checkpoints'][str(seed)]).resolve()!=Path(refit['checkpoint']['path']).resolve() or
        len(set(cfg['common_checkpoint_sha256'].values()))!=3):
        raise ValueError('Published final64/behavior256/physical384/refit binding differs')


def collect_grt(experiment,cfg,seed,refit,common,context):
    folder=experiment/f'seed_{seed}'/'round_0'
    errors=[]
    try:
        training=checked_training(folder,cfg,seed,refit,common)
    except Exception as exc:
        training=dict(completed=False,integrity_error=str(exc));errors.append('training: '+str(exc))
    tasks=folder/'requests/tasks.jsonl'
    rows=[]
    if tasks.exists() and tasks.with_name('completed.json').exists():
        try:
            physical=backend.context(experiment)
            requests,tasks=load_requests(experiment,seed,0)
            completion=read(tasks.with_name('completed.json'))
            if (len(requests)!=8 or [r['index'] for r in requests]!=list(GRT_INDICES) or
                completion.get('selection')!='all predetermined noises; no Q-selection' or
                completion.get('evaluation_never_replayed') is not True or not training.get('completed')):
                raise ValueError('Completed offline export must retain all eight eval-only noises')
            movable=context['movable'].cpu().numpy().astype(bool)
            for request in requests:
                row=dict(seed=seed,index=request['index'],noise_sha256=request.get('noise_sha256'),
                    action_sha256=request.get('action_sha256'),raw_action_sha256=request.get('raw_action_sha256'),
                    state='invalid',metrics=None,integrity_error=None,predicted_q=None,request=None)
                try:
                    sample=shared.verify_request(request,cfg,seed,training,physical)
                    row.update(baseline.inspect_outcome(request,sample,tasks,physical))
                    if row['state']=='success' and any(row['metrics'][k]<0 for k in native.METRICS
                                                      if k not in ('WNS_ns','TNS_ns')):
                        raise ValueError('Nonnegative GRT native metric is negative')
                    predicted=np.asarray(request.get('predicted_q'),float)
                    if predicted.shape!=(1,) or not np.isfinite(predicted).all():
                        raise ValueError('Exported scalar predicted Q differs')
                    raw=np.asarray(request['raw_action']);executed=np.asarray(request['executed_action'])
                    row.update(predicted_q=float(predicted[0]),request=bind(sample/'request.json'),
                        tasks=bind(tasks),actions=bind(sample/'actions.npz'),
                        executed_action=executed.tolist(),
                        legalizer_jump_rms=float(np.sqrt(np.mean((raw[movable]-executed[movable])**2))),
                        geometry_unit='RMS over movable normalized [-1,1] coordinates')
                except Exception as exc:
                    row.update(state='invalid',metrics=None,integrity_error=str(exc))
                rows.append(row)
        except Exception as exc:
            errors.append('export: '+str(exc))
    return grt_summary(cfg['final_feedback_objective'],seed,rows,training,errors,tasks if tasks.exists() else None)


def grt_summary(objective,seed,rows,training,errors,tasks=None):
    rows=sorted(rows,key=lambda r:r['index'])
    if rows and ([r['index'] for r in rows]!=list(GRT_INDICES) or
                 len({r['noise_sha256'] for r in rows})!=8):
        raise ValueError('GRT stream must retain all eight distinct actual noises')
    result=baseline.summarize(rows,8)
    if errors or not training.get('completed'):
        result['fully_accounted']=False
    return dict(objective=objective,arm=actor.ARMS[objective][0],seed=seed,round=0,
        training=training,errors=errors,tasks=bind(tasks) if tasks else None,rows=rows,
        actual_noise_by_index={str(r['index']):r['noise_sha256'] for r in rows},
        integrity_errors=errors+[r['integrity_error'] for r in rows if r['state']=='invalid'],
        projection_rms=native.describe([r.get('legalizer_jump_rms') for r in rows]),
        final_reward_not_planned_indices=list(GRT_INDICES[4:]),
        reward_note='No own/final B reward inferred for the four GRT-only indices; even grt_wl shares final feasibility',
        **result)


def measured_rewards(grt,drt,executed,context,specs):
    """Actual final feasibility for ALL targets; never backend.outcome.reward."""
    empty={o:None for o in screen.OBJECTIVES}
    state=drt['state']
    if state=='upstream_physical_failure':
        if grt['state'] not in ('physical_failure','geometry_failure') or drt.get('metrics') is not None:
            raise ValueError('Unproven final upstream failure cannot receive B4')
        return dict(feasible=False,costs=None,linear_costs=dict(empty),
                    bounded_costs={o:4. for o in empty},rewards={o:-4. for o in empty})
    if state!='completed':
        return dict(feasible=None,costs=None,linear_costs=dict(empty),bounded_costs=dict(empty),rewards=dict(empty))
    if grt['state']!='success':
        raise ValueError('Completed DRT has no successful exact GRT source')
    values=drt['metrics']
    v=np.asarray([values[k] for k in final.METRICS],float)
    if (not np.isfinite(v).all() or (v[[0,3,4,5]]<0).any() or v[5]!=np.floor(v[5])):
        raise ValueError('Actual final six native metrics are malformed')
    geometry,_=structure(context['cache'])
    score=rules(np.asarray(executed),geometry,MacroHPWL(context['cache']))
    costs=np.asarray([grt['metrics']['rWL_um'],values['DRT_WL'],max(-values['TNS'],0.),
                      values['Power'],values['StdCellArea'],*[score[k] for k in screen.EXPERT_NAMES]],float)
    if not np.isfinite(costs).all() or (costs<0).any():
        raise ValueError('Completed native/rule costs must be finite nonnegative')
    feasible=values['DRC']==0
    linear,bounded,rewards={},{},{}
    for objective,spec in specs.items():
        screen.validate_spec(spec)
        c=float(np.dot(costs/np.asarray(spec['scales']),spec['weights']))
        linear[objective]=c
        bounded[objective]=2*c/(1+c) if feasible else 4.
        rewards[objective]=-bounded[objective]
    return dict(feasible=feasible,costs=dict(zip(screen.COST_NAMES,costs.tolist())),
                linear_costs=linear,bounded_costs=bounded,rewards=rewards)


def final_summary(objective,seed,rows,training,errors):
    rows=sorted(rows,key=lambda r:r['index'])
    if ([r['index'] for r in rows]!=list(DRT_INDICES) or
        any(r['objective']!=objective or r['seed']!=seed for r in rows)):
        raise ValueError('Final stream must retain exact four predeclared indices')
    counts=Counter(r['state'] for r in rows)
    complete=not errors and training.get('completed',False) and sum(counts[s] for s in FINAL_ADMITTED)==4
    completed=[r for r in rows if r['state']=='completed']
    feasible=[r for r in rows if r.get('feasible') is True]
    own=[r for r in rows if r.get('predicted_q') is not None and r['rewards'][objective] is not None]
    p=np.asarray([r['predicted_q'] for r in own],float)
    y=np.asarray([r['rewards'][objective] for r in own],float)
    return dict(objective=objective,arm=actor.ARMS[objective][0],seed=seed,round=0,planned=4,
        counts={k:counts[k] for k in ('completed','upstream_physical_failure','missing','missing_final',
            'source_missing','source_invalid','invalid')},fully_accounted=bool(complete),
        completed=len(completed),DRC_zero=len(feasible),DRC_positive=sum(r.get('feasible') is False for r in completed),
        conditional_PPA=len(completed)!=4,training=training,rows=rows,errors=errors,
        integrity_errors=errors+[r.get('reason','invalid final evidence') for r in rows if r['state'] in ('invalid','source_invalid')],
        actual_noise_by_index={str(r['index']):r.get('noise_sha256') for r in rows},
        completed_native_metric_means={k:native.mean([r['metrics'][k] for r in completed]) for k in final.METRICS},
        DRC0_only_native_metric_means={k:native.mean([r['metrics'][k] for r in feasible]) for k in final.METRICS},
        preference_reward_means={o:native.mean([r['rewards'][o] for r in rows]) if complete else None for o in screen.OBJECTIVES},
        preference_linear_C_completed_means={o:native.mean([r['linear_costs'][o] for r in completed]) for o in screen.OBJECTIVES},
        projection_rms=native.describe([r.get('legalizer_jump_rms') for r in rows]),
        own_q_calibration=dict(n=len(own),planned=4,complete=bool(complete and len(own)==4),
            mae=float(np.abs(p-y).mean()) if len(own) else None,
            signed_prediction_minus_measurement=float((p-y).mean()) if len(own) else None,
            pearson=float(np.corrcoef(p,y)[0,1]) if len(own)>2 and np.ptp(p)>0 and np.ptp(y)>0 else None,
            interpretation='own fullTRAIN64 B target; predicted raw-action Q versus executed and true final metrics; four nested development noises'))


def collect_final(route_experiment,route_plan,grt_streams,context,specs):
    api=route_api()
    sources=api.inspect_sources(route_experiment,route_plan,seal=False)
    if len(sources)!=72:
        raise ValueError('Final source resolution must retain72 task records')
    by_grt={(s['arm'],s['seed'],r['index']):(s,r) for s in grt_streams for r in s['rows']}
    all_rows=[]
    for task,source in zip(route_plan['tasks'],sources):
        objective=next(o for o in screen.OBJECTIVES if actor.ARMS[o][0]==task['arm'])
        empty={o:None for o in screen.OBJECTIVES}
        row=dict(objective=objective,seed=task['seed'],index=task['noise_index'],state='source_missing',
            metrics=None,noise_sha256=None,feasible=None,costs=None,linear_costs=dict(empty),
            bounded_costs=dict(empty),rewards=dict(empty),predicted_q=None)
        try:
            inspected=final.inspect_final(route_experiment,route_plan,task,source)
            row.update(inspected)
            original=by_grt.get((task['arm'],task['seed'],task['noise_index']))
            if original is None:
                if row['state'] in FINAL_ADMITTED:
                    raise ValueError('Final result has no verified actor GRT request')
            else:
                stream,grt=original
                if row['state'] in FINAL_ADMITTED:
                    if (not stream['training'].get('completed') or grt['state'] not in native.ADMITTED or
                        row['noise_sha256']!=grt['noise_sha256'] or
                        source['binding']['raw_action_sha256']!=grt['raw_action_sha256'] or
                        source['binding']['action_sha256']!=grt['action_sha256'] or
                        source['binding']['request_sha256']!=grt['request']['sha256'] or
                        source['binding']['actor']['checkpoint_sha256']!=stream['training']['checkpoint_sha256']):
                        raise ValueError('Final source differs from independently checked actor/noise/raw/executed evidence')
                row.update(predicted_q=grt.get('predicted_q'),legalizer_jump_rms=grt.get('legalizer_jump_rms'),
                    grt_state=grt['state'],grt_metrics=grt['metrics'],grt_request=grt.get('request'),
                    raw_action_sha256=grt['raw_action_sha256'],action_sha256=grt['action_sha256'])
                row.update(measured_rewards(grt,row,grt.get('executed_action'),context,specs))
        except Exception as exc:
            row.update(state='invalid',metrics=None,reason=str(exc),feasible=None,costs=None,
                linear_costs=dict(empty),bounded_costs=dict(empty),rewards=dict(empty))
        all_rows.append(row)
    identities={json.dumps(r['physical_identity'],sort_keys=True) for r in all_rows if r['state']=='completed'}
    runs=[r['physical_run'] for r in all_rows if r['state']=='completed']
    duplicates={run for run in runs if runs.count(run)>1}
    if len(identities)>1 or duplicates:
        for row in all_rows:
            if row['state']=='completed' and (len(identities)>1 or row['physical_run'] in duplicates):
                row.update(state='invalid',metrics=None,reason='Final native identity differs or distinct tasks reuse one physical run',
                    feasible=None,costs=None,linear_costs={o:None for o in screen.OBJECTIVES},
                    bounded_costs={o:None for o in screen.OBJECTIVES},rewards={o:None for o in screen.OBJECTIVES})
    streams=[]
    for grt in grt_streams:
        rows=[r for r in all_rows if r['objective']==grt['objective'] and r['seed']==grt['seed']]
        streams.append(final_summary(grt['objective'],grt['seed'],rows,grt['training'],grt['errors']))
    return streams,sources


def aggregate(streams,stage):
    metrics=native.METRICS if stage=='GRT' else final.METRICS
    result=[]
    for objective in screen.OBJECTIVES:
        selected=[s for s in streams if s['objective']==objective]
        okay=(len(selected)==3 and {s['seed'] for s in selected}==set(SEEDS) and
            all(s['fully_accounted'] and not s['integrity_errors'] for s in selected))
        field='successful_native_metric_means' if stage=='GRT' else 'completed_native_metric_means'
        row=dict(objective=objective,stage=stage,planned=24 if stage=='GRT' else 12,
            planned_seeds=list(SEEDS),independent_seed_n=3 if okay else 0,aggregate_eligible=okay,
            conditional_PPA=any(s['conditional_PPA'] for s in selected),
            native_metrics={k:eligible_stats([s[field][k] for s in selected],3) for k in metrics} if okay else None)
        if stage=='GRT':
            row['counts']={k:sum(s[k] for s in selected) for k in ('planned','exported','not_exported','success',
                'physical_failure','geometry_failure','missing','invalid')}
        else:
            row['counts']={k:sum(s['counts'][k] for s in selected) for k in selected[0]['counts']} if selected else {}
            row['DRC_zero']=sum(s['DRC_zero'] for s in selected)
            row['DRC_positive']=sum(s['DRC_positive'] for s in selected)
            row['all_preference_rewards']={o:eligible_stats([s['preference_reward_means'][o] for s in selected],3)
                for o in screen.OBJECTIVES} if okay else None
        result.append(row)
    return result


def paired(streams,objective,control,stage):
    indices=GRT_INDICES if stage=='GRT' else DRT_INDICES
    metrics=native.METRICS if stage=='GRT' else final.METRICS
    good='success' if stage=='GRT' else 'completed'
    local_seeds,all_pairs=[],[]
    for seed in SEEDS:
        left=next(s for s in streams if s['objective']==objective and s['seed']==seed)
        right=next(s for s in streams if s['objective']==control and s['seed']==seed)
        rows_left={r['index']:r for r in left['rows']};rows_right={r['index']:r for r in right['rows']}
        pairs=[]
        for index in indices:
            if index not in rows_left or index not in rows_right:
                continue
            x,y=rows_left[index],rows_right[index]
            if x.get('noise_sha256') and y.get('noise_sha256') and x['noise_sha256']!=y['noise_sha256']:
                raise ValueError('Paired actual saved noise arrays differ')
            both=x['state']==y['state']==good
            if both and (not x.get('noise_sha256') or not y.get('noise_sha256')):
                raise ValueError('Completed paired rows require both actual saved noise identities')
            pair=dict(seed=seed,index=index,noise_sha256=x.get('noise_sha256'),left_state=x['state'],right_state=y['state'],
                both_native_completed=both,deltas={k:x['metrics'][k]-y['metrics'][k] for k in metrics} if both else None)
            if stage=='DRT':
                pair['preference_reward_deltas']={o:x['rewards'][o]-y['rewards'][o]
                    if x['rewards'][o] is not None and y['rewards'][o] is not None else None for o in screen.OBJECTIVES}
                pair['both_DRC0']=x.get('feasible') is True and y.get('feasible') is True
            pairs.append(pair)
        ready=(len(pairs)==len(indices) and left['fully_accounted'] and right['fully_accounted'] and
            not left['integrity_errors'] and not right['integrity_errors'])
        completed=[p for p in pairs if p['both_native_completed']]
        item=dict(seed=seed,planned=len(indices),paired=len(pairs),fully_accounted=bool(ready),
            both_native_completed=len(completed),conditional_PPA=len(completed)!=len(indices),
            native_deltas={k:native.mean([p['deltas'][k] for p in completed]) for k in metrics} if ready else None)
        if stage=='DRT':
            item['preference_reward_deltas']={o:native.mean([p['preference_reward_deltas'][o] for p in pairs])
                if ready else None for o in screen.OBJECTIVES}
        local_seeds.append(item);all_pairs.extend(pairs)
    ready=all(s['fully_accounted'] for s in local_seeds)
    result=dict(objective=objective,control=control,stage=stage,direction='objective minus control in native units',
        planned_pairs=len(indices)*3,paired=len(all_pairs),fully_accounted=ready,seeds=local_seeds,pairs=all_pairs,
        conditional_PPA=any(s['conditional_PPA'] for s in local_seeds),
        three_seed_native_deltas={k:eligible_stats([s['native_deltas'][k] if s['native_deltas'] else None for s in local_seeds],3)
            for k in metrics} if ready else None)
    if stage=='DRT':
        result['three_seed_preference_reward_deltas']={o:eligible_stats([s['preference_reward_deltas'][o] for s in local_seeds],3)
            for o in screen.OBJECTIVES} if ready else None
    return result


def verify_pairing(grt_streams):
    noises=[s['actual_noise_by_index'] for s in grt_streams if len(s['rows'])==8]
    if noises and any(n!=noises[0] for n in noises[1:]):
        raise ValueError('Actual eight noise arrays differ across six targets/three seeds')
    rows=[]
    for seed in SEEDS:
        completed=[s for s in grt_streams if s['seed']==seed and s['training'].get('completed')]
        identities=[s['training']['pairing'] for s in completed]
        if identities and any(identity!=identities[0] for identity in identities[1:]):
            raise ValueError('Common initialization/settings/64source/RNG endpoints differ across targets')
        rows.append(dict(seed=seed,verified_objectives=[s['objective'] for s in completed],
            all_six_available=len(completed)==6,identity=identities[0] if identities else None,
            caveat='same fixed update algorithm/settings and RNG endpoints, not an independently recorded full per-step drawchain'))
    return rows


def collect(experiments_root,route_experiment,route_plan_sha256):
    root,route_experiment=Path(experiments_root).resolve(),Path(route_experiment).resolve()
    preparation=read(root/'preparation.json')
    if (preparation['objectives']!=list(screen.OBJECTIVES) or preparation['automatic_selection'] is not False or
        preparation['schema']!=actor.SCHEMA or preparation['status']!='prepared_configs_only' or
        preparation['source_sha256']!=actor.sources() or preparation['online_implemented'] is not False or
        preparation['backend_calls']!=0 or preparation['actor_jobs_submitted']!=0 or
        preparation['evaluation']!=actor.FINAL_EVALUATION):
        raise ValueError('Fixed six-objective offline preparation differs')
    loaded=screen.checked_plan(Path(preparation['plan']['path']),preparation['plan']['sha256'])
    common,specs,refits,configs,evidence={},{},{},{},{}
    for objective in screen.OBJECTIVES:
        experiment=root/actor.ARMS[objective][0]
        for seed in SEEDS:
            cfg=actor.validate_config(experiment,seed)
            if cfg['final_feedback_objective']!=objective or preparation['configs'][cfg['arm']]!=bind(experiment/'config.json'):
                raise ValueError('Objective/config/preparation publication differs')
            result=screen.checked_refit(Path(cfg['final_feedback_plan']['path']),objective,seed,
                cfg['final_feedback_plan']['sha256'],loaded=loaded)
            checked_published_config(cfg,preparation,loaded[0]['inputs'],root,result,seed)
            for key in ('spec','checkpoint','completion','normalization'):
                if result[key]!=cfg['final_feedback_refits'][str(seed)][key]:
                    raise ValueError('Prepared final64 full-refit changed: '+key)
            if objective in specs and result['spec']!=specs[objective]:
                raise ValueError('FullTRAIN scales depend on model seed')
            specs[objective]=result['spec'];refits[(objective,seed)]=result
            configs[(objective,seed)]=(experiment,cfg);evidence.update(result['evidence'])
            if seed not in common:
                source=cfg['final_feedback_common'][str(seed)]['checkpoint'];bind(source['path'],source['sha256'])
                common[seed]=torch.load(source['path'],map_location='cpu',weights_only=False)
    route_plan=route_api().validate_plan(route_experiment,route_plan_sha256)
    if (route_plan['requested']!=72 or len(route_plan['tasks'])!=72 or
        route_plan['evaluation_indices']!=list(DRT_INDICES) or route_plan['rounds']!=[0] or
        route_plan['actor_seeds']!=list(SEEDS) or route_plan['detailed_route_end_iteration']!=20):
        raise ValueError('Final fixed72 first-four evaluation budget differs')
    expected_tasks=[(actor.ARMS[o][0],s,0,i) for o in screen.OBJECTIVES for s in SEEDS for i in DRT_INDICES]
    if [(t['arm'],t['seed'],t['round'],t['noise_index']) for t in route_plan['tasks']]!=expected_tasks:
        raise ValueError('Final plan task denominator/order differs')
    for objective in screen.OBJECTIVES:
        experiment,cfg=configs[(objective,0)]
        expected=dict(path=str(experiment),config_sha256=sha(experiment/'config.json'),config=cfg)
        if route_plan['actor_experiments'][cfg['arm']]!=expected:
            raise ValueError('Final route plan uses a different policy configuration')
    grt_streams=[];physical=[]
    for (objective,seed),(experiment,cfg) in configs.items():
        grt_streams.append(collect_grt(experiment,cfg,seed,refits[(objective,seed)],common[seed],loaded[2]))
        physical.append({k:v for k,v in backend.context(experiment)[2].items() if k!='config_sha256'})
    if any(value!=physical[0] for value in physical[1:]):
        raise ValueError('Native physical settings differ across final-feedback actors')
    pairing=verify_pairing(grt_streams)
    drt_streams,sources=collect_final(route_experiment,route_plan,grt_streams,loaded[2],specs)
    complete=all(s['fully_accounted'] and not s['integrity_errors'] for s in grt_streams+drt_streams)
    invalid=any(s['integrity_errors'] for s in grt_streams+drt_streams)
    contrasts=[paired(streams,o,c,stage) for stage,streams in (('GRT',grt_streams),('DRT',drt_streams))
        for c in CONTROLS for o in screen.OBJECTIVES if o!=c]
    return dict(schema='qstudy_final_feedback_actor_native_results_v1',
        status='integrity_error' if invalid else 'completed' if complete else 'partial',
        preparation=bind(root/'preparation.json'),screen_plan=preparation['plan'],
        route_plan=bind(route_experiment/'full_route_plan.json',route_plan_sha256),
        physical_context=physical[0],expected_arms=6,expected_training_seeds=18,
        expected_GRT_requests=144,expected_DRT_requests=72,
        GRT_counts={k:sum(s[k] for s in grt_streams) for k in ('planned','exported','not_exported','success',
            'physical_failure','geometry_failure','missing','invalid')},
        DRT_counts=dict(Counter(r['state'] for s in drt_streams for r in s['rows'])),
        GRT_streams=grt_streams,DRT_streams=drt_streams,GRT_multi_seed=aggregate(grt_streams,'GRT'),
        DRT_multi_seed=aggregate(drt_streams,'DRT'),paired_vs_new_controls=contrasts,
        paired_training=pairing,all_reward_specs=specs,critic_evidence=evidence,
        route_source_resolution=sources,power_training_QA=loaded[0]['inputs'].get('power_audit'),
        power_training_QA_completion=loaded[0]['inputs'].get('power_audit_completion'),
        conventions=dict(controls='new matched64 ff_grt_wl and ff_drt_wl actors; no causal oldBC reuse',
            denominator='GRT eight and final DRT four actual noises per actor seed; utilities/calibration use final-four only',
            uncertainty='three equal training-seed means; nested generation noises are not independent training runs; SD is not CI',
            reward='all six B targets recomputed from executed geometry and audited final PPA; generic backend reward is ignored',
            feasibility='DRC>0 sets B4 in all six objectives while completed native PPA/status remains visible; sealed upstream failures B4/no fakePPA; infra/missing NA',
            power='OpenSTA fixed default-activity estimate, not measured silicon/workload energy; TRAIN QA does not prove workload annotation',
            geometry='rules follow saved executed_action used by labels; native macro snapping is audited and final DEF preserves actual GRT macros',
            congestion='seven GRT metrics and six final metrics stay distinct; no invented final congestion',
            gradients='logged gradient(alphaD) versus gradient(+Q); improving direction cosine is its negative',
            promotion='all six declared targets retained; no automatic checkpoint/policy selection, fresh promotion or online updates'),
        backend_calls=0,model_inference_calls=0,actor_advanced=False,heldout_used=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('experiments-root','route-experiment','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--route-plan-sha256',required=True)
    args=parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Read-only numerical aggregation requires CPU Slurm')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',1)))
    root,route_root,output=args.experiments_root.resolve(),args.route_experiment.resolve(),args.output.resolve()
    cfg=read(root/actor.ARMS['grt_wl'][0]/'config.json')
    for protected in (root,route_root,Path(cfg['dataset_path']).resolve().parent,
                      Path(cfg['final_feedback_plan']['path']).resolve().parent):
        if output==protected or output in protected.parents or protected in output.parents:
            raise ValueError('Report output must be separate from experiment/data/screen/route originals')
    result=collect(root,route_root,args.route_plan_sha256)
    snapshot=output/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_job'+os.environ['SLURM_JOB_ID'])
    snapshot.mkdir(parents=True,exist_ok=False)
    sources={name:sha(ROOT/name) for name in SOURCE_NAMES}
    for name,digest in sources.items():
        destination=snapshot/'source_snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes());bind(destination,digest)
    result.update(snapshot=str(snapshot),slurm_job_id=os.environ['SLURM_JOB_ID'],reporter_sources=sources)
    atomic_json(snapshot/'report.json',native.clean(result))
    for name in ('GRT_streams','DRT_streams','GRT_multi_seed','DRT_multi_seed','paired_training','paired_vs_new_controls'):
        (snapshot/(name+'.csv')).write_text(csv_text(result[name]))
    files={str(path.relative_to(snapshot)):sha(path) for path in snapshot.rglob('*') if path.is_file()}
    atomic_json(snapshot/'completed.json',dict(status='completed',report_status=result['status'],files=files,
        backend_calls=0,model_inference_calls=0,actor_advanced=False))
    atomic_json(output/'latest.json',dict(report=bind(snapshot/'report.json'),completion=bind(snapshot/'completed.json'),status=result['status']))
    print(json.dumps(dict(status=result['status'],snapshot=str(snapshot),report=bind(snapshot/'report.json'),
                         GRT_counts=result['GRT_counts'],DRT_counts=result['DRT_counts'])),flush=True)


if __name__=='__main__':main()
