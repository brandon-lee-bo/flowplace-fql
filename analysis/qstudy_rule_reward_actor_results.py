"""Read-only native GRT comparison of the eight fixed rule-reward actors.

All192 requests remain in the denominator. Physical PPA is conditional on
success; independent training-seed means and same-noise differences are kept
separate from missing/invalid/failed requests. No old BC causal control, model
forward, fitting, EDA, replay, fresh selection or automatic promotion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from analysis.placement_metrics import MacroHPWL
from analysis.qstudy_rule_audit import rules, structure
from analysis.qstudy_rule_reward_results import state_sha
from fql import qstudy_backend as backend
from fql import qstudy_rule_reward_actor as actor
from fql import qstudy_rule_reward_screen as screen
from fql import qstudy_actor_report as native
from fql import qstudy_flow_baseline_report as baseline
from fql.graph_o2o_backend import load_requests
from fql.graph_o2o_archive import atomic_json
from fql.qstudy_data import bind, sha
from fql.qstudy_report import csv_text


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0,1,2)
COUNT = 8
CONTROLS = ('grt_wl','macro_hpwl')
SOURCE_NAMES = ('analysis/qstudy_rule_reward_actor_results.py',
    'analysis/qstudy_rule_reward_actor_results.slurm','analysis/qstudy_rule_reward_results.py',
    'fql/qstudy_rule_reward_actor.py','fql/qstudy_rule_reward_screen.py',
    'fql/qstudy_actor_report.py','fql/qstudy_flow_baseline_report.py',
    'fql/qstudy_backend.py','fql/graph_o2o_backend.py','docs/qstudy_rule_reward_protocol.md')


def read(path):
    return json.loads(Path(path).read_text())


def tree_equal(actual,expected,label):
    if isinstance(expected,torch.Tensor):
        if not isinstance(actual,torch.Tensor) or actual.dtype!=expected.dtype or not torch.equal(actual,expected):
            raise ValueError('Saved frozen state differs: '+label)
    elif isinstance(expected,dict):
        if not isinstance(actual,dict) or actual.keys()!=expected.keys():
            raise ValueError('Saved frozen state keys differ: '+label)
        for k in expected:tree_equal(actual[k],expected[k],label+'/'+str(k))
    elif actual!=expected:
        raise ValueError('Saved frozen state metadata differs: '+label)


def checked_training(folder,cfg,seed,refit,common):
    """Inspect saved checkpoint states and RNG only; no GraphPolicy inference."""
    curve=native.curve(folder)
    if curve.get('parse_status')=='integrity_invalid':
        raise ValueError('Corrupt actor learning curve: '+curve['integrity_error'])
    training=native.check_training(folder,cfg,seed,0,sha(folder.parents[1]/'config.json'),curve_snapshot=curve)
    if not training['completed']:
        return training
    done=read(folder/'completed.json')
    saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    if (done.get('offline_critic_frozen') is not True or done.get('online_flow_update') is not False or
        saved['step']!=1000 or saved['binding']!=read(folder/'binding.json') or
        saved['binding'].get('replay')!=[] or saved['binding'].get('parent') is not None):
        raise ValueError('Offline rule actor is not the fixed1000-update, no-replay contrast')
    tree_equal(saved['critic'],refit['model'].state_dict(),'critic')
    tree_equal(saved['flow'],common['flow'],'flow')
    if saved['actor'].keys()!=common['actor'].keys():
        raise ValueError('One-step actor architecture keys differ from common initialization')
    for k,value in saved['actor'].items():
        expected=common['actor'][k]
        if (not isinstance(value,torch.Tensor) or value.shape!=expected.shape or value.dtype!=expected.dtype or
            (value.is_floating_point() and not bool(torch.isfinite(value).all()))):
            raise ValueError('Actor checkpoint shape/dtype/finite contract differs: '+k)
    for key in ('movable','fixed'):
        tree_equal(saved['actor'][key],common['actor'][key],'actor/'+key)
    if saved['optimizers']['critic']['state'] or saved['optimizers']['flow']['state']:
        raise ValueError('Offline frozen critic/flow optimizer has updates')
    steps=[float(value['step']) for value in saved['optimizers']['actor']['state'].values()]
    if not steps or any(value!=1000 for value in steps):
        raise ValueError('Actor Adam committed-step counters differ')
    pairing=dict(seed=seed,common_sha256=cfg['common_checkpoint_sha256'][str(seed)],
        dataset_sha256=cfg['dataset_sha256'],cache_sha256=cfg['cache_sha256'],
        behavior=cfg['rule_behavior'],config={k:cfg[k] for k in actor.FIXED if k not in ('physical_failure_reward',)},
        step=saved['step'],rng=state_sha(saved['rng']),
        cuda_rng=[state_sha(value) for value in saved['cuda_rng']],
        generators={k:state_sha(value) for k,value in saved['generators'].items()})
    training.update(pairing=pairing,critic_frozen=True,flow_frozen=True,
        optimizer_updated_parameters=len(steps),adam_committed_steps=1000,
        initial_critic=refit['checkpoint'],common=cfg['rule_common'][str(seed)],
        gradient_cosine_definition='logged cosine is gradient(alpha*D) versus gradient(+q_weight*Q); desired descent/ascent update cosine is its negative, so negative logged cosine means aligned improving directions')
    return training


def measured_rewards(row,request,context,specs):
    """Match the supervised target: executed proposal rules plus real GRT WL."""
    if row['state'] in ('physical_failure','geometry_failure'):
        return dict(costs=None,rewards={o:-4. for o in screen.OBJECTIVES})
    if row['state']!='success':
        return dict(costs=None,rewards={o:None for o in screen.OBJECTIVES})
    geometry,_=structure(context['cache'])
    hpwl=MacroHPWL(context['cache'])
    score=rules(np.asarray(request['executed_action']),geometry,hpwl)
    costs=np.asarray([row['metrics']['rWL_um'],*[score[k] for k in screen.COST_NAMES[1:]]],float)
    if not np.isfinite(costs).all() or (costs<0).any():
        raise ValueError('Measured native/rule costs are unavailable or negative')
    rewards={}
    for objective,spec in specs.items():
        screen.validate_spec(spec)
        c=float((costs/np.asarray(spec['scales'])*np.asarray(spec['weights'])).sum())
        rewards[objective]=-2*c/(1+c)
    return dict(costs=dict(zip(screen.COST_NAMES,costs.tolist())),rewards=rewards)


def verify_request(row,cfg,seed,training,physical_context):
    sample=backend.checked_request(row,argparse.Namespace(seed=seed,round=0),physical_context[0],physical_context[2])
    if (row.get('arm')!=cfg['arm'] or row.get('role')!='evaluation' or row.get('round')!=0 or
        row.get('terminal_mask')!=0 or row.get('training_feedback_eligible') is not False or
        row.get('offline_train_eligible',False) is not False or row.get('noise_seed')!=4600000 or
        row.get('checkpoint_sha256')!=training['checkpoint_sha256'] or
        Path(row['source_checkpoint']).resolve()!=Path(training['checkpoint']).resolve() or
        row.get('objective_weights')!=cfg['weights'] or
        row.get('evaluation_purpose')!=cfg.get('evaluation_purpose','development_validation') or
        row.get('run_id')!=f"go2o_qs1_{cfg['run_prefix']}_s{seed}_r0_evaluation_{row['index']}"):
        raise ValueError('Rule actor eval-only source/weights/checkpoint/stream request differs')
    return sample


def stream_summary(objective,seed,rows,training,errors,tasks=None):
    rows=sorted(rows,key=lambda r:r['index'])
    if rows and (len(rows)!=COUNT or [r['index'] for r in rows]!=list(range(COUNT)) or
                 len({r['noise_sha256'] for r in rows})!=COUNT):
        raise ValueError('Rule stream must retain the exact eight distinct actual noises')
    summary=baseline.summarize(rows,COUNT)
    if errors or not training.get('completed'):
        summary['fully_accounted']=False
    successes=[r for r in rows if r['state']=='success']
    rewards=[r['rewards']['grt_wl'] for r in rows if r['state'] in native.ADMITTED]
    own=[r for r in rows if r.get('predicted_q') is not None and r['rewards'][objective] is not None]
    predicted=np.asarray([r['predicted_q'] for r in own],float)
    measured=np.asarray([r['rewards'][objective] for r in own],float)
    return dict(objective=objective,arm=actor.ARMS[objective][0],seed=seed,round=0,
        training=training,errors=errors,tasks=bind(tasks) if tasks else None,
        actual_noise_by_index={str(r['index']):r['noise_sha256'] for r in rows},rows=rows,
        integrity_errors=errors+[r['integrity_error'] for r in rows if r['state']=='invalid'],
        common_grt_B_utility_mean=native.mean(rewards) if summary['fully_accounted'] else None,
        own_reward_mean=native.mean([r['rewards'][objective] for r in rows if r['state'] in native.ADMITTED])
            if summary['fully_accounted'] else None,
        measured_rule_cost_success_means={k:native.mean([r['costs'][k] for r in successes]) for k in screen.COST_NAMES},
        own_q_calibration=dict(n=len(own),mae=float(np.abs(predicted-measured).mean()) if len(own) else None,
            signed_prediction_minus_measurement=float((predicted-measured).mean()) if len(own) else None,
            pearson=float(np.corrcoef(predicted,measured)[0,1]) if len(own)>2 and np.ptp(predicted)>0 and np.ptp(measured)>0 else None,
            interpretation='own fixed reward only; Q at raw action versus measured cost after legalizer/native flow'),
        **summary)


def collect_stream(experiment,cfg,seed,refit,common,context,specs):
    folder=experiment/f'seed_{seed}'/'round_0'
    errors=[]
    try:training=checked_training(folder,cfg,seed,refit,common)
    except Exception as exc:
        training=dict(completed=False,integrity_error=str(exc));errors.append('training: '+str(exc))
    tasks=folder/'requests/tasks.jsonl'
    if not tasks.exists() or not tasks.with_name('completed.json').exists():
        return stream_summary(cfg['rule_objective'],seed,[],training,errors)
    rows=[]
    try:
        physical=backend.context(experiment)
        requests,tasks=load_requests(experiment,seed,0)
        completion=read(tasks.with_name('completed.json'))
        if (len(requests)!=COUNT or [r['index'] for r in requests]!=list(range(COUNT)) or
            completion.get('selection')!='all predetermined noises; no Q-selection' or
            completion.get('evaluation_never_replayed') is not True or not training.get('completed')):
            raise ValueError('Export completion/count/evaluation-only or training gate differs')
        for request in requests:
            item=dict(seed=seed,index=request['index'],noise_sha256=request.get('noise_sha256'),
                action_sha256=request.get('action_sha256'),raw_action_sha256=request.get('raw_action_sha256'),
                state='invalid',metrics=None,integrity_error=None,predicted_q=None,
                costs=None,rewards={o:None for o in screen.OBJECTIVES})
            try:
                sample=verify_request(request,cfg,seed,training,physical)
                item.update(baseline.inspect_outcome(request,sample,tasks,physical))
                if item['state']=='success' and any(item['metrics'][k]<0 for k in native.METRICS if k not in ('WNS_ns','TNS_ns')):
                    raise ValueError('Nonnegative physical metric is negative')
                item.update(measured_rewards(item,request,context,specs))
                predicted=np.asarray(request.get('predicted_q'),float)
                if predicted.shape!=(1,) or not np.isfinite(predicted).all():
                    raise ValueError('Exported scalar predicted Q differs')
                item['predicted_q']=float(predicted[0])
                raw=np.asarray(request['raw_action']);executed=np.asarray(request['executed_action'])
                movable=context['movable'].cpu().numpy().astype(bool)
                item['legalizer_jump_rms']=float(np.sqrt(np.mean((raw[movable]-executed[movable])**2)))
                item['geometry_unit']='RMS over movable normalized [-1,1] coordinates'
            except Exception as exc:
                item.update(state='invalid',metrics=None,integrity_error=str(exc),costs=None,
                    rewards={o:None for o in screen.OBJECTIVES})
            rows.append(item)
    except Exception as exc:
        errors.append('export: '+str(exc))
    return stream_summary(cfg['rule_objective'],seed,rows,training,errors,tasks)


def eligible_stats(values,expected):
    result=native.describe(values)
    if result['n']!=expected:result.update(mean=None,sd=None)
    return dict(result,eligible=result['n']==expected)


def aggregate(streams):
    rows=[]
    for objective in screen.OBJECTIVES:
        matched=[s for s in streams if s['objective']==objective]
        okay=(len(matched)==3 and {s['seed'] for s in matched}==set(SEEDS) and
              all(s['fully_accounted'] and not s['integrity_errors'] for s in matched) and
              all(s['actual_noise_by_index']==matched[0]['actual_noise_by_index'] for s in matched))
        rows.append(dict(objective=objective,planned_seeds=list(SEEDS),independent_seed_n=3 if okay else 0,
            aggregate_eligible=okay,conditional_PPA=any(s['conditional_PPA'] for s in matched),
            request_accounting={k:sum(s[k] for s in matched) for k in
                ('planned','exported','not_exported','success','physical_failure','geometry_failure','missing','invalid')},
            metrics={k:eligible_stats([s['successful_native_metric_means'][k] for s in matched],3)
                     for k in native.METRICS} if okay else None,
            common_grt_B_utility=eligible_stats([s['common_grt_B_utility_mean'] for s in matched],3) if okay else None,
            own_reward=eligible_stats([s['own_reward_mean'] for s in matched],3) if okay else None,
            interpretation='three equal training-seed means; eight generation noises are nested, PPA conditional on successful native execution'))
    return rows


def paired(streams,objective,control):
    pairs,seeds=[],[]
    for seed in SEEDS:
        left=next(s for s in streams if s['objective']==objective and s['seed']==seed)
        right=next(s for s in streams if s['objective']==control and s['seed']==seed)
        same=(len(left['rows'])==len(right['rows'])==COUNT and
              left['actual_noise_by_index']==right['actual_noise_by_index'])
        if left['rows'] and right['rows'] and not same:
            raise ValueError('Paired rule actors have different actual noise arrays')
        ready=same and left['fully_accounted'] and right['fully_accounted'] and not left['integrity_errors'] and not right['integrity_errors']
        local=[]
        if same:
            for x,y in zip(left['rows'],right['rows']):
                both=x['state']==y['state']=='success'
                local.append(dict(objective=objective,control=control,seed=seed,index=x['index'],noise_sha256=x['noise_sha256'],
                    left_state=x['state'],right_state=y['state'],both_success=both,
                    deltas={k:x['metrics'][k]-y['metrics'][k] for k in native.METRICS} if both else None,
                    common_grt_B_utility_delta=x['rewards']['grt_wl']-y['rewards']['grt_wl']
                        if x['rewards']['grt_wl'] is not None and y['rewards']['grt_wl'] is not None else None))
        success=[r for r in local if r['both_success']]
        seeds.append(dict(seed=seed,planned_pairs=COUNT,paired_exported=len(local),fully_accounted=bool(ready),
            both_success=len(success),conditional_PPA=len(success)!=COUNT,
            metric_deltas={k:native.mean([r['deltas'][k] for r in success]) for k in native.METRICS} if ready else None,
            common_grt_B_utility_delta=native.mean([r['common_grt_B_utility_delta'] for r in local]) if ready else None))
        pairs.extend(local)
    complete=all(s['fully_accounted'] for s in seeds)
    return dict(objective=objective,control=control,direction='objective minus control',planned_pairs=24,
        paired_exported=len(pairs),both_success=sum(r['both_success'] for r in pairs),fully_accounted=complete,
        conditional_PPA=any(s['conditional_PPA'] for s in seeds),seeds=seeds,pairs=pairs,
        three_seed_metric_deltas={k:eligible_stats([s['metric_deltas'][k] if s['metric_deltas'] else None for s in seeds],3)
                                  for k in native.METRICS} if complete else None,
        three_seed_common_grt_B_utility_delta=eligible_stats([s['common_grt_B_utility_delta'] for s in seeds],3) if complete else None)


def verify_pairing(streams):
    noises=[s['actual_noise_by_index'] for s in streams if len(s['rows'])==COUNT]
    if noises and any(n!=noises[0] for n in noises[1:]):
        raise ValueError('Eight actual evaluation noises differ across objectives/seeds')
    rows=[]
    for seed in SEEDS:
        completed=[s for s in streams if s['seed']==seed and s['training'].get('completed')]
        identities=[s['training']['pairing'] for s in completed]
        if identities and any(value!=identities[0] for value in identities[1:]):
            raise ValueError('Rule-only training common/settings/RNG endpoints differ')
        rows.append(dict(seed=seed,verified_objectives=[s['objective'] for s in completed],
            expected_objectives=list(screen.OBJECTIVES),all_eight_available=len(completed)==8,
            common_update_identity=identities[0] if identities else None,
            caveat='same fixed algorithm/seed/budget and RNG endpoints; no separately recorded full per-step drawchain'))
    return rows


def checked_published_config(cfg,preparation,inputs,root,refit,seed):
    """Close the preparation -> dataset/cache -> exact initial-refit chain."""
    if (cfg['rule_plan']!=preparation['plan'] or cfg['rule_behavior']!=preparation['behavior'] or
        cfg['rule_common']!=preparation['common'] or cfg['rule_sources']!=preparation['source_sha256'] or
        Path(cfg['rule_preparation_root']).resolve()!=Path(root).resolve() or
        any(cfg[k]!=inputs[k] for k in ('dataset_path','dataset_sha256','cache_sha256')) or
        cfg['rule_utility_spec']!=refit['spec'] or
        cfg['critic_checkpoint_sha256'][str(seed)]!=refit['checkpoint']['sha256'] or
        Path(cfg['critic_checkpoints'][str(seed)]).resolve()!=Path(refit['checkpoint']['path']).resolve() or
        len(set(cfg['common_checkpoint_sha256'].values()))!=3):
        raise ValueError('Published actor preparation/input/initial-refit binding differs')


def collect(experiments_root):
    experiments_root=Path(experiments_root).resolve()
    preparation=read(experiments_root/'preparation.json')
    if (preparation['objectives']!=list(screen.OBJECTIVES) or preparation['automatic_selection'] is not False or
        preparation['schema']!=actor.SCHEMA or preparation['status']!='prepared_configs_only' or
        preparation['source_sha256']!=actor.sources() or preparation['online_implemented'] is not False or
        preparation['backend_calls']!=0 or preparation['actor_jobs_submitted']!=0):
        raise ValueError('Fixed eight-objective actor preparation differs')
    loaded=screen.checked_plan(Path(preparation['plan']['path']))
    bind(preparation['plan']['path'],preparation['plan']['sha256'])
    common,specs,contexts,configs,evidence={},{},{},{},{}
    for objective in screen.OBJECTIVES:
        experiment=experiments_root/actor.ARMS[objective][0]
        for seed in SEEDS:
            cfg=actor.validate_config(experiment,seed)
            if cfg['rule_objective']!=objective:
                raise ValueError('Objective directory/config differs')
            if preparation['configs'][cfg['arm']]!=bind(experiment/'config.json'):
                raise ValueError('Preparation/config publication identity differs')
            result=screen.checked_refit(Path(cfg['rule_plan']['path']),objective,seed,loaded=loaded)
            checked_published_config(cfg,preparation,loaded[0]['inputs'],experiments_root,result,seed)
            for key in ('spec','checkpoint','completion','normalization'):
                if result[key]!=cfg['rule_refits'][str(seed)][key]:
                    raise ValueError('Actor prepared full-TRAIN refit changed: '+key)
            if objective in specs and specs[objective]!=result['spec']:
                raise ValueError('Full-TRAIN reward scales changed between model seeds')
            specs[objective]=result['spec']
            contexts[(objective,seed)]=result
            configs[(objective,seed)]=(experiment,cfg)
            evidence.update(result['evidence'])
            if seed not in common:
                source=cfg['rule_common'][str(seed)]['checkpoint']
                bind(source['path'],source['sha256'])
                common[seed]=torch.load(source['path'],map_location='cpu',weights_only=False)
    streams=[]
    physical=[]
    for (objective,seed),(experiment,cfg) in configs.items():
        streams.append(collect_stream(experiment,cfg,seed,contexts[(objective,seed)],common[seed],
            loaded[2],specs))
        context=backend.context(experiment)[2]
        physical.append({k:v for k,v in context.items() if k!='config_sha256'})
    if any(value!=physical[0] for value in physical[1:]):
        raise ValueError('Native physical settings differ across rule arms')
    pairing=verify_pairing(streams)
    contrasts=[paired(streams,objective,control) for control in CONTROLS for objective in screen.OBJECTIVES if objective!=control]
    complete=all(s['fully_accounted'] and not s['integrity_errors'] for s in streams)
    invalid=any(s['integrity_errors'] for s in streams)
    return dict(schema='qstudy_rule_reward_actor_native_results_v1',status='integrity_error' if invalid else 'completed' if complete else 'partial',
        preparation=bind(experiments_root/'preparation.json'),screen_plan=preparation['plan'],
        physical_context=physical[0],expected_arms=8,expected_training_seeds=24,expected_requests=192,
        request_accounting={k:sum(s[k] for s in streams) for k in
            ('planned','exported','not_exported','success','physical_failure','geometry_failure','missing','invalid')},
        streams=streams,multi_seed=aggregate(streams),paired_vs_new_controls=contrasts,
        paired_training=pairing,all_reward_specs=specs,critic_evidence=evidence,
        conventions=dict(controls='new GRT-only and new macroHPWL-only actors; no causal reuse of old BC',
            uncertainty='three training seeds, eight matched generation noises nested within each; SD is not a confidence interval',
            reward='cost rules on saved executed coordinates plus real native GRT WL; backend may additionally snap macro tracks',
            utility='only common grt_wl B utility is comparable across arms; own Q calibration uses its own target',
            failures='confirmed physical/geometry failure reward -4 with no fabricated PPA; missing/infrastructure remains unresolved',
            stage='global route native7, not detailed-route DRC or final Table2 metrics',
            gradients='logged gradient(alphaD) versus gradient(+Q) cosine; improving update cosine is its negative',
            selection='all eight predeclared objectives, no CV filtering, checkpoint selection, promotion or fresh confirmation'),
        backend_calls=0,model_inference_calls=0,actor_advanced=False,heldout_used=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Read-only numerical aggregation requires CPU Slurm')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',1)))
    root,output=args.experiments_root.resolve(),args.output.resolve()
    first=read(root/actor.ARMS['grt_wl'][0]/'config.json')
    for protected in (root,Path(first['dataset_path']).resolve().parent,Path(first['rule_plan']['path']).resolve().parent):
        if output==protected or output in protected.parents or protected in output.parents:
            raise ValueError('Report output must be separate from original experiments/data/screen')
    result=collect(root)
    snapshot=output/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_job'+os.environ['SLURM_JOB_ID'])
    snapshot.mkdir(parents=True,exist_ok=False)
    sources={name:sha(ROOT/name) for name in SOURCE_NAMES}
    for name,digest in sources.items():
        destination=snapshot/'source_snapshot'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes());bind(destination,digest)
    result.update(snapshot=str(snapshot),slurm_job_id=os.environ['SLURM_JOB_ID'],reporter_sources=sources)
    atomic_json(snapshot/'report.json',native.clean(result))
    for name in ('multi_seed','paired_training','streams'):
        (snapshot/(name+'.csv')).write_text(csv_text(result[name]))
    (snapshot/'paired_vs_new_controls.csv').write_text(csv_text(result['paired_vs_new_controls']))
    files={str(path.relative_to(snapshot)):sha(path) for path in snapshot.rglob('*') if path.is_file()}
    atomic_json(snapshot/'completed.json',dict(status='completed',report_status=result['status'],files=files,
        backend_calls=0,model_inference_calls=0,actor_advanced=False))
    atomic_json(output/'latest.json',dict(report=bind(snapshot/'report.json'),completion=bind(snapshot/'completed.json'),status=result['status']))
    print(json.dumps(dict(status=result['status'],snapshot=str(snapshot),report=bind(snapshot/'report.json'),
                         request_accounting=result['request_accounting'])),flush=True)


if __name__=='__main__':main()
