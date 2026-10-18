"""Saved-only evidence for the sealed, four-arm scalar GRT continuation.

No checkpoint deserialization, inference, training, scheduler or EDA call.
The physical inventory has two T0 roots, four T1/T2 branches, and separate
eight-noise GRT / first-four-noise DRT estimands. Missing is never zero PPA.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics

# Import the local analysis package before the upstream loader changes sys.path.
from analysis.placement_metrics import legality  # noqa: F401
from fql import qstudy_actor_report as native
from fql import qstudy_flow_baseline_report as baseline
from fql import qstudy_final_route as final
from fql.graph_o2o_backend import load_requests
from fql.qstudy_backend import context
from tools import qstudy_scalar_online_control as control

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'qstudy_scalar_control_results_v1'
MANIFEST = Path('/scratch/bohanli/flowplace_graph/qstudy_scalar_online_control_v1/manifest.json')
MANIFEST_SHA = 'd86e233577c600b095729eff34f466c725eab7fbe02ba60166fd7957667e9016'
SEEDS = (0, 1, 2)
ARMS = ('bc', 'updated_q10', 'frozen_q10', 'updated_q30')
RUN_ID_TRANSLATION = "run_id='go2o_'+original.run_id; every other task field unchanged"
GRT_STATES = ('success', 'geometry_failure', 'physical_failure', 'missing', 'not_exported', 'invalid')
DRT_STATES = ('completed', 'upstream_physical_failure', 'missing_final', 'source_missing', 'source_invalid', 'invalid')
SOURCE_NAMES = ('analysis/qstudy_scalar_control_results.py',
    'analysis/qstudy_scalar_control_results.slurm', 'fql/qstudy_flow_baseline_report.py',
    'fql/qstudy_actor_report.py', 'fql/qstudy_backend.py', 'fql/qstudy_final_route.py',
    'fql/graph_o2o_final_evidence.py', 'tools/qstudy_scalar_online_control.py')


def read(path):
    return native.read(path)


def bind(path, digest=None):
    item = baseline.fingerprint(path)
    if digest is not None and item['sha256'] != digest:
        raise ValueError('Changed pinned evidence: '+str(path))
    return item


def statistics_of(values):
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
        sd=statistics.stdev(values) if len(values)>1 else None)


def checked_manifest(path, digest):
    if digest != MANIFEST_SHA or Path(path).resolve() != MANIFEST:
        raise ValueError('Only the sealed four-arm scientific manifest is supported')
    return control.checked_manifest(path, digest)


def check_branch(plan, key, cfg, decl):
    """Bind each real export to the predeclared branch, not just its label."""
    stage = int(key[-1]); output = Path(plan['output'])
    expected_request = output/('requests' if stage==0 else 'continuation_v1/requests')/('export_'+key+'.json')
    if (cfg['arm'] != 'soc_'+key or cfg['streams'] != plan['streams'][str(stage)] or
            cfg['seeds'] != list(SEEDS) or cfg['evaluation_purpose'] != 'development_validation' or
            cfg['prospective_request'] != bind(expected_request) or decl['request'] != cfg['prospective_request']):
        raise ValueError('Export differs from fixed physical branch/streams/request')
    if stage==0:
        source = plan['source_metadata']['bc' if key=='bc_t0' else 'fql']
        parents = cfg['parent_identities']
        if any(parents[str(s)]['checkpoint'] != source['parents'][str(s)]['checkpoint'] for s in SEEDS):
            raise ValueError('T0 is not its actual predeclared BC/FQL checkpoint')
        return
    arm = key.rsplit('_t',1)[0]
    identities = cfg['verified_children' if stage==1 else 'verified_seconds']
    for seed in SEEDS:
        identity = identities[str(seed)]; request = cfg['parent_training'][str(seed)]
        folder = output/f'training/cycle_{stage}'/arm/f'seed_{seed}'
        if (request['request'] != bind(output/'continuation_v1/requests'/f'train{stage}_{arm}_{seed}.json') or
                request['checkpoint'] != bind(folder/'checkpoint.pt') or
                identity['training_request'] != request['request'] or identity['checkpoint'] != request['checkpoint'] or
                identity['continuation_binding']['settings'] != plan['arms'][arm]):
            raise ValueError('Actual continuation checkpoint/settings differ from declared branch')
        training = read(request['request']['path']); done = read(folder/'completed.json')
        parent_key=('bc_t0' if arm=='bc' else 'fql_t0') if stage==1 else arm+'_t1'
        parent=output/'physical'/parent_key
        _, parent_tasks=load_requests(parent,seed,stage-1)
        if (training['settings'] != plan['arms'][arm] or training['seed'] != seed or
                Path(training['output']) != folder or done['status'] != 'completed' or done['steps'] != 600 or
                done['checkpoint'] != request['checkpoint'] or
                done['continuation_binding'] != identity['continuation_binding'] or
                Path(training['experiment']) != parent or
                training['config_sha256'] != native.sha(parent/'config.json') or
                training['declaration_sha256'] != native.sha(parent/'declaration.json') or
                training['tasks_sha256'] != native.sha(parent_tasks) or
                done['binding'] != read(folder/'binding.json')):
            raise ValueError('Continuation is not its completed actual600-update prefix')
        if stage==2:
            first = identity['first_continuation']['first_continuation']
            if first['continuation_binding']['settings'] != plan['arms'][arm]:
                raise ValueError('T2 changed first-cycle settings')
            if done['replay_counts'] != [8,8] or done['cumulative_steps'] != identity['cumulative_steps']:
                raise ValueError('T2 lost the complete C0+C1/cumulative-step evidence')


def pending_row(spec, reason='not yet exported'):
    return dict(spec, state='not_exported', metrics=None, reason=reason,
        noise_sha256=None, raw_action_sha256=None, action_sha256=None)


def merge_native_row(expected, item, **evidence):
    # Native inspectors already return request/actions and some identity keys.
    # Never use overlapping **kwargs or silently replace the sealed inventory.
    for key in ('seed','index','role','sample_dir'):
        if item.get(key) != expected[key]:
            raise ValueError('Native inspector changed fixed request identity: '+key)
    result=dict(expected);result.update(item);result.update(evidence)
    return result


def collect_grt_stream(plan, specs):
    """Verify one complete saved export, then audit native outcomes read-only."""
    spec=specs[0]; experiment=Path(spec['experiment']); seed=spec['seed']; stage=spec['round']
    result=dict(physical_key=spec['physical_key'], seed=seed, round=stage, errors=[], evidence={})
    rows=[pending_row(s) for s in specs]
    marker=experiment/f'seed_{seed}/round_{stage}/requests/completed.json'
    if not (experiment/'config.json').is_file() or not marker.is_file():
        result.update(rows=rows, export_complete=False)
        return result
    try:
        exporter=importlib.import_module(control.export_module(spec['physical_key']))
        cfg, decl, physical=exporter.settings(experiment, native.sha(experiment/'config.json'))
        check_branch(plan, spec['physical_key'], cfg, decl)
        requests=exporter.verify_export(experiment,seed,cfg,decl,physical)
        saved,tasks=load_requests(experiment,seed,stage)
        if requests != saved or [(r['role'],r['index']) for r in saved] != [(s['role'],s['index']) for s in specs]:
            raise ValueError('Saved export differs from exact physical manifest population')
        backend=context(experiment)
        for position,(expected,request) in enumerate(zip(specs,requests)):
            sample=Path(request['sample_dir']).resolve()
            if (str(sample)!=expected['sample_dir'] or request['noise_seed']!=expected['noise_stream_seed'] or
                    request['arm']!=expected['arm'] or request['round']!=stage or request['seed']!=seed or
                    request['training_feedback_eligible']!=expected['training_feedback_eligible']):
                raise ValueError('Native request differs from fixed role/noise/source namespace')
            item=baseline.inspect_outcome(request,sample,tasks,backend)
            if item['state']=='success' and any(item['metrics'][k]<0 for k in native.METRICS
                                                if k not in ('WNS_ns','TNS_ns')):
                raise ValueError('Negative nonnegative native metric')
            rows[position]=merge_native_row(expected,item,
                request=bind(sample/'request.json'),tasks=bind(tasks),actions=bind(sample/'actions.npz'),
                export_completion=bind(marker),config=bind(experiment/'config.json'),
                predicted_q=request.get('predicted_q'),checkpoint_sha256=request['checkpoint_sha256'],
                source_checkpoint=bind(request['source_checkpoint'],request['checkpoint_sha256']),
                dataset_sha256=cfg['dataset_sha256'],physical_flow_revision=backend[2]['physical_flow_revision'],
                def_sha256=request.get('def_sha256'))
        result.update(export_complete=True,evidence=dict(config=bind(experiment/'config.json'),
            declaration=bind(experiment/'declaration.json'),tasks=bind(tasks),completion=bind(marker)))
    except Exception as exc:
        # Reject every derived row of a stream with corrupt source/completion.
        result['errors'].append(str(exc));result['export_complete']=False
        rows=[dict(pending_row(s),state='invalid',reason=str(exc)) for s in specs]
    result['rows']=rows
    return result


def noise_gate(rows):
    evaluations=defaultdict(set); collections=defaultdict(set)
    for row in rows:
        if not row.get('noise_sha256') or row['state']=='invalid':
            continue
        target=evaluations[row['index']] if row['role']=='evaluation' else collections[(row['round'],row['seed'],row['index'])]
        target.add(row['noise_sha256'])
    errors=[]
    if any(len(v)>1 for v in evaluations.values()): errors.append('Actual matched evaluation arrays differ')
    if any(len(v)>1 for v in collections.values()): errors.append('Actual paired collection arrays differ')
    es=set().union(*evaluations.values()) if evaluations else set()
    cs=set().union(*collections.values()) if collections else set()
    if es & cs: errors.append('Evaluation noise entered a collection stream')
    if len(es)!=len(evaluations): errors.append('Repeated actual evaluation array across indices')
    return dict(eligible=not errors,errors=errors,available_evaluation_indices=sorted(evaluations),
        actual_evaluation_noise_by_index={str(i):next(iter(v)) for i,v in evaluations.items() if len(v)==1},
        note='Missing streams remain pending; this verifies saved actual arrays without regenerating noise')


def reject_aliased_grt_runs(rows):
    runs=defaultdict(list)
    for row in rows:
        if row['state'] in native.ADMITTED and row.get('native',{}).get('physical_run'):
            runs[str(Path(row['native']['physical_run']).resolve())].append(row)
    for duplicate in runs.values():
        if len(duplicate)>1:
            for row in duplicate:
                row.update(state='invalid',metrics=None,reason='Duplicate physical GRT run across unique requests')


def check_final_source(task, grt, binding):
    if grt['role']!='evaluation' or grt['state'] not in native.ADMITTED:
        raise ValueError('DRT source is not its admitted evaluation GRT request')
    out=read(grt['outcome']['path'])
    expected=dict(key=task['key'],arm=task['arm'],seed=task['seed'],round=task['round'],noise_index=task['noise_index'],
        status=grt['state'],source_outcome=task['source_outcome'],source_outcome_sha256=grt['outcome']['sha256'],
        request_sha256=grt['request']['sha256'],tasks_sha256=grt['tasks']['sha256'],
        export_completion_sha256=grt['export_completion']['sha256'],actions_npz_sha256=grt['actions']['sha256'],
        raw_action_sha256=grt['raw_action_sha256'],action_sha256=grt['action_sha256'],
        noise_sha256=grt['noise_sha256'],noise_seed=grt['noise_stream_seed'],
        config_sha256=grt['config']['sha256'],dataset_sha256=grt['dataset_sha256'],
        physical_flow_revision=grt['physical_flow_revision'],def_sha256=grt['def_sha256'],
        failure_class=out.get('failure_class'),native=native.native_fingerprints(out),GRT_metrics=grt['metrics'])
    if any(binding.get(k)!=v for k,v in expected.items()):
        raise ValueError('Sealed DRT continuation differs from independently audited GRT source')
    actor=binding.get('actor',{})
    if actor.get('checkpoint')!=grt['source_checkpoint']:
        raise ValueError('DRT source actor is not the actual saved stage checkpoint')


def collect_drt(plan, grt_rows, route_binding=None):
    output=Path(plan['output']); route=output/'route'; declared=plan['physical_drt_requests']
    evidence=None
    if route_binding:
        evidence=bind(route_binding['path'],route_binding['sha256']); physical_plan=read(evidence['path'])
        original_digest=hashlib.sha256(json.dumps(declared,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        translated=[dict(task,run_id='go2o_'+task['run_id']) for task in declared]
        if (Path(evidence['path'])!=route/'full_route_plan.json' or physical_plan.get('tasks')!=translated or
                physical_plan.get('base_manifest')!=bind(MANIFEST,MANIFEST_SHA) or
                physical_plan.get('run_id_translation')!=RUN_ID_TRANSLATION or
                physical_plan.get('original_tasks_sha256')!=original_digest or
                physical_plan.get('design')!='bp_fe' or physical_plan.get('detailed_route_end_iteration')!=20 or
                physical_plan.get('backend_threads')!=8 or physical_plan.get('backend_storage')!='scratch'):
            raise ValueError('Route plan differs from the sealed72-task physical budget')
        declared=translated
    else:
        physical_plan=dict(design='bp_fe')
    sources={(r['physical_key'],r['seed'],r['index']):r for r in grt_rows if r['role']=='evaluation'}
    rows=[]
    for task in declared:
        row=dict(task,state='source_missing',metrics=None,noise_sha256=None)
        grt=sources[(task['group'],task['seed'],task['noise_index'])]
        sidecar=route/'source_bindings'/(task['key']+'.json')
        try:
            if grt['state'] not in native.ADMITTED:
                row['state']='source_invalid' if grt['state']=='invalid' else 'source_missing'
            elif not route_binding or not sidecar.is_file():
                row.update(state='source_missing',reason='Explicit pinned route plan and sealed source binding required')
            else:
                binding=read(sidecar);check_final_source(task,grt,binding)
                source=dict(state='resolved',binding=binding,binding_sha256=native.sha(sidecar))
                row.update(final.inspect_final(route,physical_plan,task,source))
                row.update(source_binding=bind(sidecar),grt_request=grt['request'],grt_outcome=grt['outcome'],
                    raw_action_sha256=grt['raw_action_sha256'],action_sha256=grt['action_sha256'])
        except Exception as exc:
            row.update(state='invalid',metrics=None,reason=str(exc))
        rows.append(row)
    # Aliased physical runs cannot be counted as independent measurements.
    by_run=defaultdict(list)
    for row in rows:
        if row['state']=='completed': by_run[row['physical_run']].append(row)
    for duplicate in by_run.values():
        if len(duplicate)>1:
            for row in duplicate: row.update(state='invalid',metrics=None,reason='Duplicate physical DRT run')
    return rows,evidence


def group_summary(rows, stage, key, seed, role, noise_eligible=True):
    expected=8 if stage=='GRT' else 4
    states=GRT_STATES if stage=='GRT' else DRT_STATES
    success='success' if stage=='GRT' else 'completed'
    admitted=native.ADMITTED if stage=='GRT' else {'completed','upstream_physical_failure'}
    metric_names=native.METRICS if stage=='GRT' else final.METRICS
    counts=Counter(r['state'] for r in rows)
    indices=[r['index'] if stage=='GRT' else r['noise_index'] for r in rows]
    if len(rows)!=expected or sorted(indices)!=list(range(expected)) or any(r['seed']!=seed for r in rows):
        raise ValueError('Wrong fixed per-seed noise denominator')
    good=[r for r in rows if r['state']==success]
    accounted=all(r['state'] in admitted for r in rows)
    return dict(stage=stage,physical_key=key,seed=seed,round=rows[0]['round'],role=role,planned=expected,
        counts={s:counts[s] for s in states},fully_accounted=accounted,
        all_success=len(good)==expected,noise_eligible=noise_eligible,conditional_PPA=len(good)!=expected,
        success_n=len(good),successful_metric_means={m:statistics_of([r['metrics'][m] for r in good])['mean'] for m in metric_names},
        actual_noise_by_index={str(r['index'] if stage=='GRT' else r['noise_index']):r.get('noise_sha256') for r in rows},
        DRC_zero=sum(r['metrics']['DRC']==0 for r in good) if stage=='DRT' else None,
        DRC_positive=sum(r['metrics']['DRC']>0 for r in good) if stage=='DRT' else None)


def aggregate_groups(seed_groups):
    buckets=defaultdict(list)
    for row in seed_groups: buckets[(row['stage'],row['physical_key'],row['role'])].append(row)
    groups=[]
    for (stage,key,role),rows in sorted(buckets.items()):
        rows=sorted(rows,key=lambda r:r['seed']);metrics=native.METRICS if stage=='GRT' else final.METRICS
        complete=([r['seed'] for r in rows]==list(SEEDS) and all(r['fully_accounted'] and r['noise_eligible'] for r in rows))
        all_success=complete and all(r['all_success'] for r in rows)
        values={}
        if complete:
            for metric in metrics:
                stat=statistics_of([r['successful_metric_means'][metric] for r in rows])
                values[metric]=(dict(stat,eligible=True) if stat['n']==3 else
                    dict(n=stat['n'],mean=None,sd=None,eligible=False))
        groups.append(dict(stage=stage,physical_key=key,round=rows[0]['round'],role=role,
            planned_seeds=list(SEEDS),accounted_seed_n=sum(r['fully_accounted'] for r in rows),
            success_seed_n=sum(r['all_success'] for r in rows),fully_accounted=complete,
            all_success=all_success,conditional_PPA=not all_success,
            planned=sum(r['planned'] for r in rows),counts=dict(sum((Counter(r['counts']) for r in rows),Counter())),
            metrics=values if complete else None,
            interpretation='Equal-weight successful-conditional means within each training seed, then sample SD across three seed means; no policy mean when requests remain unresolved'))
    return groups


def comparison_pairs(round_number):
    if round_number==0: return [('fql_t0','bc_t0')]
    suffix='_t'+str(round_number)
    return [(a+suffix,b+suffix) for a,b in
        (('updated_q10','frozen_q10'),('updated_q30','updated_q10'),
         ('updated_q10','bc'),('frozen_q10','bc'),('updated_q30','bc'))]


def matched_deltas(rows, stage, noise_eligible=True):
    size=8 if stage=='GRT' else 4;metrics=native.METRICS if stage=='GRT' else final.METRICS
    successful='success' if stage=='GRT' else 'completed'
    index_name='index' if stage=='GRT' else 'noise_index'
    lookup={(r['physical_key'] if stage=='GRT' else r['group'],r['seed'],r[index_name]):r
            for r in rows if stage!='GRT' or r['role']=='evaluation'}
    result=[]
    for round_number in ((0,1,2) if stage=='GRT' else (0,2)):
        for lhs,rhs in comparison_pairs(round_number):
            seeds=[]
            for seed in SEEDS:
                pairs=[(lookup[(lhs,seed,i)],lookup[(rhs,seed,i)]) for i in range(size)]
                matched=[(a,b) for a,b in pairs if a['state']==successful and b['state']==successful and
                    a.get('noise_sha256') is not None and a['noise_sha256']==b.get('noise_sha256')]
                eligible=noise_eligible and len(matched)==size
                seeds.append(dict(seed=seed,planned_pairs=size,both_success_same_noise_n=len(matched),
                    eligible=eligible,deltas={m:statistics.mean(a['metrics'][m]-b['metrics'][m] for a,b in matched)
                                             for m in metrics} if eligible else None))
            full=all(r['eligible'] for r in seeds)
            result.append(dict(stage=stage,round=round_number,lhs=lhs,rhs=rhs,
                fixed_primary_endpoint=round_number==2,per_seed=seeds,eligible=full,
                statistics={m:statistics_of([r['deltas'][m] for r in seeds]) for m in metrics} if full else None,
                sign='lhs minus rhs in native units; positive favors signed WNS/TNS, negative favors other metrics',
                note='Complete same-noise pairs only; failures/NA remain in separate full request accounting; shared FQL T0 is one root'))
    return result


def build_report(plan, grt_streams, drt_rows, route_evidence):
    grt=[r for s in grt_streams for r in s['rows']]
    if (len(grt)!=384 or len({r['key'] for r in grt})!=384 or len(drt_rows)!=72 or
            len({r['key'] for r in drt_rows})!=72):
        raise ValueError('Physical request inventory is incomplete or duplicated')
    gate=noise_gate(grt);seed_groups=[]
    for stage,rows,key_field,role_field in (('GRT',grt,'physical_key','role'),('DRT',drt_rows,'group',None)):
        groups=defaultdict(list)
        for row in rows:groups[(row[key_field],row['seed'],row[role_field] if role_field else 'evaluation')].append(row)
        for (key,seed,role),subset in sorted(groups.items()):
            seed_groups.append(group_summary(subset,stage,key,seed,role,gate['eligible']))
    errors=[e for s in grt_streams for e in s['errors']]+gate['errors']
    invalid=[r for r in grt+drt_rows if r['state'] in ('invalid','source_invalid')]
    resolved=all(r['state'] in native.ADMITTED for r in grt) and all(r['state'] in ('completed','upstream_physical_failure') for r in drt_rows)
    return dict(schema=VERSION,status='complete' if resolved and not errors and not invalid else 'partial',
        manifest=bind(MANIFEST,MANIFEST_SHA),route_plan=route_evidence,design='bp_fe',fresh_confirmation=False,
        scope='Saved-only development policy means; no model/EDA/scheduler calls, winner selection or replay',
        budgets=dict(unique_GRT=384,unique_DRT=72,logical_GRT_representations=480,
            T0_physical_roots=2,FQL_shared_T0_branches=3,evaluation_GRT_noises=8,evaluation_DRT_noises=4),
        GRT_counts=dict(Counter(r['state'] for r in grt)),DRT_counts=dict(Counter(r['state'] for r in drt_rows)),
        infrastructure_note='No PPA or B4 is fabricated for unresolved infrastructure work; missing outcomes remain missing. No scheduler state is inferred from missing files.',
        integrity_errors=errors+[r.get('integrity_error',r.get('reason','invalid evidence')) for r in invalid],
        noise_gate=gate,export_streams=[{k:v for k,v in s.items() if k!='rows'} for s in grt_streams],
        physical_grt_rows=grt,physical_drt_rows=drt_rows,seed_groups=seed_groups,
        policy_groups=aggregate_groups([s for s in seed_groups if s['role']=='evaluation']),
        collection_groups=aggregate_groups([s for s in seed_groups if s['role']=='collect']),
        matched_deltas=matched_deltas(grt,'GRT',gate['eligible'])+matched_deltas(drt_rows,'DRT',gate['eligible']),
        native_units=dict(GRT=dict(rWL_um='um',WNS_ns='ns',TNS_ns='ns',horizontal_overflow_sum='GRT demand-capacity sum',
            vertical_overflow_sum='GRT demand-capacity sum',Power_W='W',StdCellArea_um2='um^2'),DRT=final.UNITS),
        limitations=['R2 is fixed primary; R0/R1 are fully retained, never best-round selection.',
            'Three independent training seeds; eight/four nested development noises are not independent training repeats.',
            'GRT eight-noise and DRT first-four statistics are distinct estimands, not stage-change causal effects.',
            'No final congestion is invented; power uses fixed tool activity assumptions, not workload measurements.',
            'Source/optimizer tensor certificates are produced by training/export prepares; this report rehashes their saved bindings without loading model tensors.'])


def write_csv(path, rows):
    rows=[{k:json.dumps(v,sort_keys=True,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows]
    if not rows:return
    names=sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,names);writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=MANIFEST)
    parser.add_argument('--manifest-sha256',default=MANIFEST_SHA)
    parser.add_argument('--route-plan',type=Path)
    parser.add_argument('--route-plan-sha256')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Numerical saved-report reduction requires bounded CPU Slurm')
    if bool(args.route_plan)!=bool(args.route_plan_sha256):raise ValueError('Explicit route plan and SHA must be supplied together')
    plan=checked_manifest(args.manifest,args.manifest_sha256)
    output=args.output.resolve(); experiment=Path(plan['output'])
    if output==experiment or output in experiment.parents or experiment in output.parents:
        raise ValueError('Report output must be outside the physical experiment tree')
    groups=defaultdict(list)
    for spec in plan['physical_grt_requests']:groups[(spec['physical_key'],spec['seed'])].append(spec)
    streams=[collect_grt_stream(plan,specs) for specs in groups.values()]
    grt=[r for s in streams for r in s['rows']]
    reject_aliased_grt_runs(grt)
    route=bind(args.route_plan,args.route_plan_sha256) if args.route_plan else None
    drt,route=collect_drt(plan,grt,route)
    report=build_report(plan,streams,drt,route)
    report['reporter_source_sha256']={name:native.sha(ROOT/name) for name in SOURCE_NAMES}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_job'+os.environ['SLURM_JOB_ID']
    folder=output/'snapshots'/stamp;folder.mkdir(parents=True,exist_ok=False)
    control.atomic(folder/'report.json',report)
    for name,rows in (('seed_groups',report['seed_groups']),('policy_groups',report['policy_groups']),
                      ('matched_deltas',report['matched_deltas']),('physical_grt_rows',grt),('physical_drt_rows',drt)):
        write_csv(folder/(name+'.csv'),rows)
    control.atomic(folder/'completed.json',dict(status='completed_readonly_reduction',result_status=report['status'],
        manifest=report['manifest'],files={p.name:native.sha(p) for p in sorted(folder.iterdir()) if p.is_file()},
        model_calls=0,backend_calls=0))
    control.atomic(output/'latest.json',dict(path=str(folder),report_sha256=native.sha(folder/'report.json'),
        completed_sha256=native.sha(folder/'completed.json')))
    print(json.dumps(dict(status=report['status'],report=bind(folder/'report.json'),completed=bind(folder/'completed.json'))))


if __name__=='__main__':main()
