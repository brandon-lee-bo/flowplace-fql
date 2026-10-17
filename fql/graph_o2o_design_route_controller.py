"""Run all predeclared full-route comparisons as their measured GRT sources arrive."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import ROOT
from .graph_o2o_design_policy_data import measured_context
from .graph_o2o_controller import LIVE, RESTARTABLE, states, submit_unique


def physical_run_id(plan, key):
    prefix = plan.get('run_id_prefix', 'go2o_drt_v1')
    name = prefix + '_' + key
    if 'tasks' in plan:
        matches = [task for task in plan['tasks'] if task['key'] == key]
        if len(matches) != 1:
            raise ValueError('Explicit task namespace is missing or duplicated')
        name = matches[0]['run_id']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError('Invalid full-route namespace')
    return name


def requested(experiment, plan):
    if 'tasks' in plan:
        result = []
        for row in plan['tasks']:
            task = dict(key=row['key'], method=row.get('method', row['group']),
                        group=row['group'], index=row['noise_index'], output=row['output'])
            if ('source_outcome' in row) == ('source_run' in row):
                raise ValueError('Explicit route task needs exactly one source kind')
            if 'source_outcome' in row:
                task['source_outcome'] = row['source_outcome']
            else:
                task.update(source_run=row['source_run'],
                            source_manifest_sha256=row['source_manifest_sha256'],
                            source_metrics_sha256=row['source_metrics_sha256'])
            result.append(task)
        if len(result) != plan['requested'] or len({row['key'] for row in result}) != len(result):
            raise ValueError('Explicit full-route denominator or task keys changed')
        for task in result:
            if output_path(experiment, task) != experiment/'full_route'/task['key']:
                raise ValueError('Explicit output must match the fixed experiment task')
        return result
    reference=json.loads((experiment/'best_train_reference.json').read_text())
    result=[dict(key='best_train',method='best_observed_TRAIN',group='best_train',index=0,
                 source_run=reference['physical_run'])]
    for group in plan['groups']:
        for i in group['indices']:
            key=f"{group['arm']}_r{group['round']}_n{i}"
            path=(experiment/f"seed_{plan['seed']}"/f"round_{group['round']}"/'requests'/
                  group['arm']/group['role']/str(i)/'outcome.json')
            result.append(dict(key=key,method=group['arm']+'_round_'+str(group['round']),
                               group=group['arm']+'_r'+str(group['round']),index=i,source_outcome=str(path)))
    if len(result)!=plan['requested_final_layouts']:
        raise ValueError('Incomplete full-route request denominator')
    return result


def output_path(experiment, task):
    return Path(task.get('output', experiment/'full_route'/task['key'])).resolve()


def route_cap(plan):
    cap = plan.get('detailed_route_end_iteration', plan.get('route_end_iteration'))
    if not isinstance(cap, int) or cap <= 0:
        raise ValueError('Missing or invalid predeclared detailed-route cap')
    return cap


def resolve_source(task):
    if 'source_run' in task:
        for field, name in [('source_manifest_sha256', 'manifest.json'),
                            ('source_metrics_sha256', 'metrics.json')]:
            if field in task and sha(Path(task['source_run'])/name) != task[field]:
                raise ValueError('Explicit source-run evidence changed: '+field)
        return task['source_run'],None
    path=Path(task['source_outcome'])
    if not path.exists():
        return None,None
    outcome=json.loads(path.read_text())
    if outcome['status']!='success':
        if outcome['status'] not in ['physical_failure','geometry_failure']:
            raise ValueError('Unknown upstream outcome')
        return None,dict(status='upstream_physical_failure',source_outcome=str(path),
                         source_outcome_sha256=sha(path),failure_class=outcome['failure_class'],metrics=None)
    return outcome['physical_run'],None


def aggregate(experiment, plan, tasks, journal):
    records=[]
    for task in tasks:
        entry=journal['tasks'][task['key']]
        if entry['status']=='upstream_physical_failure':
            result=entry['outcome']
        else:
            path=output_path(experiment, task)/'outcome.json'
            result=json.loads(path.read_text())
            run=Path(result['physical_run'])
            if (sha(run/'manifest.json')!=result['physical_manifest_sha256'] or
                sha(run/'metrics.json')!=result['metrics_sha256']):
                raise ValueError('Full-route evidence changed')
        records.append(dict(task=task,result=result))
    groups={}
    keys=['DRT_WL','WNS','TNS','Power','StdCellArea','DRC']
    for group in dict.fromkeys(t['group'] for t in tasks):
        rows=[r for r in records if r['task']['group']==group]
        good=[r for r in rows if r['result']['status']=='completed']
        means={k:statistics.mean(r['result']['metrics'][k] for r in good) for k in keys} if good else None
        groups[group]=dict(requested=len(rows),completed=len(good),
            drc_clean=sum(r['result']['route_drc_clean'] for r in good),
            mean=means if len(rows)==len(good) else None,successful_only_mean=means)
    seed = plan.get('seed', 0)
    report=dict(status='all_requested_outcomes_accounted',scope=f"Single training seed{seed}, fixed {plan['design']} netlist; common cap{route_cap(plan)} detailed route",
        plan_sha256=sha(experiment/'full_route_plan.json'),requested=len(tasks),groups=groups,records=records,
        independent_training_seeds=1,not_historical_table2=True,
        next_work='Audit effects and Q calibration, then replication/justified improvements. This report does not complete the research goal.')
    atomic_json(experiment/'full_route'/'summary.json',report)
    lines=['# FlowPlace＋Graph FQL：统一详细布线比较','',
        f"{plan['design']}，训练 seed={seed}；每组四个固定评估噪声。统一 OpenROAD/Nangate45、GRT 检查点延续和详细布线 cap{route_cap(plan)}。",
        '此组与历史 Table 2 cap4 分开，旧任务仍保持暂停。','',
        '| 方法 | 完成/请求 | DRC=0 | rWL（μm） | WNS（ns） | TNS（ns） | 功耗（W） | 面积（μm²） | DRC 均值 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for name,value in groups.items():
        metrics=value['mean']
        cells=' | '.join(f'{metrics[k]:.6g}' for k in keys) if metrics else '— | — | — | — | — | —'
        lines.append(f"| {name} | {value['completed']}/{value['requested']} | {value['drc_clean']} | {cells} |")
    if any(task['key'] == 'best_train' for task in tasks):
        lines += ['', 'best_train 是仅从 64 个离线训练布局中选出的一个固定参考，不是四次独立重复。']
    lines += ['', 'best64 和 best80 各是一份根据 GRT 从已测 TRAIN 数据中选出的固定布局；不作为四次独立重复。best80 只使用最终策略实际用过的16个自采标签。', '', '失败保留在分母中；组内有失败时不报告完整组均值。非零 DRC 原样保留。',
              '四个噪声不等于四个训练 seed，仍需多 seed 验证。','']
    (experiment/'full_route'/'summary.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=report['status'],groups=groups)),flush=True)


def run(args):
    experiment=Path(args.experiment).resolve()
    plan_path=experiment/'full_route_plan.json';plan=json.loads(plan_path.read_text())
    cfg,binding,_=measured_context(experiment)
    if (plan['design']!=cfg['design'] or plan['config_sha256']!=sha(experiment/'config.json') or
            plan['dataset_binding_sha256']!=sha(experiment/'dataset_binding.json')):
        raise ValueError('Final-route plan crossed its frozen design/data context')
    tasks=requested(experiment,plan)
    route_cap(plan)
    concurrency = args.concurrency if args.concurrency is not None else plan.get('concurrency', 1)
    if concurrency < 1:
        raise ValueError('Detailed-route concurrency must be positive')
    if len({physical_run_id(plan, task['key']) for task in tasks}) != len(tasks):
        raise ValueError('Full-route physical run names are not unique')
    state_path=experiment/'route_controller.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
    else:
        initial = {}
        if any(task['key'] == 'best_train' for task in tasks):
            reference_path = experiment/'full_route/best_train/outcome.json'
            reused = plan.get('reused_reference')
            if reused and reference_path.exists() and sha(reference_path) == reused['sha256']:
                if sha(reused['path']) != reused['sha256']:
                    raise ValueError('Reused fixed TRAIN reference changed')
                initial['best_train'] = dict(status='completed', jobs=[],
                    outcome_path=str(reference_path), outcome_sha256=sha(reference_path), reused_reference=True)
            elif args.initial_reference_job:
                initial['best_train'] = dict(status='running', jobs=[str(args.initial_reference_job)])
            else:
                raise ValueError('A grouped plan needs the fixed reference job or a verified reused result')
        state=dict(version=1, plan_sha256=sha(plan_path), tasks=initial)
    if state['plan_sha256']!=sha(plan_path):
        raise ValueError('Full-route plan changed during execution')
    state['controller_job_id']=os.environ['SLURM_JOB_ID']
    if state.get('concurrency', concurrency) != concurrency:
        raise ValueError('Detailed-route concurrency changed while resuming')
    state['concurrency'] = concurrency
    state['backend_storage'] = plan.get('backend_storage', 'bulk')
    restart=int(os.environ.get('SLURM_RESTART_COUNT','0'))
    source_key=os.environ['SLURM_JOB_ID']+(f'_restart_{restart}'if restart else'')
    snapshot=experiment/'route_controller_sources'/source_key
    snapshot.mkdir(parents=True,exist_ok=False)
    captured={}
    for name in ['graph_o2o_design_route_controller.py','graph_o2o_design_route_controller.slurm',
                 'graph_o2o_design_policy_data.py',
                 'graph_o2o_controller.py','graph_o2o_full_route.py']:
        shutil.copyfile(Path(__file__).with_name(name),snapshot/name)
        captured[name]=sha(snapshot/name)
    state.setdefault('source_snapshots',{})[source_key]=dict(
        path=str(snapshot),sha256=captured)
    save=lambda:atomic_json(state_path,state)
    save()
    try:
        while True:
            pending=[];active=0
            for task in tasks:
                entry=state['tasks'].setdefault(task['key'],dict(status='waiting_for_grt',jobs=[]))
                if entry['status'] in ['completed','upstream_physical_failure']:
                    continue
                if entry['status']=='running':
                    try:
                        observed=states(entry['jobs'][-1])
                    except (OSError, subprocess.SubprocessError) as error:
                        entry['last_observation_error']=repr(error)
                        save();active+=1;continue
                    entry['last_states']=observed;save()
                    if any(s in LIVE or s=='ACCOUNTING_PENDING' for s in observed):
                        active+=1;continue
                    if all(s=='COMPLETED' for s in observed):
                        path=output_path(experiment, task)/'outcome.json'
                        result=json.loads(path.read_text())
                        if result['status']!='completed':
                            raise ValueError('Completed job lacks complete physical outcome')
                        entry.update(status='completed',outcome_path=str(path),outcome_sha256=sha(path));save()
                        continue
                    if not all(s in RESTARTABLE or s=='COMPLETED' for s in observed):
                        raise RuntimeError(f"Full-route {entry['jobs'][-1]} ended {observed}; inspect original evidence")
                    entry['status']='waiting_for_grt'
                source,failure=resolve_source(task)
                if failure:
                    if 'tasks' in plan:
                        folder=output_path(experiment,task);folder.mkdir(parents=True,exist_ok=True)
                        atomic_json(folder/'outcome.json',failure)
                    entry.update(status='upstream_physical_failure',outcome=failure);save()
                elif source:
                    pending.append((task,entry,source))
            if all(e['status'] in ['completed','upstream_physical_failure'] for e in state['tasks'].values()) and len(state['tasks'])==len(tasks):
                aggregate(experiment,plan,tasks,state)
                state['stage']='all_requested_full_route_outcomes_accounted';save();return
            for task,entry,source in pending[:max(0,concurrency-active)]:
                key=task['key'];name='go2odrt_'+sha(plan_path)[:8]+'_'+key+'_'+str(len(entry['jobs']))
                source_args = (['--source-outcome',task['source_outcome']] if 'tasks' in plan and 'source_outcome' in task
                               else ['--source-run',source])
                job=submit_unique(name,['--exclude=natural-selection,enterprise'],[str(ROOT/'fql/graph_o2o_full_route.slurm'),
                    *source_args,'--run-id',physical_run_id(plan,key),'--method',task['method'],
                    '--storage',plan.get('backend_storage','bulk'),
                    '--output',str(output_path(experiment,task)),'--end-iter',str(route_cap(plan))])
                entry['jobs'].append(job);entry['status']='running';save()
            state['stage']='waiting_for_sources_or_active_detailed_route';save()
            time.sleep(15)
    except BaseException as error:
        state['last_error']=repr(error);state['attention_required']=True;save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',required=True)
    p.add_argument('--initial-reference-job')
    p.add_argument('--concurrency', type=int)
    args=p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use Slurm')
    run(args)
