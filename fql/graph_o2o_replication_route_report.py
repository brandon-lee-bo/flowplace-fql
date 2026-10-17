"""Audit final PPA across training seeds, averaging noises within each seed."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .graph_o2o_archive import atomic_json,sha
from .graph_o2o_final_evidence import audit_final_outcome, LEGACY_REPEAT_PLAN

KEYS=['DRT_WL','WNS','TNS','Power','StdCellArea','DRC']
GROUPS=['flowplace_r0','bc_r0','fql_r0','bc_r2','fql_r2']


def write_csv(path,rows):
    with path.open('w')as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(args):
    plan_path=Path(args.plan).resolve();plan=json.loads(plan_path.read_text())
    experiments={0:Path(plan['source_experiment']),**{int(k):Path(v)for k,v in plan['experiments'].items()}}
    if sorted(experiments)!=plan['independent_training_seeds']or len(experiments)<2:
        raise ValueError('Missing predeclared independent training seeds')
    grt_path=Path(args.grt_evidence).resolve();grt=json.loads(grt_path.read_text())
    if (grt['status']!='requested_seed_pilots_accounted'or grt['plan_sha256']!=sha(plan_path)or
            grt['independent_training_seeds']!=sorted(experiments)):
        raise ValueError('Missing actual independently trained offline-to-online evidence')
    rows=[];layouts=[];bindings={};common_flow=None;reference=None
    for seed,exp in sorted(experiments.items()):
        trained=grt['evidence'][str(seed)]
        if (Path(trained['experiment']).resolve()!=exp.resolve()or
                trained['controller_sha256']!=sha(exp/f'controller_seed_{seed}.json')or
                trained['config_sha256']!=sha(exp/'config.json')):
            raise ValueError('Actual trained seed/config/controller differs from the GRT audit')
        rp=exp/'full_route_plan.json';path=exp/'full_route/summary.json'
        route_plan=json.loads(rp.read_text());report=json.loads(path.read_text())
        if (report['status']!='all_requested_outcomes_accounted'or report['plan_sha256']!=sha(rp)or
                report['requested']!=21 or len(report['records'])!=21 or
                route_plan['seed']!=seed or route_plan['detailed_route_end_iteration']!=20):
            raise ValueError('Incomplete or changed full-route experiment')
        expected={(group,index)for group in GROUPS for index in range(4)}|{('best_train',0)}
        actual={(r['task']['group'],r['task']['index'])for r in report['records']}
        if actual!=expected or len(actual)!=len(report['records']):
            raise ValueError('Final comparisons lost or duplicated a requested layout')
        for row in report['records']:
            task,result=row['task'],row['result'];metrics=result.get('metrics')or{}
            if result['status']=='completed':
                physical=Path(result['physical_run'])
                checked=audit_final_outcome(result,design='bp_fe',route_cap=20,legacy_plan=LEGACY_REPEAT_PLAN)
                flow=checked['physical_identity']
                if common_flow is None:common_flow=flow
                if flow!=common_flow:raise ValueError('Final physical evaluator differs across seeds/methods')
            elif result['status']!='upstream_physical_failure':
                raise ValueError('Unresolved final measurement cannot enter aggregate')
            if task['group']=='best_train':
                if reference is None:reference=result
                if result!=reference:raise ValueError('The single shared fixed TRAIN reference differs')
            layouts.append(dict(seed=seed,group=task['group'],noise_index=task['index'],
                status=result['status'],**{k:metrics.get(k)for k in KEYS},
                physical_run=result.get('physical_run'),physical_manifest_sha256=result.get('physical_manifest_sha256'),
                legacy_exact_repeat_evidence=checked['legacy_exact_repeat_evidence'] if result['status']=='completed'else None))
        for group in GROUPS:
            selected=[row for row in report['records']if row['task']['group']==group]
            completed=[row for row in selected if row['result']['status']=='completed']
            complete=len(completed)==len(selected)
            values={k:statistics.mean(r['result']['metrics'][k]for r in completed)if complete else None for k in KEYS}
            rows.append(dict(seed=seed,group=group,requested=4,completed=len(completed),
                drc_clean=sum(r['result']['route_drc_clean']for r in completed),**values))
        bindings[str(seed)]=dict(experiment=str(exp),route_summary_sha256=sha(path),route_plan_sha256=sha(rp))
    groups=[]
    for group in GROUPS:
        selected=[r for r in rows if r['group']==group]
        complete=all(row['completed']==row['requested']for row in selected)
        mean={k:statistics.mean(r[k]for r in selected)if complete else None for k in KEYS}
        sd={k:statistics.stdev(r[k]for r in selected)if complete else None for k in KEYS}
        groups.append(dict(group=group,training_seeds=len(selected),requested=4*len(selected),
            completed=sum(r['completed']for r in selected),drc_clean=sum(r['drc_clean']for r in selected),
            mean=mean,sd_across_training_seed_means=sd,
            frozen_generator=group=='flowplace_r0'))
    output=plan_path.parent/'reports'/('route_job_'+os.environ['SLURM_JOB_ID'])
    output.mkdir(parents=True,exist_ok=False);shutil.copyfile(__file__,output/Path(__file__).name)
    atomic_json(output/'evidence.json',dict(status='all_predeclared_training_seeds_accounted',
        plan_sha256=sha(plan_path),source_sha256=sha(__file__),bindings=bindings,groups=groups,
        common_physical_flow=common_flow,single_shared_best_TRAIN_reference=reference,
        physical_audit_source_sha256=sha(Path(__file__).with_name('graph_o2o_final_evidence.py')),
        legacy_exact_repeat_plan_sha256=sha(LEGACY_REPEAT_PLAN),
        actual_training_evidence=dict(path=str(grt_path),sha256=sha(grt_path)),
        scope='bp_fe, common cap20; noises averaged within training seed before across-seed statistics',
        frozen_FlowPlace_not_independently_retrained=True,goal_completed=False))
    write_csv(output/'per_layout.csv',layouts);write_csv(output/'per_training_seed.csv',rows)
    lines=['# Flow-Q：多个训练 seed 的最终布线结果','',
        '统一 bp_fe、OpenROAD/Nangate45、八线程和详细布线 cap20，原始宏坐标在后端继续保持。',
        '每个训练 seed 内先平均四个预先固定的评估噪声，再汇总训练 seed；标准差来自训练 seed 均值。',
        '原始 FlowPlace 为同一冻结生成器，没有重新训练；最佳 TRAIN 是同一个实测参考，不重复计为新样本。','',
        '| 方法 | 完成/请求 | DRC=0 | rWL（m） | seed 间标准差（m） | WNS（ns） | TNS（ns） | 功耗（W） | 面积（μm²） | DRC均值 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for group in groups:
        m=group['mean'];sd=group['sd_across_training_seed_means']
        values=([m['DRT_WL']/1e6,sd['DRT_WL']/1e6,*[m[k]for k in KEYS[1:]]]
                if m['DRT_WL']is not None else[None]*7)
        cells=' | '.join('—'if v is None else f'{v:.6g}'for v in values)
        lines.append(f"| {group['group']} | {group['completed']}/{group['requested']} | {group['drc_clean']} | {cells} |")
    lines+=['','所有失败保留在请求分母中；存在失败时不报告完整组均值。非零 DRC 不剔除。',
        '线长、时序和功耗分别判断，不能用一项改善推断全面 PPA 改善。此报告仍是单电路结果。','']
    (output/'summary.md').write_text('\n'.join(lines))
    fig,ax=plt.subplots(figsize=(8,4.5))
    for i,group in enumerate(groups):
        if group['mean']['DRT_WL']is None:continue
        y=group['mean']['DRT_WL']/1e6;sd=group['sd_across_training_seed_means']['DRT_WL']/1e6
        ax.errorbar(i,y,yerr=sd,fmt='o',capsize=5,color='#3972a2')
        within=[r['DRT_WL']/1e6 for r in rows if r['group']==group['group']and r['DRT_WL']is not None]
        ax.scatter([i]*len(within),within,marker='x',color='#a95843',alpha=.75)
    ax.set_xticks(range(len(groups)),['FlowPlace','BC offline','FQL offline','BC online','FQL online'])
    ax.set_ylabel('Detailed-route wirelength (m)')
    ax.set_title(f'bp_fe; {len(experiments)} training seeds; common cap20')
    ax.grid(axis='y',alpha=.2);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(output/('final_rwl.'+ext),dpi=180)
    plt.close(fig)
    print(json.dumps(dict(status='reported',output=str(output),goal_completed=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True)
    p.add_argument('--grt-evidence',required=True)
    args=p.parse_args()
    if 'SLURM_JOB_ID'not in os.environ:raise RuntimeError('Aggregate final evidence through Slurm')
    run(args)
