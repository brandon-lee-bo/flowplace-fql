"""Audit the completed single-seed pilot and export tables and scientific plots.

This report is an experiment artifact, not a completion claim for the broader
research goal. Four evaluation noises do not become four training seeds.
"""
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

from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_collect import collect


def save_csv(path, rows):
    if not rows:
        return
    with path.open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def fmt(value, digits=6):
    return '—' if value is None else f'{value:.{digits}g}'


def main(args):
    exp=Path(args.experiment).resolve()
    cfg=json.loads((exp/'config.json').read_text())
    seed=cfg['seed']
    rounds=[collect(exp,seed,i) for i in range(cfg['online_rounds']+1)]
    route_path=exp/'full_route/summary.json';route=json.loads(route_path.read_text())
    if (route['requested']!=21 or len(route['records'])!=21 or
        route['plan_sha256']!=sha(exp/'full_route_plan.json')):
        raise ValueError('Incomplete or changed full-route plan')
    full_rows=[]
    for row in route['records']:
        task,result=row['task'],row['result']
        values=result.get('metrics') or {}
        if result['status']=='completed':
            run=Path(result['physical_run'])
            if (sha(run/'manifest.json')!=result['physical_manifest_sha256'] or
                sha(run/'metrics.json')!=result['metrics_sha256']):
                raise ValueError('Final physical evidence changed')
        full_rows.append(dict(group=task['group'],index=task['index'],status=result['status'],
            rWL_m=None if values.get('DRT_WL') is None else values['DRT_WL']/1e6,
            WNS_ns=values.get('WNS'),TNS_ns=values.get('TNS'),Power_W=values.get('Power'),
            StdCellArea_um2=values.get('StdCellArea'),DRC=values.get('DRC'),
            run=result.get('physical_run'),manifest_sha256=result.get('physical_manifest_sha256')))
    training={}
    for number in range(1,cfg['online_rounds']+1):
        for arm in ['bc','fql']:
            path=exp/f'seed_{seed}/round_{number}/{arm}/completed.json'
            row=json.loads(path.read_text())
            if row['online_training_rows']!=number*cfg['online_samples']:
                raise ValueError('Unexpected actual online training data count')
            if not all(row['parameter_changes'][name]['changed_parameter_tensors']>0
                       for name in ['flow','actor','critic']):
                raise ValueError('Online checkpoint lacks an actual model update')
            training[f'{arm}/round_{number}']=dict(completion_sha256=sha(path),details=row)
    reference_path=exp/'best_replay_reference/report.json'
    reference=json.loads(reference_path.read_text())
    if reference['status']!='completed':
        raise ValueError('Equal-budget search references have not completed')
    folder=exp/'reports'/('pilot_job_'+os.environ['SLURM_JOB_ID']);folder.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,folder/Path(__file__).name)
    atomic_json(folder/'evidence.json',dict(config_sha256=sha(exp/'config.json'),
        grt_rounds=rounds,full_route_report_sha256=sha(route_path),
        search_reference_report_sha256=sha(reference_path),online_training=training,
        goal_completed=False,scope='One training seed, fixed bp_fe netlist, four evaluation noises per method'))
    save_csv(folder/'detailed_route_per_layout.csv',full_rows)
    grt=[]
    for number,report in enumerate(rounds):
        for arm in ['bc','fql']:
            group=report['groups'][arm+'/evaluation'];mean=group['mean'];std=group['sample_std']
            grt.append(dict(round=number,arm=arm,available_measured_TRAIN_layouts=64+number*cfg['online_samples'],
                completed=group['successful'],requested=group['count'],
                rWL_m=None if mean is None else mean['rWL_um']/1e6,
                noise_std_m=None if std is None else std['rWL_um']/1e6,
                WNS_ns=None if mean is None else mean['WNS_ns'],TNS_ns=None if mean is None else mean['TNS_ns']))
    save_csv(folder/'global_route_learning_curve.csv',grt)
    fig,ax=plt.subplots(figsize=(6.5,4))
    for arm,color in [('bc','#4774b7'),('fql','#b54c3e')]:
        rows=[r for r in grt if r['arm']==arm and r['rWL_m'] is not None]
        ax.errorbar([r['available_measured_TRAIN_layouts'] for r in rows],[r['rWL_m'] for r in rows],
            yerr=[r['noise_std_m'] for r in rows],marker='o',capsize=4,label=arm.upper(),color=color)
    original=rounds[0]['groups']['flowplace/evaluation']['mean']
    if original:
        ax.axhline(original['rWL_um']/1e6,color='#646464',ls='--',label='Original FlowPlace')
    ax.set(xlabel='Measured layouts available to TRAIN/replay',ylabel='Global-route wirelength (m)',
           title='bp_fe, training seed 0; SD across 4 fixed noises')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(folder/('grt_learning_curve.'+ext),dpi=180)
    plt.close(fig)
    groups=[('flowplace_r0','FlowPlace'),('bc_r0','BC offline'),('fql_r0','FQL offline'),
            ('bc_r2','BC online'),('fql_r2','FQL online')]
    fig,ax=plt.subplots(figsize=(8,4.2))
    for i,(key,label) in enumerate(groups):
        rows=[r for r in full_rows if r['group']==key]
        successful=[r for r in rows if r['status']=='completed']
        if len(successful)!=len(rows):
            ax.text(i,.04,'Incomplete',rotation=90,ha='center',transform=ax.get_xaxis_transform());continue
        y=[r['rWL_m'] for r in successful]
        ax.errorbar(i,statistics.mean(y),yerr=statistics.stdev(y),fmt='s',capsize=5,color='#333333')
        ax.scatter([i+(.04*(j-1.5)) for j in range(len(y))],y,
            c=['#b54c3e' if r['DRC'] else '#4774b7' for r in successful],s=32)
    best=route['groups']['best_train']['mean']
    if best:
        ax.axhline(best['DRT_WL']/1e6,color='#578456',ls='--',label='One fixed best-TRAIN layout')
    ax.set_xticks(range(len(groups)),[label for _,label in groups])
    ax.set(ylabel='Detailed-route wirelength (m)',title='Common cap20; red points have nonzero DRC')
    ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(folder/('detailed_route_comparison.'+ext),dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.4,4.8))
    palette=['#777777','#4774b7','#b54c3e','#68a7c5','#dd9852']
    for (key,label),color in zip(groups,palette):
        successful=[r for r in full_rows if r['group']==key and r['status']=='completed']
        for row in successful:
            ax.scatter(row['rWL_m'],abs(row['TNS_ns']),color=color,
                       marker='x' if row['DRC'] else 'o',s=42)
            ax.annotate(str(row['index']),(row['rWL_m'],abs(row['TNS_ns'])),
                        xytext=(3,3),textcoords='offset points',fontsize=8,color=color)
        ax.scatter([],[],color=color,label=label)
    for row in full_rows:
        if row['group']=='best_train' and row['status']=='completed':
            ax.scatter(row['rWL_m'],abs(row['TNS_ns']),marker='*',s=130,
                       color='#578456',label='Fixed best-TRAIN reference')
    ax.set(xlabel='Detailed-route wirelength (m)',ylabel='Absolute total negative slack (ns)',
           title='Lower left is better; labels are noise indices, x means nonzero DRC')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(folder/('detailed_route_timing_tradeoff.'+ext),dpi=180)
    plt.close(fig)
    lines=['# FlowPlace 与 Flow-Q 单电路初步实验结果','',
        'bp_fe，训练 seed=0。每种策略使用四个固定评估噪声；噪声不是独立训练 seed。',
        '真实反馈进入 replay，两个在线阶段均核验了 flow、一步策略和 Q 参数更新。','',
        '## 全局布线学习过程','',
        '| 阶段 | 方法 | 已用于训练的真实布局数 | 成功/请求 | GRT 线长（m） | 噪声标准差（m） | WNS（ns） | TNS（ns） |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in grt:
        lines.append(f"| {row['round']} | {row['arm']} | {row['available_measured_TRAIN_layouts']} | "+
            f"{row['completed']}/{row['requested']} | "+' | '.join(fmt(row[k]) for k in
                ['rWL_m','noise_std_m','WNS_ns','TNS_ns'])+' |')
    lines += ['', '每种方法另有192条未标记 teacher 行为数据。评估布局不进入训练；最后额外采集的8个布局不计入已用训练量。',
        '', '## 统一详细布线结果','',
        '| 方法 | 完成/请求 | DRC=0 | rWL（m） | WNS（ns） | TNS（ns） | 功耗（W） | 面积（μm²） | DRC 均值 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for key,label in [('best_train','最佳 TRAIN 单个参考'),*groups]:
        group=route['groups'][key];mean=group['mean']
        cells='— | — | — | — | — | —' if mean is None else ' | '.join(fmt(v) for v in
            [mean['DRT_WL']/1e6,mean['WNS'],mean['TNS'],mean['Power'],mean['StdCellArea'],mean['DRC']])
        lines.append(f"| {label} | {group['completed']}/{group['requested']} | {group['drc_clean']} | {cells} |")
    lines += ['', '统一 OpenROAD/Nangate45、8线程和详细布线 cap20。与历史 Table2 cap4 分开；有失败时不报告完整组均值。',
              '非零 DRC 和负时序余量按原值保留，指标变好不等于时序收敛或制造签核完成。',
              '', '## 相同采集预算的布局保留对照','',
              '| 数据来源 | 成功/候选 | 选中布局 | 最终 rWL（m） | DRC |',
              '| --- | --- | --- | --- | --- |']
    for arm,choice in reference['choices'].items():
        outcome=reference['outcomes'][arm]['outcome']
        lines.append(f"| {arm} | {choice['successful']}/{choice['requested']} | {choice['selected']['key']} | "+
                     f"{outcome['metrics']['DRT_WL']/1e6:.6f} | {outcome['metrics']['DRC']} |")
    lines += ['', '各取64个 TRAIN 加自身16个已采集标签，只按实测 GRT 线长选一个布局。相同布局的复用不增加独立样本数。',
              '', '原始逐布局数据见同目录 CSV，证据和参数变化见 evidence.json。仍需根据实际效果决定改进，并扩展独立训练 seed 和电路。','']
    (folder/'notion_results.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='single_seed_pilot_reported',folder=str(folder),goal_completed=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True)
    args=parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:raise RuntimeError('Use Slurm')
    main(args)
