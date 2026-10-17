"""Compare equal-update-budget offline continuation and actual online FQL."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import statistics
import numpy as np

from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import load_requests
from .graph_o2o_collect import audit_outcome


def run(experiment):
    cfg=json.loads((experiment/'config.json').read_text());source=Path(cfg['source_experiment'])
    if sha(source/'config.json')!=cfg['source_config_sha256']:
        raise ValueError('Changed main experiment')
    if sha(source/'seed_0/round_0/fql/checkpoint.pt')!=cfg['parent_checkpoint_sha256']:
        raise ValueError('Different offline parent')
    training=[]
    parent_sha=cfg['parent_checkpoint_sha256']
    for stage in [1,2]:
        path=experiment/f'training_stage_{stage}/completed.json'
        record=json.loads(path.read_text())
        if record['online_training_rows']!=0 or record['measured_train_rows']!=64:
            raise ValueError('Offline-only control consumed new labels')
        if sha(path.with_name('checkpoint.pt'))!=record['checkpoint_sha256']:
            raise ValueError('Control checkpoint changed')
        if record['parent_checkpoint_sha256']!=parent_sha:
            raise ValueError('Control did not continue the paired offline checkpoint chain')
        logs=[json.loads(line)for line in path.with_name('training.jsonl').read_text().splitlines()]
        if logs[-1]['step']!=cfg['online_update_steps'] or any(row['online_training_rows'] for row in logs):
            raise ValueError('Actual extra-update budget or offline-only training differs')
        parent_sha=record['checkpoint_sha256']
        training.append(dict(path=str(path),sha256=sha(path),record=record))
    rows,tasks=load_requests(experiment,0,0)
    if [row['index'] for row in rows]!=list(range(cfg['evaluation_samples'])):
        raise ValueError('Incorrect paired-noise denominator')
    for row in rows:
        original=source/f"seed_0/round_2/requests/fql/evaluation/{row['index']}/actions.npz"
        with np.load(Path(row['sample_dir'])/'actions.npz',allow_pickle=False)as control, np.load(original,allow_pickle=False)as online:
            if not np.array_equal(control['noise'],online['noise']):
                raise ValueError('Control and online policies used different evaluation noises')
    scale=json.loads((source/'seed_0/round_0/training_status.json').read_text())['reward_scale']
    grt=[audit_outcome(row,cfg,scale) for row in rows]
    final=[]
    for i in range(cfg['evaluation_samples']):
        path=experiment/f'full_route/noise_{i}/outcome.json';record=json.loads(path.read_text())
        if record['status']=='completed':
            run=Path(record['physical_run'])
            if (sha(run/'manifest.json')!=record['physical_manifest_sha256'] or
                sha(run/'metrics.json')!=record['metrics_sha256'] or record['spec']['route_end_iteration']!=20):
                raise ValueError('Changed final physical evidence or route budget')
        elif record['status']!='upstream_physical_failure':
            raise ValueError('Unresolved final physical outcome')
        final.append(record)
    main=json.loads((source/'full_route/summary.json').read_text())
    if main['plan_sha256']!=sha(source/'full_route_plan.json'):
        raise ValueError('Changed primary route cohort')
    summaries=[]
    for key,label in [('fql_r0','FQL 离线'),('fql_r2','FQL 在线两轮')]:
        group=main['groups'][key]
        summaries.append(dict(method=label,requested=group['requested'],completed=group['completed'],
            drc_clean=group['drc_clean'],**(group['mean']or{key:None for key in ['DRT_WL','WNS','TNS','Power','StdCellArea','DRC']})))
    good=[row for row in final if row['status']=='completed']
    means={key:statistics.mean(row['metrics'][key]for row in good) for key in ['DRT_WL','WNS','TNS','Power','StdCellArea','DRC']} if len(good)==len(final) else {key:None for key in ['DRT_WL','WNS','TNS','Power','StdCellArea','DRC']}
    summaries.append(dict(method='FQL 只继续离线训练',requested=len(final),completed=len(good),
        drc_clean=sum(row['route_drc_clean']for row in good),**means))
    online,control=summaries[1:]
    improvement=None if online['DRT_WL'] is None or control['DRT_WL'] is None else 100*(control['DRT_WL']-online['DRT_WL'])/control['DRT_WL']
    report=dict(status='all_requested_accounted',scope='One bp_fe training seed; four paired evaluation noises',
        config_sha256=sha(experiment/'config.json'),extra_updates_per_arm=1200,
        online_new_TRAIN_rows=16,control_new_TRAIN_rows=0,training=training,
        control_grt_outcomes=[dict(path=row['outcome_path'],sha256=row['outcome_sha256'])for row in grt],
        control_full_route=final,main_full_route_report_sha256=sha(source/'full_route/summary.json'),
        source_sha256=sha(__file__),
        summaries=summaries,online_rwl_improvement_over_extra_offline_percent=improvement,
        goal_completed=False,selection='Same frozen offline parent, final update step and four noises; no best-noise selection')
    atomic_json(experiment/'comparison.json',report)
    shutil.copyfile(__file__,experiment/('report_source_job_'+os.environ['SLURM_JOB_ID']+'.py'))
    with (experiment/'comparison.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    lines=['# 新反馈与额外训练步数的对照','',
        '同一个离线 FQL checkpoint 出发，继续离线训练与在线训练均额外更新1200步。在线版使用16个新增真实标签；离线继续版只用原有64个标签。两者均保留相同的192条额外行为数据。',
        '最终四个评估噪声固定，统一 OpenROAD/Nangate45、8线程和详细布线 cap20。','',
        '| 方法 | 完成/请求 | DRC=0 | rWL（m） | WNS（ns） | TNS（ns） | 功耗（W） | 面积（μm²） |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in summaries:
        values=[None if row['DRT_WL']is None else row['DRT_WL']/1e6,row['WNS'],row['TNS'],row['Power'],row['StdCellArea']]
        cells=' | '.join('—'if value is None else f'{value:.6g}'for value in values)
        lines.append(f"| {row['method']} | {row['completed']}/{row['requested']} | {row['drc_clean']} | {cells} |")
    lines += ['', '有失败时不报告完整组均值；非零 DRC 保留在原始结果。单 seed 的四个噪声不能证明收益稳定。',
              '比较的是“新增反馈”与“只多训练”的区别，尚需更多独立训练 seed 与电路。','']
    (experiment/'summary.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='reported',online_rwl_improvement_over_extra_offline_percent=improvement)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment',required=True)
    args=parser.parse_args()
    if 'SLURM_JOB_ID'not in os.environ:raise RuntimeError('Use Slurm')
    run(Path(args.experiment).resolve())
