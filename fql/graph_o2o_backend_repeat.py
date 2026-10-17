"""Check backend variation using exactly repeated, already frozen DEF inputs.

These repeats are tool measurements, not new noises or training seeds. They
never enter replay and do not select layouts or checkpoints.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics

from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import load_requests
from .graph_o2o_collect import audit_outcome, METRICS


def export(experiment):
    cfg=json.loads((experiment/'config.json').read_text())
    source=Path(cfg['source_experiment'])
    if sha(source/'config.json')!=cfg['source_config_sha256']:
        raise ValueError('Frozen source configuration changed')
    folder=experiment/'seed_0/round_0/requests'
    if (folder/'completed.json').exists():
        raise FileExistsError('Already exported; use existing tasks')
    folder.mkdir(parents=True,exist_ok=True)
    records=[]
    for arm in ['bc','fql']:
        original=source/f'seed_0/round_0/requests/{arm}/evaluation/0'
        request=json.loads((original/'request.json').read_text())
        if sha(original/'outcome.json')!=cfg['source_outcome_sha256'][arm]:
            raise ValueError('Frozen source outcome changed')
        audit_outcome(request,cfg,cfg['reward_scale'])
        for i in range(cfg['repeats_per_input']):
            dest=folder/arm/'evaluation'/str(i);dest.mkdir(parents=True,exist_ok=False)
            shutil.copyfile(original/'actions.npz',dest/'actions.npz')
            row=dict(request,index=i,sample_dir=str(dest),
                run_id=f'go2o_toolrepeat_v1_{arm}_repeat_{i}',
                original_request=str(original/'request.json'),original_request_sha256=sha(original/'request.json'),
                original_outcome=str(original/'outcome.json'),original_outcome_sha256=sha(original/'outcome.json'),
                repeat_semantics='Exact same raw action, legalized coordinates and DEF; no new policy inference')
            atomic_json(dest/'request.json',row);records.append(row)
    tasks=folder/'tasks.jsonl';tasks.write_text(''.join(json.dumps(row)+'\n' for row in records))
    atomic_json(folder/'completed.json',dict(count=len(records),tasks_sha256=sha(tasks),
        config_sha256=sha(experiment/'config.json'),training_feedback_eligible=False,
        all_requested_required=True,actual_policy_inference_calls=0))
    print(json.dumps(dict(status='exported_exact_input_repeats',count=len(records))),flush=True)


def report(experiment):
    cfg=json.loads((experiment/'config.json').read_text())
    requests,tasks=load_requests(experiment,0,0)
    outcomes=[audit_outcome(row,cfg,cfg['reward_scale']) for row in requests]
    groups={}
    for arm in ['bc','fql']:
        selected=[row for row in outcomes if row['arm']==arm]
        # The reference path belongs to the immutable request, not new outcome metadata.
        original_path=Path(next(row['original_outcome'] for row in requests if row['arm']==arm))
        original=json.loads(original_path.read_text())
        successful=[row for row in selected if row['status']=='success']
        statistics_by_metric={}
        if len(successful)==len(selected):
            for metric in METRICS:
                values=[row['metrics'][metric] for row in successful]
                statistics_by_metric[metric]=dict(min=min(values),max=max(values),
                    mean=statistics.mean(values),sample_std=statistics.stdev(values),
                    original=original['metrics'][metric],
                    exact_original_match=all(value==original['metrics'][metric] for value in values))
        groups[arm]=dict(requested=len(selected),successful=len(successful),
            failures=[row['failure_class'] for row in selected if row['status']!='success'],
            metrics=statistics_by_metric,
            records=[dict(outcome=row['outcome_path'],sha256=row['outcome_sha256']) for row in selected])
    atomic_json(experiment/'report.json',dict(status='all_requested_accounted',groups=groups,
        config_sha256=sha(experiment/'config.json'),tasks_sha256=sha(tasks),
        interpretation='Repeated physical evaluations of two fixed inputs; not independent policy/noise replicates',
        training_feedback_replayed=False,source_sha256=sha(__file__)))
    lines=['# 固定输入的后端重复性检查','',
        '每种方法固定同一个已有布局，完整重复三次 GRT；没有重新推理或重新合法化，不进入训练。','',
        '| 固定输入 | 成功/请求 | 原 GRT 线长（μm） | 重跑最小值 | 重跑最大值 | 标准差 |','| --- | --- | --- | --- | --- | --- |']
    for arm,group in groups.items():
        metric=group['metrics'].get('rWL_um')
        cells='— | — | — | —' if metric is None else ' | '.join(f'{metric[k]:.6g}' for k in ['original','min','max','sample_std'])
        lines.append(f"| {arm}，评估噪声0 | {group['successful']}/{group['requested']} | {cells} |")
    lines += ['', '结论仅覆盖这两个固定输入、当前工具和八线程设置。其他 PPA 和拥塞指标原值见 report.json。','']
    (experiment/'summary.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='reported',groups=groups)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--phase',choices=['export','report'],required=True)
    args=parser.parse_args()
    if args.phase=='report' and 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use Slurm for physical evidence aggregation')
    experiment=Path(args.experiment).resolve()
    (export if args.phase=='export' else report)(experiment)
