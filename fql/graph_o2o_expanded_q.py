"""Compare 64/384-label critics on identical real, never-replayed layouts.

Diagnostic only: no fitting, model selection, policy updates or backend work.
The existing CPU/GPU training-MAE reproduction guard applies to every critic.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from .critic_gate import load_train_validation
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import ROOT, load_requests
from .graph_o2o_collect import audit_outcome
from .graph_o2o_coverage_q import score_checkpoint
from .graph_o2o_expanded_data import audited_expansion


def run(args):
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','1')))
    path=Path(args.plan).resolve();plan=json.loads(path.read_text())
    roots={name:Path(value).resolve()for name,value in plan['experiments'].items()}
    configs={name:json.loads((root/'config.json').read_text())for name,root in roots.items()}
    for name,root in roots.items():
        if sha(root/'config.json')!=plan['config_sha256'][name]:
            raise ValueError('The fixed-cohort experiment configuration changed')
    original,manifest,_=load_train_validation(ROOT/'datasets/bp_fe_grt_v1/dataset.npz')
    extra,binding=audited_expansion(roots['expanded384'],configs['expanded384'])
    expanded=dict(original,train=dict(original['train']))
    expanded['train']['raw_action']=np.concatenate([original['train']['raw_action'],
        np.asarray([r['raw_action']for r in extra],dtype=np.float32).reshape(320,-1)])
    expanded['train']['reward']=np.concatenate([original['train']['reward'],
        np.asarray([r['reward']for r in extra],dtype=np.float32)])
    if len(expanded['train']['reward'])!=384 or binding['reward_scale']!=manifest['reward_scale_train_only']:
        raise ValueError('Expanded fixed-cohort critic fit context differs')
    cache_path=ROOT/'baseline/runtime/benchmark_cache/bp_fe.pt'
    if sha(cache_path)!=manifest['source_proxy_manifest']['provenance']['cache_sha256']:
        raise ValueError('Graph-cache identity changed')
    cache=torch.load(cache_path,map_location='cpu',weights_only=False)
    cohorts={};bindings={}
    for declared in plan['evaluation_cohorts']:
        name,number,arm=declared['experiment'],declared['round'],declared['arm']
        requests,_=load_requests(roots[name],0,number)
        selected=[r for r in requests if r['role']=='evaluation'and r['arm']==arm]
        if [r['index']for r in selected]!=[0,1,2,3]or any(r['training_feedback_eligible']for r in selected):
            raise ValueError('Fixed evaluation denominator or replay eligibility changed')
        key=f'{name}/{arm}/r{number}'
        cohorts[key]=[audit_outcome(r,configs[name],manifest['reward_scale_train_only'])for r in selected]
        bindings[key]=[dict(path=r['outcome_path'],sha256=r['outcome_sha256'])for r in cohorts[key]]
    output=Path(plan['output']);output.mkdir(parents=True,exist_ok=True)
    records={};table=[];predictions=[]
    for name,root in roots.items():
        data=original if name=='original64'else expanded
        cfg=configs[name]
        for number in [0,1,2]:
            replay=json.loads((root/f'seed_0/replay_round_{number}.json').read_text())if number else None
            if replay and (replay['config_sha256']!=sha(root/'config.json')or
                    replay['for_training_round']!=number or replay['reward_scale']!=manifest['reward_scale_train_only']):
                raise ValueError('A saved fit uses a different replay/normalization')
            for arm in ['bc','fql']:
                rows=replay['arms'][arm]if replay else[]
                if len(rows)!=number*8:raise ValueError('Actual online replay budget changed')
                for row in rows:
                    outcome_path=Path(row['outcome_path'])
                    request=json.loads((outcome_path.parent/'request.json').read_text())
                    measured=audit_outcome(request,cfg,manifest['reward_scale_train_only'])
                    if (row['outcome_sha256']!=measured['outcome_sha256']or
                            row['raw_action']!=measured['raw_action']or row['reward']!=measured['reward']or
                            measured['role']!='collect'or not measured['training_feedback_eligible']or
                            measured['arm']!=arm):
                        raise ValueError('Saved critic fit contains unbound feedback')
                key=f'{name}/{arm}/r{number}'
                checkpoint=root/f'seed_0/round_{number}/{arm}/checkpoint.pt'
                completion=json.loads(checkpoint.with_name('completed.json').read_text())
                if name=='expanded384'and completion['expanded_training_binding_sha256']!=sha(root/'expanded_training_binding.json'):
                    raise ValueError('The expanded checkpoint did not bind all320 extra labels')
                record,values=score_checkpoint(checkpoint,cfg,data,cache,rows,cohorts)
                if record['training_rows'] != (64 if name == 'original64' else 384) + number*8:
                    raise ValueError('Recomputed training denominator differs')
                records[key]=record
                predictions +=[dict(fitted_critic=key,**value)for value in values]
                for cohort,metrics in record['scores'].items():
                    table.append(dict(fitted_critic=key,cohort=cohort,
                        **{k:metrics[k]for k in ['mae','rmse','spearman_r','optimism_bias','failed_physical_cases']}))
    snapshot=output/('source_job_'+os.environ['SLURM_JOB_ID']);snapshot.mkdir(exist_ok=False)
    for name in ['graph_o2o_expanded_q.py','graph_o2o_coverage_q.py','graph_o2o_models.py']:
        shutil.copyfile(Path(__file__).with_name(name),snapshot/name)
    atomic_json(output/'evidence.json',dict(status='fixed_cohort_critics_audited',
        plan_sha256=sha(path),original_dataset_sha256=manifest['dataset_sha256'],
        expanded_training_binding_sha256=sha(roots['expanded384']/'expanded_training_binding.json'),
        cache_sha256=sha(cache_path),cohort_outcomes=bindings,fits=records,
        no_fit_or_policy_selection=True,evaluation_feedback_replayed=False,heldout_labels_used=False,
        scope='Single bp_fe training seed; same actions across all fitted critics; reward MAE is not policy quality',
        goal_completed=False,slurm_job_id=os.environ['SLURM_JOB_ID']))
    for filename,rows in [('cohort_metrics.csv',table),('per_action_predictions.csv',predictions)]:
        with(output/filename).open('w')as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fixed=plan['primary_display_cohort']
    lines=['# 扩大离线标签后的固定布局 Q 检查','',
        '同一组真实布局分别交给64标签和384标签方案的各阶段 Q，避免将评估布局变化误认为模型改善。',
        '每个模型均在 CPU 重算其实际 TRAIN MAE，并核对训练时的保存值；所有评估反馈不进入 replay。',
        '模型之间数据预算不同，预测误差下降仍需另行验证是否转化为策略与最终 PPA 改善。','',
        f'下表固定使用 `{fixed}`，全部预定布局和逐布局预测另存为 CSV。','',
        '| Q 拟合数据与阶段 | MAE | 排序相关 | 乐观偏差 |',
        '| --- | --- | --- | --- |']
    for row in table:
        if row['cohort']!=fixed:continue
        rank='—'if row['spearman_r']is None else f"{row['spearman_r']:.4f}"
        lines.append(f"| {row['fitted_critic']} | {row['mae']:.6f} | {rank} | {row['optimism_bias']:.6f} |")
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='completed',output=str(output),fits=len(records),cohorts=len(cohorts))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True)
    args=p.parse_args()
    if 'SLURM_JOB_ID'not in os.environ:raise RuntimeError('Run checkpoint diagnostics through Slurm')
    run(args)
