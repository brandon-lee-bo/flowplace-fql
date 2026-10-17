"""Measure previously unlabeled TRAIN families without changing their actions.

These are fixed offline behavior examples, not on-policy feedback and not new
synthetic reward labels. Source validation/heldout families remain untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import ROOT, load_requests, validate_request
from .graph_o2o_collect import audit_outcome, summary


def sources(experiment):
    exp = Path(experiment).resolve()
    cfg = json.loads((exp/'config.json').read_text())
    plan = json.loads((exp/'selection_plan.json').read_text())
    if plan['config_sha256'] != sha(exp/'config.json'):
        raise ValueError('Offline expansion configuration changed')
    for item in plan['source_bindings'].values():
        if sha(item['path']) != item['sha256']:
            raise ValueError('Frozen expansion source changed: '+item['path'])
    source = Path(plan['source_bindings']['behavior_dataset']['path'])
    with np.load(source, allow_pickle=False) as a:
        arrays = {key:a[key].copy() for key in [
            'raw_action', 'action', 'noise', 'critic_parent_index', 'variant', 'critic_split', 'split']}
    selected = plan['selected_source_rows']
    if len(selected) != 320 or len(set(selected)) != len(selected):
        raise ValueError('Expansion must retain all 320 frozen requested rows')
    parents = arrays['critic_parent_index'][selected]
    if (set(parents.tolist()) != set(range(32,192)) or
            not np.all(arrays['critic_split'][selected] == 0) or
            not np.all(arrays['split'][parents] == 0) or
            set(arrays['variant'][selected].tolist()) != {0,1}):
        raise ValueError('Expansion crossed the predeclared TRAIN-family/variant boundary')
    pairs = list(zip(parents.tolist(), arrays['variant'][selected].tolist()))
    if len(set(pairs)) != 320:
        raise ValueError('Repeated TRAIN family/variant')
    return exp, cfg, plan, arrays


def capture(exp, stage):
    folder=exp/'sources'/(stage+'_job_'+os.environ['SLURM_JOB_ID'])
    folder.mkdir(parents=True, exist_ok=False)
    for name in ['graph_o2o_offline_expand.py','graph_o2o_offline_expand.slurm',
                 'graph_o2o_collect.py','graph_o2o_backend.py']:
        shutil.copyfile(Path(__file__).with_name(name),folder/name)
    atomic_json(folder/'sha256.json',{p.name:sha(p)for p in folder.iterdir()if p.is_file()})


def export(exp, cfg, plan, arrays):
    import torch
    from analysis.placement_metrics import MacroHPWL, legality
    from baseline.api import CHECKPOINT, load_problem, save_sample
    from backend.scripts.export_macro_def import convert

    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    problem=load_problem('bp_fe','cpu')
    metric=MacroHPWL(problem.benchmark_info)
    dest=exp/'seed_0/round_0/requests'
    if (dest/'completed.json').exists():
        rows,_=load_requests(exp,0,0)
        for row in rows:validate_request(row)
        return
    dest.mkdir(parents=True,exist_ok=True)
    checkpoint_sha=sha(CHECKPOINT)
    if checkpoint_sha != plan['source_bindings']['pretrained_checkpoint']['sha256']:
        raise ValueError('The source teacher checkpoint changed')
    records=[]
    for index,source_row in enumerate(plan['selected_source_rows']):
        parent=int(arrays['critic_parent_index'][source_row]);variant=int(arrays['variant'][source_row])
        folder=dest/'frozen_behavior/collect'/str(index)
        if (folder/'request.json').exists():
            row=json.loads((folder/'request.json').read_text())
            if row['source_row_index'] != source_row or row['selection_plan_sha256'] != sha(exp/'selection_plan.json'):
                raise ValueError('Interrupted export differs from the frozen selection')
            validate_request(row);records.append(row);continue
        folder.mkdir(parents=True,exist_ok=True)
        raw=arrays['raw_action'][source_row].reshape(-1,2)
        action=arrays['action'][source_row].reshape(-1,2)
        noise=arrays['noise'][parent].reshape(-1,2)
        geometry=legality(action,metric.sizes)
        row=dict(arm='frozen_behavior',seed=0,round=0,role='collect',index=index,
            graph='bp_fe',terminal_mask=0,raw_action=raw.tolist(),executed_action=action.tolist(),
            predicted_q=None,source_checkpoint=str(CHECKPOINT),checkpoint_sha256=checkpoint_sha,
            geometry=geometry,action_sha256=hashlib.sha256(np.asarray(action,dtype='<f4').reshape(-1).tobytes()).hexdigest(),
            sample_dir=str(folder),run_id=f'go2o_offexpand_v1_row{source_row:04d}',
            stage='global_route_only',training_feedback_eligible=True,macro_hpwl=float(metric(action)),
            data_collection_kind='offline_augmentation_of_frozen_TRAIN_behavior',not_online_policy_feedback=True,
            source_row_index=source_row,source_noise_family=parent,source_variant=variant,
            source_dataset_sha256=plan['source_bindings']['behavior_dataset']['sha256'],
            selection_plan_sha256=sha(exp/'selection_plan.json'),
            legalizer='Reuse exact already-legalized source action; no second legalization and no old proxy reward reuse')
        np.savez_compressed(folder/'actions.npz',noise=noise,raw_action=raw,action=action)
        if geometry['legal']:
            save_sample(folder,torch.from_numpy(action),torch.from_numpy(noise),problem,row)
            path=folder/f'bp_fe_job{os.environ["SLURM_JOB_ID"]}.def'
            audit=convert('bp_fe',folder/'macro_physical.json',path)
            row.update(def_path=str(path),def_sha256=sha(path),rounded_geometry=audit['legal_no_halo'])
            if not audit['legal_no_halo']:row['pre_backend_failure']='rounded_DEF_geometry'
        else:
            row['pre_backend_failure']='normalized_geometry'
        atomic_json(folder/'request.json',row);records.append(row)
    tasks=dest/'tasks.jsonl'
    tasks.write_text(''.join(json.dumps(row)+'\n'for row in records))
    atomic_json(dest/'completed.json',dict(status='completed',count=len(records),
        tasks_sha256=sha(tasks),config_sha256=sha(exp/'config.json'),slurm_job_id=os.environ['SLURM_JOB_ID']))
    print(json.dumps(dict(status='exported',requests=len(records),new_train_families=160)),flush=True)


def report(exp,cfg,plan,arrays):
    rows,_=load_requests(exp,0,0)
    if [row['source_row_index']for row in rows] != plan['selected_source_rows']:
        raise ValueError('Measured denominator or source rows changed')
    scale=plan['reward_scale_frozen_original_TRAIN']
    outcomes=[];replay=[]
    for request in rows:
        validate_request(request)
        source=request['source_row_index']
        if (not np.array_equal(np.asarray(request['raw_action'],dtype=np.float32).reshape(-1),arrays['raw_action'][source]) or
                not np.array_equal(np.asarray(request['executed_action'],dtype=np.float32).reshape(-1),arrays['action'][source])):
            raise ValueError('Scored action differs from the fixed offline source')
        row=audit_outcome(request,cfg,scale);outcomes.append(row)
        replay.append(dict(raw_action=row['raw_action'],executed_action=row['executed_action'],reward=row['reward'],
            status=row['status'],terminal_mask=0,outcome_path=row['outcome_path'],outcome_sha256=row['outcome_sha256'],
            action_sha256=row['action_sha256'],source_row_index=source,source_noise_family=request['source_noise_family'],
            source_variant=request['source_variant'],metrics=row['metrics']))
    dest=exp/'report';dest.mkdir(exist_ok=True)
    atomic_json(dest/'offline_replay.json',dict(status='all_requested_accounted',
        collection_kind='additional_offline_TRAIN_labels_not_on_policy',config_sha256=sha(exp/'config.json'),
        selection_plan_sha256=sha(exp/'selection_plan.json'),reward_scale=scale,rows=replay,
        new_TRAIN_families=160,new_requested_labels=320,validation_heldout_used=False))
    result=dict(status='all_requested_accounted',requested=len(rows),summary=summary(outcomes),
        selection_plan_sha256=sha(exp/'selection_plan.json'),offline_replay_sha256=sha(dest/'offline_replay.json'),
        original_training_labels=64,expanded_training_requested_labels=384,new_TRAIN_families=160,
        validation_heldout_used=False,online_feedback=False,slurm_job_id=os.environ['SLURM_JOB_ID'])
    atomic_json(dest/'evidence.json',result)
    (dest/'summary.md').write_text('\n'.join([
        '# 增加离线 TRAIN 的真实后端标签','',
        '原始 64 个标签覆盖 32 个训练噪声家族；本次给其余 160 个 TRAIN 家族各补充两个固定布局的真实评分，共 320 个请求。',
        '原始动作与已合法化坐标均不改变，所有请求经过同一原生导入、八线程 OpenROAD/Nangate45 GRT 流程。',
        '不读取原合成数据的 proxy reward，不使用验证或测试家族，也不将这些固定行为数据称为在线策略反馈。','',
        f"成功 {result['summary']['successful']}/{len(rows)}；失败保留在分母和 replay 中，不编造 PPA。",
        '这一步只增加真实训练数据，不代表新的策略质量已经改善。','']))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--stage',choices=['export','report'],required=True)
    args=parser.parse_args()
    if 'SLURM_JOB_ID'not in os.environ:raise RuntimeError('Export and audit through Slurm')
    exp,cfg,plan,arrays=sources(args.experiment);capture(exp,args.stage)
    (export if args.stage=='export'else report)(exp,cfg,plan,arrays)
