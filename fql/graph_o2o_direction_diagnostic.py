"""Measure whether the saved Q's local action gradients survive legalization.

Four predeclared, previously untrained collection actions are perturbed along
the Q gradient and a fixed random direction, each with plus/minus signs. These
are diagnostics, never replay, policy selection or independent training seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from analysis.placement_metrics import MacroHPWL, legality
from baseline.api import legalize, save_sample
from backend.scripts.export_macro_def import convert
from .core import critic_diagnostics
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import load_requests
from .graph_o2o_collect import audit_outcome
from .graph_o2o_models import FeatureCritics
from .graph_o2o_train import initialize_problem


def export(args, exp, cfg, source):
    folder = exp/'seed_0/round_0/requests'
    if (folder/'completed.json').exists():
        load_requests(exp, 0, 0)
        return
    problem, data, manifest, cache, train, reward, _, _, _ = initialize_problem(args)
    saved = torch.load(cfg['checkpoint'], map_location='cuda', weights_only=False)
    if sha(cfg['checkpoint']) != cfg['checkpoint_sha256']:
        raise ValueError('Frozen gradient critic changed')
    critic = FeatureCritics(cache,train,reward,saved['critic']['movable'],cfg['critic_width']).cuda()
    critic.load_state_dict(saved['critic']);critic.eval();del saved
    source_cfg=json.loads((source/'config.json').read_text())
    old, _=load_requests(source,0,2)
    bases=[r for r in old if r['arm']=='fql' and r['role']=='collect' and r['index'] in cfg['base_indices']]
    if [r['index'] for r in bases] != cfg['base_indices']:
        raise ValueError('Changed predeclared base-action denominator')
    metric=MacroHPWL(problem.benchmark_info)
    generator=torch.Generator(device='cuda').manual_seed(cfg['direction_seed'])
    requests=[]
    for base in bases:
        observed=audit_outcome(base,source_cfg,manifest['reward_scale_train_only'])
        raw=torch.tensor(base['raw_action'],device='cuda',dtype=torch.float32).requires_grad_()
        q=critic(raw[None]).mean()
        gradient=torch.autograd.grad(q,raw)[0]*critic.movable[:,None]
        random=torch.randn(raw.shape,device='cuda',generator=generator)*critic.movable[:,None]
        for kind,vector in [('gradient',gradient),('random',random)]:
            norm=vector[critic.movable].square().mean().sqrt()
            if not torch.isfinite(norm) or norm<=0:
                raise ValueError('A diagnostic direction has zero/nonfinite norm')
            direction=vector/norm
            for sign in [-1,1]:
                index=base['index']*2+int(sign==1)
                target=folder/kind/'diagnostic'/str(index)
                if (target/'request.json').exists():
                    requests.append(json.loads((target/'request.json').read_text()));continue
                target.mkdir(parents=True,exist_ok=True)
                with torch.no_grad():
                    action_raw=(raw.detach()+sign*cfg['radius_rms']*direction).clamp(-1,1)
                    action_raw=torch.where(critic.movable[:,None],action_raw,raw.detach())
                    action=legalize(action_raw,problem)
                    predicted=critic(action_raw[None])[0].cpu().tolist()
                values=action.cpu().numpy();geometry=legality(values,metric.sizes)
                row=dict(arm=kind,seed=0,round=0,role='diagnostic',index=index,graph='bp_fe',
                    base_index=base['index'],sign=sign,base_outcome_path=observed['outcome_path'],
                    base_outcome_sha256=observed['outcome_sha256'],base_reward=observed['reward'],
                    base_predicted_q=float(q),direction_seed=cfg['direction_seed'],
                    radius_rms=cfg['radius_rms'],proposal_kind='controlled perturbation, not a new trained policy',
                    raw_action=action_raw.cpu().tolist(),executed_action=values.tolist(),predicted_q=predicted,
                    source_checkpoint=cfg['checkpoint'],checkpoint_sha256=cfg['checkpoint_sha256'],
                    geometry=geometry,terminal_mask=0,training_feedback_eligible=False,
                    action_sha256=hashlib.sha256(np.asarray(values,dtype='<f4').reshape(-1).tobytes()).hexdigest(),
                    sample_dir=str(target),run_id=f"go2o_direction_v1_b{base['index']}_{kind}_{'plus'if sign==1 else'minus'}",
                    stage='global_route_only',macro_hpwl=float(metric(values)),
                    actual_raw_step_rms=float((action_raw-raw.detach())[critic.movable].square().mean().sqrt()),
                    executed_step_rms=float((action-torch.tensor(base['executed_action'],device='cuda'))[critic.movable].square().mean().sqrt()))
                with np.load(Path(base['sample_dir'])/'actions.npz',allow_pickle=False)as original:
                    noise=original['noise'].copy()
                np.savez_compressed(target/'actions.npz',noise=noise,raw_action=action_raw.cpu().numpy(),action=values)
                if geometry['legal']:
                    save_sample(target,action,torch.tensor(noise,device='cuda'),problem,row)
                    path=target/('bp_fe_job'+os.environ['SLURM_JOB_ID']+'.def')
                    audit=convert('bp_fe',target/'macro_physical.json',path)
                    row.update(def_path=str(path),def_sha256=sha(path),rounded_geometry=audit['legal_no_halo'])
                    if not audit['legal_no_halo']:row['pre_backend_failure']='rounded_DEF_geometry'
                else:row['pre_backend_failure']='normalized_geometry'
                atomic_json(target/'request.json',row);requests.append(row)
    if len(requests)!=16:
        raise ValueError('Incomplete direction-comparison requests')
    task_path=folder/'tasks.jsonl'
    task_path.write_text(''.join(json.dumps(r)+'\n'for r in requests))
    atomic_json(folder/'completed.json',dict(status='completed',count=16,tasks_sha256=sha(task_path),
        config_sha256=sha(exp/'config.json'),slurm_job_id=os.environ['SLURM_JOB_ID']))
    print(json.dumps(dict(status='exported',count=16,policy_changed=False)),flush=True)


def report(args, exp, cfg, source):
    import csv
    requests,_=load_requests(exp,0,0)
    manifest=json.loads((Path(__file__).resolve().parents[1]/'datasets/bp_fe_grt_v1/manifest.json').read_text())
    outcomes={};bindings=[]
    for request in requests:
        path=Path(request['base_outcome_path'])
        if sha(path)!=request['base_outcome_sha256']:
            raise ValueError('Base measurement changed')
        row=audit_outcome(request,cfg,manifest['reward_scale_train_only'])
        outcomes[(request['base_index'],request['arm'],request['sign'])]=(request,row)
        bindings.append(dict(path=row['outcome_path'],sha256=row['outcome_sha256']))
    pairs=[]
    for index in cfg['base_indices']:
        for kind in ['gradient','random']:
            pr,plus=outcomes[(index,kind,1)];mr,minus=outcomes[(index,kind,-1)]
            success=plus['status']==minus['status']=='success'
            predicted=float(np.mean(plus['predicted_q'])-np.mean(minus['predicted_q']))
            measured=plus['reward']-minus['reward'] if success else None
            pairs.append(dict(base_index=index,direction=kind,plus_status=plus['status'],minus_status=minus['status'],
                predicted_plus_minus=predicted,measured_plus_minus=measured,
                prediction_sign_agrees=None if measured is None else bool(predicted*measured>0),
                plus_vs_base_reward=None if not success else plus['reward']-pr['base_reward'],
                minus_vs_base_reward=None if not success else minus['reward']-mr['base_reward'],
                plus_executed_step_rms=pr['executed_step_rms'],minus_executed_step_rms=mr['executed_step_rms'],
                same_legalized_layout=pr['action_sha256']==mr['action_sha256']))
    folder=exp/'report';folder.mkdir(exist_ok=True)
    atomic_json(folder/'evidence.json',dict(status='all_requested_accounted',requested=16,
        config_sha256=sha(exp/'config.json'),pairs=pairs,outcome_bindings=bindings,
        scope='Four fixed collection-action neighborhoods, not new policy or independent-seed results',
        labels_entered_replay=False,policy_changed=False,source_sha256=sha(__file__)))
    with(folder/'pairs.csv').open('w')as handle:
        writer=csv.DictWriter(handle,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    lines=['# Q 梯度在真实执行流程中的局部检查','',
        '固定第二轮 FQL 的四个未用于该轮训练的采集布局。分别沿 Q 梯度和固定随机方向做等 RMS 的正负扰动。',
        '这些新标签不进入 replay；失败保留，GRT 不代表最终详细布线 PPA。','',
        '| 原采集布局 | 方向 | Q 预测的正减负差 | 实测 reward 正减负差 | 方向一致 | 正负合法化布局相同 |',
        '| --- | --- | --- | --- | --- | --- |']
    for row in pairs:
        value='—'if row['measured_plus_minus']is None else f"{row['measured_plus_minus']:.6f}"
        lines.append(f"| {row['base_index']} | {row['direction']} | {row['predicted_plus_minus']:.6f} | {value} | {row['prediction_sign_agrees']} | {row['same_legalized_layout']} |")
    lines+=['','reward 越大越好；正减负为正表示正方向更好。只根据四个邻域不能宣称总体稳定性。','']
    (folder/'summary.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='reported',path=str(folder))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True);parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--stage',choices=['export','report'],required=True)
    args=parser.parse_args()
    if 'SLURM_JOB_ID'not in os.environ or args.seed!=0:raise RuntimeError('Use Slurm; pilot seed0 only')
    exp=Path(args.experiment).resolve();cfg=json.loads((exp/'config.json').read_text());source=Path(cfg['source_experiment'])
    if sha(source/'config.json')!=cfg['source_config_sha256']:raise ValueError('Changed parent configuration')
    folder=exp/'sources'/('job_'+os.environ['SLURM_JOB_ID']);folder.mkdir(parents=True,exist_ok=False)
    for name in ['graph_o2o_direction_diagnostic.py','graph_o2o_models.py','physical_ridge.py','analytic.py']:
        shutil.copyfile(Path(__file__).with_name(name),folder/name)
    if args.stage=='export':export(args,exp,cfg,source)
    else:report(args,exp,cfg,source)


if __name__=='__main__':main()
