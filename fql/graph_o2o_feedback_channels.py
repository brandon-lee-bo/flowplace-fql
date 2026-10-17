"""Separate new feedback in behavior-flow training from new feedback in Q fitting.

Both missing cells of the existing 2x2 use the same frozen round2 FQL parent,
600 updates, real replay and paired diagnostic noises. This is a mechanism
ablation, not independent-seed confirmation or a new selected algorithm.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from analysis.placement_metrics import MacroHPWL, legality
from baseline.api import legalize, save_sample
from backend.scripts.export_macro_def import convert
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import ROOT, load_requests, validate_request
from .graph_o2o_collect import audit_outcome, summary
from .graph_o2o_coverage import context as coverage_context, replay as coverage_replay
from .graph_o2o_final_evidence import audit_final_outcome
from .graph_o2o_models import actor_objective
from .graph_o2o_train import initialize_problem, construct, update, save_point, parameter_changes, critic_stats


def context(args):
    exp = Path(args.experiment).resolve()
    cfg = json.loads((exp / 'config.json').read_text())
    plan = json.loads((exp / 'experiment_plan.json').read_text())
    if sha(exp / 'config.json') != plan['config_sha256']:
        raise ValueError('Feedback-channel configuration changed')
    for name, digest in plan['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Feedback-channel source changed: ' + name)
    for item in plan['bindings'].values():
        if sha(item['path']) != item['sha256']:
            raise ValueError('Frozen feedback-channel input changed: ' + item['path'])
    coverage = Path(cfg['coverage_experiment'])
    _, coverage_cfg, original = coverage_context(argparse.Namespace(experiment=str(coverage)))
    if any(cfg[k] != coverage_cfg[k] for k in plan['matched_hyperparameters']):
        raise ValueError('A supposedly matched hyperparameter differs')
    if cfg['training_arms'] != {'fresh_behavior_old_q': [48, 16], 'old_behavior_fresh_q': [16, 48]}:
        raise ValueError('The predeclared feedback-channel cells changed')
    return exp, cfg, plan, coverage, coverage_cfg, original


def train(args, exp, cfg, plan, coverage, coverage_cfg, original):
    if not torch.cuda.is_available():
        raise RuntimeError('Train and export with an allocated GPU')
    problem, _, manifest, cache, x, y, vx, vy, behavior = initialize_problem(args)
    if manifest['dataset_sha256'] != cfg['offline_dataset_sha256']:
        raise ValueError('Initial measured dataset changed')
    feedback, replay_path = coverage_replay(coverage, coverage_cfg, original, manifest['reward_scale_train_only'])
    replay = feedback['arms']['fql']
    if len(replay) != 48 or any(r['role'] != 'collect' or not r['training_feedback_eligible'] for r in replay):
        raise ValueError('Feedback must be the 48 actual FQL collect labels')
    atomic_json(exp / 'training_data.json', dict(coverage_replay=str(replay_path),
        coverage_replay_sha256=sha(replay_path), rows=replay,
        prior_prefix_count=16, evaluation_feedback_replayed=False))
    parent = Path(cfg['parent_checkpoints']['fql']['path'])
    for arm, (behavior_count, q_count) in cfg['training_arms'].items():
        folder = exp / 'seed_0/round_0' / arm
        folder.mkdir(parents=True, exist_ok=True)
        marker = folder / 'completed.json'
        if marker.exists():
            if sha(folder / 'checkpoint.pt') != json.loads(marker.read_text())['checkpoint_sha256']:
                raise ValueError('Completed ablation checkpoint changed')
            continue
        resume = folder / 'resume.pt'
        saved = torch.load(resume if resume.exists() else parent, map_location='cuda', weights_only=False)
        expected = sha(exp / 'config.json') if resume.exists() else cfg['source_config_sha256']
        if saved['config_sha256'] != expected:
            raise ValueError('Resume or parent experiment mismatch')
        if resume.exists() and (saved['replay_sha256'] != sha(replay_path) or
                saved['parent_checkpoint_sha256'] != sha(parent) or saved['arm'] != arm):
            raise ValueError('Resumed ablation data or parent differs')
        flow, actor, critic, opts = construct(problem, cache, x, y, saved, cfg)
        raw = torch.tensor([r['raw_action'] for r in replay], device='cuda').reshape(-1, len(problem.x), 2)
        rewards = torch.tensor([r['reward'] for r in replay], device='cuda', dtype=torch.float32)
        good = torch.tensor([r['status'] == 'success' for r in replay[:behavior_count]], device='cuda')
        xx, yy = torch.cat([x, raw[:q_count]]), torch.cat([y, rewards[:q_count]])
        bb = torch.cat([behavior, raw[:behavior_count][good]])
        begin = saved.get('iteration', 0) if resume.exists() else 0
        if not resume.exists():
            torch.manual_seed(cfg['update_seed'])
        started = time.perf_counter()
        with (folder / 'training.jsonl').open('a') as log:
            for i in range(begin, cfg['online_update_steps']):
                n = cfg['batch_size']; online_n = round(n * cfg['online_batch_fraction'])
                index = torch.randint(len(x), (n,), device='cuda')
                index[:online_n] = len(x) + torch.randint(q_count, (online_n,), device='cuda')
                bindex = torch.randint(len(behavior), (n,), device='cuda')
                if len(bb) > len(behavior):
                    bindex[:online_n] = len(behavior) + torch.randint(len(bb) - len(behavior), (online_n,), device='cuda')
                z = 2 * torch.rand(n, len(problem.x), 2, device='cuda') - 1
                zbc = 2 * torch.rand_like(z) - 1; t = torch.rand(n, device='cuda')
                aloss, distill, aq = actor_objective(actor, flow, critic, z,
                    alpha=cfg['alpha'], q_weight=1., flow_steps=cfg['flow_steps'])
                floss = flow.flow_loss(bb[bindex], zbc, t)
                qloss = (critic(xx[index]) - yy[index, None]).square().mean()
                update(aloss, opts['actor'], actor)
                update(floss, opts['flow'], flow)
                update(qloss, opts['critic'], critic)
                if (i + 1) % 50 == 0 or i + 1 == cfg['online_update_steps']:
                    row = dict(arm=arm, step=i + 1, actor_loss=float(aloss), flow_loss=float(floss),
                        critic_loss=float(qloss), distill=float(distill), q=float(aq),
                        behavior_online_rows=behavior_count, q_online_rows=q_count,
                        critic_train=critic_stats(critic, xx, yy.cpu().numpy()),
                        critic_validation=critic_stats(critic, vx, vy), seconds=time.perf_counter()-started)
                    log.write(json.dumps(row) + '\n'); log.flush(); print(json.dumps(row), flush=True)
                if (i + 1) % 250 == 0 or i + 1 == cfg['online_update_steps']:
                    save_point(folder, 'resume.pt', flow, actor, critic, opts,
                        dict(iteration=i + 1, arm=arm, seed=0, config_sha256=sha(exp/'config.json'),
                            replay_sha256=sha(replay_path), parent_checkpoint_sha256=sha(parent)))
            save_point(folder, 'checkpoint.pt', flow, actor, critic, opts,
                dict(iteration=cfg['online_update_steps'], arm=arm, seed=0,
                    config_sha256=sha(exp/'config.json'), replay_sha256=sha(replay_path),
                    parent_checkpoint_sha256=sha(parent)))
        before = torch.load(parent, map_location='cpu', weights_only=False)
        changes = {name: parameter_changes(module, before[name]) for name, module in
                   [('flow', flow), ('actor', actor), ('critic', critic)]}
        if any(v['changed_parameter_tensors'] <= 0 for v in changes.values()):
            raise ValueError('A feedback-channel branch did not update all networks')
        atomic_json(marker, dict(status='completed', arm=arm, checkpoint_sha256=sha(folder/'checkpoint.pt'),
            parent_checkpoint=str(parent), parent_checkpoint_sha256=sha(parent),
            config_sha256=sha(exp/'config.json'), replay_sha256=sha(replay_path),
            behavior_online_rows=behavior_count, q_online_rows=q_count, behavior_rows=len(bb),
            measured_q_rows=len(xx), online_training_rows=q_count, actual_update_steps=cfg['online_update_steps'],
            parameter_changes=changes, critic_train=critic_stats(critic, xx, yy.cpu().numpy()),
            slurm_job_id=os.environ['SLURM_JOB_ID'], gpu_model=torch.cuda.get_device_name(),
            torch_version=torch.__version__, seconds=time.perf_counter()-started,
            evaluation_feedback_replayed=False))
        del flow, actor, critic, opts, saved, before
        torch.cuda.empty_cache()
    export(exp, cfg, problem, cache, x, y)


def export(exp, cfg, problem, cache, x, y):
    dest = exp / 'seed_0/round_0/requests'; dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'completed.json').exists():
        rows, _ = load_requests(exp, 0, 0)
        for row in rows: validate_request(row)
        return
    metric = MacroHPWL(problem.benchmark_info); records = []
    reference = Path(cfg['coverage_experiment']) / 'seed_0/round_1/requests/fql/evaluation'
    for arm in cfg['training_arms']:
        checkpoint = exp / 'seed_0/round_0' / arm / 'checkpoint.pt'
        saved = torch.load(checkpoint, map_location='cuda', weights_only=False)
        _, actor, critic, _ = construct(problem, cache, x, y, saved, cfg)
        generator = torch.Generator(device='cuda').manual_seed(cfg['evaluation_noise_seed'])
        noises = 2*torch.rand(cfg['evaluation_samples'], len(problem.x), 2, generator=generator, device='cuda')-1
        for i, noise in enumerate(noises):
            folder = dest / arm / 'evaluation' / str(i); folder.mkdir(parents=True, exist_ok=True)
            if (folder/'request.json').exists():
                row = json.loads((folder/'request.json').read_text()); validate_request(row)
                records.append(row); continue
            with np.load(reference/str(i)/'actions.npz', allow_pickle=False) as old:
                if not np.array_equal(old['noise'], noise.cpu().numpy()):
                    raise ValueError('Factorial evaluation noises are not exactly paired')
            with torch.no_grad():
                raw = actor(noise[None])[0]; q = critic(raw[None]).cpu().tolist()[0]
                action = legalize(raw, problem)
            values = action.cpu().numpy(); geometry = legality(values, metric.sizes)
            row = dict(arm=arm, seed=0, round=0, role='evaluation', index=i, graph='bp_fe',
                noise_seed=cfg['evaluation_noise_seed'], terminal_mask=0, raw_action=raw.cpu().tolist(),
                executed_action=values.tolist(), predicted_q=q, source_checkpoint=str(checkpoint),
                checkpoint_sha256=sha(checkpoint), geometry=geometry,
                action_sha256=hashlib.sha256(np.asarray(values,dtype='<f4').reshape(-1).tobytes()).hexdigest(),
                sample_dir=str(folder), run_id=f'go2o_channels_v1_{arm}_n{i}',
                stage='global_route_only', training_feedback_eligible=False, macro_hpwl=float(metric(values)))
            np.savez_compressed(folder/'actions.npz',noise=noise.cpu().numpy(),raw_action=raw.cpu().numpy(),action=values)
            if geometry['legal']:
                save_sample(folder, action, noise, problem, row)
                path = folder / ('bp_fe_job'+os.environ['SLURM_JOB_ID']+'.def')
                audit = convert('bp_fe', folder/'macro_physical.json', path)
                row.update(def_path=str(path), def_sha256=sha(path), rounded_geometry=audit['legal_no_halo'])
                if not audit['legal_no_halo']: row['pre_backend_failure']='rounded_DEF_geometry'
            else: row['pre_backend_failure']='normalized_geometry'
            atomic_json(folder/'request.json', row); records.append(row)
        del saved, actor, critic
        torch.cuda.empty_cache()
    tasks = dest/'tasks.jsonl'; tasks.write_text(''.join(json.dumps(row)+'\n' for row in records))
    atomic_json(dest/'completed.json', dict(status='completed', count=len(records),
        tasks_sha256=sha(tasks), config_sha256=sha(exp/'config.json'), slurm_job_id=os.environ['SLURM_JOB_ID']))
    print(json.dumps(dict(status='trained_and_exported', requested=len(records))), flush=True)


def report(exp, cfg, final=False):
    import csv
    from .graph_o2o_design_policy_report import actual_weight_changes
    from .graph_o2o_coverage_q import score_checkpoint
    from .critic_gate import load_train_validation
    from baseline.api import RUNTIME
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    coverage=Path(cfg['coverage_experiment'])
    scale=json.loads((ROOT/'datasets/bp_fe_grt_v1/manifest.json').read_text())['reward_scale_train_only']
    new, _=load_requests(exp,0,0); old, _=load_requests(coverage,0,1)
    parent_path=Path(cfg['parent_checkpoints']['fql']['path'])
    parent=torch.load(parent_path,map_location='cpu',weights_only=False)
    updates={}
    for arm,(behavior_count,q_count) in cfg['training_arms'].items():
        folder=exp/'seed_0/round_0'/arm
        completion=json.loads((folder/'completed.json').read_text())
        if (completion['checkpoint_sha256']!=sha(folder/'checkpoint.pt') or
                completion['parent_checkpoint_sha256']!=sha(parent_path) or
                completion['replay_sha256']!=sha(coverage/'training_replay.json') or
                completion['behavior_online_rows']!=behavior_count or completion['q_online_rows']!=q_count or
                completion['actual_update_steps']!=600 or completion['measured_q_rows']!=64+q_count):
            raise ValueError('Actual ablation training/data budget changed')
        saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
        updates[arm]=actual_weight_changes(saved,parent,completion['parameter_changes'])
        del saved
    del parent
    groups={};cohorts={}
    for name in ['fql_frozen_replay','fresh_behavior_old_q','old_behavior_fresh_q','fql','bc']:
        rows=[r for r in (new if name in cfg['training_arms'] else old) if r['arm']==name and r['role']=='evaluation']
        if [r['index'] for r in rows]!=list(range(4)):raise ValueError('Missing paired evaluation layout')
        measured=[audit_outcome(r,cfg,scale) for r in rows]
        for request in rows:validate_request(request)
        cohorts[name]=measured
        group=dict(grt=summary(measured),bindings=[dict(path=r['outcome_path'],sha256=r['outcome_sha256']) for r in measured])
        if final:
            outcomes=[]
            for row in rows:
                key=(name+'_n'+str(row['index'])) if name in cfg['training_arms'] else ('after_'+name+'_n'+str(row['index']))
                p=(exp if name in cfg['training_arms'] else coverage)/'full_route'/key/'outcome.json'
                outcome=json.loads(p.read_text())
                if outcome['status']=='completed':audit_final_outcome(outcome,design='bp_fe',route_cap=20)
                elif outcome['status']!='upstream_physical_failure':raise ValueError('Unresolved route')
                outcomes.append(dict(path=str(p),sha256=sha(p),outcome=outcome))
            good=[r['outcome'] for r in outcomes if r['outcome']['status']=='completed']
            keys=['DRT_WL','WNS','TNS','Power','StdCellArea','DRC']
            group['final']=dict(requested=4,completed=len(good),drc_clean=sum(r['route_drc_clean'] for r in good),
                mean={k:statistics.mean(r['metrics'][k] for r in good) for k in keys} if len(good)==4 else None,
                bindings=outcomes)
        groups[name]=group
    output=exp/('final_report' if final else 'grt_report');output.mkdir(exist_ok=True)
    data,_,_=load_train_validation(ROOT/'datasets/bp_fe_grt_v1/dataset.npz')
    cache=torch.load(RUNTIME/'benchmark_cache/bp_fe.pt',map_location='cpu',weights_only=False)
    replay=json.loads((exp/'training_data.json').read_text())['rows']
    fixed_q={}
    for name in ['fql_frozen_replay','fresh_behavior_old_q','old_behavior_fresh_q','fql']:
        checkpoint=(exp/'seed_0/round_0'/name/'checkpoint.pt' if name in cfg['training_arms'] else
                    coverage/'seed_0/round_1'/name/'checkpoint.pt')
        count=cfg['training_arms'][name][1] if name in cfg['training_arms'] else (16 if name=='fql_frozen_replay' else 48)
        fixed_q[name],_=score_checkpoint(checkpoint,cfg,data,cache,replay[:count],cohorts)
    atomic_json(output/'fixed_cohort_q.json',fixed_q)
    atomic_json(output/'evidence.json',dict(status='all_requested_accounted',groups=groups,
        config_sha256=sha(exp/'config.json'),plan_sha256=sha(exp/'experiment_plan.json'),
        actual_weight_changes=updates,fixed_cohort_q_sha256=sha(output/'fixed_cohort_q.json'),
        scope='bp_fe seed0 factorial mechanism diagnostic; same parent and four already-observed paired noises; no algorithm selection or independent confirmation',
        evaluation_feedback_replayed=False,goal_completed=False,slurm_job_id=os.environ['SLURM_JOB_ID']))
    table=[]
    for name,row in groups.items():
        result=row['final'] if final else row['grt'];m=result['mean']
        table.append(dict(arm=name,requested=4,completed=result.get('completed',result.get('successful')),
            rWL_m=None if m is None else m['DRT_WL' if final else 'rWL_um']/1e6,
            WNS_ns=None if m is None else m['WNS' if final else 'WNS_ns'],
            TNS_ns=None if m is None else m['TNS' if final else 'TNS_ns'],
            Power_W=None if m is None else m['Power' if final else 'Power_W'],
            StdCellArea_um2=None if m is None else m['StdCellArea' if final else 'StdCellArea_um2'],
            DRC_clean=result.get('drc_clean')))
    with (output/'summary.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# 新增反馈分别进入行为 flow 和 Q 的对照','',
        '同一个第二轮 FQL 父模型、相同600步更新；old 为16条在线反馈，fresh 为48条。',
        '四个配对噪声仅用于诊断，已在原对照中观察过；没有用于训练或挑选模型，不能作为新算法的独立验证。','',
        '| 方法 | 完成/请求 | '+('最终 rWL' if final else 'GRT 线长')+'（m） | WNS（ns） | TNS（ns） | 功耗（W） | 面积（μm²） | DRC=0 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in table:
        values=[row['rWL_m'],row['WNS_ns'],row['TNS_ns'],row['Power_W'],row['StdCellArea_um2'],row['DRC_clean']]
        lines.append('| '+row['arm']+f" | {row['completed']}/4 | "+' | '.join('—' if v is None else f'{v:.6g}' for v in values)+' |')
    lines+=['','完整分母及失败保留。GRT 结果不能代表详细布线 DRC；线长改善不能推出全面 PPA 改善。','']
    (output/'summary.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='reported',output=str(output))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',required=True)
    p.add_argument('--stage',choices=['train','report','final-report'],required=True);p.add_argument('--seed',type=int,default=0)
    args=p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ or args.seed!=0:raise RuntimeError('Run this fixed-seed diagnostic through Slurm')
    exp,cfg,plan,coverage,coverage_cfg,original=context(args)
    if args.stage=='train':
        snapshot=exp/'sources'/('train_'+os.environ['SLURM_JOB_ID']+'_restart_'+os.environ.get('SLURM_RESTART_COUNT','0'))
        snapshot.mkdir(parents=True,exist_ok=False)
        for name in plan['source_sha256']:
            dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        train(args,exp,cfg,plan,coverage,coverage_cfg,original)
    else:report(exp,cfg,final=args.stage=='final-report')


if __name__=='__main__':main()
