"""Continue the measured pilot with more fresh feedback and matched controls.

Freeze both round-2 policies, collect 24 additional layouts per policy, then
update BC/FQL with 48 cumulative online labels. A third FQL arm keeps only
the parent's 16 labels while taking the same 600 updates. Four new paired
evaluation noises are measured before and after; they never enter replay.
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
from baseline.api import CHECKPOINT, RUNTIME, legalize, save_sample
from backend.scripts.export_macro_def import convert
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import load_requests
from .graph_o2o_collect import audit_outcome, summary
from .graph_o2o_models import actor_objective
from .graph_o2o_train import initialize_problem, construct, update, save_point, parameter_changes, critic_stats
from .core import critic_diagnostics


def context(args):
    exp = Path(args.experiment).resolve()
    cfg = json.loads((exp/'config.json').read_text())
    source = Path(cfg['source_experiment'])
    if sha(source/'config.json') != cfg['source_config_sha256']:
        raise ValueError('The frozen parent configuration changed')
    parent_status=json.loads((source/'seed_0/round_2/training_status.json').read_text())
    if (sha(RUNTIME/'benchmark_cache/bp_fe.pt')!=parent_status['graph_cache_sha256']or
            sha(CHECKPOINT)!=parent_status['pretrained_sha256']):
        raise ValueError('Graph inputs or pretrained weights differ from the frozen parents')
    for arm, row in cfg['parent_checkpoints'].items():
        completion = json.loads(Path(row['path']).with_name('completed.json').read_text())
        if (sha(row['path']) != row['sha256'] or completion['checkpoint_sha256'] != row['sha256']
                or completion['online_training_rows'] != 16 or completion['arm'] != arm):
            raise ValueError('The frozen parent policy or its training provenance changed')
    return exp, cfg, source


def snapshot(exp, stage):
    dest = exp/'sources'/(stage+'_job_'+os.environ['SLURM_JOB_ID'])
    dest.mkdir(parents=True, exist_ok=False)
    names = ['graph_o2o_coverage.py', 'graph_o2o_coverage.slurm', 'graph_o2o_coverage_report.slurm',
             'graph_o2o_coverage_q.py', 'core.py', 'critic_gate.py',
             'graph_o2o_train.py', 'graph_o2o_models.py', 'graph_o2o_collect.py',
             'physical_ridge.py', 'analytic.py']
    for name in names:
        shutil.copyfile(Path(__file__).with_name(name), dest/name)
    hashes={name:sha(dest/name) for name in names}
    root=Path(__file__).resolve().parents[1]
    for name in ['baseline/api.py','backend/scripts/export_macro_def.py','analysis/placement_metrics.py']:
        target=dest/'shared'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,target);hashes['shared/'+name]=sha(target)
    atomic_json(dest/'source_hashes.json', hashes)


def export(args, exp, cfg, problem, cache, x, y, number):
    dest = exp/f'seed_0/round_{number}/requests'
    marker = dest/'completed.json'
    if marker.exists():
        load_requests(exp, 0, number)
        return
    dest.mkdir(parents=True, exist_ok=True)
    metric = MacroHPWL(problem.benchmark_info)
    arms = ['bc', 'fql'] if number == 0 else ['bc', 'fql', 'fql_frozen_replay']
    records = []
    for arm in arms:
        checkpoint = (Path(cfg['parent_checkpoints'][arm]['path']) if number == 0
                      else exp/f'seed_0/round_1/{arm}/checkpoint.pt')
        saved = torch.load(checkpoint, map_location='cuda', weights_only=False)
        _, actor, critic, _ = construct(problem, cache, x, y, saved, cfg)
        for role in (['collect', 'evaluation'] if number == 0 else ['evaluation']):
            count = cfg['additional_collection_samples'] if role == 'collect' else cfg['evaluation_samples']
            seed = cfg['collection_noise_seed'] if role == 'collect' else cfg['evaluation_noise_seed']
            generator = torch.Generator(device='cuda').manual_seed(seed)
            noises = 2*torch.rand(count, len(problem.x), 2, generator=generator, device='cuda')-1
            for i, noise in enumerate(noises):
                folder = dest/arm/role/str(i)
                if (folder/'request.json').exists():
                    old = json.loads((folder/'request.json').read_text())
                    if old['checkpoint_sha256'] != sha(checkpoint):
                        raise ValueError('Interrupted export used a different policy')
                    records.append(old)
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                with torch.no_grad():
                    raw = actor(noise[None])[0]
                    q = critic(raw[None]).cpu().tolist()[0]
                    action = legalize(raw, problem)
                values = action.cpu().numpy()
                geometry = legality(values, metric.sizes)
                row = dict(arm=arm, seed=0, round=number, role=role, index=i,
                    noise_seed=seed, graph='bp_fe', terminal_mask=0, raw_action=raw.cpu().tolist(),
                    executed_action=values.tolist(), predicted_q=q, source_checkpoint=str(checkpoint),
                    checkpoint_sha256=sha(checkpoint), geometry=geometry,
                    action_sha256=hashlib.sha256(np.asarray(values,dtype='<f4').reshape(-1).tobytes()).hexdigest(),
                    sample_dir=str(folder), run_id=f'go2o_cov_v1_s0_r{number}_{arm}_{role}_{i}',
                    stage='global_route_only', training_feedback_eligible=role=='collect',
                    macro_hpwl=float(metric(values)))
                np.savez_compressed(folder/'actions.npz', noise=noise.cpu().numpy(),
                                    raw_action=raw.cpu().numpy(), action=values)
                if geometry['legal']:
                    save_sample(folder, action, noise, problem, row)
                    path = folder/('bp_fe_job'+os.environ['SLURM_JOB_ID']+'.def')
                    audit = convert('bp_fe', folder/'macro_physical.json', path)
                    row.update(def_path=str(path), def_sha256=sha(path), rounded_geometry=audit['legal_no_halo'])
                    if not audit['legal_no_halo']: row['pre_backend_failure']='rounded_DEF_geometry'
                else:
                    row['pre_backend_failure']='normalized_geometry'
                atomic_json(folder/'request.json', row)
                records.append(row)
        del actor, critic, saved
        torch.cuda.empty_cache()
    tasks = dest/'tasks.jsonl'
    tasks.write_text(''.join(json.dumps(row)+'\n' for row in records))
    atomic_json(marker, dict(status='completed', count=len(records), tasks_sha256=sha(tasks),
                            config_sha256=sha(exp/'config.json'), slurm_job_id=os.environ['SLURM_JOB_ID']))
    print(json.dumps(dict(status='exported', round=number, requests=len(records))), flush=True)


def replay(exp, cfg, source, scale):
    prior = {arm:[] for arm in ['bc','fql']}
    added = {arm:[] for arm in ['bc','fql']}
    old_cfg = json.loads((source/'config.json').read_text())
    for number in range(3):
        rows, _ = load_requests(source, 0, number)
        for row in rows:
            if row['role']=='collect' and row['training_feedback_eligible']:
                out = audit_outcome(row, old_cfg, scale)
                (prior if number<2 else added)[row['arm']].append(out)
    rows, _ = load_requests(exp, 0, 0)
    for row in rows:
        if row['role']=='collect' and row['training_feedback_eligible']:
            added[row['arm']].append(audit_outcome(row, cfg, scale))
    for arm in prior:
        if len(prior[arm])!=16 or len(added[arm])!=32:
            raise ValueError('Incomplete fixed feedback budget')
        if len({row['run_id']for row in prior[arm]+added[arm]})!=48:
            raise ValueError('A logical collection sample was counted twice')
    arms = {arm:prior[arm]+added[arm] for arm in prior}
    arms['fql_frozen_replay'] = prior['fql']
    result = dict(version=1, config_sha256=sha(exp/'config.json'), reward_scale=scale,
                  evaluation_feedback_replayed=False, initial_online_rows=16,
                  additional_online_rows=32, source_config_sha256=cfg['source_config_sha256'], arms=arms)
    path = exp/'training_replay.json'
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError('Immutable replay changed')
    if not path.exists(): atomic_json(path, result)
    return result, path


def train(args, exp, cfg, source, problem, manifest, cache, x, y, vx, vy, behavior):
    feedback, replay_path = replay(exp, cfg, source, manifest['reward_scale_train_only'])
    for arm in ['bc','fql','fql_frozen_replay']:
        folder = exp/f'seed_0/round_1/{arm}'
        folder.mkdir(parents=True, exist_ok=True)
        marker = folder/'completed.json'
        if marker.exists():
            if sha(folder/'checkpoint.pt')!=json.loads(marker.read_text())['checkpoint_sha256']:
                raise ValueError('A completed continuation checkpoint changed')
            continue
        parent = Path(cfg['parent_checkpoints']['bc' if arm=='bc' else 'fql']['path'])
        scratch = Path('/scratch/bohanli/flowplace_graph/coverage_training')/sha(exp/'config.json')/arm
        scratch.mkdir(parents=True, exist_ok=True)
        resume = scratch/'resume.pt'
        saved = torch.load(resume if resume.exists() else parent, map_location='cuda', weights_only=False)
        expected = sha(exp/'config.json') if resume.exists() else cfg['source_config_sha256']
        if saved['config_sha256']!=expected:
            raise ValueError('Resume/parent belongs to another experiment')
        if resume.exists() and (saved['replay_sha256']!=sha(replay_path) or saved['parent_checkpoint_sha256']!=sha(parent)):
            raise ValueError('Resumed replay or parent changed')
        flow, actor, critic, opts = construct(problem, cache, x, y, saved, cfg)
        rows = feedback['arms'][arm]
        raw = torch.tensor([row['raw_action']for row in rows], device='cuda').reshape(-1,len(problem.x),2)
        reward = torch.tensor([row['reward']for row in rows], device='cuda', dtype=torch.float32)
        good = torch.tensor([row['status']=='success'for row in rows], device='cuda')
        xx, yy, bb = torch.cat([x,raw]), torch.cat([y,reward]), torch.cat([behavior,raw[good]])
        begin = saved.get('iteration',0) if resume.exists() else 0
        if not resume.exists(): torch.manual_seed(cfg['update_seed'])
        start = time.perf_counter()
        with (folder/'training.jsonl').open('a') as log:
            for i in range(begin,cfg['online_update_steps']):
                n=cfg['batch_size'];online_n=round(n*cfg['online_batch_fraction'])
                index=torch.randint(len(x),(n,),device='cuda')
                index[:online_n]=len(x)+torch.randint(len(raw),(online_n,),device='cuda')
                bindex=torch.randint(len(behavior),(n,),device='cuda')
                if len(bb)>len(behavior):
                    bindex[:online_n]=len(behavior)+torch.randint(len(bb)-len(behavior),(online_n,),device='cuda')
                z=2*torch.rand(n,len(problem.x),2,device='cuda')-1
                zbc=2*torch.rand_like(z)-1;t=torch.rand(n,device='cuda')
                aloss,distill,aq=actor_objective(actor,flow,critic,z,alpha=cfg['alpha'],
                    q_weight=float(arm!='bc'),flow_steps=cfg['flow_steps'])
                floss=flow.flow_loss(bb[bindex],zbc,t)
                qloss=(critic(xx[index])-yy[index,None]).square().mean()
                update(aloss,opts['actor'],actor);update(floss,opts['flow'],flow);update(qloss,opts['critic'],critic)
                if (i+1)%50==0 or i+1==cfg['online_update_steps']:
                    row=dict(step=i+1,arm=arm,actor_loss=float(aloss),flow_loss=float(floss),
                        critic_loss=float(qloss),distill=float(distill),q=float(aq),
                        online_training_rows=len(rows),new_feedback_rows=len(rows)-16,
                        critic_train=critic_stats(critic,xx,yy.cpu().numpy()),
                        critic_validation=critic_stats(critic,vx,vy),
                        seconds=time.perf_counter()-start)
                    log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                if (i+1)%250==0 or i+1==cfg['online_update_steps']:
                    metadata=dict(iteration=i+1,arm=arm,seed=0,config_sha256=sha(exp/'config.json'),
                        replay_sha256=sha(replay_path),parent_checkpoint_sha256=sha(parent),online_rows=len(rows))
                    save_point(scratch,'resume.pt',flow,actor,critic,opts,metadata)
                    save_point(scratch,f'weights_step_{i+1}.pt',flow,actor,critic,opts,metadata,full=False)
            save_point(folder,'checkpoint.pt',flow,actor,critic,opts,
                dict(iteration=cfg['online_update_steps'],arm=arm,seed=0,config_sha256=sha(exp/'config.json'),
                     replay_sha256=sha(replay_path),parent_checkpoint_sha256=sha(parent),online_rows=len(rows)))
        before=torch.load(parent,map_location='cpu',weights_only=False)
        changes={name:parameter_changes(module,before[name])for name,module in [('flow',flow),('actor',actor),('critic',critic)]}
        atomic_json(marker,dict(status='completed',arm=arm,checkpoint_sha256=sha(folder/'checkpoint.pt'),
            parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),replay_sha256=sha(replay_path),
            online_training_rows=len(rows),new_feedback_rows=len(rows)-16,total_measured_rows=len(xx),
            behavior_rows=len(bb),parameter_changes=changes,critic_train=critic_stats(critic,xx,yy.cpu().numpy()),
            critic_validation=critic_stats(critic,vx,vy),slurm_job_id=os.environ['SLURM_JOB_ID'],
            gpu_model=torch.cuda.get_device_name(),torch_version=torch.__version__,
            intermediate_checkpoint_storage='scratch, final checkpoint durable in fact_data',
            intermediate_checkpoints={str(p):sha(p)for p in scratch.glob('*.pt')},seconds=time.perf_counter()-start))
        del flow,actor,critic,opts,saved,before;torch.cuda.empty_cache()
    export(args,exp,cfg,problem,cache,x,y,1)


def report(args, exp, cfg, source):
    import csv
    from .graph_o2o_coverage_q import fixed_cohort_report
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    manifest=json.loads((Path(__file__).resolve().parents[1]/'datasets/bp_fe_grt_v1/manifest.json').read_text())
    feedback,replay_path=replay(exp,cfg,source,manifest['reward_scale_train_only'])
    training_evidence={}
    for arm in ['bc','fql','fql_frozen_replay']:
        folder=exp/f'seed_0/round_1/{arm}';path=folder/'completed.json'
        completion=json.loads(path.read_text());count=16 if arm=='fql_frozen_replay' else 48
        parent=cfg['parent_checkpoints']['bc'if arm=='bc'else'fql']
        if (sha(folder/'checkpoint.pt')!=completion['checkpoint_sha256']or
            completion['parent_checkpoint_sha256']!=parent['sha256']or
            completion['replay_sha256']!=sha(replay_path)or
            completion['online_training_rows']!=count or completion['new_feedback_rows']!=count-16 or
            completion['total_measured_rows']!=64+count):
            raise ValueError('Actual continuation data, checkpoint or parent chain differs')
        if any(completion['parameter_changes'][name]['changed_parameter_tensors']<=0
               for name in ['flow','actor','critic']):
            raise ValueError('Continuation lacks actual updates to all three networks')
        logs=[json.loads(line)for line in(folder/'training.jsonl').read_text().splitlines()]
        if logs[-1]['step']!=cfg['online_update_steps']or any(row['online_training_rows']!=count for row in logs):
            raise ValueError('Actual continuation update budget differs')
        training_evidence[arm]=dict(path=str(path),sha256=sha(path),record=completion)
    groups={};bindings=[];cohorts={}
    for number in [0,1]:
        rows,_=load_requests(exp,0,number)
        if len(rows)!=(56 if number==0 else 12):
            raise ValueError('Changed total candidate denominator')
        for arm in dict.fromkeys(row['arm']for row in rows):
            requests=[row for row in rows if row['arm']==arm and row['role']=='evaluation']
            if [row['index']for row in requests]!=list(range(cfg['evaluation_samples'])):
                raise ValueError('Incomplete fixed evaluation-noise denominator')
            parent_arm='bc'if arm=='bc'else'fql'
            for row in requests:
                reference=exp/f"seed_0/round_0/requests/{parent_arm}/evaluation/{row['index']}/actions.npz"
                with np.load(Path(row['sample_dir'])/'actions.npz',allow_pickle=False)as current,np.load(reference,allow_pickle=False)as before:
                    if not np.array_equal(current['noise'],before['noise']):
                        raise ValueError('Before/after evaluation noises differ')
            outcomes=[audit_outcome(row,cfg,manifest['reward_scale_train_only'])for row in requests]
            key=('before/'if number==0 else 'after/')+arm
            cohorts[key]=outcomes
            groups[key]=summary(outcomes)
            good=[row for row in outcomes if row['status']=='success']
            if good:
                predictions=np.asarray([row['predicted_q']for row in good])
                rewards=np.asarray([row['reward']for row in good])
                groups[key]['on_policy_Q']=dict(critic_diagnostics(predictions,rewards),
                    optimism_bias=float((predictions.mean(-1)-rewards).mean()),
                    scope='Successful measured evaluation cases only; all failures retained separately')
            bindings += [dict(path=row['outcome_path'],sha256=row['outcome_sha256'])for row in outcomes]
    output=exp/('full_route_comparison'if args.full_route else 'grt_comparison')
    output.mkdir(exist_ok=True)
    fixed_q=fixed_cohort_report(exp,cfg,feedback,cohorts,output)
    route=None
    if args.full_route:
        plan=json.loads((exp/'full_route_plan.json').read_text());route={}
        for item in plan['tasks']:
            path=Path(item['output'])/'outcome.json';row=json.loads(path.read_text())
            if row['status']=='completed':
                root=Path(row['physical_run'])
                if sha(root/'manifest.json')!=row['physical_manifest_sha256']or sha(root/'metrics.json')!=row['metrics_sha256']or row['spec']['route_end_iteration']!=20:
                    raise ValueError('Changed physical evidence or route budget')
            elif row['status']!='upstream_physical_failure':raise ValueError('Unresolved final outcome')
            route.setdefault(item['group'],[]).append(row)
            bindings.append(dict(path=str(path),sha256=sha(path)))
    atomic_json(output/'report.json',dict(status='all_requested_accounted',scope='bp_fe seed0; four new paired evaluation noises',
        config_sha256=sha(exp/'config.json'),groups=groups,full_route=route,outcome_bindings=bindings,
        training_updates_per_arm=cfg['online_update_steps'],
        online_labels=dict(parent=16,bc=48,fql=48,fql_frozen_replay=16),
        replay_sha256=sha(replay_path),training_evidence=training_evidence,
        fixed_cohort_q_path=str(output/'fixed_cohort_q.json'),
        fixed_cohort_q_sha256=sha(output/'fixed_cohort_q.json'),
        evaluation_feedback_replayed=False,goal_completed=False,source_sha256=sha(__file__)))
    lines=['# 增加在线反馈覆盖的对照','',
        '固定原实验第二轮的 BC/FQL 策略，增加每种方法24个新采集布局，并加入上一轮未用于训练的8个布局。',
        'BC/FQL 各用64个原有标签和48个累计在线标签；FQL 固定 replay 对照保留原有16个在线标签。三种方法都再更新600步。',
        '四个新评估噪声在更新前后配对，结果不进入训练；仍然只有一个独立训练 seed。','',
        '| 方法 | 成功/请求 | GRT 线长（m） | WNS（ns） | TNS（ns） | 功耗（W） | 面积（μm²） |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    csv_rows=[]
    for name,group in groups.items():
        m=group['mean'];values=[None]*5 if m is None else [m['rWL_um']/1e6,m['WNS_ns'],m['TNS_ns'],m['Power_W'],m['StdCellArea_um2']]
        lines.append('| '+name+f" | {group['successful']}/{group['count']} | "+' | '.join('—'if v is None else f'{v:.6g}'for v in values)+' |')
        csv_rows.append(dict(method=name,requested=group['count'],successful=group['successful'],**(m or{})))
    lines += ['', '| 方法 | Q MAE | Q 偏差 | Q 排序相关 |',
              '| --- | --- | --- | --- |']
    for name,group in groups.items():
        q=group.get('on_policy_Q',{})
        values=[q.get('mae'),q.get('optimism_bias'),q.get('spearman_r')]
        lines.append('| '+name+' | '+' | '.join('—'if v is None else f'{v:.6g}'for v in values)+' |')
    lines += ['', '以下让各阶段 Q 评估完全相同的更新后 FQL 布局，排除布局样本变化的影响。',
              '这些布局从未进入任一 replay；全部五组固定布局的预测和误差另存为 CSV。','',
              '| Q 的拟合阶段 | 固定布局上的 MAE | 乐观偏差 | 排序相关 |',
              '| --- | --- | --- | --- |']
    for name in ['before/fql','after/fql','after/fql_frozen_replay']:
        score=fixed_q['fits'][name]['scores']['after/fql']
        values=[score['mae'],score['optimism_bias'],score['spearman_r']]
        lines.append('| '+name+' | '+' | '.join('—'if v is None else f'{v:.6g}'for v in values)+' |')
    if route:
        lines += ['', '统一详细布线 cap20；非零 DRC 保留，有失败时不报告完整组均值。','',
                  '| 方法 | 完成/请求 | DRC=0 | rWL（m） | WNS（ns） | TNS（ns） | 功耗（W） | 面积（μm²） |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for name,rows in route.items():
            good=[r for r in rows if r['status']=='completed']
            keys=['DRT_WL','WNS','TNS','Power','StdCellArea']
            values=[statistics.mean(r['metrics'][key]for r in good)for key in keys]if len(good)==len(rows)else[None]*5
            if values[0]is not None:values[0]/=1e6
            lines.append('| '+name+f" | {len(good)}/{len(rows)} | {sum(r['route_drc_clean']for r in good)} | "+' | '.join('—'if v is None else f'{v:.6g}'for v in values)+' |')
    lines += ['', '新反馈是否有效，应比较 after/fql 与 after/fql_frozen_replay；RL 是否有效，应同时比较匹配 BC。',
              'GRT 线长不等于最终详细布线线长，也不能支持详细布线 DRC 结论。正负结果全部保留。','']
    (output/'summary.md').write_text('\n'.join(lines))
    columns=list(dict.fromkeys(k for row in csv_rows for k in row))
    with(output/'summary.csv').open('w')as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(csv_rows)
    print(json.dumps(dict(status='reported',path=str(output))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',required=True);p.add_argument('--stage',choices=['export','train','report'],required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--full-route',action='store_true')
    args=p.parse_args()
    if 'SLURM_JOB_ID'not in os.environ or args.seed!=0:raise RuntimeError('Use Slurm; this continuation is paired to seed0')
    exp,cfg,source=context(args);snapshot(exp,args.stage)
    if args.stage=='report':return report(args,exp,cfg,source)
    if not torch.cuda.is_available():raise RuntimeError('Request a GPU')
    problem,data,manifest,cache,x,y,vx,vy,behavior=initialize_problem(args)
    if manifest['dataset_sha256']!=cfg['offline_dataset_sha256']:raise ValueError('Offline labels changed')
    if args.stage=='export':export(args,exp,cfg,problem,cache,x,y,0)
    else:train(args,exp,cfg,source,problem,manifest,cache,x,y,vx,vy,behavior)


if __name__=='__main__':main()
