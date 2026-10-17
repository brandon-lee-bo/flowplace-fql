"""Freeze the expanded-data reference and audit actual offline-to-online work."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import statistics

import torch

# Load project analysis before the upstream adapter can shadow its package.
from .graph_o2o_train import ROOT
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import load_requests
from .graph_o2o_collect import audit_outcome, collect
from .graph_o2o_expanded_data import audited_expansion
from .graph_o2o_design_policy_report import actual_weight_changes
from .graph_o2o_offline_seed_snapshot import legacy_actual_changes
from .graph_o2o_final_evidence import audit_final_outcome, LEGACY_REPEAT_PLAN


def immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('A frozen expanded-data record changed: '+str(path))
    else:
        atomic_json(path, value)


def audit_training_phase(root, cfg, arm, number, count, *, legacy_offline=False):
    """Read actual parent/child tensors; preserve the original metadata gap."""
    root = Path(root)
    folder = root / 'seed_0' / f'round_{number}' / arm
    path = folder / 'completed.json'
    done = json.loads(path.read_text())
    child_path = folder / 'checkpoint.pt'
    expected_parent = (folder.parent / 'common.pt' if number == 0 else
                       root / 'seed_0' / f'round_{number-1}' / arm / 'checkpoint.pt')
    if (done['status'] != 'completed' or done['arm'] != arm or done['seed'] != 0 or
            done['round'] != number or done['train_rows'] != count + number * 8 or
            done['online_training_rows'] != number * 8 or
            done['checkpoint_sha256'] != sha(child_path)):
        raise ValueError('Actual checkpoint/data/update identity differs')
    legacy = (legacy_offline and number == 0 and count == 64 and
              'parameter_changes' not in done and 'parent_checkpoint_sha256' not in done)
    if not legacy:
        if (Path(done['parent_checkpoint']).resolve() != expected_parent.resolve() or
                done['parent_checkpoint_sha256'] != sha(expected_parent)):
            raise ValueError('Actual parent binding differs')
    parent_sha = sha(expected_parent)
    current = torch.load(child_path, map_location='cpu', weights_only=False)
    parent = torch.load(expected_parent, map_location='cpu', weights_only=False)
    steps = cfg['offline_steps'] if number == 0 else cfg['online_update_steps']
    if (current['config_sha256'] != sha(root / 'config.json') or
            current['iteration'] != steps or current['seed'] != 0):
        raise ValueError('Actual checkpoint does not match the fixed training stage')
    if number and current['replay_sha256'] != sha(root / 'seed_0' / f'replay_round_{number}.json'):
        raise ValueError('Actual online checkpoint does not bind its replay')
    changes = (legacy_actual_changes(current, parent) if legacy else
               actual_weight_changes(current, parent, done['parameter_changes']))
    if sha(expected_parent) != parent_sha or sha(child_path) != done['checkpoint_sha256']:
        raise ValueError('Checkpoint changed during independent audit')
    result = dict(path=str(path), sha256=sha(path), record=done,
                  actual_weight_changes=changes, actual_checkpoint_iteration=steps,
                  actual_parent_checkpoint_sha256=parent_sha,
                  parent_binding_recorded_at_training_time=not legacy)
    if legacy:
        result['legacy_parent_note'] = (
            'Original round0 completion predates parent-hash and parameter-delta fields. '
            'Differences are recomputed from retained common.pt now. This is a '
            'retrospective parent hash, not historical runtime provenance.')
    return result


def check_reference_training(exp, plan):
    """Exercise the corrected reporter on six real existing checkpoints."""
    root = Path(plan['reference_experiment'])
    if sha(root / 'config.json') != plan['reference_config_sha256']:
        raise ValueError('Original reference changed')
    cfg = json.loads((root / 'config.json').read_text())
    rows = [dict(round=number, arm=arm, **audit_training_phase(
        root, cfg, arm, number, 64, legacy_offline=True))
        for number in range(3) for arm in ['bc', 'fql']]
    if sum(not row['parent_binding_recorded_at_training_time'] for row in rows) != 2:
        raise ValueError('Unexpected legacy provenance scope')
    summary = json.loads((root / 'full_route/summary.json').read_text())
    audits = [audit_final_outcome(row['result'], design='bp_fe', route_cap=20,
                                  legacy_plan=LEGACY_REPEAT_PLAN)
              for row in summary['records'] if row['result']['status'] == 'completed']
    if len(audits) != 21 or any(a['physical_identity'] != audits[0]['physical_identity'] for a in audits):
        raise ValueError('Original final physical cohort differs')
    folder = exp / 'report_schema_check' / ('job_' + os.environ['SLURM_JOB_ID'])
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(folder / 'evidence.json', dict(
        status='six_actual_reference_training_stages_and_21_native_routes_audited',
        report_source_sha256=sha(__file__), plan_sha256=sha(plan['_path']),
        actual_training=rows, native_route_audits=audits,
        no_training_or_physical_work_repeated=True,
        expanded_training_completed=False, slurm_job_id=os.environ['SLURM_JOB_ID']))
    print(json.dumps(dict(status='reference_schema_check_passed', output=str(folder))), flush=True)


def offline_candidates(exp, cfg):
    extra, binding = audited_expansion(exp, cfg)
    rows = []
    label_path = ROOT/'datasets/bp_fe_grt_v1/backend_labels.jsonl'
    for line in label_path.read_text().splitlines():
        item = json.loads(line)
        if item['critic_split'] != 0:
            continue
        run = Path(item['manifest_path']).parent
        if (sha(run/'manifest.json') != item['manifest_sha256'] or
                sha(run/'metrics.json') != item['metrics_sha256'] or
                json.loads((run/'metrics.json').read_text())['rWL_um'] != item['rWL_um']):
            raise ValueError('Original TRAIN reference label changed')
        rows.append(dict(key=f"original_{item['row_index']:04d}", status='success',
            rWL_um=item['rWL_um'], source_run=str(run),
            physical_manifest_sha256=item['manifest_sha256'], metrics_sha256=item['metrics_sha256']))
    if len(rows) != 64:
        raise ValueError('Original TRAIN reference denominator changed')
    for item in extra:
        out = json.loads(Path(item['outcome_path']).read_text())
        rows.append(dict(key=f"extra_{item['source_row_index']:04d}", status=item['status'],
            rWL_um=(item['metrics'] or {}).get('rWL_um'), source_run=out.get('physical_run'),
            physical_manifest_sha256=out.get('physical_manifest_sha256'),
            metrics_sha256=out.get('metrics_sha256'), outcome_path=item['outcome_path'],
            outcome_sha256=item['outcome_sha256']))
    if len(rows) != 384 or len({r['key']for r in rows}) != 384:
        raise ValueError('Expanded TRAIN reference denominator changed')
    return rows, binding, label_path


def choice(rows):
    good = [r for r in rows if r['status'] == 'success']
    if not good:
        raise ValueError('No physically successful reference candidate')
    return dict(requested=len(rows), successful=len(good), failed=len(rows)-len(good),
                selected=min(good,key=lambda r:(r['rWL_um'],r['key'])), candidates=rows)


def select_offline(exp, cfg, plan):
    rows, binding, labels = offline_candidates(exp,cfg)
    folder=exp/'reference_selection';folder.mkdir(exist_ok=True)
    selection=dict(kind='best_measured_expanded_offline_TRAIN', rule='Minimum GRT rWL; stable source key tie break; final PPA not used',
        config_sha256=sha(exp/'config.json'), expanded_training_binding_sha256=sha(exp/'expanded_training_binding.json'),
        original_labels_sha256=sha(labels), choice=choice(rows), online_feedback=False)
    immutable_json(folder/'offline384.json',selection)
    selected=selection['choice']['selected']
    tasks=[dict(key='best_offline384', group='best_offline384',noise_index=0,
        source_run=selected['source_run'],source_manifest_sha256=selected['physical_manifest_sha256'],
        source_metrics_sha256=selected['metrics_sha256'],
        output=str(exp/'full_route/best_offline384'),run_id=cfg['run_id_prefix']+'_drt_best_offline384')]
    for arm,number in [('flowplace',0),('bc',0),('fql',0),('bc',2),('fql',2)]:
        for i in range(4):
            key=f'{arm}_r{number}_n{i}'
            tasks.append(dict(key=key,group=f'{arm}_r{number}',noise_index=i,
                source_outcome=str(exp/f'seed_0/round_{number}/requests/{arm}/evaluation/{i}/outcome.json'),
                output=str(exp/'full_route'/key),run_id=cfg['run_id_prefix']+'_drt_'+key))
    route_plan=dict(scope='bp_fe expanded offline-label FQL; twenty fixed policy evaluations plus one selected384TRAIN reference',
        config_sha256=sha(exp/'config.json'),seed=0,requested=21,route_end_iteration=20,
        backend_storage='scratch',reference_selection_sha256=sha(folder/'offline384.json'),tasks=tasks)
    immutable_json(exp/'full_route_plan.json',route_plan)
    immutable_json(exp/'training_replay_reference_plan.json',dict(config_sha256=sha(exp/'config.json'),
        expanded_training_binding_sha256=sha(exp/'expanded_training_binding.json'),
        offline_labels_sha256=sha(labels),offline_TRAIN_candidates=384,
        seed=0,arms=['bc','fql'],online_collection_rounds=[0,1],online_candidates_per_arm=16,
        detailed_route_end_iteration=20,run_id_prefix=cfg['run_id_prefix']+'_drt'))
    print(json.dumps(dict(status='expanded_reference_frozen',requested=384,selected=selected,
                          route_plan_sha256=sha(exp/'full_route_plan.json'))),flush=True)


def report(exp,cfg,plan,full_route=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    candidates,binding,labels=offline_candidates(exp,cfg)
    source=Path(plan['reference_experiment'])
    if sha(source/'config.json')!=plan['reference_config_sha256']:
        raise ValueError('Original64-label comparison changed')
    training=[];table=[];groups={};selections={}
    for name,root,count in [('original64',source,64),('expanded384',exp,384)]:
        config=json.loads((root/'config.json').read_text())
        controller=json.loads((root/'controller_seed_0.json').read_text())
        if (controller['stage']!='pilot_completed_requires_full_route_and_replication' or
                controller['config_sha256']!=sha(root/'config.json')):
            raise ValueError('Missing actual complete offline-to-online controller')
        for number in range(3):
            summary=collect(root,0,number)
            requests,_=load_requests(root,0,number)
            for arm in ['bc','fql']:
                path=root/f'seed_0/round_{number}/{arm}/completed.json'
                checked=audit_training_phase(root,config,arm,number,count,
                    legacy_offline=name=='original64')
                done=checked['record']
                if name=='expanded384' and done['expanded_training_binding_sha256']!=sha(exp/'expanded_training_binding.json'):
                    raise ValueError('Actual expanded checkpoint does not bind384 labels')
                training.append(dict(experiment=name,**checked))
                g=summary['groups'][arm+'/evaluation'];groups[f'{name}/{arm}/r{number}']=g
                m=g['mean']or{}
                table.append(dict(experiment=name,round=number,arm=arm,offline_labels=count,
                    used_online_labels=number*8,requested=g['count'],successful=g['successful'],
                    rWL_m=m.get('rWL_um',0)/1e6 if m else None,WNS_ns=m.get('WNS_ns'),
                    TNS_ns=m.get('TNS_ns'),Power_W=m.get('Power_W'),StdCellArea_um2=m.get('StdCellArea_um2')))
            if name=='expanded384' and any(r['role']=='evaluation'and r['training_feedback_eligible']for r in requests):
                raise ValueError('Evaluation feedback entered expanded replay')
    for arm in ['bc','fql']:
        online=[]
        for number in [0,1]:
            requests,_=load_requests(exp,0,number)
            for request in requests:
                if request['arm']==arm and request['role']=='collect':
                    out=audit_outcome(request,cfg,binding['reward_scale'])
                    online.append(dict(key=f"online_r{number}_{request['index']:04d}",status=out['status'],
                        rWL_um=(out['metrics']or{}).get('rWL_um'),source_run=out.get('physical_run'),
                        physical_manifest_sha256=out.get('physical_manifest_sha256'),metrics_sha256=out.get('metrics_sha256'),
                        outcome_path=out['outcome_path'],outcome_sha256=out['outcome_sha256']))
        if len(online)!=16:raise ValueError('Best400 reference must use exactly the16 actually used own online labels')
        selections[arm]=choice(candidates+online)
    folder=exp/('final_comparison'if full_route else 'comparison');folder.mkdir(exist_ok=True)
    immutable_json(exp/'reference_selection/online400.json',dict(config_sha256=sha(exp/'config.json'),
        rule='Minimum GRT rWL over384TRAIN plus16 own used online labels; no final PPA selection',choices=selections))
    route_reports={};search_reports={};common_flow=None;native_audits=[]
    if full_route:
        for name,root in [('original64',source),('expanded384',exp)]:
            path=root/'full_route/summary.json';data=json.loads(path.read_text())
            if (data['status']!='all_requested_outcomes_accounted' or
                    data['plan_sha256']!=sha(root/'full_route_plan.json') or data['requested']!=21):
                raise ValueError('Missing all21 actual final-route outcomes')
            for record in data['records']:
                out=record['result']
                if out['status']=='completed':
                    run=Path(out['physical_run']);manifest=json.loads((run/'manifest.json').read_text())
                    if (sha(run/'manifest.json')!=out['physical_manifest_sha256'] or
                            sha(run/'metrics.json')!=out['metrics_sha256'] or
                            out['spec']['route_end_iteration']!=20 or int(manifest['num_cores'])!=8 or
                            not out['final_macros_match_audited_grt']):
                        raise ValueError('Final PPA/tool/cap20/macro evidence differs')
                    audited=audit_final_outcome(out,design='bp_fe',route_cap=20,
                        legacy_plan=LEGACY_REPEAT_PLAN)
                    native_audits.append(audited)
                    flow=audited['physical_identity']
                    if common_flow is None:common_flow=flow
                    if common_flow!=flow:raise ValueError('Different final physical evaluators')
                elif out['status']!='upstream_physical_failure':
                    raise ValueError('Unresolved final physical outcome')
            route_reports[name]=dict(path=str(path),sha256=sha(path),groups=data['groups'])
            search_path=root/'best_replay_reference/report.json'
            search=json.loads(search_path.read_text())
            if search['status']!='completed':raise ValueError('Best-measured replay control is incomplete')
            expected=80 if name=='original64'else 400
            for arm in ['bc','fql']:
                selected=search['choices'][arm]
                out=search['outcomes'][arm]['outcome'];run=Path(out['physical_run'])
                manifest=json.loads((run/'manifest.json').read_text())
                audited=audit_final_outcome(out,design='bp_fe',route_cap=20,
                    legacy_plan=LEGACY_REPEAT_PLAN)
                if (selected['requested']!=expected or out['status']!='completed' or
                        Path(out['spec']['source_run']).resolve()!=Path(selected['selected']['source_run']).resolve() or
                        out['spec']['source_manifest_sha256']!=selected['selected']['physical_manifest_sha256'] or
                        out['spec']['route_end_iteration']!=20 or int(manifest['num_cores'])!=8 or
                        not out['final_macros_match_audited_grt'] or
                        sha(run/'manifest.json')!=out['physical_manifest_sha256'] or
                        sha(run/'metrics.json')!=out['metrics_sha256'] or
                        audited['physical_identity']!=common_flow):
                    raise ValueError('Best-measured replay reference provenance/budget differs')
                native_audits.append(audited)
            search_reports[name]=dict(path=str(search_path),sha256=sha(search_path),report=search)
    atomic_json(folder/'evidence.json',dict(status='expanded_offline_to_online_accounted',plan_sha256=sha(Path(plan['_path'])),
        expanded_training_binding_sha256=sha(exp/'expanded_training_binding.json'),training=training,groups=groups,
        best_measured_reference_selection_sha256=sha(exp/'reference_selection/online400.json'),
        scope='One bp_fe training seed; original64 and expanded384 differ in real label budget and TRAIN-only critic normalization; behavior targets unchanged',
        final_route_reports=route_reports,best_measured_reference_reports=search_reports,common_final_flow=common_flow,
        actual_native_route_audits=native_audits,report_source_sha256=sha(__file__),
        report_schema_amendment_sha256=sha(Path(plan['_path']).parent/'report_schema_amendment_v2.json'),
        detailed_route_DRC='See exact final-route groups; nonzero counts retained'if full_route else None,goal_completed=False))
    with(folder/'per_stage.csv').open('w')as handle:
        writer=csv.DictWriter(handle,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# 扩大离线实测标签后的 Flow-Q','',
        '原始64标签与扩展384标签使用相同256个行为目标、预训练图网络和更新预算；新标签也用于 Q 的 TRAIN-only 归一化。',
        '384标签方案的离线评分预算更大，不能声称与64标签方案同测量成本。每种方案内部的 BC/FQL 使用相同数据预算。',
        '先离线训练，再各收集两轮真实反馈并更新；每轮8条，最终策略实际使用16条。最后额外采集的8条不算该 checkpoint 已用数据。',
        ('前表是单电路单训练 seed 的 GRT 结果，最终详细布线结果见后表；四个固定噪声不当作四个独立 seed。'
         if full_route else '这是单电路单训练 seed 的 GRT 结果，四个固定评估噪声不当作四个独立 seed，不提供最终 DRC 结论。'),'',
        '| 数据 | 阶段 | 方法 | 已用在线标签 | 成功/请求 | GRT线长（m） | WNS（ns） | TNS（ns） |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in table:
        metrics='— | — | —'if r['rWL_m']is None else f"{r['rWL_m']:.6f} | {r['WNS_ns']:.6f} | {r['TNS_ns']:.3f}"
        lines.append(f"| {r['experiment']} | {r['round']} | {r['arm']} | {r['used_online_labels']} | {r['successful']}/{r['requested']} | {metrics} |")
    if route_reports:
        lines+=['','以下为统一 cap20 的详细布线结果；失败和非零 DRC 保留。最佳实测布局是单个参考，不是四次独立重复。','',
            '| 数据 | 布局组 | 完成/请求 | DRC=0 | 最终线长（m） | WNS（ns） | TNS（ns） | 功耗（W） | 面积（μm²） |',
            '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for name,data in route_reports.items():
            for group,r in data['groups'].items():
                m=r['mean'];values='— | — | — | — | —'if m is None else f"{m['DRT_WL']/1e6:.6f} | {m['WNS']:.6f} | {m['TNS']:.3f} | {m['Power']:.6f} | {m['StdCellArea']:.1f}"
                lines.append(f"| {name} | {group} | {r['completed']}/{r['requested']} | {r['drc_clean']} | {values} |")
        for name,data in search_reports.items():
            for arm,result in data['report']['outcomes'].items():
                m=result['outcome']['metrics'];budget=data['report']['choices'][arm]['requested']
                values=f"{m['DRT_WL']/1e6:.6f} | {m['WNS']:.6f} | {m['TNS']:.3f} | {m['Power']:.6f} | {m['StdCellArea']:.1f}"
                lines.append(f"| {name} | best{budget}/{arm} | 1/1 | {int(m['DRC']==0)} | {values} |")
    lines+=['','全部失败保留在分母；同预算直接选择最佳实测布局的候选和选择另存为 reference_selection。','']
    (folder/'summary.md').write_text('\n'.join(lines))
    fig,ax=plt.subplots(figsize=(7,4.5))
    for name,style in [('original64','--'),('expanded384','-')]:
        for arm,color in [('bc','#4774b7'),('fql','#b54c3e')]:
            subset=[r for r in table if r['experiment']==name and r['arm']==arm]
            ax.plot([r['used_online_labels']for r in subset],[r['rWL_m']for r in subset],style,marker='o',color=color,label=name+' '+arm)
    ax.set(xlabel='Own online labels actually used',ylabel='Global-route wirelength (m)',title='bp_fe; one training seed; four fixed diagnostic noises')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(folder/('label_coverage_curve.'+suffix),dpi=180)
    plt.close(fig)
    print(json.dumps(dict(status='expanded_comparison_reported',output=str(folder),goal_completed=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--stage',choices=['select-offline','report','final-report','audit-reference-training'],required=True)
    args=parser.parse_args()
    if 'SLURM_JOB_ID'not in os.environ:raise RuntimeError('Audit data/checkpoints and render plots through Slurm')
    path=Path(args.plan).resolve();plan=json.loads(path.read_text());plan['_path']=str(path)
    exp=Path(plan['experiment']).resolve();cfg=json.loads((exp/'config.json').read_text())
    if sha(exp/'config.json')!=plan['config_sha256']:raise ValueError('Expanded experiment configuration changed')
    amendment=json.loads((path.parent/'report_schema_amendment_v2.json').read_text())
    if amendment['original_plan_sha256']!=sha(path):raise ValueError('Report amendment belongs to a different plan')
    for name,digest in amendment['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Amended reporting/audit source changed: '+name)
    for name,digest in plan['training_source_hashes'].items():
        registered=amendment.get('prior_controller_requeue_revision',{})
        if name==registered.get('path'):
            if digest!=registered['original_plan_sha256']:
                raise ValueError('Controller amendment differs from the original plan')
            digest=registered['current_sha256']
        if sha(ROOT/name)!=digest:raise ValueError('Frozen training/controller source changed: '+name)
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    snapshot=exp/'report_sources'/(args.stage+'_job_'+os.environ['SLURM_JOB_ID']);snapshot.mkdir(parents=True,exist_ok=False)
    for name in ['graph_o2o_expanded_report.py','graph_o2o_expanded_data.py']:
        shutil.copyfile(Path(__file__).with_name(name),snapshot/name)
    if args.stage=='audit-reference-training':check_reference_training(exp,plan)
    elif args.stage=='select-offline':select_offline(exp,cfg,plan)
    else:report(exp,cfg,plan,full_route=args.stage=='final-report')
