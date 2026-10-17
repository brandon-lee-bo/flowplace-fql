"""Audit complete independent-seed pilots without pooling noises as seeds."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import statistics

from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import load_requests
from .graph_o2o_collect import collect


def report(plan_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    experiments = {0: Path(plan['source_experiment'])}
    experiments.update({int(k): Path(v) for k, v in plan['experiments'].items()})
    if sorted(experiments) != plan['independent_training_seeds']:
        raise ValueError('Missing or unexpected independent training seed')
    reference = json.loads((experiments[0] / 'config.json').read_text())
    ignored = {'seed', 'scope', 'backend_concurrency'}
    common_config = {k: v for k, v in reference.items() if k not in ignored}
    rows, evidence, shared_inputs = [], {}, None
    for seed, exp in sorted(experiments.items()):
        cfg = json.loads((exp / 'config.json').read_text())
        if cfg['seed'] != seed or {k: v for k, v in cfg.items() if k not in ignored} != common_config:
            raise ValueError('Replication algorithm/data configuration changed')
        if len(experiments) < 2 or cfg['evaluation_noise_seed'] != plan['paired_evaluation_noise_seed']:
            raise ValueError('Independent seeds or paired-noise design changed')
        controller = json.loads((exp / f'controller_seed_{seed}.json').read_text())
        if (controller['config_sha256'] != sha(exp / 'config.json') or
                controller['stage'] != 'pilot_completed_requires_full_route_and_replication'):
            raise ValueError('Pilot controller has not completed all real online stages')
        training = []
        for number in range(cfg['online_rounds'] + 1):
            folder = exp / f'seed_{seed}/round_{number}'
            status_path = folder / 'training_status.json'
            status = json.loads(status_path.read_text())
            if (status['status'] != 'completed' or status['seed'] != seed or
                    status['config_sha256'] != sha(exp / 'config.json') or
                    not status['no_heldout_labels_used']):
                raise ValueError('Incomplete or unbound training phase')
            inputs = {k: status[k] for k in ['graph_cache_sha256', 'pretrained_sha256',
                      'offline_dataset_sha256', 'reward_scale']}
            inputs['policy_source_sha256'] = status['source']['files']['fql/graph_o2o_models.py']['sha256']
            inputs['adapter_source_sha256'] = status['source']['files']['baseline/api.py']['sha256']
            if shared_inputs is None:
                shared_inputs = inputs
            if inputs != shared_inputs:
                raise ValueError('Graph, pretrained policy, data, reward or policy implementation differs')
            for arm in ['bc', 'fql']:
                path = folder / arm / 'completed.json'
                completion = json.loads(path.read_text())
                if (completion['seed'] != seed or completion['round'] != number or
                        completion['online_training_rows'] != number * cfg['online_samples'] or
                        completion['train_rows'] != 64 + number * cfg['online_samples'] or
                        completion['checkpoint_sha256'] != sha(path.with_name('checkpoint.pt'))):
                    raise ValueError('Actual checkpoint or online data budget differs')
                if number and not all(completion['parameter_changes'][k]['changed_parameter_tensors'] > 0
                                      for k in ['flow', 'actor', 'critic']):
                    raise ValueError('Online stage lacks actual updates to all three networks')
                training.append(dict(path=str(path), sha256=sha(path), details=completion))
            measured = collect(exp, seed, number)
            requests, _ = load_requests(exp, seed, number)
            if any(r['role'] == 'evaluation' and (r['training_feedback_eligible'] or
                   r['noise_seed'] != plan['paired_evaluation_noise_seed']) for r in requests):
                raise ValueError('Evaluation feedback/noise contract changed')
            for arm in ['bc', 'fql']:
                group = measured['groups'][arm + '/evaluation']
                if group['count'] != plan['evaluation_samples_per_arm']:
                    raise ValueError('Evaluation denominator changed')
                metrics = group['mean'] or {}
                rows.append(dict(seed=seed, round=number, arm=arm,
                    training_layouts=64 + number * cfg['online_samples'],
                    requested=group['count'], successful=group['successful'],
                    rWL_m=None if not metrics else metrics['rWL_um'] / 1e6,
                    WNS_ns=metrics.get('WNS_ns'), TNS_ns=metrics.get('TNS_ns'),
                    Power_W=metrics.get('Power_W'), StdCellArea_um2=metrics.get('StdCellArea_um2')))
        evidence[str(seed)] = dict(experiment=str(exp), config_sha256=sha(exp / 'config.json'),
            controller_sha256=sha(exp / f'controller_seed_{seed}.json'),
            backend_storage=controller.get('backend_storage', 'bulk'), training=training)

    groups = []
    for number in range(reference['online_rounds'] + 1):
        for arm in ['bc', 'fql']:
            selected = [r for r in rows if r['round'] == number and r['arm'] == arm]
            complete = len(selected) == len(experiments) and all(r['rWL_m'] is not None for r in selected)
            mean = ({k: statistics.mean(r[k] for r in selected)
                     for k in ['rWL_m', 'WNS_ns', 'TNS_ns', 'Power_W', 'StdCellArea_um2']}
                    if complete else None)
            groups.append(dict(round=number, arm=arm, independent_seeds=len(selected),
                successful_layouts=sum(r['successful'] for r in selected),
                requested_layouts=sum(r['requested'] for r in selected), mean=mean,
                across_training_seed_rWL_std_m=statistics.stdev(r['rWL_m'] for r in selected) if complete else None))
    output = plan_path.parent / 'reports' / ('grt_job_' + os.environ['SLURM_JOB_ID'])
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / Path(__file__).name)
    atomic_json(output / 'evidence.json', dict(status='requested_seed_pilots_accounted',
        plan_sha256=sha(plan_path), source_sha256=sha(__file__), shared_inputs=shared_inputs,
        independent_training_seeds=sorted(experiments), evidence=evidence, groups=groups,
        scope='bp_fe GRT only; noise samples averaged within each training seed first',
        detailed_route_DRC=None, goal_completed=False))
    with (output / 'per_training_seed.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ['# 独立训练 seed 的 Flow-Q 复现', '',
        '同一 bp_fe 电路、预训练权重与离线数据，各训练 seed 独立完成离线和两轮真实反馈更新。',
        '新增 seed 的物理输出存入共享 scratch；工具、工艺、输入和八线程流程保持一致，并单独验证固定布局结果。',
        '先在每个训练 seed 内对四个固定噪声求均值，再汇总训练 seed；噪声不当作独立训练重复。',
        '以下是 GRT 指标，不能代表最终详细布线 PPA 或 DRC。', '',
        '| 阶段 | 方法 | 独立训练 seed 数 | 成功/请求 | GRT 线长（m） | 训练 seed 间标准差（m） | WNS（ns） | TNS（ns） |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for g in groups:
        m = g['mean']
        values = ['—'] * 4 if m is None else [f"{m['rWL_m']:.6f}",
            f"{g['across_training_seed_rWL_std_m']:.6f}", f"{m['WNS_ns']:.6f}", f"{m['TNS_ns']:.3f}"]
        lines.append(f"| {g['round']} | {g['arm']} | {g['independent_seeds']} | {g['successful_layouts']}/{g['requested_layouts']} | " + ' | '.join(values) + ' |')
    lines += ['', f'失败保留在分母；不选择最佳 seed、噪声或 checkpoint。当前 {len(experiments)} 个训练 seed 仍不足以证明跨电路的普遍稳定性。',
              '完整逐 seed 的功耗、面积和模型/数据证据分别见 CSV 与 evidence.json。', '']
    (output / 'summary.md').write_text('\n'.join(lines))
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for arm, color in [('bc', '#4774b7'), ('fql', '#b54c3e')]:
        for seed in experiments:
            subset = [r for r in rows if r['seed'] == seed and r['arm'] == arm]
            ax.plot([r['training_layouts'] for r in subset], [r['rWL_m'] for r in subset],
                    color=color, alpha=.32, linewidth=1)
        subset = [g for g in groups if g['arm'] == arm and g['mean'] is not None]
        ax.errorbar([64 + g['round'] * reference['online_samples'] for g in subset],
            [g['mean']['rWL_m'] for g in subset],
            yerr=[g['across_training_seed_rWL_std_m'] for g in subset],
            color=color, marker='o', capsize=4, label=arm.upper())
    ax.set(xlabel='Measured layouts used by each policy', ylabel='Global-route wirelength (m)',
           title=f"bp_fe; {len(experiments)} independent training seeds; SD across seed means")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(output / ('learning_curve.' + ext), dpi=180)
    plt.close(fig)
    print(json.dumps(dict(status='replication_reported', output=str(output), goal_completed=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Audit full checkpoints and plot through Slurm')
    report(args.plan)
