"""Preserve and inspect completed graph-policy training logs without retraining.

Mini-batch losses, fixed validation errors and downstream placement quality are
different measurements. These figures describe optimization and generalization;
they neither select a checkpoint nor establish a PPA improvement.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bound(item):
    path = Path(item['path'])
    if sha(path) != item['sha256']:
        raise ValueError('Completed training input changed: ' + str(path))
    return path


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(plan_path):
    plan = json.loads(plan_path.read_text())
    if sha(__file__) != plan['source_sha256']:
        raise ValueError('Registered curve implementation changed')
    curves, stages, bindings, jobs = [], [], [], set()
    for experiment in plan['experiments']:
        cfg_path = load_bound(experiment['config'])
        cfg = json.loads(cfg_path.read_text())
        bindings.append(experiment['config'])
        for item in experiment['stages']:
            log_path, completed_path = load_bound(item['log']), load_bound(item['completed'])
            bindings.extend([item['log'], item['completed']])
            completed = json.loads(completed_path.read_text())
            stage, arm = item['round'], item['arm']
            total = cfg['offline_steps'] if stage == 0 else cfg['online_update_steps']
            offset = 0 if stage == 0 else cfg['offline_steps'] + (stage - 1) * cfg['online_update_steps']
            if (completed['status'] != 'completed' or completed['round'] != stage
                    or completed['seed'] != experiment['seed'] or completed['arm'] != arm
                    or completed['online_training_rows'] != stage * 8
                    or completed['critic_train']['count'] != completed['train_rows']
                    or completed['critic_validation']['count'] != 16):
                raise ValueError('Wrong completed arm/stage or feedback count')
            rows = [json.loads(line) for line in log_path.read_text().splitlines() if line]
            if [r['step'] for r in rows] != list(range(50, total + 1, 50)):
                raise ValueError('Missing, repeated or unfinished logged training steps')
            for row in rows:
                numeric = ('actor_loss', 'flow_loss', 'critic_loss', 'distill', 'q', 'seconds')
                if any(not math.isfinite(row[k]) for k in numeric):
                    raise ValueError('Non-finite training measurement')
                if any(row[k] < 0 for k in ('flow_loss', 'critic_loss', 'distill', 'seconds')):
                    raise ValueError('Negative squared loss or elapsed time')
                if row['online_training_rows'] != stage * 8:
                    raise ValueError('Logged feedback count differs from stage')
                expected = cfg['alpha'] * row['distill'] - (row['q'] if arm == 'fql' else 0)
                if not math.isclose(row['actor_loss'], expected, rel_tol=2e-6, abs_tol=2e-7):
                    raise ValueError('Actor loss differs from configured distillation/Q objective')
                curves.append(dict(experiment=experiment['name'], seed=experiment['seed'],
                    arm=arm, round=stage, cumulative_step=offset + row['step'], **row))
            final = rows[-1]
            stages.append(dict(experiment=experiment['name'], seed=experiment['seed'], arm=arm,
                round=stage, logged_rows=len(rows), final_step=total,
                used_online_rows=completed['online_training_rows'], train_rows=completed['train_rows'],
                critic_train_MAE=completed['critic_train']['mae'],
                critic_validation_MAE=completed['critic_validation']['mae'],
                critic_validation_Spearman=completed['critic_validation']['spearman_r'],
                final_batch_flow_loss=final['flow_loss'], final_batch_distill_MSE=final['distill'],
                final_batch_critic_MSE=final['critic_loss'], reported_policy_Q=final['q'],
                checkpoint_sha256=completed['checkpoint_sha256'], training_job=completed['slurm_job_id']))
            jobs.add(completed['slurm_job_id'])
    result = subprocess.run(['sacct', '-j', ','.join(sorted(jobs)), '-X', '-n', '-P',
        '-o', 'JobIDRaw,State,ExitCode'], check=True, capture_output=True, text=True)
    completed_jobs = {row.split('|')[0] for row in result.stdout.splitlines()
                      if row and row.split('|')[1:3] == ['COMPLETED', '0:0']}
    if not jobs <= completed_jobs:
        raise ValueError('Actual training jobs are not all successfully completed')
    output = Path(plan['output']) / ('job_' + os.environ['SLURM_JOB_ID'])
    output.mkdir(parents=True, exist_ok=False)
    for index, item in enumerate(bindings):
        path = load_bound(item)
        copied = output / f'input_{index:03d}_{path.name}'
        shutil.copyfile(path, copied)
        if sha(copied) != item['sha256'] or sha(path) != item['sha256']:
            raise ValueError('Input changed during durable preservation')
    shutil.copyfile(plan_path, output / 'plan.json')
    shutil.copyfile(__file__, output / Path(__file__).name)
    (output / 'actual_training_jobs.sacct.txt').write_text(result.stdout)
    write_csv(output / 'training_curves.csv', curves)
    write_csv(output / 'stage_metrics.csv', stages)
    plot(output, plan['experiments'], curves, stages)
    evidence = dict(status='completed_training_curves_and_fixed_validation_preserved',
        experiments=len(plan['experiments']), arm_stages=len(stages), logged_rows=len(curves),
        stages=stages, input_bindings=bindings, plan_sha256=sha(plan_path),
        source_sha256=sha(__file__), slurm_job_id=os.environ['SLURM_JOB_ID'],
        independent_training_jobs=sorted(jobs), validation_rows_per_stage=16,
        checkpoint_selection_unchanged=True, new_training_or_EDA_runs=0,
        checkpoint_weights_not_reloaded_by_this_report=True)
    (output / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    lines = ['# 训练曲线与固定验证集', '',
        f'全部{len(stages)}个arm/stage、{len(jobs)}个实际训练作业均已完成；保存原始日志与配置。',
        '曲线是每50步记录的单个mini-batch值，并非完整数据集误差或最终PPA；没有据此挑选checkpoint。', '',
        '| 实验 | 阶段 | 方法 | TRAIN MAE | 固定16条验证 MAE | 验证 Spearman |',
        '| --- | --- | --- | --- | --- | --- |']
    for row in stages:
        lines.append(f"| {row['experiment']} | {row['round']} | {row['arm']} | "
            f"{row['critic_train_MAE']:.6f} | {row['critic_validation_MAE']:.6f} | "
            f"{row['critic_validation_Spearman']:.4f} |")
    lines += ['', '误差单位是各电路冻结归一化后的reward；不能直接跨电路比较绝对功耗或线长。',
        '阶段TRAIN误差包含该checkpoint实际使用的离线与自采在线标签；每阶段数据量见stage_metrics.csv。',
        '每个电路三个阶段的验证布局保持固定且不进入训练。它与策略新生成布局的Q诊断是不同分布，不能互相替代。',
        '扩大标签组改变TRAIN特征归一化和训练样本数，因此不是仅添加样本而其他统计量完全不变的对照。',
        '低训练loss或验证相关不能证明策略改进；应结合独立实际布局和完整布线对照。', '']
    (output / 'summary.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=evidence['status'], output=str(output), arm_stages=len(stages),
                          logged_rows=len(curves))), flush=True)


def plot(output, experiments, curves, stages):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'bc': '#2875b2', 'fql': '#df711c'}
    fig, axes = plt.subplots(len(experiments), 3, figsize=(13, 13), squeeze=False)
    for i, experiment in enumerate(experiments):
        for col, (key, title) in enumerate([('flow_loss', 'Behavior flow MSE'),
                ('distill', 'Actor distillation MSE'), ('critic_loss', 'Critic batch MSE')]):
            ax = axes[i, col]
            for arm in ['bc', 'fql']:
                rows = [r for r in curves if r['experiment'] == experiment['name'] and r['arm'] == arm]
                ax.semilogy([r['cumulative_step'] for r in rows],
                            [r[key] if r[key] > 0 else float('nan') for r in rows],
                            label=arm.upper(), color=colors[arm], linewidth=1)
            for step in [1000, 1600]:
                ax.axvline(step, color='grey', linestyle=':', linewidth=.8)
            ax.set_title(experiment['name'] + ' / ' + title, fontsize=9)
            ax.grid(alpha=.2)
            if i == len(experiments) - 1:
                ax.set_xlabel('Updates after common initialization')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Logged mini-batch losses; dotted lines begin online updates', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .97))
    for suffix in ['png', 'pdf']:
        fig.savefig(output / ('training_losses.' + suffix), dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(len(experiments), 2, figsize=(11, 12), squeeze=False)
    for i, experiment in enumerate(experiments):
        for arm in ['bc', 'fql']:
            rows = [r for r in stages if r['experiment'] == experiment['name'] and r['arm'] == arm]
            axes[i, 0].semilogy([r['round'] for r in rows], [r['critic_train_MAE'] for r in rows],
                '-o', color=colors[arm], label=arm.upper() + ' TRAIN', markersize=3)
            axes[i, 0].semilogy([r['round'] for r in rows], [r['critic_validation_MAE'] for r in rows],
                '--o', color=colors[arm], label=arm.upper() + ' validation', markersize=3)
            axes[i, 1].plot([r['round'] for r in rows], [r['critic_validation_Spearman'] for r in rows],
                '-o', color=colors[arm], label=arm.upper(), markersize=3)
        axes[i, 0].set_title(experiment['name'] + ' / reward MAE', fontsize=10)
        axes[i, 1].set_title(experiment['name'] + ' / validation ranking', fontsize=10)
        axes[i, 1].set_ylim(-1.05, 1.05)
        for ax in axes[i]:
            ax.set_xticks([0, 1, 2], ['offline', 'online 1', 'online 2'])
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle('Fixed validation layouts are excluded from training', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .97))
    for suffix in ['png', 'pdf']:
        fig.savefig(output / ('critic_generalization.' + suffix), dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use Slurm for plotting and training-log analysis')
    run(args.plan.resolve())
