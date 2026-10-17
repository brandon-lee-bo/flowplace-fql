"""Independently audit second-design training, online replay and final PPA."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import statistics

import numpy as np
import torch

from .core import critic_diagnostics
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import load_requests, validate_request
from .graph_o2o_design_collect import audit_outcome
from .graph_o2o_design_policy_data import load_dataset, measured_context
from .graph_o2o_design_route_controller import requested, output_path
from .graph_o2o_final_evidence import audit_final_outcome

KEYS = ['DRT_WL', 'WNS', 'TNS', 'Power', 'StdCellArea', 'DRC']


def actual_weight_changes(checkpoint, parent, declared):
    result = {}
    for module in ['flow', 'actor', 'critic']:
        before, after = parent[module], checkpoint[module]
        if set(before) != set(after):
            raise ValueError('Parent/child model inventories differ')
        changed, square, maximum = 0, 0., 0.
        for name, value in after.items():
            old = before[name]
            if value.shape != old.shape or value.dtype != old.dtype:
                raise ValueError('A trained tensor changed its shape/type')
            if not value.is_floating_point():
                if not torch.equal(value, old):
                    raise ValueError('Training changed a fixed nonfloating graph buffer')
                continue
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite checkpoint tensor')
            if torch.equal(value, old):
                continue
            changed += 1
            delta = value.double() - old.double()
            square += float(delta.square().sum())
            maximum = max(maximum, float(delta.abs().max()))
        values = dict(changed_parameter_tensors=changed, parameter_delta_l2=math.sqrt(square),
                      max_absolute_parameter_delta=maximum)
        if not changed or maximum <= 0:
            raise ValueError('Expected network did not actually update: ' + module)
        reference = declared[module]
        if changed != reference['changed_parameter_tensors']:
            raise ValueError('Actual changed tensors disagree with the training record: ' + module)
        for field in ['parameter_delta_l2', 'max_absolute_parameter_delta']:
            if not math.isclose(values[field], reference[field], rel_tol=1e-8, abs_tol=1e-12):
                raise ValueError('Actual weight change disagrees with training metadata: ' + field)
        result[module] = values
    return result


def run(args):
    exp = Path(args.experiment).resolve()
    cfg, binding, verification = measured_context(exp)
    load_dataset(exp)
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    restart = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
    key = os.environ['SLURM_JOB_ID'] + (f'_restart_{restart}' if restart else '')
    output = exp / 'reports' / ('job_' + key)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / Path(__file__).name)
    controller_path = exp / 'controller_seed_0.json'
    controller = json.loads(controller_path.read_text())
    if controller['stage'] != 'pilot_completed_requires_full_route_and_replication':
        raise ValueError('The actual offline-to-online controller has not completed')
    measured, training, eval_noises = [], [], {}
    for number in range(cfg['online_rounds'] + 1):
        rows, task_path = load_requests(exp, 0, number)
        expected = 28 if number == 0 else 24
        if len(rows) != expected:
            raise ValueError('A requested stage lost samples')
        expected_samples = {(arm, role, index) for arm in ['bc', 'fql']
            for role, count in [('collect', 8), ('evaluation', 4)] for index in range(count)}
        if number == 0:
            expected_samples |= {('flowplace', 'evaluation', index) for index in range(4)}
        if {(row['arm'], row['role'], row['index']) for row in rows} != expected_samples:
            raise ValueError('Stage sample identities differ from the predeclared comparison')
        for row in rows:
            folder = validate_request(row)
            out = audit_outcome(row, cfg, binding['reward_scale'])
            if (row['graph'] != cfg['design'] or row['dataset_binding_sha256'] != sha(exp / 'dataset_binding.json') or
                    row['training_feedback_eligible'] != (row['role'] == 'collect')):
                raise ValueError('A measured action crossed its training/evaluation context')
            with np.load(folder / 'actions.npz', allow_pickle=False) as arrays:
                noise = arrays['noise'].copy()
            if row['role'] == 'evaluation':
                old = eval_noises.setdefault(row['index'], noise)
                np.testing.assert_array_equal(old, noise)
            raw = np.asarray(row['raw_action'])
            action = np.asarray(row['executed_action'])
            measured.append(dict(round=number, arm=row['arm'], role=row['role'], index=row['index'],
                status=out['status'], raw_to_execution_RMS=float(np.sqrt(np.square(raw-action).mean())),
                reward=out['reward'], predicted_q_mean=statistics.mean(row['predicted_q']) if row['predicted_q'] else None,
                outcome_path=out['outcome_path'], outcome_sha256=out['outcome_sha256'],
                request_sha256=out['request_sha256'], metrics=out['metrics']))
        if number:
            replay_path = exp / 'seed_0' / f'replay_round_{number}.json'
            replay = json.loads(replay_path.read_text())
            for arm in ['bc', 'fql']:
                expected_paths = {item['outcome_path'] for item in measured
                    if item['round'] < number and item['arm'] == arm and item['role'] == 'collect'}
                actual_paths = {item['outcome_path'] for item in replay['arms'][arm]}
                if actual_paths != expected_paths or len(replay['arms'][arm]) != number * 8:
                    raise ValueError('Online replay omitted feedback or consumed evaluation/future samples')
        for arm in ['bc', 'fql']:
            folder = exp / 'seed_0' / f'round_{number}' / arm
            path = folder / 'completed.json'
            done = json.loads(path.read_text())
            checkpoint_path = folder / 'checkpoint.pt'
            parent_path = Path(done['parent_checkpoint'])
            if (sha(checkpoint_path) != done['checkpoint_sha256'] or
                    sha(parent_path) != done['parent_checkpoint_sha256'] or
                    done['train_rows'] != 64 + number * 8 or done['online_training_rows'] != number * 8):
                raise ValueError('Checkpoint/data/update provenance differs')
            current = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            parent = torch.load(parent_path, map_location='cpu', weights_only=False)
            total = cfg['offline_steps'] if number == 0 else cfg['online_update_steps']
            if (current['config_sha256'] != sha(exp / 'config.json') or current['iteration'] != total or
                    current['dataset_binding_sha256'] != sha(exp / 'dataset_binding.json')):
                raise ValueError('Actual checkpoint does not match its predeclared training stage')
            if number and current['replay_sha256'] != sha(exp / 'seed_0' / f'replay_round_{number}.json'):
                raise ValueError('Online checkpoint did not bind the actual measured replay')
            changes = actual_weight_changes(current, parent, done['parameter_changes'])
            training.append(dict(round=number, arm=arm, completed_path=str(path), completed_sha256=sha(path),
                checkpoint_path=str(checkpoint_path), checkpoint_sha256=sha(checkpoint_path),
                parent_checkpoint_sha256=sha(parent_path), actual_weight_changes=changes,
                used_online_labels=number * 8, training_steps=total,
                critic_train=done['critic_train'], critic_validation=done['critic_validation']))
            del current, parent
    route_plan_path = exp / 'full_route_plan.json'
    route_plan = json.loads(route_plan_path.read_text())
    tasks = requested(exp, route_plan)
    if route_plan['requested'] != 23 or route_plan['detailed_route_end_iteration'] != 20:
        raise ValueError('The final physical comparison changed its scope or repair budget')
    route_rows, physical_identity = [], None
    for task in tasks:
        path = output_path(exp, task) / 'outcome.json'
        out = json.loads(path.read_text())
        row = dict(group=task['group'], noise_index=task['index'], status=out['status'],
            outcome_path=str(path), outcome_sha256=sha(path),
            **{key: (out.get('metrics') or {}).get(key) for key in KEYS})
        if out['status'] == 'completed':
            run = Path(out['physical_run'])
            checked = audit_final_outcome(out, design=cfg['design'], route_cap=20)
            identity = checked['physical_identity']
            if physical_identity is None:
                physical_identity = identity
            if identity != physical_identity:
                raise ValueError('Final physical settings differ between methods')
            row.update(physical_run=str(run), physical_manifest_sha256=sha(run / 'manifest.json'),
                final_macro_count=checked['final_macro_count'],
                final_DEF_reparsed=checked['actual_macro_DEF_reparsed'],
                native_route_log_sha256=checked['native_route_log_sha256'])
        elif out['status'] != 'upstream_physical_failure':
            raise ValueError('An unresolved final failure cannot be treated as PPA')
        route_rows.append(row)
    groups = {}
    for group in dict.fromkeys(row['group'] for row in route_rows):
        rows = [row for row in route_rows if row['group'] == group]
        completed = [row for row in rows if row['status'] == 'completed']
        mean = {key: statistics.mean(row[key] for row in completed) for key in KEYS} if completed else None
        groups[group] = dict(requested=len(rows), completed=len(completed),
            drc_clean=sum(row['DRC'] == 0 for row in completed),
            mean=mean if len(completed) == len(rows) else None, successful_only_mean=mean)
    q_checks = {}
    for number in range(3):
        rows = [row for row in measured if row['round'] == number and row['arm'] == 'fql' and row['role'] == 'evaluation']
        prediction = np.asarray([[row['predicted_q_mean']] for row in rows])
        target = np.asarray([row['reward'] for row in rows])
        q_checks[str(number)] = critic_diagnostics(prediction, target)
    report = dict(status='second_fixed_design_training_online_and_final_PPA_audited', design=cfg['design'],
        config_sha256=sha(exp / 'config.json'), dataset_binding_sha256=sha(exp / 'dataset_binding.json'),
        implementation_plan_sha256=sha(exp / 'implementation_plan.json'),
        controller_sha256=sha(controller_path), final_route_plan_sha256=sha(route_plan_path),
        requested_backend_outcomes=76, actual_backend_outcomes=len(measured), requested_final_routes=23,
        initial_TRAIN=64, initial_validation=16, final_used_online_labels_per_arm=16,
        final_unused_collect_labels_per_arm=8, groups=groups, actual_training=training,
        Q_fixed_noise_evaluation=q_checks, measurements=measured, final_layouts=route_rows,
        common_physical_identity=physical_identity, independent_training_seeds=1,
        noises_are_not_training_seeds=True, cross_design_zero_shot_claim=False,
        heldout_labels_used=False, old_Table2_resumed=False, goal_completed=False,
        source_sha256=sha(__file__),
        physical_audit_source_sha256=sha(Path(__file__).with_name('graph_o2o_final_evidence.py')),
        slurm_job_id=os.environ['SLURM_JOB_ID'])
    if len(measured) != 76:
        raise ValueError('Final policy interaction denominator changed')
    atomic_json(output / 'evidence.json', report)
    with (output / 'per_layout.csv').open('w') as handle:
        keys = ['group', 'noise_index', 'status', *KEYS, 'outcome_path', 'outcome_sha256']
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction='ignore')
        writer.writeheader(); writer.writerows(route_rows)
    lines = ['# 第二电路 bp_be：FlowPlace＋Flow-Q', '',
        '同一 OpenROAD/Nangate45、八线程、详细布线 cap20；宏块保持实际 GRT 导入位置。',
        '每组四个固定生成噪声，只有一个独立训练 seed。两电路分别训练，不代表跨电路零样本迁移。', '',
        '| 方法 | 完成/请求 | DRC=0 | rWL（m） | WNS（ns） | TNS（ns） | 功耗（W） | 面积（μm²） |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for name, group in groups.items():
        mean = group['mean']
        values = [mean['DRT_WL'] / 1e6, *[mean[key] for key in KEYS[1:5]]] if mean else [None] * 5
        cells = ' | '.join('—' if value is None else f'{value:.6g}' for value in values)
        lines.append(f"| {name} | {group['completed']}/{group['requested']} | {group['drc_clean']} | {cells} |")
    lines += ['', '训练包含64个真实 TRAIN 标签、16个独立验证标签和192个无标签 TRAIN teacher。',
        '最终策略在两个在线阶段累计使用各自16条新反馈；最后一轮额外8条采集未用于这个最终策略。',
        '已独立读取父子 checkpoint，核验 BC 与 FQL 各阶段的 flow、actor、critic 均实际更新。',
        'best64、best80 各为一份根据已测 GRT 选出的布局，保留全部请求成本；没有按最终 PPA 挑选。',
        '失败留在分母中，组内有失败时不报告全组均值。详细布线 DRC 与时序分别判断。', '']
    (output / 'summary.md').write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(output / 'evidence.json'), summary=str(output / 'summary.md'), groups=groups)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run checkpoint and physical audits under Slurm')
    run(args)
