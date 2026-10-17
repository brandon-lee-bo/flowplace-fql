"""Measure on-policy Q calibration, execution changes and real online updates.

Read only TRAIN labels and outcomes already requested by the pilot. This is
diagnosis, never checkpoint/noise selection or a new source of replay data.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .core import critic_diagnostics
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import ROOT, load_requests
from .graph_o2o_collect import audit_outcome


def mean_or_none(values):
    return float(np.mean(values)) if len(values) else None


def describe(rows, expected, train_raw, train_exec):
    good = [r for r in rows if r['status'] == 'success']
    labeled = [r for r in good if r['predicted_q'] is not None]
    report = dict(requested=expected, measured=len(rows), successful=len(good),
                  all_requested_successful=len(good) == expected,
                  failures=[dict(index=r['index'], failure_class=r.get('failure_class'))
                            for r in rows if r['status'] != 'success'],
                  unique_executed_layouts=len({r['action_sha256'] for r in rows}))
    if labeled:
        predictions = np.asarray([r['predicted_q'] for r in labeled])
        rewards = np.asarray([r['reward'] for r in labeled])
        diagnostics = critic_diagnostics(predictions, rewards)
        error = predictions.mean(-1)-rewards
        diagnostics.update(optimism_bias=float(error.mean()),
                           overestimated_count=int((error > 0).sum()),
                           twin_range_mean=float(np.ptp(predictions, axis=-1).mean()),
                           comparison_scope='Measured successful cases only; failures remain above')
        report['on_policy_Q'] = diagnostics
    if rows:
        raw = np.asarray([r['raw_action'] for r in rows]).reshape(len(rows), -1)
        executed = np.asarray([r['executed_action'] for r in rows]).reshape(len(rows), -1)
        displacement = np.sqrt(np.square(raw-executed).mean(-1))
        raw_distance = np.sqrt(np.square(raw[:, None]-train_raw[None]).mean(-1))
        exec_distance = np.sqrt(np.square(executed[:, None]-train_exec[None]).mean(-1))
        report['action_geometry'] = dict(
            coordinates='Normalized canvas [-1,1], RMS over macro x/y coordinates',
            raw_to_legalized_rms_mean=float(displacement.mean()),
            raw_to_legalized_rms_max=float(displacement.max()),
            nearest_offline_TRAIN_raw_rms_mean=float(raw_distance.min(-1).mean()),
            nearest_offline_TRAIN_executed_rms_mean=float(exec_distance.min(-1).mean()),
            raw_action_variation=float(raw.std(0).mean()),
            executed_action_variation=float(executed.std(0).mean()),
            native_backend_legal_count=sum(bool(r.get('backend_macro_legal', False)) for r in good))
    # Incomplete groups never become apparently complete successful-only means.
    report['rWL_um_mean'] = mean_or_none([r['metrics']['rWL_um'] for r in good]) if len(good) == expected else None
    return report


def run(args):
    experiment = Path(args.experiment).resolve()
    cfg = json.loads((experiment/'config.json').read_text())
    data_path = ROOT/'datasets/bp_fe_grt_v1/dataset.npz'
    manifest = json.loads(data_path.with_name('manifest.json').read_text())
    if sha(data_path) != manifest['dataset_sha256']:
        raise ValueError('Changed offline dataset')
    with np.load(data_path, allow_pickle=False) as arrays:
        train = arrays['critic_split'] == 0
        train_raw = arrays['raw_action'][train].reshape(int(train.sum()), -1)
        train_exec = arrays['action'][train].reshape(int(train.sum()), -1)
    folder = experiment/'diagnostics'/('seed_'+str(args.seed)+'_job_'+os.environ['SLURM_JOB_ID'])
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, folder/'graph_o2o_diagnostics.py')
    report = dict(created_unix=time.time(), slurm_job_id=os.environ['SLURM_JOB_ID'],
                  config_sha256=sha(experiment/'config.json'), source_sha256=sha(__file__),
                  dataset_sha256=manifest['dataset_sha256'], seed=args.seed, rounds={},
                  evaluation_feedback_replayed=False, heldout_labels_used=False,
                  scope='One fixed bp_fe circuit, one training seed; noises are not training replicates')
    csv_rows = []
    for number in range(cfg['online_rounds']+1):
        stage = experiment/f'seed_{args.seed}'/f'round_{number}'
        marker = stage/'requests/completed.json'
        if not marker.exists():
            report['rounds'][str(number)] = dict(status='not_exported')
            continue
        requests, tasks_path = load_requests(experiment, args.seed, number)
        outcomes = []
        for request in requests:
            path = Path(request['sample_dir'])/'outcome.json'
            if not path.exists():
                continue
            row = audit_outcome(request, cfg, manifest['reward_scale_train_only'])
            outcomes.append(row)
            predicted = mean_or_none(row['predicted_q']) if row['predicted_q'] is not None else None
            csv_rows.append(dict(round=number, arm=row['arm'], role=row['role'], index=row['index'],
                status=row['status'], reward=row['reward'], predicted_Q=predicted,
                prediction_error=None if predicted is None else predicted-row['reward'],
                rWL_um=(row['metrics'] or {}).get('rWL_um'),
                WNS_ns=(row['metrics'] or {}).get('WNS_ns'),
                TNS_ns=(row['metrics'] or {}).get('TNS_ns'),
                action_sha256=row['action_sha256'], outcome=row['outcome_path'], outcome_sha256=row['outcome_sha256']))
        current = dict(status='complete' if len(outcomes) == len(requests) else 'incomplete',
                       requested=len(requests), measured=len(outcomes), tasks_sha256=sha(tasks_path), groups={}, updates={})
        for arm, role in dict.fromkeys((r['arm'], r['role']) for r in requests):
            rows = [r for r in outcomes if r['arm'] == arm and r['role'] == role]
            expected = sum(r['arm'] == arm and r['role'] == role for r in requests)
            current['groups'][arm+'/'+role] = describe(rows, expected, train_raw, train_exec)
        for arm in ['bc', 'fql']:
            path = stage/arm/'completed.json'
            if not path.exists():
                continue
            completion = json.loads(path.read_text())
            if sha(stage/arm/'checkpoint.pt') != completion['checkpoint_sha256']:
                raise ValueError('Completed training checkpoint changed')
            changes = completion.get('parameter_changes', {})
            verified = None if not changes else all(changes[name]['changed_parameter_tensors'] > 0
                                                       for name in ['flow', 'actor', 'critic'])
            if number > 0 and verified is not True:
                raise ValueError('Online completion lacks verified flow/actor/Q parameter updates')
            current['updates'][arm] = dict(online_training_rows=completion['online_training_rows'],
                all_three_modules_changed=verified, parameter_changes=changes,
                completion_path=str(path), completion_sha256=sha(path),
                checkpoint_sha256=completion['checkpoint_sha256'],
                critic_validation=completion['critic_validation'])
        report['rounds'][str(number)] = current
    atomic_json(folder/'report.json', report)
    if csv_rows:
        with (folder/'per_layout.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(csv_rows[0]))
            writer.writeheader();writer.writerows(csv_rows)
    lines=['# Graph FQL：Q 校准与在线训练核验','',
           '仅统计已经评测的预定样本；没有用这些诊断选择最好噪声或 checkpoint。',
           'reward 是归一化的负 GRT 线长；Q 高估表示预测过于乐观。','',
           '| 轮次 | 方法/用途 | 成功/请求 | rWL（μm） | Q MAE | Q 偏差 | Q 排序相关 | 合法化位移 RMS |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
    def fmt(value):
        return '—' if value is None else f'{value:.5g}'
    for number, stage in report['rounds'].items():
        for group, result in stage.get('groups', {}).items():
            q=result.get('on_policy_Q',{});geometry=result.get('action_geometry',{})
            metrics=[result['rWL_um_mean'],q.get('mae'),q.get('optimism_bias'),q.get('spearman_r'),
                     geometry.get('raw_to_legalized_rms_mean')]
            lines.append(f"| {number} | {group} | {result['successful']}/{result['requested']} | "+
                         ' | '.join(map(fmt,metrics))+' |')
    lines += ['', '四个评估噪声不等于四个训练 seed；当前诊断不能支持跨电路结论。',
              '合法化位移使用归一化坐标，实际后端还另行核验轨道对齐后的宏几何。','']
    (folder/'summary.md').write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(folder/'report.json'), round_status={k:v['status'] for k,v in report['rounds'].items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use Slurm')
    run(args)
