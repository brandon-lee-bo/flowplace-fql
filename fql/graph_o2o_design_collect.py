"""Audit measured outcomes, build arm-local replay, and summarize fixed samples."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import ROOT, load_requests
from .graph_o2o_design_policy_data import measured_context

METRICS = ['rWL_um', 'WNS_ns', 'TNS_ns', 'Power_W', 'StdCellArea_um2',
           'horizontal_overflow_sum', 'vertical_overflow_sum']


def audit_outcome(request, cfg, scale):
    folder = Path(request['sample_dir'])
    path = folder/'outcome.json'
    out = json.loads(path.read_text())
    if (out['request_sha256'] != sha(folder/'request.json') or
        out['action_sha256'] != request['action_sha256'] or
        out['raw_action'] != request['raw_action'] or
        out['executed_action'] != request['executed_action'] or
        out['checkpoint_sha256'] != request['checkpoint_sha256']):
        raise ValueError('Outcome/action/checkpoint binding changed')
    if (out['reward_scale'] != scale or out['terminal_mask'] != 0 or
            out['graph'] != request['graph'] or out['dataset_binding_sha256'] != request['dataset_binding_sha256']):
        raise ValueError('Reward scale/terminal semantics changed')
    if 'backend_macro_geometry_path' in out:
        audit_path = Path(out['backend_macro_geometry_path'])
        if sha(audit_path) != out['backend_macro_geometry_sha256']:
            raise ValueError('Actual backend geometry evidence changed')
        geometry = json.loads(audit_path.read_text())
        if geometry['legal_no_halo'] != out['backend_macro_legal']:
            raise ValueError('Backend geometry status mismatch')
    if out['status'] == 'success':
        run = Path(out['physical_run'])
        if (sha(run/'manifest.json') != out['physical_manifest_sha256'] or
            sha(run/'metrics.json') != out['metrics_sha256'] or
            json.loads((run/'metrics.json').read_text()) != out['metrics']):
            raise ValueError('Measured PPA provenance changed')
        expected = -out['metrics']['rWL_um']/scale
    elif out['status'] in ['geometry_failure', 'physical_failure']:
        if out['metrics'] is not None or not out['failure_utility_only']:
            raise ValueError('Failure must have no numerical PPA')
        if out['failure_class'] not in ['normalized_geometry', 'rounded_DEF_geometry', 'cts_hold_repair_max_buffer_count',
                                         'backend_macro_overlap_or_bounds']:
            raise ValueError('Unsupported physical failure class')
        expected = cfg['physical_failure_reward']
    else:
        raise ValueError('Unresolved infrastructure/parser failure cannot enter replay')
    if not math.isclose(out['reward'], expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('Reward does not equal the measured outcome/declared failure utility')
    return dict(out, outcome_path=str(path), outcome_sha256=sha(path))


def summary(rows):
    good = [r for r in rows if r['status'] == 'success']
    means = {key: statistics.mean(r['metrics'][key] for r in good) for key in METRICS} if good else None
    std = {key: statistics.stdev(r['metrics'][key] for r in good) if len(good)>1 else 0. for key in METRICS} if good else None
    return dict(count=len(rows), successful=len(good), failed=len(rows)-len(good),
                all_cases_successful=len(rows)==len(good),
                mean=means if len(rows)==len(good) else None,
                sample_std=std if len(rows)==len(good) else None,
                successful_only_mean=means, successful_only_sample_std=std,
                failures=[dict(index=r['index'], status=r['status'], failure_class=r.get('failure_class'))
                          for r in rows if r['status'] != 'success'])


def paired(left, right):
    if [r['index'] for r in left] != [r['index'] for r in right]:
        raise ValueError('Paired evaluation indices differ')
    differences = []
    for a, b in zip(left, right):
        if a['status'] == b['status'] == 'success':
            differences.append(dict(index=a['index'],
                rwl_improvement_percent=100*(a['metrics']['rWL_um']-b['metrics']['rWL_um'])/a['metrics']['rWL_um'],
                reward_prediction_error=None if b['predicted_q'] is None else
                  statistics.mean(b['predicted_q'])-b['reward']))
    return dict(expected_pairs=len(left), successful_pairs=len(differences), pairs=differences,
                mean_rwl_improvement_percent=statistics.mean(r['rwl_improvement_percent'] for r in differences)
                  if len(differences)==len(left) and differences else None)


def collect(experiment, seed, round_number):
    experiment = Path(experiment).resolve()
    cfg = json.loads((experiment/'config.json').read_text())
    cfg, binding, _ = measured_context(experiment)
    manifest = json.loads(Path(binding['manifest_path']).read_text())
    scale = binding['reward_scale']
    all_rows = {}
    for number in range(round_number+1):
        requests, path = load_requests(experiment, seed, number)
        for row in requests:
            if (row['graph'] != cfg['design'] or row['dataset_binding_sha256'] != sha(experiment/'dataset_binding.json')):
                raise ValueError('Replay request crossed its frozen design/data context')
        outcomes = [audit_outcome(r, cfg, scale) for r in requests]
        all_rows[number] = outcomes
    replay = dict(version=1, seed=seed, for_training_round=round_number+1,
                  reward_scale=scale, offline_dataset_sha256=manifest['dataset_sha256'],
                  includes_collection_rounds=list(range(round_number+1)),
                  config_sha256=sha(experiment/'config.json'), arms={})
    for arm in ['bc', 'fql']:
        eligible = [r for number in all_rows for r in all_rows[number]
                    if r['arm']==arm and r['role']=='collect' and r['training_feedback_eligible']]
        if len(eligible) != (round_number+1)*cfg['online_samples']:
            raise ValueError('Incomplete collection budget')
        if len({r['run_id'] for r in eligible}) != len(eligible):
            raise ValueError('Repeated logical candidate in replay')
        replay['arms'][arm] = [dict(raw_action=r['raw_action'], executed_action=r['executed_action'],
            reward=r['reward'], status=r['status'], terminal_mask=0, run_id=r['run_id'],
            collection_round=r['round'], outcome_path=r['outcome_path'], outcome_sha256=r['outcome_sha256'],
            action_sha256=r['action_sha256'], metrics=r['metrics']) for r in eligible]
    replay_path = experiment/f'seed_{seed}'/f'replay_round_{round_number+1}.json'
    if replay_path.exists():
        if json.loads(replay_path.read_text()) != replay:
            raise ValueError('Existing immutable replay differs')
    else:
        atomic_json(replay_path, replay)
    current = all_rows[round_number]
    report = dict(version=1, seed=seed, round=round_number, stage='global_route_only',
                  not_final_table2=True, detailed_route_DRC=None,
                  reward='-rWL / fixed offline TRAIN mean; failures separately counted',
                  groups={}, paired={}, replay_path=str(replay_path), replay_sha256=sha(replay_path),
                  evaluation_replayed=False, heldout_labels_used=False,
                  scope='Single fixed netlist; four predetermined diagnostic noises are not four independent training seeds',
                  outcomes=[dict(path=r['outcome_path'], sha256=r['outcome_sha256']) for r in current])
    evaluation = {}
    for arm in ['bc', 'fql'] + (['flowplace'] if round_number==0 else []):
        for role in ['collect', 'evaluation']:
            rows = [r for r in current if r['arm']==arm and r['role']==role]
            if rows:
                report['groups'][arm+'/'+role] = summary(rows)
                if role=='evaluation':
                    evaluation[arm] = rows
    report['paired']['fql_vs_matched_bc'] = paired(evaluation['bc'], evaluation['fql'])
    original = [r for r in all_rows[0] if r['arm']=='flowplace' and r['role']=='evaluation']
    report['paired']['fql_vs_original_flowplace'] = paired(original, evaluation['fql'])
    if round_number:
        offline = [r for r in all_rows[0] if r['arm']=='fql' and r['role']=='evaluation']
        report['paired']['online_fql_vs_offline_fql'] = paired(offline, evaluation['fql'])
    folder = experiment/f'seed_{seed}'/f'round_{round_number}'
    atomic_json(folder/'summary.json', report)
    lines = [f'# Graph FQL：{cfg["design"]}，seed={seed}，round={round_number}', '',
        '当前是 CTS＋全局布线评估，**不是最终详细布线 PPA，也没有详细布线 DRC 结论**。', '',
        '每组四个预先固定的评估噪声；不选择最佳噪声，评估结果不进入 replay。', '',
        '| 方法 / 样本用途 | 成功 / 总数 | rWL（μm） | WNS（ns） | TNS（ns） | 功耗（W） | 面积（μm²） |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for group, result in report['groups'].items():
        values = result['mean']
        metrics = ' | '.join(f'{values[key]:.6g}' for key in METRICS[:5]) if values else '— | — | — | — | —'
        lines.append(f"| {group} | {result['successful']}/{result['count']} | {metrics} |")
    lines += ['', '发生失败时上表不报告全体均值；成功样本条件均值和全部失败记录保留在 JSON。', '',
              f"已生成下一轮 replay：BC {len(replay['arms']['bc'])} 条，FQL {len(replay['arms']['fql'])} 条；仅来自各自 collect 样本。", '',
              '这些是单电路、单训练 seed 的阶段性证据。仍需要完整布线和更多 seed 验证。', '']
    (folder/'summary.md').write_text('\n'.join(lines))
    print(json.dumps(dict(summary=str(folder/'summary.json'), replay=str(replay_path), paired=report['paired'])), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--round', type=int, default=0)
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run audit/aggregation under Slurm')
    collect(args.experiment, args.seed, args.round)


if __name__ == '__main__':
    main()
