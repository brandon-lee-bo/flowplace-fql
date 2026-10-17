"""Audit and summarize saved Q predictions on the same measured proposals.

No checkpoint loading, fitting, candidate selection or new physical work.
Identical BC requests across label budgets are counted once in the pooled
diagnostic; the originally declared four-case primary cohort stays unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
from scipy.stats import spearmanr

from fql.graph_o2o_archive import atomic_json, sha
from fql.graph_o2o_collect import audit_outcome


def measure(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    error = prediction - target
    rank = (float(spearmanr(prediction, target).statistic)
            if np.std(prediction) > 0 and np.std(target) > 0 else None)
    return dict(count=len(target), mae=float(np.abs(error).mean()),
                rmse=float(np.sqrt(np.square(error).mean())),
                optimism_bias=float(error.mean()), spearman_r=rank)


def run(plan_path):
    plan = json.loads(plan_path.read_text())
    if sha(__file__) != plan['implementation_sha256']:
        raise ValueError('Transfer summary implementation changed')
    root = Path(__file__).resolve().parents[1]
    for name, digest in plan['dependency_sha256'].items():
        if sha(root / name) != digest:
            raise ValueError('Transfer audit dependency changed: ' + name)
    for item in plan['inputs'].values():
        if sha(item['path']) != item['sha256']:
            raise ValueError('Saved fixed-cohort evidence changed')
    source = json.loads(Path(plan['inputs']['report']['path']).read_text())
    declared = json.loads(Path(plan['inputs']['cohort_plan']['path']).read_text())
    if (source['status'] != 'fixed_cohort_critics_audited'
            or source['plan_sha256'] != plan['inputs']['cohort_plan']['sha256']
            or not source['no_fit_or_policy_selection']
            or source['evaluation_feedback_replayed']):
        raise ValueError('Wrong diagnostic or contaminated fixed evaluation')
    primary = declared['primary_display_cohort']
    if len(source['fits']) != 12 or len(source['cohort_outcomes']) != 13:
        raise ValueError('Declared critic/cohort denominator changed')
    configs = {}
    for name, experiment_root in declared['experiments'].items():
        path = Path(experiment_root) / 'config.json'
        if sha(path) != declared['config_sha256'][name]:
            raise ValueError('Critic experiment configuration changed')
        configs[name] = json.loads(path.read_text())
    output = Path(plan['output']) / ('job_' + os.environ['SLURM_JOB_ID'])
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / 'inputs'
    inputs.mkdir()
    copies = {}

    def retain(path):
        path = Path(path)
        key = sha(path)
        target = inputs / (key + '_' + path.name)
        if not target.exists():
            shutil.copyfile(path, target)
        if sha(target) != key or sha(path) != key:
            raise ValueError('Small evidence copy changed')
        copies[str(path)] = dict(copy=str(target), sha256=key)

    cases, proposals = {}, {}
    for cohort, bindings in source['cohort_outcomes'].items():
        if len(bindings) != 4:
            raise ValueError('Lost fixed evaluation cases')
        for index, binding in enumerate(bindings):
            path = Path(binding['path'])
            if sha(path) != binding['sha256']:
                raise ValueError('Bound measured outcome changed')
            request_path = path.parent / 'request.json'
            request = json.loads(request_path.read_text())
            outcome = audit_outcome(request, configs[cohort.split('/')[0]],
                                    json.loads(path.read_text())['reward_scale'])
            if (outcome['index'] != index or outcome['role'] != 'evaluation'
                    or outcome['training_feedback_eligible'] or outcome['status'] != 'success'):
                raise ValueError('Expected complete never-replayed measured cohort')
            identity = json.dumps(outcome['raw_action'], separators=(',', ':'))
            signature = (outcome['executed_action'], request['def_sha256'], outcome['reward'])
            if identity in proposals and proposals[identity] != signature:
                raise ValueError('Repeated raw proposal has different execution or target')
            proposals[identity] = signature
            cases[(cohort, index)] = dict(proposal=identity, outcome=outcome)
            for small in [path, request_path]:
                retain(small)
    if len(cases) != 52:
        raise ValueError('Logical evaluation denominator changed')
    with Path(plan['inputs']['predictions']['path']).open() as handle:
        rows = list(csv.DictReader(handle))
    values = {}
    for row in rows:
        key = row['fitted_critic'], row['cohort'], int(row['index'])
        if key in values or key[0] not in source['fits'] or key[1:] not in cases:
            raise ValueError('Repeated or unknown saved prediction')
        measured = cases[key[1:]]['outcome']
        if (row['outcome_sha256'] != measured['outcome_sha256']
                or row['outcome_path'] != measured['outcome_path']
                or row['status'] != measured['status']
                or float(row['target']) != measured['reward']):
            raise ValueError('Prediction does not bind the same measured proposal')
        values[key] = float(row['prediction'])
        if not math.isfinite(values[key]):
            raise ValueError('Nonfinite saved Q prediction')
    if len(values) != 12 * 52:
        raise ValueError('Incomplete critic by proposal comparison')
    pooled, per_unique = {}, {}
    for fitted, record in source['fits'].items():
        if set(record['scores']) != set(source['cohort_outcomes']):
            raise ValueError('A fitted critic lost a declared cohort')
        for cohort, saved in record['scores'].items():
            computed = measure([values[fitted, cohort, i] for i in range(4)],
                               [cases[cohort, i]['outcome']['reward'] for i in range(4)])
            for name in ['mae', 'rmse', 'optimism_bias', 'spearman_r']:
                same = (computed[name] == saved[name] if computed[name] is None or saved[name] is None
                        else math.isclose(computed[name], saved[name], abs_tol=2e-6, rel_tol=2e-5))
                if not same:
                    raise ValueError('Saved per-cohort metric does not reproduce: ' + name)
        unique = {}
        for (cohort, index), case in cases.items():
            prediction = values[fitted, cohort, index]
            old = unique.setdefault(case['proposal'], (prediction, case['outcome']['reward']))
            if not math.isclose(old[0], prediction, rel_tol=0, abs_tol=1e-6):
                raise ValueError('Identical proposal has different predictions from the same Q')
        # Use the first identical case, never the best duplicate outcome.
        per_unique[fitted] = list(unique.values())
        pooled[fitted] = measure(*zip(*unique.values()))
    left, right = 'original64/fql/r2', 'expanded384/fql/r2'
    comparisons = []
    for cohort in source['cohort_outcomes']:
        a, b = source['fits'][left]['scores'][cohort], source['fits'][right]['scores'][cohort]
        comparisons.append(dict(cohort=cohort, original_mae=a['mae'], expanded_mae=b['mae'],
                                original_spearman=a['spearman_r'], expanded_spearman=b['spearman_r']))
    for item in plan['inputs'].values():
        retain(item['path'])
    retain(plan_path)
    retain(__file__)
    for name in plan['dependency_sha256']:
        retain(root / name)
    report = dict(status='same_proposal_critic_transfer_recomputed_without_duplicate_BC',
        plan_sha256=sha(plan_path), declared_primary_cohort=primary,
        logical_cases=len(cases), unique_raw_proposals=len(proposals),
        pooled_unique_metrics=pooled, final_critic_cohort_comparison=comparisons,
        primary_metrics={key: value['scores'][primary] for key, value in source['fits'].items()},
        input_copies=copies, source_report_sha256=plan['inputs']['report']['sha256'],
        scope='Single bp_fe training seed, observed fixed diagnostic proposals; no fitting, selection, new EDA, extra independent seeds or policy-quality proof.',
        slurm_job_id=os.environ['SLURM_JOB_ID'], goal_completed=False)
    atomic_json(output / 'evidence.json', report)
    lines = ['# 同一批布局上的 Q 诊断', '',
             f'13组共52条逻辑评估观测，按原始提案去重后为{len(proposals)}条；重复 BC 不当作额外样本或 seed。',
             '所有观测均未进入 replay。逐组指标已从624条保存预测和绑定的真实后端奖励重算。', '',
             '| Q | 去重后 MAE | 去重后排序相关 | 原定四布局 MAE | 原定四布局排序相关 |',
             '| --- | --- | --- | --- | --- |']
    for name in declared['experiments']:
        for number in range(3):
            key = f'{name}/fql/r{number}'
            a, b = pooled[key], source['fits'][key]['scores'][primary]
            lines.append(f"| {key} | {a['mae']:.6f} | {a['spearman_r']:.4f} | {b['mae']:.6f} | {b['spearman_r']:.4f} |")
    lines += ['', f'原定主展示组保持 `{primary}`，未按结果更换。',
              '四条样本的排序相关较粗；去重汇总仍是同一训练 seed 的机制诊断，不代表新的独立验证集或最终 PPA。', '']
    (output / 'summary.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharex=True, sharey=True)
    all_values = [value for key in [left, right] for pair in per_unique[key] for value in pair]
    limits = (min(all_values) - .015, max(all_values) + .015)
    for ax, key in zip(axes, [left, right]):
        prediction, target = zip(*per_unique[key])
        ax.scatter(target, prediction, s=22, alpha=.75)
        ax.plot(limits, limits, '--', color='gray', linewidth=1)
        ax.set(xlim=limits, ylim=limits, xlabel='Measured GRT reward',
               title=f"{key}\nMAE={pooled[key]['mae']:.4f}; rank={pooled[key]['spearman_r']:.3f}")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Predicted Q')
    fig.suptitle(f'Same {len(proposals)} unique proposals; no fitting or selection')
    fig.tight_layout()
    for suffix in ['png', 'pdf']:
        fig.savefig(output / ('fixed_critic_transfer.' + suffix), dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(dict(output=str(output), status=report['status'],
                          logical_cases=len(cases), unique_raw_proposals=len(proposals))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use Slurm for report auditing and scientific plots')
    run(args.plan.resolve())
