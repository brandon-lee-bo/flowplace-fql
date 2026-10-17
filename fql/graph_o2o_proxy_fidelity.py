"""Audit whether measured GRT rewards agree with actual final routed quality.

This is a read-only diagnostic on every task in frozen existing comparisons.
It neither adds labels to replay nor selects policies/layouts from final PPA.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics

import numpy as np

from .core import average_rank, correlation
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_final_evidence import audit_final_outcome, LEGACY_REPEAT_PLAN


def associations(rows):
    if not rows:
        return dict(count=0)
    grt = np.asarray([r['GRT_rWL_um'] for r in rows])
    drt = np.asarray([r['DRT_rWL_um'] for r in rows])
    return dict(count=len(rows), Pearson_r=correlation(grt, drt),
        Spearman_r=correlation(average_rank(grt), average_rank(drt)),
        median_DRT_over_GRT=statistics.median(drt / grt),
        WNS_Pearson_r=correlation([r['GRT_WNS_ns'] for r in rows], [r['DRT_WNS_ns'] for r in rows]),
        TNS_Pearson_r=correlation([r['GRT_TNS_ns'] for r in rows], [r['DRT_TNS_ns'] for r in rows]))


def run(args):
    path = Path(args.plan).resolve()
    plan = json.loads(path.read_text())
    for name, digest in plan['source_sha256'].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError('Diagnostic source changed: ' + name)
    for item in plan['input_plans']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('A predeclared physical comparison changed')
    key = os.environ['SLURM_JOB_ID'] + '_restart_' + os.environ.get('SLURM_RESTART_COUNT', '0')
    output = Path(plan['output']) / ('job_' + key)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / Path(__file__).name)
    rows, actual, physical = [], [], None
    for task in plan['tasks']:
        outcome_path = Path(task['outcome'])
        if not outcome_path.exists():
            rows.append(dict(task, status='not_yet_accounted', metrics=None))
            continue
        out = json.loads(outcome_path.read_text())
        entry = dict(task, status=out['status'], outcome_sha256=sha(outcome_path))
        if out['status'] == 'upstream_physical_failure':
            source = Path(out['source_outcome'])
            original = json.loads(source.read_text())
            if (sha(source) != out['source_outcome_sha256'] or out['metrics'] is not None or
                    original['status'] not in ['geometry_failure', 'physical_failure'] or
                    original['metrics'] is not None or original['failure_class'] != out['failure_class']):
                raise ValueError('A final failure lacks its actual upstream physical evidence')
            rows.append(dict(entry, failure_class=out['failure_class'], metrics=None))
            continue
        checked = audit_final_outcome(out, design='bp_fe', route_cap=20, legacy_plan=LEGACY_REPEAT_PLAN)
        identity = checked['physical_identity']
        if physical is None:
            physical = identity
        if identity != physical:
            raise ValueError('This analysis requires the same design and physical settings')
        g, d = checked['source_metrics'], out['metrics']
        source = Path(checked['source_run'])
        geometry = json.loads((source / 'backend_macro_audit.json').read_text())
        macro_key = dict(macros=geometry['actual_macros'], dbu_per_um=geometry['dbu_per_um'],
                         core_dbu=geometry['core_dbu'], physical_identity=identity)
        semantic_key = hashlib.sha256(json.dumps(macro_key, sort_keys=True).encode()).hexdigest()
        values = dict(GRT_rWL_um=g['rWL_um'], DRT_rWL_um=d['DRT_WL'],
            GRT_WNS_ns=g['WNS_ns'], DRT_WNS_ns=d['WNS'], GRT_TNS_ns=g['TNS_ns'], DRT_TNS_ns=d['TNS'],
            GRT_power_W=g['Power_W'], DRT_power_W=d['Power'], GRT_area_um2=g['StdCellArea_um2'],
            DRT_area_um2=d['StdCellArea'], DRT_DRC=d['DRC'])
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError('Completed physical pair lacks finite metrics')
        record = dict(entry, **values, macro_geometry_identity=semantic_key,
            physical_run=str(Path(out['physical_run']).resolve()),
            final_def_sha256=checked['final_def_sha256'],
            final_macro_count=checked['final_macro_count'],
            legacy_exact_repeat_evidence=checked['legacy_exact_repeat_evidence'])
        rows.append(record); actual.append(record)
    missing = sum(row['status'] == 'not_yet_accounted' for row in rows)
    if args.require_complete and missing:
        raise ValueError('Final diagnostic still lacks ' + str(missing) + ' predeclared outcomes')
    policy = [row for row in actual if row['policy_evaluation']]
    unique = []
    for key in dict.fromkeys(row['macro_geometry_identity'] for row in policy):
        duplicates = [row for row in policy if row['macro_geometry_identity'] == key]
        unique.append(dict(macro_geometry_identity=key, measurements=len(duplicates),
            **{name: statistics.mean(row[name] for row in duplicates) for name in
               ['GRT_rWL_um', 'DRT_rWL_um', 'GRT_WNS_ns', 'DRT_WNS_ns', 'GRT_TNS_ns', 'DRT_TNS_ns']}))
    cohorts = {}
    for cohort in dict.fromkeys(task['cohort'] for task in plan['tasks']):
        selected = [row for row in rows if row['cohort'] == cohort]
        valid = [row for row in selected if row['status'] == 'completed' and row['policy_evaluation']]
        cohorts[cohort] = dict(requested=len(selected), statuses=dict(collections.Counter(r['status'] for r in selected)),
            policy_evaluations=associations(valid), complete=all(r['status'] != 'not_yet_accounted' for r in selected))
    index = {(row['cohort'], row['group'], row['noise_index']): row for row in actual}
    pairs = []
    for comparison in plan['paired_comparisons']:
        for noise in range(4):
            a = index.get((comparison['cohort'], comparison['left'], noise))
            b = index.get((comparison['cohort'], comparison['right'], noise))
            if a is None or b is None:
                pairs.append(dict(comparison, noise_index=noise, status='not_a_completed_pair'))
                continue
            grt = 100 * (a['GRT_rWL_um'] - b['GRT_rWL_um']) / a['GRT_rWL_um']
            drt = 100 * (a['DRT_rWL_um'] - b['DRT_rWL_um']) / a['DRT_rWL_um']
            pairs.append(dict(comparison, noise_index=noise, status='completed',
                GRT_rWL_improvement_percent=grt, DRT_rWL_improvement_percent=drt,
                improvement_direction_agrees=bool(np.sign(grt) == np.sign(drt))))
    pair_done = [row for row in pairs if row['status'] == 'completed']
    report = dict(status='all_requested_outcomes_accounted' if not missing else 'partial_read_only_snapshot',
        plan_sha256=sha(path), source_sha256=plan['source_sha256'], requested=len(rows),
        statuses=dict(collections.Counter(row['status'] for row in rows)), cohorts=cohorts,
        unique_macro_policy_evaluations=associations(unique), unique_policy_layouts=len(unique),
        observed_policy_evaluation_measurements=len(policy), repeated_layouts_averaged_before_correlation=True,
        paired_comparisons_requested=len(pairs), completed_pairs=len(pair_done),
        pairs_with_agreeing_direction=sum(row['improvement_direction_agrees'] for row in pair_done),
        common_physical_identity=physical, measurements=rows, paired_comparisons=pairs,
        no_new_training_or_EDA=True, no_replay_or_policy_selection=True,
        scope='Descriptive fixed-design diagnostics; paired noises and repeated frozen baselines are not independent training seeds',
        goal_completed=False, slurm_job_id=os.environ['SLURM_JOB_ID'])
    atomic_json(output / 'evidence.json', report)
    for filename, data in [('paired_comparisons.csv', pairs), ('per_layout.csv', rows)]:
        fields = list(dict.fromkeys(key for row in data for key in row))
        with (output / filename).open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(data)
    lines = ['# GRT 奖励与最终详细布线是否一致', '',
        '逐项核验真实源布局、工具指纹、原生布线日志和最终 DEF；所有请求及未完成项保留。',
        '这是诊断，不用于 replay、选模型或根据最终 PPA 挑布局；多个噪声不等于独立训练 seed。', '',
        '| 实验 | 已完成/请求 | 策略评估布局数 | GRT–DRT 线长 Pearson | Spearman | DRT/GRT 中位数 |',
        '| --- | --- | --- | --- | --- | --- |']
    for name, cohort in cohorts.items():
        stat = cohort['policy_evaluations']
        values = [stat.get(k) for k in ['Pearson_r', 'Spearman_r', 'median_DRT_over_GRT']]
        cells = ' | '.join('—' if v is None else f'{v:.5f}' for v in values)
        lines.append(f"| {name} | {cohort['statuses'].get('completed', 0)}/{cohort['requested']} | {stat['count']} | {cells} |")
    u = report['unique_macro_policy_evaluations']
    lines += ['', f"按实际宏坐标去重后，共{len(unique)}个策略评估布局，原始测量{len(policy)}次。",
        '同一宏布局的多次测量先取均值，再计算描述性相关；已选出的 TRAIN 参考不混入策略相关。',
        f"BC/FQL 配对对照已完成{len(pair_done)}/{len(pairs)}对，其中{report['pairs_with_agreeing_direction']}对在 GRT 与 DRT 上的改善方向一致。",
        '尚未完成的任务不能视为物理失败；非零 DRC 原样报告。线长相关不代表时序、功耗或 DRC 同时改善。', '']
    (output / 'summary.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for name in cohorts:
        selected = [row for row in policy if row['cohort'] == name]
        if selected:
            ax.scatter([r['GRT_rWL_um']/1e6 for r in selected], [r['DRT_rWL_um']/1e6 for r in selected],
                label=name, s=25, alpha=.7)
    ax.set_xlabel('Global-route wirelength (m)'); ax.set_ylabel('Detailed-route wirelength (m)')
    ax.set_title('Same physical flow / cap20; descriptive layout pairs'); ax.grid(alpha=.2)
    if policy: ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ['png', 'pdf']: fig.savefig(output / ('grt_to_drt.' + ext), dpi=180)
    plt.close(fig)
    print(json.dumps(dict(output=str(output), statuses=report['statuses'], unique_policy=u,
        complete_pairs=len(pair_done), direction_agrees=report['pairs_with_agreeing_direction'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--require-complete', action='store_true')
    args = p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run physical evidence analysis through Slurm')
    run(args)
