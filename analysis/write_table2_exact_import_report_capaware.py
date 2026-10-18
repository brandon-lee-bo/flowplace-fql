#!/usr/bin/env python3
"""Write a Chinese completion report only from a fully audited 12-row cohort."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics

METRICS = [('rWL_million_um', 'rWL (百万 µm)', 6), ('WNS_ns', 'WNS (ns)', 6),
           ('TNS_ns', 'TNS (ns)', 4), ('Power_W', '功耗 (W)', 6),
           ('CellArea_um2', '标准单元面积 (µm²)', 1)]
DESIGNS = ['ariane133', 'ariane136', 'bp', 'bp_be', 'bp_fe', 'swerv_wrapper']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('summary', 'macro', 'connectivity', 'plots'):
        p.add_argument('--' + name, required=True, type=Path)
        p.add_argument('--' + name + '-sha256', required=True)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Submit through Slurm'
    sources = {}
    for name in ('summary', 'macro', 'connectivity', 'plots'):
        path, expected = getattr(args, name), getattr(args, name + '_sha256')
        assert sha(path) == expected, 'Evidence changed: ' + name
        sources[name] = {'path': str(path.resolve()), 'sha256': expected}
    report, macro, connectivity, plots = [json.loads(getattr(args, n).read_text())
                                         for n in ('summary', 'macro', 'connectivity', 'plots')]
    assert report['validated_count'] == report['requested_count'] == 12
    assert all(report[k] is True for k in ('comparison_ready', 'same_shared_backend'))
    assert report['primary_comparison_ready'] is False
    assert macro['all_macros_preserved'] is True and macro['preserved_count'] == 12
    assert macro['complete'] is True and macro['checked'] == macro['requested'] == 12
    assert macro['summary_sha256'] == plots['summary_sha256'] == args.summary_sha256
    assert macro['all_checked_final_macros_legal'] is True
    assert macro['all_checked_final_match_native_macroplacement'] is True
    assert connectivity['success'] is True and connectivity['validated_count'] == 6
    assert plots['rows'] == 12
    for name, expected in plots['output_sha256'].items():
        assert sha(args.plots.parent / name) == expected, 'Plot artifact changed'
    for row in connectivity['rows']:
        assert sha(row['summary']) == row['summary_sha256']
    rows = {(r['method'], r['design']): r for r in report['rows']}
    macros = {(r['method'], r['design']): r for r in macro['rows']}
    assert len(rows) == len(macros) == 12
    ordered = [rows[(m, d)] for m in ('FlowPlace', 'DREAMPlace') for d in DESIGNS]
    for row in ordered:
        assert row['status'] == 'validated'
        task = json.loads((Path(row['run']) / 'manifest.json').read_text())['task']
        assert task['diagnostic'] is True and task['not_primary_table2_result'] is True
        assert task['macro_import_policy'] == 'preserve_generator_dbu_coordinates_v1'
        assert task['recipe_label'] == 'released_flow_exact_macro_import_v1'
        run = Path(row['run'])
        for file, key in [('manifest.json', 'manifest_sha256'), ('metrics.json', 'metrics_sha256'),
                          ('fullflow_audit.json', 'fullflow_audit_sha256')]:
            assert sha(run / file) == row[key], 'Validated evidence changed'
        for file, expected in row['source_report_sha256'].items():
            assert sha(run / file) == expected
        m = macros[(row['method'], row['design'])]
        assert m['run_id'] == row['run_id'] and m['input_def_sha256'] == row['def_sha256']
        assert m['native_macroplacement']['final_matches_native'] is True
        assert m['final_macro_geometry']['legal_no_halo'] is True
        assert all(math.isfinite(row['measured'][k]) and row['paper_rounded'][k] != 0 for k, _, _ in METRICS)
    output = args.output.resolve()
    assert output.is_relative_to(Path('/fact_data/bohanli/flowplace_fql_backend/table2_reports'))
    output.mkdir(parents=True, exist_ok=False)
    for name, item in sources.items():
        shutil.copyfile(item['path'], output / ('source_' + name + '.json'))
    shutil.copyfile(__file__, output / Path(__file__).name)
    lines = ['# FlowPlace Table 2：统一精确宏坐标导入对照组', '',
             '12/12 组已完成带统一宏导入兼容修正的 OpenROAD fullflow，并通过原始报告、输入与工具哈希审计。'
             '两方法使用同一后端，均保留生成器原始整数 DBU 坐标。此为独立兼容性对照组，不能冒充未经修改的作者流程或替换其单行；可作为明确标注的后续算法比较 baseline。'
             '**不等于逐项复现论文原始数字，也不等于 DRC 签核通过。**', '',
             '## 结果与论文逐项对照', '', '每格为“本次结果 / 论文四舍五入值”。WNS、TNS 为负，越接近 0 越好；'
             '其他三项越小越好。DRC 为本次详细布线剩余违例数。', '',
             '| 方法 | 电路 | rWL (百万 µm) | WNS (ns) | TNS (ns) | 功耗 (W) | 标准单元面积 (µm²) | DRC |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in ordered:
        cells = [f'{row["measured"][k]:.{n}f} / {row["paper_rounded"][k]:g}' for k, _, n in METRICS]
        lines.append('| ' + ' | '.join([row['method'], row['design']] + cells + [str(row['DRC'])]) + ' |')
    reconciled = [r for r in ordered if r.get('posthoc_drc_cap_reconciliation')]
    lines += ['', '## 事后核对：DRC 导出条目上限', '',
              '原运行的 manifest、Slurm 状态和 fullflow_audit 保持原样。部分运行物理流程完成，'
              '但原核对错误地假设 DRC 文本报告没有导出上限。单独核对冻结版本源码、完整原生路由计数、'
              '逐类别计数及每类 10,000 条导出上限后，才纳入这些行。所有 PPA 和完整 DRC 数值保持不变。', '',
              '| 方法 | 电路 | 原运行状态 | 完整 DRC | 导出条目 |', '| --- | --- | --- | ---: | ---: |']
    for row in reconciled:
        e = row['posthoc_drc_cap_reconciliation']
        lines.append('| ' + ' | '.join([row['method'], row['design'], row['original_run_status'],
                     str(e['full_drc_count']), str(sum(e['exported_type_counts'].values()))]) + ' |')
    if not reconciled:
        lines.append('本组没有需要该事后核对的行。')
    lines += ['', '## 与论文的数值距离', '',
              '相对偏差为 `100 × |本次 − 论文| / |论文|`，每个方法均包含六个电路，'
              '没有按成功子集或优选 seed 计算。这里衡量复现距离，不衡量算法优劣。'
              '接近 0 的论文 slack 会放大相对偏差，因此同时报告绝对误差。', '',
              '| 方法 | 指标 | 平均绝对相对偏差 | 最大绝对相对偏差 | 平均绝对误差（该指标单位） |',
              '| --- | --- | ---: | ---: | ---: |']
    errors = {}
    for method in ('FlowPlace', 'DREAMPlace'):
        members = [r for r in ordered if r['method'] == method]
        assert len(members) == 6
        errors[method] = {}
        for k, label, _ in METRICS:
            absolute = [abs(r['measured'][k] - r['paper_rounded'][k]) for r in members]
            relative = [100 * e / abs(r['paper_rounded'][k]) for r, e in zip(members, absolute)]
            values = {'count': 6, 'mean_absolute_relative_percent': statistics.mean(relative),
                      'max_absolute_relative_percent': max(relative), 'mean_absolute_error': statistics.mean(absolute)}
            errors[method][k] = values
            lines.append(f'| {method} | {label} | {statistics.mean(relative):.2f}% | {max(relative):.2f}% | {statistics.mean(absolute):.6g} |')
    lines += ['', '## 实际使用的流程', '',
              '- FlowPlace：发布 checkpoint，全局 seed300 随机流，20 步、uniform prior、原 hard-constraint sampler；六个生成布局都通过独立几何检查。',
              '- DREAMPlace：public 4.1.0 BB/两阶段、seed1000、完整公开合法化，六设计同一 GPU 安装。保留一次已验证并冻结的 Abacus 浮点行索引修复。',
              '- 后端：作者发布 flow、Nangate45 库、缓存综合网表与 SDC；OpenROAD `v2.0-17598-ga008522d8`、8 线程、发布 GPL0/0、详细布线 cap4。两方法共同从宏位置重做标准单元放置和后续实现。',
              '- 保留原生 place_macro 命令后，统一以 OpenDB 恢复全部生成器坐标及原生 LOCKED 状态，避免轨道吸附引入重叠；没有搜索或依据 PPA 修补位置。'
              f'完整 {macro["macro_count_checked"]} 个最终宏均无重叠、无 core 越界，并逐 DBU 对应生成器原始坐标、方向及恢复后的 OpenDB 日志。',
              '- 六电路的发布 DEF 与缓存综合网表已逐实例/master 和信号网端点分组核对。此检查证明结构连接一致，不是功能形式等价证明。', '',
              '## 仍然存在的限制（本组并非原样复现）', '',
              '1. 额外使用统一宏导入兼容修正，原流程结果独立保留；该差异不由本次 PPA 数值决定。作者准确 OpenROAD 二进制、原始生成 RNG 历史及样本选择规则未公开；本次工具版本按发布 flow 的源码时期选择，未按 PPA 选版本。',
              '2. 作者独立 DREAMPlace baseline 的完整配置、初始化和 strong-legalizer 对应实现尚未匹配。public4.1 重构不能宣称与论文原 baseline 逐项相同。',
              '3. 论文描述 timing/congestion-driven GPL，但发布配置为 0/0。主表保留发布设置；工具×GPL 的固定布局诊断另列，不能替换更接近论文的主表行。',
              '4. 原生 GPU 非确定性与 hard projection 可能放大同 seed 的位置差异。没有据 PPA 挑选或混用不同设计的生成流。',
              '5. 发布 cap4 流程仍有详细布线 DRC；宏合法与全芯片可制造性不同。保留实际 DRC 及 PDN 等日志，不能称为签核洁净。',
              '6. FlowPlace 与 DREAMPlace 生成使用不同 GPU；本报告不作生成耗时加速比，也不借用其他三个方法的论文数字重算五方法排名。', '',
              '## 可追溯结果', '',
              f'- 原始完整汇总：[{args.summary.name}]({args.summary.resolve()})',
              f'- 最终宏审计：[{args.macro.name}]({args.macro.resolve()})',
              f'- 连接审计：[{args.connectivity.name}]({args.connectivity.resolve()})',
              f'- [论文距离图]({args.plots.parent / "paper_distance.png"}) / [原单位散点图]({args.plots.parent / "paper_scatter.png"})', '',
              '| 方法 | 电路 | Slurm task | 原始结果目录 |', '| --- | --- | --- | --- |']
    for row in ordered:
        task = row['slurm_job_id'] + '_' + str(row['task_index'])
        lines.append(f'| {row["method"]} | {row["design"]} | {task} | [{row["run_id"]}]({row["run"]}) |')
    lines += ['', '被抢占的旧尝试、同输入基础设施重试绑定及 scheduler 耗时均保留在原始汇总；'
              '压缩归档的中间数据库可恢复，失败尝试不会因归档被改记为成功。', '']
    (output / 'report.md').write_text('\n'.join(lines))
    (output / 'numerical_distance.json').write_text(json.dumps(errors, indent=2) + '\n')
    for name, item in sources.items():
        assert sha(item['path']) == item['sha256'], 'Evidence changed during report generation'
    manifest = {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'runner_sha256': sha(__file__),
                'sources': sources, 'requested_count': 12, 'validated_count': 12,
                'posthoc_drc_cap_reconciled_count': len(reconciled),
                'original_failure_records_preserved': True,
                'macro_count_checked': macro['macro_count_checked'],
                'all_route_drc_clean': report['all_route_drc_clean'],
                'output_sha256': {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'validated_count': 12}))


if __name__ == '__main__':
    main()
