#!/usr/bin/env python3
"""Visualize all twelve audited Table 2 outcomes without selecting runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METRICS = [('rWL_million_um', 'rWL (million um)'), ('WNS_ns', 'WNS (ns)'),
           ('TNS_ns', 'TNS (ns)'), ('Power_W', 'Power (W)'), ('CellArea_um2', 'Cell area (um2)')]
DESIGNS = ['ariane133', 'ariane136', 'bp', 'bp_be', 'bp_fe', 'swerv_wrapper']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True, type=Path)
    parser.add_argument('--summary-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Submit through Slurm'
    assert sha(args.summary) == args.summary_sha256, 'Summary identity changed'
    report = json.loads(args.summary.read_text())
    assert report['validated_count'] == 12 and report['comparison_ready'] is True
    assert report['primary_comparison_ready'] is False and report['same_shared_backend'] is True
    for row in report['rows']:
        task = json.loads((Path(row['run']) / 'manifest.json').read_text())['task']
        assert task['diagnostic'] is True and task['not_primary_table2_result'] is True
        assert task['macro_import_policy'] == 'preserve_generator_dbu_coordinates_v1'
        assert task['recipe_label'] == 'released_flow_exact_macro_import_v1'
    rows = {(r['method'], r['design']): r for r in report['rows']}
    assert len(rows) == 12
    ordered = [rows[(method, design)] for method in ('FlowPlace', 'DREAMPlace') for design in DESIGNS]
    for row in ordered:
        assert row['status'] == 'validated'
        run = Path(row['run'])
        assert sha(run / 'metrics.json') == row['metrics_sha256']
        assert sha(run / 'fullflow_audit.json') == row['fullflow_audit_sha256']
        for relative, value in row['source_report_sha256'].items():
            assert sha(run / relative) == value, 'Audited raw report changed'
    output = args.output.resolve()
    assert output.is_relative_to(Path('/fact_data/bohanli/flowplace_fql_backend/table2_reports'))
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.summary, output / 'source_summary.json')
    shutil.copyfile(__file__, output / Path(__file__).name)
    measured = np.array([[row['measured'][key] for key, _ in METRICS] for row in ordered])
    paper = np.array([[row['paper_rounded'][key] for key, _ in METRICS] for row in ordered])
    assert np.isfinite(measured).all() and np.isfinite(paper).all() and (paper != 0).all()
    errors = 100 * np.abs(measured - paper) / np.abs(paper)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(9.5, 8))
    image = ax.imshow(np.minimum(errors, 50), cmap='YlOrRd', vmin=0, vmax=50, aspect='auto')
    ax.set_xticks(range(5), [label for _, label in METRICS])
    ax.set_yticks(range(12), [r['method'] + ' / ' + r['design'] for r in ordered])
    ax.axhline(5.5, color='white', linewidth=3)
    for i in range(12):
        for j in range(5):
            ax.text(j, i, f'{errors[i, j]:.1f}%', ha='center', va='center',
                    color='white' if errors[i, j] >= 32 else '#222222')
    fig.colorbar(image, ax=ax, label='Absolute difference / |published value| (%)', extend='max')
    ax.set_title('Exact-import compatibility cohort vs Table 2; all 12 outcomes', pad=15)
    fig.text(.5, .035, 'Color saturates at 50%; labels show full errors. Timing ratios can amplify small published slack.',
             ha='center', fontsize=8)
    fig.text(.5, .015, 'Reproduction error is not algorithm quality. Fullflow completion does not imply zero DRC.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    for suffix in ('png', 'pdf'):
        fig.savefig(output / ('paper_distance.' + suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for index, (key, label) in enumerate(METRICS):
        ax = axes.flat[index]
        for method, color, marker in [('FlowPlace', '#2467a4', 'o'), ('DREAMPlace', '#c46924', '^')]:
            members = [r for r in ordered if r['method'] == method]
            x = np.array([r['paper_rounded'][key] for r in members])
            y = np.array([r['measured'][key] for r in members])
            ax.scatter(x, y, label=method, color=color, marker=marker, alpha=.8)
        lower = min(paper[:, index].min(), measured[:, index].min())
        upper = max(paper[:, index].max(), measured[:, index].max())
        pad = max((upper - lower) * .05, 1e-9)
        ax.plot([lower - pad, upper + pad], [lower - pad, upper + pad], '--', color='#888888', linewidth=1)
        ax.set(xlabel='Published', ylabel='Reproduced', title=label)
        ax.ticklabel_format(style='sci', axis='both', scilimits=(-3, 4))
        ax.grid(alpha=.15)
    axes.flat[5].axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[5].legend(handles, labels, loc='center', frameon=False)
    fig.suptitle('Exact-import compatibility cohort: routed metrics vs paper', fontsize=13)
    fig.text(.5, .015, 'Uniform coordinate-preserving import; same backend for both methods. Exact author binary, seed history and DMP baseline settings are unavailable.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .96))
    for suffix in ('png', 'pdf'):
        fig.savefig(output / ('paper_scatter.' + suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)
    assert sha(args.summary) == args.summary_sha256
    manifest = {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'summary_sha256': args.summary_sha256,
                'source_summary': str(args.summary.resolve()), 'rows': 12, 'metrics_per_row': 5,
                'runner_sha256': sha(__file__), 'all_route_drc_clean': report['all_route_drc_clean'],
                'output_sha256': {p.name: sha(p) for p in output.iterdir() if p.is_file()},
                'notes': ['Separate uniform exact-import compatibility cohort; never substitute for original-flow rows.',
                          'No run selection, interpolation, timing rescaling, or new PPA calculation.',
                          'Paper-distance colors saturate at 50%; numerical annotations are not clipped.']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'rows': 12, 'success': True}))


if __name__ == '__main__':
    main()
