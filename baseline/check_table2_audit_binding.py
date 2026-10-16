"""Bounded check of the actual Table 2 summarizer using copied bp_fe inputs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run this numerical check through Slurm')
    output = ROOT / 'baseline/artifacts/table2_audit_binding_v1'
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'analysis/summarize_table2.py'
    original = ROOT / 'backend/results/released_bp_fe_03'
    bound = [source, original / 'metrics.json', original / 'fullflow_audit.json']
    before = {str(path): sha(path) for path in bound}
    results = []
    for name, tamper in [('unchanged', False), ('changed_drt_wl', True)]:
        fixture = output / name
        (fixture / 'analysis').mkdir(parents=True)
        (fixture / 'backend/assets').mkdir(parents=True)
        (fixture / 'backend/results').mkdir()
        copied_source = fixture / 'analysis/summarize_table2.py'
        shutil.copy2(source, copied_source)
        shutil.copy2(ROOT / 'backend/assets/paper_table2_flowplace.json',
                     fixture / 'backend/assets/paper_table2_flowplace.json')
        for index, design in enumerate(['ariane133', 'ariane136', 'bp', 'bp_be', 'swerv_wrapper']):
            run_id = f'released_{design}_job1783836_{index}'
            (fixture / 'backend/results' / run_id).symlink_to(ROOT / 'backend/results' / run_id,
                                                           target_is_directory=True)
        run = fixture / 'backend/results/released_bp_fe_03'
        run.mkdir()
        for filename in ['manifest.json', 'metrics.json', 'fullflow_audit.json']:
            shutil.copy2(original / filename, run / filename)
        (run / 'orfs_work').symlink_to(original / 'orfs_work', target_is_directory=True)
        if tamper:
            metrics = json.loads((run / 'metrics.json').read_text())
            metrics['DRT_WL'] += 1.0
            (run / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        command = [sys.executable, str(copied_source), '--array-job', '1783836',
                   '--output', str(fixture / 'summary')]
        completed = subprocess.run(command, text=True, capture_output=True)
        (fixture / 'stdout.txt').write_text(completed.stdout)
        (fixture / 'stderr.txt').write_text(completed.stderr)
        report = json.loads((fixture / 'summary/report.json').read_text())
        row = next(row for row in report['designs'] if row['design'] == 'bp_fe')
        if tamper:
            assert row['status'] == 'failed_or_unavailable', row
            assert row['error'] == 'Current DRT_WL differs from the audited metric', row
            assert 'measured' not in row, row
        else:
            assert row['status'] == 'completed', row
        results.append({'case': name, 'command': command, 'exit_code': completed.returncode,
                        'bp_fe_status': row['status'], 'bp_fe_error': row.get('error'),
                        'source_sha256': sha(copied_source), 'report_sha256': sha(fixture / 'summary/report.json')})
    after = {str(path): sha(path) for path in bound}
    assert before == after, 'Original source or bp_fe inputs changed during the check'
    report = {'status': 'passed', 'slurm_job_id': os.environ['SLURM_JOB_ID'],
              'scope': 'Two executions of an unchanged source copy: real copied bp_fe metrics/audit accepted, then only copied DRT_WL increased by 1 and rejected. Other designs are read-only links and may retain their original incomplete outcomes.',
              'original_files_unchanged': before == after, 'source_and_original_hashes': before,
              'cases': results}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
