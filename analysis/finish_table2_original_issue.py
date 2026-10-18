#!/usr/bin/env python3
"""Preserve original-flow geometry failure while publishing its complete PPA table."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'backend/plans/table2_background_v1'
REPORTS = Path('/fact_data/bohanli/flowplace_fql_backend/table2_reports')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert os.environ.get('SLURM_JOB_ID')
    os.chdir(ROOT)
    previous = json.loads((BASE / 'postprocess_v1.json').read_text())
    assert previous['status'] == 'failed', 'Original strict postprocess did not fail as expected'
    job = previous['children']['macro']['job']
    raw = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', job,
                                   '-o', 'JobID%64,State,ExitCode'], text=True)
    exact = [r.split('|') for r in raw.splitlines() if r.split('|')[0] == job]
    assert len(exact) == 1 and exact[0][1:3] == ['FAILED', '1:0'], raw
    summary = Path(previous['summary']['path'])
    assert sha(summary) == previous['summary']['sha256']
    macro = REPORTS / 'macro_preservation_completed_v1/summary.json'
    evidence = json.loads(macro.read_text())
    assert evidence['complete'] is True and evidence['checked'] == 12
    assert evidence['all_checked_final_match_native_macroplacement'] is True
    assert evidence['all_checked_final_macros_legal'] is False
    # A deliberate report of failure must never turn a missing/crashed audit into a result.
    assert any(not r['final_macro_geometry']['legal_no_halo'] for r in evidence['rows'])
    plot_job = previous['children']['plots']['job']
    deadline = time.monotonic() + 180
    while True:
        raw = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', plot_job,
                                       '-o', 'JobID%64,State,ExitCode'], text=True)
        exact = [r.split('|') for r in raw.splitlines() if r.split('|')[0] == plot_job]
        if len(exact) == 1 and exact[0][1:3] == ['COMPLETED', '0:0']:
            break
        assert time.monotonic() < deadline and all(r[1] in ('RUNNING', 'PENDING', 'COMPLETING') for r in exact)
        time.sleep(10)
    writer = ROOT / 'analysis/write_table2_original_issue_report.py'
    expected = os.environ['TABLE2_ISSUE_WRITER_SHA256']
    assert sha(writer) == expected
    argv = ['sbatch', '--parsable', 'analysis/write_table2_original_issue_report.slurm', expected]
    for name, path in [('summary', summary), ('macro', macro),
                       ('plots', REPORTS / 'reproduction_plots_completed_v1/manifest.json'),
                       ('connectivity', REPORTS / 'connectivity_v1/summary.json')]:
        argv += ['--' + name, str(path), '--' + name + '-sha256', sha(path)]
    output = REPORTS / 'original_flow_issue_report_v1'
    argv += ['--output', str(output)]
    child = subprocess.check_output(argv, text=True).strip().split(';')[0]
    journal = {'status': 'report_submitted_macro_geometry_failed',
               'macro_geometry_qualified': False, 'baseline_qualified': False,
               'job': child, 'command': argv, 'writer_sha256': expected,
               'controller_sha256': sha(__file__), 'output': str(output)}
    (BASE / 'original_issue_report_submission_v1.json').write_text(json.dumps(journal, indent=2) + '\n')
    print(json.dumps(journal))


if __name__ == '__main__':
    main()
