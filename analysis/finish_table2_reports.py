#!/usr/bin/env python3
"""Postprocess the existing frozen primary batch; never rerun or tune a placer."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'backend/plans/table2_background_v1'
REPORTS = Path('/fact_data/bohanli/flowplace_fql_backend/table2_reports')
SOURCES = {
    'analysis/audit_table2_macro_preservation.py': '521fe0f12165be6590ca407b4bc52509cf6e014ce77a496e7011c664f3cc37e4',
    'analysis/plot_table2_reproduction.py': '5a9d7555c4804cb8e0d222d74f946da2131fa29dd6d6639f341212843422a001',
    'analysis/write_table2_completion_report.py': 'f1805a7813e4309820696dc09bc2c5f029592bccfb9f1cd48d5ede2ec336b34e',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def wait_job(job):
    while True:
        raw = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', str(job),
                                       '-o', 'JobID%64,State%30,ExitCode'], text=True)
        rows = [line.split('|') for line in raw.splitlines()]
        exact = [r for r in rows if r[0] == str(job)]
        if exact:
            state = exact[0][1].split()[0].rstrip('+')
            if state == 'COMPLETED':
                assert exact[0][2] == '0:0', 'Nonzero job: ' + str(job)
                return
            if state not in ('RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED'):
                raise RuntimeError('Postprocess prerequisite ' + str(job) + ': ' + state)
        time.sleep(30)


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Submit through Slurm'
    os.chdir(ROOT)
    journal = BASE / 'postprocess_v1.json'
    with (BASE / '.postprocess.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not journal.exists(), 'Inspect existing postprocess journal before recovery'
        state = {'status': 'waiting_for_final_audit', 'slurm_job_id': os.environ['SLURM_JOB_ID'],
                 'created_unix': time.time(), 'sources': SOURCES, 'children': {}}
        save(journal, state)
        try:
            for path, expected in SOURCES.items():
                assert sha(ROOT / path) == expected, 'Postprocess source changed'
            batch = json.loads((BASE / 'state.json').read_text())
            assert batch['status'] == 'audit_submitted'
            state['final_audit_job'] = batch['final_audit_job']
            save(journal, state)
            wait_job(batch['final_audit_job'])
            summary = REPORTS / 'background_completed_v1/report/summary.json'
            info = json.loads(summary.read_text())
            assert info['validated_count'] == 12 and info['primary_comparison_ready'] is True
            state['summary'] = {'path': str(summary), 'sha256': sha(summary)}
            save(journal, state)

            def submit(name, argv):
                job = subprocess.check_output(['sbatch', '--parsable', '--job-name=t2finish_' + name] + argv,
                                              text=True).strip().split(';')[0]
                state['children'][name] = {'job': job, 'command': argv}
                save(journal, state)
                return job

            macro = REPORTS / 'macro_preservation_completed_v1'
            plots = REPORTS / 'reproduction_plots_completed_v1'
            macro_job = submit('macro', ['analysis/audit_table2_macro_preservation.slurm', str(summary),
                sha(summary), str(macro), SOURCES['analysis/audit_table2_macro_preservation.py']])
            plot_job = submit('plots', ['analysis/plot_table2_reproduction.slurm', str(summary),
                sha(summary), str(plots), SOURCES['analysis/plot_table2_reproduction.py']])
            state['status'] = 'auditing_macros_and_plotting'
            save(journal, state)
            wait_job(macro_job)
            wait_job(plot_job)
            connectivity = REPORTS / 'connectivity_v1/summary.json'
            output = REPORTS / 'completion_report_v1'
            argv = ['analysis/write_table2_completion_report.slurm', SOURCES['analysis/write_table2_completion_report.py']]
            for name, path in [('summary', summary), ('macro', macro / 'summary.json'),
                               ('plots', plots / 'manifest.json'), ('connectivity', connectivity)]:
                argv += ['--' + name, str(path), '--' + name + '-sha256', sha(path)]
            argv += ['--output', str(output)]
            report_job = submit('report', argv)
            state['status'] = 'writing_report'
            save(journal, state)
            wait_job(report_job)
            state.update(status='completed', completed_unix=time.time(), output=str(output))
            save(journal, state)
            print(json.dumps({'status': 'completed', 'output': str(output)}), flush=True)
        except Exception as exc:
            state.update(status='failed', error=repr(exc), failed_unix=time.time())
            save(journal, state)
            raise


if __name__ == '__main__':
    main()
