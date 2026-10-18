#!/usr/bin/env python3
"""Lightweight read-only status for the two separate Table 2 cohorts."""
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    print(time.strftime('%Y-%m-%d %H:%M:%S %Z'))
    pause_path = ROOT / 'backend/plans/USER_PAUSED_TABLE2.json'
    paused = pause_path.is_file() and json.loads(pause_path.read_text()).get('paused') is True
    if paused:
        pause = json.loads(pause_path.read_text())
        print(f'USER PAUSED: {pause["time_hkt"]}; saved run/controller states below are historical, not live execution.')
        print(f'Pause journal: {pause["journal"]}')
    remaining = int(os.getxattr('/fact_data/bohanli', 'ceph.quota.max_bytes')) - int(
        os.getxattr('/fact_data/bohanli', 'ceph.dir.rbytes'))
    print(f'Persistent quota remaining: {remaining / 2**30:.2f} GiB')
    for cohort in ('table2_background_v1', 'table2_exact_import_v1'):
        path = ROOT / 'backend/plans' / cohort / 'state.json'
        state = json.loads(path.read_text())
        completed, active, pending, physical = [], [], 0, 0
        for label, entry in state['selected'].items():
            run = ROOT / 'backend/results' / entry['task']['run_id']
            manifest = run / 'manifest.json'
            if not manifest.is_file():
                pending += 1
                continue
            record = json.loads(manifest.read_text())
            physical += int(record.get('success') is True and record.get('evaluation_complete') is True)
            if record['status'] == 'completed':
                completed.append(label)
                continue
            if record.get('success') is True and record.get('evaluation_complete') is True:
                active.append((label, entry['job'], 'physical_complete_original_audit_failed'))
                continue
            logs = list((run / 'orfs_work/logs').glob('*/*/eval/*.log'))
            latest = max(logs, key=lambda p: p.stat().st_mtime) if logs else None
            stage = latest.name if latest else 'no native log'
            age = int(time.time() - latest.stat().st_mtime) if latest else None
            run_status = 'user_paused_incomplete' if paused else record['status']
            active.append((label, entry['job'], run_status, stage, age))
        print(f'{cohort}: physical_complete={physical}/12, original_audit_passed={len(completed)}/12, not_started={pending}, '
              f'supervisor={"user_paused" if paused else state["status"]}, error={state.get("last_supervisor_error", state.get("error"))}')
        for row in active:
            print('  ', *row)
        for report in ('capacity_finish_v1.json', 'capaware_finish_v1.json', 'postprocess_v1.json', 'original_issue_report_submission_v1.json'):
            p = path.parent / report
            if p.is_file():
                r = json.loads(p.read_text())
                print('  ', report, r.get('status'), r.get('error'), r.get('output'))


if __name__ == '__main__':
    main()
