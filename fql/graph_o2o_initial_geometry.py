"""Read-only physical geometry audit of offline TRAIN and validation labels."""
import argparse
import json
import os
from pathlib import Path
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import ROOT
from .graph_o2o_geometry import audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True)
    args=p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use Slurm')
    rows=[]
    source=ROOT/'datasets/bp_fe_grt_v1/backend_labels.jsonl'
    with source.open() as stream:
        for line in stream:
            record=json.loads(line)
            if record['critic_split'] not in [0,1]:
                continue  # No heldout power/timing/reward analysis or audit.
            run=Path(record['manifest_path']).parent
            if sha(run/'manifest.json')!=record['manifest_sha256']:
                raise ValueError('Original offline manifest changed')
            result=audit(run)
            rows.append(dict(row_index=record['row_index'], split=record['critic_split'],
                macro_count=result['macro_count'], backend_legal=result['legal_no_halo'],
                changed_from_input=len(result['input_to_actual_changes']),
                audit_path=str(run/'backend_macro_audit.json'), audit_sha256=sha(run/'backend_macro_audit.json')))
            if len(rows)%10==0:
                print(json.dumps(dict(audited=len(rows), legal=sum(r['backend_legal'] for r in rows))), flush=True)
    if len(rows)!=80 or sum(r['split']==0 for r in rows)!=64:
        raise ValueError('Expected 64 TRAIN plus 16 validation layouts')
    report=dict(status='passed' if all(r['backend_legal'] for r in rows) else 'physical_geometry_issue',
                scope='Offline TRAIN and validation actual saved GRT macro geometry only; no PPA rerun',
                rows=rows, count=len(rows), legal=sum(r['backend_legal'] for r in rows),
                source_backend_labels_sha256=sha(source), slurm_job_id=os.environ['SLURM_JOB_ID'],
                heldout_audited=False, original_manifests_and_metrics_unchanged=True)
    path=Path(args.experiment)/'offline_geometry_report.json'
    atomic_json(path, report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}), flush=True)
    if report['status']!='passed':
        raise RuntimeError('Offline physical macro legality must be handled before trusting reward labels')


if __name__=='__main__':
    main()
