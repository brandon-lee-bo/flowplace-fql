"""Compare fixed candidate outcomes with already measured exact teacher references."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregate',required=True,type=Path)
    args=parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:raise RuntimeError('Use Slurm')
    root=Path(__file__).resolve().parents[1]
    aggregate=args.aggregate
    b=json.loads(aggregate.read_text())
    if b['status'] not in ['completed','completed_with_physical_failures']:raise ValueError('Audit must finish')
    rows=[]
    for line in (root/'datasets/bp_fe_grt_v1/backend_labels.jsonl').read_text().splitlines():
        row=json.loads(line);r=row['request_record']
        if r['family'] not in [192,193] or r['variant']!=0:continue
        if row['critic_split']!=1:raise ValueError('Expected existing validation reference')
        for key in ['metrics','def']:
            if sha(row[key+'_path'])!=row[key+'_sha256']:raise ValueError('Reference changed')
        if any(x['backend_consistency']!=row['backend_consistency'] for x in b['records']):
            raise ValueError('Different physical evaluator')
        values=json.loads(Path(row['metrics_path']).read_text())
        rows.append(dict(family=r['family'],source_row=row['row_index'],metrics_path=row['metrics_path'],
            metrics_sha256=row['metrics_sha256'],def_sha256=row['def_sha256'],values=values))
    if sorted(x['family'] for x in rows)!=[192,193]:raise ValueError('Missing/duplicated reference family')
    keys=['rWL_um','WNS_ns','TNS_ns','Power_W','StdCellArea_um2']
    summary=dict(status='completed',slurm_job_id=os.environ['SLURM_JOB_ID'],
        label='Existing original FlowPlace outputs with the dataset collection final relegalization',
        mean_of_two_references={k:float(np.mean([x['values'][k] for x in rows])) for k in keys},
        references=rows,independent_reference_measurements=2,reference_replicated_across_training_seeds=False,
        candidate_mean={},paired_by_noise=[],source_aggregate_sha256=sha(aggregate),
        scope='Exploratory existing validation references, not new runs or Table 2 detailed-route PPA.')
    for name in ['bc_continued','fql_physical_alpha_1']:
        selected=[r for r in b['records'] if r['configuration']==name]
        if len(selected)!=6:raise ValueError('Incomplete candidate denominator')
        summary['candidate_mean'][name]={k:float(np.mean([x['values'][k] for x in selected]))
            if all(x['status']=='success' and x['values'][k] is not None for x in selected) else None for k in keys}
        for reference in rows:
            matching=[x for x in selected if x['index']==reference['family']]
            summary['paired_by_noise'].append(dict(method=name,noise=reference['family'],training_seeds=[x['training_seed'] for x in matching],
                teacher_rwl_um=reference['values']['rWL_um'],student_seed_mean_rwl_um=float(np.mean([x['values']['rWL_um'] for x in matching]))
                if all(x['status']=='success' for x in matching) else None))
    out=Path('/fact_data/bohanli/flowplace_fql/fresh_reference_comparison_v1');out.mkdir(parents=True,exist_ok=False)
    (out/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='references'},indent=2),flush=True)


if __name__=='__main__':main()
