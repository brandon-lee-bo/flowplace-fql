"""Read-only locked fresh GRT results; no policy, fitting, selection or EDA.

All seven native metrics are retained. PPA means are success-conditional;
unresolved eight-request denominators withhold complete policy/three-seed
means. Public FlowPlace is ONE frozen checkpoint with eight noise outputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import torch

from . import qstudy_fresh_native as finish
from . import qstudy_fresh_eval as fresh
from . import qstudy_actor_report as native
from .graph_o2o_archive import atomic_json, sha
from .qstudy_bounded_utility import assert_common_scale, derive_cost_utility, validate_spec
from .qstudy_report import clean, csv_text

ROOT=fresh.ROOT
UTILITY_VERSION='fresh_common_bounded_utilities_v1'
SOURCE_NAMES=tuple(dict.fromkeys((*finish.SOURCE_NAMES,'fql/qstudy_fresh_report.py',
    'fql/qstudy_bounded_utility.py','fql/qstudy_report.py')))


def utility_specs(lock):
    """Read declared common TRAIN scales; never infer them from fresh labels."""
    path=fresh.checked_file(lock['common_train_utility_reference'])
    value=fresh.read(path)
    if value.get('version')==UTILITY_VERSION:
        if set(value)!={'version','utility_specs'} or not isinstance(value['utility_specs'],dict) or not value['utility_specs']:
            raise ValueError('Malformed explicit common utility inventory')
        specs={name:validate_spec(spec) for name,spec in value['utility_specs'].items()}
        if any(not isinstance(k,str) or not k for k in specs):
            raise ValueError('Common utility names must be explicit nonempty strings')
    else:
        specs={'locked':validate_spec(value)}
    reference=next(iter(specs.values()))
    for spec in specs.values():
        assert_common_scale(reference,spec)
    return specs


def measured(row,spec):
    values=(torch.tensor([[row['metrics'][k] for k in native.METRICS]],dtype=torch.float64)
            if row['state']=='success' else None)
    target=derive_cost_utility(values,[row['state']],spec)
    if not bool(target['valid'][0]):
        return dict(valid=False,C=None,B=None,utility=None)
    c=(float(np.sum(np.asarray(spec['weights'],dtype=np.longdouble)*
        np.asarray(native.costs(row['metrics']),dtype=np.longdouble)/np.asarray(spec['scales'],dtype=np.longdouble)))
        if row['state']=='success' else None)
    return dict(valid=True,C=c,B=float(target['cost'][0]),utility=float(target['utility'][0]))


def summarize_record(record,cfg,physical,specs):
    rows=[]
    if record['status']=='exported':
        _,verification,_=finish.backend.context(record['experiment'])
        for row in record['rows']:
            item=finish.inspect_outcome(row,record['tasks'],cfg,physical,verification)
            item['bounded']={name:measured(item,spec) for name,spec in specs.items()}
            rows.append(item)
    expected=fresh.COUNT
    counts={s:sum(r['state']==s for r in rows) for s in
        ('success','geometry_failure','physical_failure','missing','invalid')}
    complete=len(rows)==expected and sum(counts[s] for s in native.ADMITTED)==expected
    success=[r for r in rows if r['state']=='success']
    return dict(stream=record['stream']['id'],purpose=record['stream']['purpose'],
        kind=record['stream']['kind'],seed=record['seed'],experiment=record['experiment'],
        planned=expected,exported=len(rows),not_exported=expected-len(rows),**counts,
        fully_accounted=bool(complete),conditional_PPA=len(success)!=expected,
        successful_PPA_denominator=len(success),
        successful_native_metric_means={k:native.mean([r['metrics'][k] for r in success]) for k in native.METRICS},
        generation_noise_statistics_success_only={k:native.describe([r['metrics'][k] for r in success]) for k in native.METRICS},
        measured_objectives={name:dict(
            C_success_mean=native.mean([r['bounded'][name]['C'] for r in success]),
            B_all_request_mean=native.mean([r['bounded'][name]['B'] for r in rows]) if complete else None,
            utility_all_request_mean=native.mean([r['bounded'][name]['utility'] for r in rows]) if complete else None)
            for name in specs},
        macro_legality={k:dict(true=sum(r.get(k) is True for r in rows),false=sum(r.get(k) is False for r in rows),
            unknown=sum(r.get(k) is None for r in rows)) for k in
            ['requested_geometry_legal','rounded_geometry_legal','backend_macro_legal']},
        actual_noise_by_index={str(r['index']):r['noise_sha256'] for r in rows},rows=rows,
        export_completion_sha256=record.get('export_completion_sha256'),tasks_sha256=record.get('tasks_sha256'))


def aggregate(records,stream,specs):
    selected=[r for r in records if r['stream']==stream['id']]
    public=stream['kind']=='public_official20'
    complete=({r['seed'] for r in selected}==set(stream['seeds']) and
        all(r['fully_accounted'] for r in selected) and
        all(r['actual_noise_by_index']==selected[0]['actual_noise_by_index'] for r in selected))
    result=dict(stream=stream['id'],purpose=stream['purpose'],kind=stream['kind'],
        planned_seeds=stream['seeds'],accounted_seeds=[r['seed'] for r in selected if r['fully_accounted']],
        aggregate_eligible=bool(complete),independent_training_seed_n=0 if public else 3 if complete else 0,
        public_frozen_checkpoint_n=1 if public else 0,
        request_accounting={k:sum(r[k] for r in selected) for k in ['planned','exported','not_exported',
            'success','geometry_failure','physical_failure','missing','invalid']},
        conditional_PPA=any(r['conditional_PPA'] for r in selected),metrics=None,objectives=None,
        uncertainty='ONE public checkpoint; SD across successful noise outputs only, not training-seed SD' if public else
            'Sample SD of three independent policy-training seed means; generation noise is shared, not new training replicas')
    if complete:
        if public:
            result['metrics']=selected[0]['generation_noise_statistics_success_only']
            result['objectives']={name:dict(**selected[0]['measured_objectives'][name],
                generation_noise_B=native.describe([r['bounded'][name]['B'] for r in selected[0]['rows']]),
                training_seed_sd=None) for name in specs}
        else:
            result['metrics']={k:(native.describe([r['successful_native_metric_means'][k] for r in selected])
                if all(r['successful_native_metric_means'][k] is not None for r in selected) else
                dict(n=sum(r['successful_native_metric_means'][k] is not None for r in selected),mean=None,sd=None))
                for k in native.METRICS}
            result['objectives']={name:{k:native.describe([r['measured_objectives'][name][k] for r in selected])
                for k in ['B_all_request_mean','utility_all_request_mean']} for name in specs}
    return result


def paired(records,lock,specs):
    candidate=next(s for s in lock['streams'] if s['purpose']=='candidate')
    lookup={(r['stream'],r['seed']):r for r in records}
    pairs=[];seed_rows=[];summary=[]
    for reference in (s for s in lock['streams'] if s['id']!=candidate['id']):
        for seed in candidate['seeds']:
            a=lookup[(candidate['id'],seed)]
            b=lookup[(reference['id'],0 if reference['kind']=='public_official20' else seed)]
            eligible=a['fully_accounted'] and b['fully_accounted'] and a['actual_noise_by_index']==b['actual_noise_by_index']
            values=[]
            for left,right in zip(a['rows'],b['rows']):
                if left['index']!=right['index'] or left['noise_sha256']!=right['noise_sha256']:
                    raise ValueError('Fresh paired output does not use the same actual saved noise/index')
                both=left['state']=='success' and right['state']=='success'
                row=dict(candidate=candidate['id'],reference=reference['id'],seed=seed,index=left['index'],
                    noise_sha256=left['noise_sha256'],candidate_state=left['state'],reference_state=right['state'],
                    both_successful=both,
                    metric_delta_candidate_minus_reference={k:left['metrics'][k]-right['metrics'][k] if both else None
                        for k in native.METRICS},
                    bounded_utility_delta={name:(left['bounded'][name]['utility']-right['bounded'][name]['utility']
                        if left['bounded'][name]['valid'] and right['bounded'][name]['valid'] else None) for name in specs})
                values.append(row);pairs.append(row)
            seed_rows.append(dict(candidate=candidate['id'],reference=reference['id'],seed=seed,
                aggregate_eligible=bool(eligible),both_successful_pairs=sum(r['both_successful'] for r in values),
                paired_request_denominator=fresh.COUNT,
                metric_deltas={k:native.mean([r['metric_delta_candidate_minus_reference'][k] for r in values])
                    if eligible else None for k in native.METRICS},
                metric_deltas_conditioned_on_both_success=True,
                bounded_utility_deltas={name:native.mean([r['bounded_utility_delta'][name] for r in values])
                    if eligible else None for name in specs},
                reference_is_single_public_checkpoint=reference['kind']=='public_official20'))
        selected=[r for r in seed_rows if r['reference']==reference['id']]
        full=len(selected)==3 and all(r['aggregate_eligible'] for r in selected)
        stats=lambda values:native.describe(values) if full and all(v is not None for v in values) else dict(n=0,mean=None,sd=None)
        summary.append(dict(candidate=candidate['id'],reference=reference['id'],aggregate_eligible=bool(full),
            seeds=[r['seed'] for r in selected],metric_deltas={k:stats([r['metric_deltas'][k] for r in selected]) for k in native.METRICS},
            metric_deltas_conditioned_on_both_success=True,
            bounded_utility_deltas={name:stats([r['bounded_utility_deltas'][name] for r in selected]) for name in specs},
            interpretation='Three candidate-training seed differences vs ONE fixed public checkpoint; descriptive reference comparison'
                if reference['kind']=='public_official20' else 'Paired three independent training-seed differences on shared saved noise'))
    return pairs,seed_rows,summary


def call_accounting(records):
    attempts=[];runs=set();manifest_runs=set()
    for record in records:
        for row in record['rows']:
            sample=Path(row['sample_dir'])
            for path in sorted((sample/'backend_attempts').glob('*/attempt.json')):
                attempt=fresh.read(path)
                if attempt['request_sha256']!=sha(sample/'request.json') or attempt['tasks_sha256']!=record['tasks_sha256']:
                    raise ValueError('Attempt changed request/tasks identity in fresh call accounting')
                run=Path(attempt['physical_run']);runs.add(str(run))
                if (run/'manifest.json').is_file(): manifest_runs.add(str(run))
                attempts.append(dict(metadata=fresh.binding(path),physical_run=str(run),job_id=attempt['job_id']))
    return dict(logical_exported_requests=sum(len(r['rows']) for r in records),declared_worker_attempt_records=len(attempts),
        unique_declared_physical_run_ids=len(runs),unique_physical_calls_with_manifest=len(manifest_runs),
        physical_runs_without_manifest=len(runs-manifest_runs),attempts=attempts,
        interpretation='Native manifests prove physical invocations. An attempt without a manifest is unresolved call evidence, not zero cost; retries are not independent policy samples')


def collect(selection_path,selection_sha):
    checked=finish.checked_exports(selection_path,selection_sha,require_complete=False)
    lock,configs,physicals,declared,noise_info,records,dispatch,inventory=checked
    specs=utility_specs(lock)
    streams=[summarize_record(r,configs[r['stream']['id']],physicals[r['stream']['id']],specs) for r in records]
    for record in streams:
        if record['rows'] and record['actual_noise_by_index']!={str(i):v for i,v in enumerate(noise_info['noise_sha256'])}:
            raise ValueError('Fresh report actual shared noise inventory differs')
    pairs,seeds,comparisons=paired(streams,lock,specs)
    aggregate_rows=[aggregate(streams,s,specs) for s in lock['streams']]
    for path,digest in inventory.items():
        if sha(Path(path))!=digest:
            raise ValueError('Fresh export changed during read-only reporting')
    return dict(version='fresh_native_report_v1',selection=fresh.binding(selection_path),
        declaration=fresh.binding(Path(lock['output_root'])/'declaration.json'),
        status='invalid' if any(r['invalid'] for r in streams) else 'completed' if all(r['fully_accounted'] for r in streams) else 'partial',
        native_revision=lock['physical_flow_revision'],design=lock['design'],specs=specs,
        expected_requests=sum(fresh.COUNT*len(s['seeds']) for s in lock['streams']),
        export_inventory=inventory,noise=noise_info,streams=streams,aggregates=aggregate_rows,
        paired_rows=pairs,paired_seed_deltas=seeds,paired_summary=comparisons,
        call_accounting=call_accounting(records),report_source_sha256={n:sha(ROOT/n) for n in SOURCE_NAMES},
        conventions=dict(PPA='Success-conditional GRT means with failure/missing denominators; no final detailed route/Table2 claim',
            utility='Same locked TRAIN-only scales/reference across all methods; confirmed failuresB4, unresolved has no target',
            public='One released checkpoint and eight actual noises; never three independent public training seeds',
            confirmation='No automatic improvement/tradeoff decision or checkpoint/seed selection from this report',
            DRT='Only indices0..3 with repair cap20 were preselected; detailed-routing execution is not implemented here'),
        optimizer_updates=0,model_forward_calls=0,layout_generation_calls=0,backend_calls=0,
        training_feedback_eligible=False,offline_train_eligible=False,selected_winner=None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True);parser.add_argument('--selection-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Saved-array/native reporting requires bounded CPU Slurm')
    result=collect(args.selection,args.selection_sha256)
    output=args.output.resolve();source=Path(fresh.read(args.selection)['output_root']).resolve()
    if output==source or output in source.parents or source in output.parents:
        raise ValueError('Fresh report destination must be separate from frozen streams/selection output')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    snapshot=output/(stamp+'_job'+os.environ['SLURM_JOB_ID']);snapshot.mkdir(parents=True,exist_ok=False)
    result.update(snapshot=str(snapshot),slurm_job_id=os.environ['SLURM_JOB_ID'])
    atomic_json(snapshot/'report.json',clean(result))
    for name in ['streams','aggregates','paired_rows','paired_seed_deltas','paired_summary']:
        (snapshot/(name+'.csv')).write_text(csv_text(result[name]))
    (snapshot/'reporter_source.py').write_bytes(Path(__file__).read_bytes())
    atomic_json(snapshot/'completed.json',dict(status='completed',report_status=result['status'],
        files={p.name:sha(p) for p in snapshot.iterdir() if p.is_file()},selection=result['selection'],
        backend_calls=0,model_forward_calls=0,training_feedback_eligible=False))
    print(json.dumps(dict(status=result['status'],snapshot=str(snapshot))),flush=True)


if __name__=='__main__':
    main()
