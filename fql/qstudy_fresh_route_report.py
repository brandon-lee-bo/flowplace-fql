"""Read-only fresh cap20 detailed-route evidence, fixed four-request denominator.

Six final native metrics, including nonzero DRC, are preserved. GRT congestion
and its bounded utility are not relabeled as final-route measurements. Missing
or unknown failed attempts never receive fictitious PPA/failure targets.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

from . import qstudy_fresh_route as route
from . import qstudy_fresh_eval as fresh
from . import qstudy_actor_report as native
from .graph_o2o_archive import atomic_json, sha
from .qstudy_report import clean, csv_text

ROOT=route.ROOT
ADMITTED=('completed','upstream_physical_failure')
STATES=(*ADMITTED,'missing_GRT','missing_DRT','invalid')
SOURCE_NAMES=tuple(dict.fromkeys((*route.SOURCE_NAMES,'fql/qstudy_fresh_route_report.py','fql/qstudy_report.py')))


def summarize(rows,stream,seed):
    selected=[r for r in rows if r['stream']==stream['id'] and r['seed']==seed]
    if sorted(r['index'] for r in selected)!=fresh.DRT_INDICES:
        raise ValueError('Final DRT four-request denominator changed')
    good=[r for r in selected if r['state']=='completed']
    counts={s:sum(r['state']==s for r in selected) for s in STATES}
    complete=sum(counts[s] for s in ADMITTED)==4
    means={k:native.mean([r['metrics'][k] for r in good]) for k in route.METRICS}
    return dict(stream=stream['id'],purpose=stream['purpose'],kind=stream['kind'],seed=seed,
        planned=4,counts=counts,fully_accounted=complete,all_four_PPA_complete=len(good)==4,
        successful_PPA_denominator=len(good),conditional_PPA=len(good)!=4,
        complete_policy_PPA_mean=means if len(good)==4 else None,
        successful_only_PPA_mean=means,noise_output_statistics_success_only={k:native.describe(
            [r['metrics'][k] for r in good]) for k in route.METRICS},
        drc_zero=sum(r['metrics']['DRC']==0 for r in good),
        drc_zero_fraction_of_all_requests=sum(r['metrics']['DRC']==0 for r in good)/4,
        actual_noise_by_index={str(r['index']):r['noise_sha256'] for r in selected})


def aggregate(seed_rows,stream):
    selected=[r for r in seed_rows if r['stream']==stream['id']]
    public=stream['kind']=='public_official20'
    complete=(len(selected)==len(stream['seeds']) and {r['seed'] for r in selected}==set(stream['seeds']) and
        all(r['fully_accounted'] for r in selected))
    all_ppa=complete and all(r['all_four_PPA_complete'] for r in selected)
    some_per_seed=complete and all(r['successful_PPA_denominator'] for r in selected)
    result=dict(stream=stream['id'],purpose=stream['purpose'],kind=stream['kind'],planned_seeds=stream['seeds'],
        independent_training_seed_n=0 if public else len(stream['seeds']),public_frozen_checkpoint_n=1 if public else 0,
        aggregate_eligible=complete,conditional_PPA=not all_ppa,
        request_accounting=dict(planned=4*len(stream['seeds']),**{
            s:sum(r['counts'][s] for r in selected) for s in STATES}),
        successful_PPA_denominator=sum(r['successful_PPA_denominator'] for r in selected),
        complete_policy_PPA_statistics=None,success_conditional_PPA_statistics=None)
    if public:
        result['uncertainty']='ONE frozen public checkpoint; SD across at most four noise outputs, not training seeds'
        if complete:
            stats=selected[0]['noise_output_statistics_success_only']
            result['success_conditional_PPA_statistics']=stats
            if all_ppa:result['complete_policy_PPA_statistics']=stats
    else:
        result['uncertainty']='Sample SD across three independent training-seed means; four noises per policy'
        if some_per_seed:
            stats={k:native.describe([r['successful_only_PPA_mean'][k] for r in selected]) for k in route.METRICS}
            result['success_conditional_PPA_statistics']=stats
            if all_ppa:result['complete_policy_PPA_statistics']=stats
    return result


def paired(rows,streams):
    candidates=[s for s in streams if s['purpose']=='candidate']
    if len(candidates)!=1:raise ValueError('Expected exactly one locked candidate')
    candidate=candidates[0];lookup={(r['stream'],r['seed'],r['index']):r for r in rows}
    pairs=[];per_seed=[];summaries=[]
    for ref in streams:
        if ref['id']==candidate['id']:continue
        current=[]
        for seed in candidate['seeds']:
            bucket=[]
            for index in fresh.DRT_INDICES:
                a=lookup[(candidate['id'],seed,index)]
                b=lookup[(ref['id'],0 if ref['kind']=='public_official20' else seed,index)]
                if a['noise_sha256']!=b['noise_sha256']:
                    raise ValueError('DRT comparison requires the same actual saved noise')
                both=a['state']=='completed' and b['state']=='completed'
                row=dict(candidate=candidate['id'],reference=ref['id'],seed=seed,index=index,
                    noise_sha256=a['noise_sha256'],candidate_state=a['state'],reference_state=b['state'],
                    fully_accounted=a['state'] in ADMITTED and b['state'] in ADMITTED,
                    both_successful=both,metric_deltas={k:a['metrics'][k]-b['metrics'][k] for k in route.METRICS} if both else None)
                pairs.append(row);bucket.append(row)
            valid=[p for p in bucket if p['both_successful']];complete=all(p['fully_accounted'] for p in bucket)
            row=dict(candidate=candidate['id'],reference=ref['id'],seed=seed,planned_pairs=4,
                fully_accounted=complete,successful_pairs=len(valid),metric_deltas_conditioned_on_both_success=True,
                metric_deltas={k:native.mean([p['metric_deltas'][k] for p in valid]) for k in route.METRICS} if complete and valid else None)
            per_seed.append(row);current.append(row)
        complete=len(current)==3 and all(p['fully_accounted'] for p in current)
        summaries.append(dict(candidate=candidate['id'],reference=ref['id'],aggregate_eligible=complete,
            planned_pairs=12,successful_pairs=sum(p['successful_pairs'] for p in current),
            metric_deltas_conditioned_on_both_success=True,
            metric_deltas={k:native.describe([p['metric_deltas'][k] for p in current]) for k in route.METRICS}
                if complete and all(p['metric_deltas'] is not None for p in current) else None,
            interpretation='Candidate minus reference; three training-seed paired means'+
                (' against ONE fixed public checkpoint, descriptive' if ref['kind']=='public_official20' else '')))
    return pairs,per_seed,summaries


def call_accounting(plan):
    attempts=[];physical=set()
    for task in plan['tasks']:
        folder=Path(task['output']);spec=folder/'spec.json'
        for path in sorted(folder.glob('attempt_*/binding.json')):
            value=fresh.read(path)
            if not spec.exists() or value['spec_sha256']!=sha(spec):
                raise ValueError('DRT attempt differs from saved physical specification')
            run=Path(value['physical_run']);manifest=run/'manifest.json'
            row=dict(task=task['key'],binding=fresh.binding(path),job=value['slurm_job_id'],physical_run=str(run),
                manifest=fresh.binding(manifest) if manifest.exists() else None)
            if manifest.exists():physical.add(str(run.resolve()))
            attempts.append(row)
    return dict(logical_requests=plan['requested'],attempt_records=len(attempts),
        unique_physical_calls_with_manifest=len(physical),attempts=attempts,
        note='A missing manifest/unknown interruption is unresolved, not a zero-cost physical failure')


def collect(path,expected_sha):
    plan,checked=route.checked_plan(path,expected_sha)
    sources=route.inspect_sources(plan,checked,seal=False);rows=route.inspect_finals(plan,sources)
    if len(rows)!=len(sources):raise ValueError('Fresh final/source denominator changed')
    for row,source in zip(rows,sources):
        row.update(GRT_metrics=source.get('metrics'),GRT_failure_class=source.get('failure_class'),
            source_binding=source.get('source_binding'),GRT_outcome=source.get('outcome'))
    streams=checked[0]['streams']
    seeds=[summarize(rows,s,seed) for s in streams for seed in s['seeds']]
    pairs,pair_seeds,pair_summary=paired(rows,streams)
    complete=all(r['state'] in ADMITTED for r in rows)
    return dict(version=route.VERSION,status='all_requested_outcomes_accounted' if complete else 'partial',
        plan=fresh.binding(path),selection=plan['selection'],declaration=plan['declaration'],
        source_sha256={n:sha(ROOT/n) for n in SOURCE_NAMES},export_inventory=plan['export_inventory'],
        requested=plan['requested'],indices=fresh.DRT_INDICES,route_cap=20,backend_threads=8,
        route_settings=plan['route_settings'],records=rows,seed_means=seeds,
        stream_statistics=[aggregate(seeds,s) for s in streams],paired_rows=pairs,
        paired_seed_means=pair_seeds,paired_statistics=pair_summary,call_accounting=call_accounting(plan),
        integrity_errors=[dict(key=r['key'],error=r.get('integrity_error')) for r in rows if r['state']=='invalid'],
        final_congestion=None,utility_scope='No DRT congestion or bounded utility inferred from GRT labels',
        model_forward_calls=0,training_updates=0,backend_calls=0,feedback_eligible=False,
        interpretation='Preselected fresh evaluation only. Complete status accounts failures; inspect eligibility and denominators. No winner selection.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run report in bounded CPU Slurm')
    result=collect(args.plan.resolve(),args.plan_sha256)
    folder=args.output.resolve()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    folder.mkdir(parents=True,exist_ok=False);atomic_json(folder/'report.json',clean(result))
    (folder/'requests.csv').write_text(csv_text([{k:v for k,v in row.items() if not isinstance(v,(dict,list))}|
        (row['metrics'] or {}) for row in result['records']]))
    print(dict(status=result['status'],report=str(folder/'report.json'),integrity_errors=len(result['integrity_errors'])),flush=True)


if __name__=='__main__':main()
