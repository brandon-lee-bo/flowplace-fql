"""Read-only bounded actor / unchanged measured BC-reference comparison.

No model forward, refit, selection, physical evaluation, or replay occurs.
Measured utilities are recomputed for EVERY arm under EACH full-TRAIN spec.
Old BC Q is discarded. Reused BC layouts are a measured paired reference;
the old source-study certificate does not certify a new bounded-Q BC update.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from . import qstudy_actor_report as native
# Preserve the local analysis package before upstream adds analysis.py paths.
from . import qstudy_backend as backend
from . import qstudy_bounded_actor as adapter
from .qstudy_bounded_utility import assert_common_scale, derive_cost_utility, validate_spec
from .qstudy_data import bind, load_dataset

ROOT = Path(__file__).resolve().parents[1]
OBJECTIVES = tuple(adapter.ARMS)
ARMS = ('BC', *OBJECTIVES)
SEEDS, COUNT = (0, 1, 2), 8
SETTINGS = ('design','cache_sha256','seeds','alpha','offline_actor_steps','batch_size',
    'flow_steps','actor_lr','flow_lr','online_flow_update','offline_critic_update',
    'evaluation_samples','evaluation_noise_seed','evaluation_purpose','backend_threads',
    'backend_storage','common_checkpoint_sha256')
SOURCE_NAMES = tuple(dict.fromkeys(('fql/qstudy_bounded_actor_report.py','fql/qstudy_actor_report.py',
    'fql/qstudy_bounded_actor.py','fql/qstudy_bounded_utility.py','fql/qstudy_backend.py',
    'fql/qstudy_data.py','fql/graph_o2o_backend.py',*adapter.SOURCE_NAMES)))


def plans_and_specs(root, bc):
    """Validate declarations and full-TRAIN scales; never load model weights."""
    plans, specs, evidence = {}, {}, {}
    reference = native.read(bc/'config.json')
    if (reference.get('arm') != 'bc' or reference.get('q_weight') != 0 or
        reference.get('seeds') != list(SEEDS) or reference.get('evaluation_samples') != COUNT or
        reference.get('offline_actor_steps') != 1000 or reference.get('online_flow_update') is not False or
        reference.get('offline_critic_update') is not False):
        raise ValueError('BC reference is not the existing fixed-flow offline zero-Q plan')
    original,_ = load_dataset(Path(reference['dataset_path']))
    evidence['BC_dataset'] = bind(reference['dataset_path'],reference['dataset_sha256'])
    behavior = adapter.behavior_identity(original)
    plans['BC'] = (bc, reference)
    for objective,(arm,_) in adapter.ARMS.items():
        path = root/arm
        cfg = None
        for seed in SEEDS:
            current = adapter.validate_config(path,seed,0)
            if cfg is not None and cfg != current:
                raise ValueError('Config changed during declaration validation')
            cfg = current
            key = str(seed)
            selected = cfg['bounded_refits'][key]
            if (bind(cfg['critic_checkpoints'][key],cfg['critic_checkpoint_sha256'][key]) != selected['checkpoint'] or
                bind(selected['completion']['path'],selected['completion']['sha256']) != selected['completion']):
                raise ValueError('Bounded full-TRAIN refit artifact differs')
            done = native.read(selected['completion']['path'])
            if (done['utility_spec'] != cfg['bounded_utility_spec'] or
                done['normalization'] != selected['normalization'] or
                done['binding']['fit'] != 'refit' or done['binding']['train_indices'] != list(range(384))):
                raise ValueError('Bounded refit/spec/normalization differs')
            evidence[f'{objective}_seed{seed}_checkpoint'] = selected['checkpoint']
            evidence[f'{objective}_seed{seed}_completed'] = selected['completion']
            bind(cfg['common_checkpoints'][key],cfg['common_checkpoint_sha256'][key])
            bind(reference['common_checkpoints'][key],reference['common_checkpoint_sha256'][key])
        if any(cfg.get(k) != reference.get(k) for k in SETTINGS):
            raise ValueError('Bounded actor and BC reference effective inputs differ')
        if cfg['bounded_behavior'] != behavior:
            raise ValueError('BC reference original256 targets differ')
        data,_ = load_dataset(Path(cfg['dataset_path']))
        evidence[f'{objective}_dataset'] = bind(cfg['dataset_path'],cfg['dataset_sha256'])
        if (adapter.behavior_identity(data) != behavior or any(not np.array_equal(data[k], original[k])
                for k in ('behavior_action','behavior_family'))):
            raise ValueError('BC reference exact original256 targets/families differ')
        context = adapter.screen.arrays(data['train'],'train')
        context['binding'] = dict(dataset_sha256=cfg['dataset_sha256'])
        if len(context['x']) != 384:
            raise ValueError('Bounded utility must use full384 TRAIN reference')
        expected,_ = adapter.screen.specification(context,list(range(384)),objective)
        spec = validate_spec(cfg['bounded_utility_spec'])
        if spec != expected:
            raise ValueError('Utility spec differs from native full-TRAIN construction')
        specs[objective] = spec
        plans[objective] = (path,cfg)
    for spec in specs.values():
        assert_common_scale(specs['wl'],spec)
    if len({p for p,_ in plans.values()}) != len(ARMS):
        raise ValueError('Duplicate bounded/control experiment')
    for arm,(path,_) in plans.items():
        evidence[arm+'_config'] = bind(path/'config.json')
    return plans,specs,evidence,dict(
        kind='unchanged_preexisting_measured_BC_layout_reference',
        matched_effective_settings=list(SETTINGS), original256_behavior=behavior,
        paired_native_reference_eligible_requires='Complete same-seed/index/actual-noise native outcomes',
        controlled_effective_update_eligible=False, new_bounded_BC_training=False,
        old_source_certificate_extended=False, new_physical_measurements=0,
        limitation='No new bounded zero-Q finite-gradient/update-equivalence certificate. Existing BC native layouts are reused unchanged; comparisons are descriptive paired physical references, not a newly certified bounded BC rerun.')


def measured(row,spec):
    status = row['state']
    values = [[row['metrics'][k] for k in native.METRICS]] if status == 'success' else None
    target = derive_cost_utility(None if values is None else torch.tensor(values,dtype=torch.float64),[status],spec)
    if not bool(target['valid'][0]):
        return dict(valid=False,C=None,B=None,utility=None)
    c = None
    if status == 'success':
        positive = native.costs(row['metrics'])
        # These fixed production scales/weights use ordinary physical units.
        # Long-double arithmetic also avoids intermediate overflow in the report.
        c = float(np.sum(np.asarray(spec['weights'],dtype=np.longdouble)*
            np.asarray(positive,dtype=np.longdouble)/np.asarray(spec['scales'],dtype=np.longdouble)))
    return dict(valid=True,C=c,B=float(target['cost'][0]),utility=float(target['utility'][0]))


def calibration(rows,objective):
    pairs = [(r.get('predicted_scalar_q'),r['bounded'][objective]['utility']) for r in rows
             if r['bounded'][objective]['valid'] and r.get('predicted_scalar_q') is not None]
    p,y = (np.asarray(v,dtype=float) for v in zip(*pairs)) if pairs else (np.array([]),np.array([]))
    rho = float(spearmanr(p,y).statistic) if len(p)>2 and np.ptp(p)>0 and np.ptp(y)>0 else None
    i,j = np.triu_indices(len(p),1)
    distinct = np.abs(y[i]-y[j]) > 1e-10
    all_prediction = np.asarray([-r['predicted_scalar_q'] for r in rows if r.get('predicted_scalar_q') is not None],dtype=float)
    return dict(n=len(p),eligible_native_rows=sum(r['bounded'][objective]['valid'] for r in rows),
        mae=float(np.abs(p-y).mean()) if len(p) else None,
        bias_prediction_minus_native=float((p-y).mean()) if len(p) else None,
        spearman=rho,distinct_native_pairs=int(distinct.sum()),
        layout_pair_accuracy=float(np.mean((p[i[distinct]]-p[j[distinct]])*(y[i[distinct]]-y[j[distinct]])>0)) if distinct.any() else None,
        saved_scalar_predictions=len(all_prediction),predicted_B_minimum=float(all_prediction.min()) if len(all_prediction) else None,
        predicted_B_maximum=float(all_prediction.max()) if len(all_prediction) else None,
        predicted_B_outside_success_training_range=int(((all_prediction<0)|(all_prediction>2)).sum()),
        range_note='[0,2] is the measured successful target range, not a clamp on unrestricted model predictions; a true failure has measured B=4. Only the saved mean scalar, not individual ensemble members, is available',
        definition='Saved scalar Q=-mean of two scalar-B ensemble members vs OWN objective native utility=-B; no Q1/Q2 interpretation or cross-objective Q comparison')


def evaluation_stream(record,arm,specs):
    errors = list(record.get('errors',[]))
    training = record.get('training',{})
    if training.get('integrity_error'):
        errors.append('Training integrity: '+training['integrity_error'])
    rows = [dict(r) for r in record['rows'] if r.get('role') == 'evaluation']
    indices,noises = [r['index'] for r in rows],[r.get('noise_sha256') for r in rows]
    if rows and (len(rows)!=COUNT or set(indices)!=set(range(COUNT)) or len(set(indices))!=COUNT):
        errors.append('Evaluation export differs from fixed eight-index plan')
    if rows and (any(not isinstance(n,str) or not n for n in noises) or len(set(noises))!=len(rows)):
        errors.append('Actual evaluation noise hashes absent or duplicated')
    for row in rows:
        if row['state']=='success' and any(type((row.get('metrics') or {}).get(k)) not in (int,float) or
                not math.isfinite(row['metrics'][k]) for k in native.METRICS):
            row.update(state='invalid',metrics=None,integrity_error='Seven finite native metrics required')
        if row['state']=='success' and (native.costs(row['metrics'])<0).any():
            row.update(state='invalid',metrics=None,integrity_error='Negative native physical cost')
        if row['state']!='success':
            row['metrics']=None
        q = row.pop('predicted_q',None)
        row['predicted_scalar_q'] = q if arm in OBJECTIVES and type(q) in (int,float) and math.isfinite(q) else None
        row.pop('measured_utility',None)  # Frozen reporter's unbounded legacy calculation is NOT B.
        row['bounded'] = {objective:measured(row,spec) for objective,spec in specs.items()}
    counts = {s:sum(r['state']==s for r in rows) for s in ('success','geometry_failure','physical_failure','missing','invalid')}
    complete = not errors and training.get('completed') is True and len(rows)==COUNT and sum(counts[s] for s in native.ADMITTED)==COUNT
    successful = [r for r in rows if r['state']=='success']
    return dict(arm=arm,seed=record['seed'],round=0,experiment=record['experiment'],
        training=training,errors=errors,tasks_sha256=record.get('tasks_sha256'),
        export_completion_sha256=record.get('export_completion_sha256'),
        planned=COUNT,exported=len(rows),not_exported=max(COUNT-len(rows),0),**counts,
        fully_accounted=bool(complete),integrity_invalid=bool(errors or counts['invalid']),
        actual_noise_by_index={str(r['index']):r['noise_sha256'] for r in rows},
        conditional_PPA=len(successful)!=COUNT,successful_PPA_denominator=len(successful),
        successful_native_metric_means={k:native.mean([r['metrics'][k] for r in successful]) for k in native.METRICS},
        measured_objectives={o:dict(C_success_mean=native.mean([r['bounded'][o]['C'] for r in successful]),
            B_all_request_mean=native.mean([r['bounded'][o]['B'] for r in rows]) if complete else None,
            utility_all_request_mean=native.mean([r['bounded'][o]['utility'] for r in rows]) if complete else None) for o in specs},
        own_objective_q_calibration=calibration(rows,arm) if arm in OBJECTIVES else None,
        raw_to_legalizer_rms=native.describe([r.get('raw_to_legalizer_rms') for r in rows]),rows=rows)


def full_stat(values):
    result = native.describe(values)
    return dict(result,eligible=True) if result['n']==len(SEEDS) else dict(n=result['n'],mean=None,sd=None,eligible=False)


def aggregate(streams):
    lookup = {(s['arm'],s['seed']):s for s in streams}
    if len(lookup)!=len(streams) or set(lookup)!={(a,s) for a in ARMS for s in SEEDS}:
        raise ValueError('Exactly four arms x three declared independent seeds are required')
    arms,pairs,contrasts = [],[],[]
    for arm in ARMS:
        selected = [lookup[(arm,s)] for s in SEEDS]
        ready = all(s['fully_accounted'] for s in selected) and all(s['actual_noise_by_index']==selected[0]['actual_noise_by_index'] for s in selected)
        arms.append(dict(arm=arm,aggregate_eligible=ready,planned_seeds=list(SEEDS),independent_seed_n=3 if ready else 0,
            request_accounting={k:sum(s[k] for s in selected) for k in ('planned','exported','not_exported','success','geometry_failure','physical_failure','missing','invalid')},
            conditional_PPA=any(s['conditional_PPA'] for s in selected),
            native_metrics={k:full_stat([s['successful_native_metric_means'][k] for s in selected]) for k in native.METRICS} if ready else None,
            measured_objectives={o:{k:full_stat([s['measured_objectives'][o][k] for s in selected])
                for k in ('C_success_mean','B_all_request_mean','utility_all_request_mean')} for o in OBJECTIVES} if ready else None,
            own_objective_q_calibration={k:full_stat([s['own_objective_q_calibration'][k] for s in selected])
                for k in ('mae','bias_prediction_minus_native','spearman','layout_pair_accuracy')} if ready and arm in OBJECTIVES else None))
    for baseline,target in combinations(ARMS,2):
        per_seed=[]
        selected = [lookup[(a,s)] for a in (target,baseline) for s in SEEDS]
        shared = all(s['actual_noise_by_index']==selected[0]['actual_noise_by_index'] for s in selected)
        for seed in SEEDS:
            a,b=lookup[(target,seed)],lookup[(baseline,seed)]
            aa,bb=({r['index']:r for r in side['rows']} for side in (a,b))
            same = len(aa)==len(bb)==COUNT and a['actual_noise_by_index']==b['actual_noise_by_index']
            ready = same and a['fully_accounted'] and b['fully_accounted']
            local=[]
            for index in sorted(set(aa)&set(bb)):
                ar,br=aa[index],bb[index]
                matched=ar['noise_sha256']==br['noise_sha256']
                admitted=matched and not a['integrity_invalid'] and not b['integrity_invalid'] and ar['state'] in native.ADMITTED and br['state'] in native.ADMITTED
                success=admitted and ar['state']==br['state']=='success'
                pair=dict(target=target,baseline=baseline,seed=seed,index=index,same_actual_noise=matched,
                    pair_admitted=admitted,both_successful=success,target_state=ar['state'],baseline_state=br['state'],
                    actual_noise_sha256=ar['noise_sha256'] if matched else None,
                    metric_delta_target_minus_baseline={k:ar['metrics'][k]-br['metrics'][k] for k in native.METRICS} if success else None,
                    utility_deltas={o:ar['bounded'][o]['utility']-br['bounded'][o]['utility'] for o in OBJECTIVES} if admitted else None,
                    target_request_sha256=ar.get('request_sha256'),baseline_request_sha256=br.get('request_sha256'),
                    target_outcome_sha256=ar.get('outcome_sha256'),baseline_outcome_sha256=br.get('outcome_sha256'),
                    target_raw_action_sha256=ar.get('raw_action_sha256'),baseline_raw_action_sha256=br.get('raw_action_sha256'),
                    target_executed_action_sha256=ar.get('action_sha256'),baseline_executed_action_sha256=br.get('action_sha256'))
                local.append(pair);pairs.append(pair)
            per_seed.append(dict(seed=seed,fully_accounted=ready,same_actual_noise_plan=same,
                planned_pairs=COUNT,admitted_pairs=sum(r['pair_admitted'] for r in local),successful_pairs=sum(r['both_successful'] for r in local),
                metric_deltas_conditioned_on_both_success=True,conditional_PPA=sum(r['both_successful'] for r in local)!=COUNT,
                utility_deltas={o:native.mean([r['utility_deltas'][o] for r in local if r['pair_admitted']]) for o in OBJECTIVES} if ready else None,
                metric_deltas={k:native.mean([r['metric_delta_target_minus_baseline'][k] for r in local if r['both_successful']]) for k in native.METRICS} if ready else None))
        ready=shared and all(s['fully_accounted'] for s in per_seed)
        contrasts.append(dict(target=target,baseline=baseline,aggregate_eligible=ready,
            paired_native_reference_eligible=ready,controlled_effective_update_eligible=ready and baseline!='BC',
            comparison_scope='descriptive existing measured BC reference; no bounded BC certificate' if baseline=='BC' else 'matched bounded actor objective intervention',
            common_noise_plan_across_seeds=shared,independent_seed_n=3 if ready else 0,
            metric_deltas_conditioned_on_both_success=True,conditional_PPA=any(s['conditional_PPA'] for s in per_seed),
            utility_deltas={o:full_stat([s['utility_deltas'][o] for s in per_seed]) for o in OBJECTIVES} if ready else None,
            metric_deltas={k:full_stat([s['metric_deltas'][k] for s in per_seed]) for k in native.METRICS} if ready else None,
            per_seed=per_seed))
    return arms,pairs,contrasts


def run(args):
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Read-only native/array report requires bounded CPU Slurm')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','1')))
    plans,specs,evidence,reuse = plans_and_specs(args.bounded_root.resolve(),args.bc.resolve())
    streams,bindings=[],{}
    for arm,(experiment,cfg) in plans.items():
        physical = backend.context(experiment)
        bindings[arm]=dict(config=bind(experiment/'config.json'),dataset=bind(cfg['dataset_path'],cfg['dataset_sha256']),physical=physical[2])
        for seed in SEEDS:
            # The frozen inspector has a guard before checked_existing: a missing
            # successful native geometry audit becomes invalid, never OpenROAD.
            record=native.inspect_round(experiment,cfg,seed,0,np.asarray(specs['wl']['scales']),physical)
            streams.append(evaluation_stream(record,arm,specs))
    if len({b['physical']['physical_flow_revision'] for b in bindings.values()})!=1:
        raise ValueError('Physical evaluator revision differs across bounded/reference arms')
    arms,pairs,contrasts=aggregate(streams)
    invalid=any(s['integrity_invalid'] for s in streams)
    status='integrity_invalid' if invalid else 'complete' if all(s['fully_accounted'] for s in streams) else 'partial'
    output=args.output.resolve()
    if any(output==p or output.is_relative_to(p) or p.is_relative_to(output) for p,_ in plans.values()):
        raise ValueError('Report must be separate from immutable actor/native artifacts')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    snapshot=output/(stamp+'_job'+os.environ['SLURM_JOB_ID'])
    snapshot.mkdir(parents=True,exist_ok=False)
    report=dict(schema='qstudy_bounded_actor_native_report_v1',status=status,created_utc=stamp,
        slurm_job_id=os.environ['SLURM_JOB_ID'],stage='global_route_only',development_only=True,
        backend_calls=0,model_forward_calls=0,critic_fits=0,optimizer_updates=0,selected_winner=None,
        plans=bindings,declaration_evidence=evidence,fullTRAIN_utility_specs=specs,BC_reuse=reuse,
        definitions=dict(native_metrics=list(native.METRICS),planned_requests='Three bounded objectives x three seeds x eight noises =72 plus unchanged BC24 reference',
            physical_costs='Native signed WNS/TNS -> max(-slack,0); all seven native metrics retained; no DRT/DRC inferred',
            utility='Each arm measured under EACH common-fullTRAIN-scale objective: C=sum(w*c/S), success B=2C/(1+C), confirmed failure B=4; unresolved/invalid no utility',
            Q='Saved scalar mean-ensemble Q predicts only its OWN objective -B. No cross-objective Q comparison; old BC Q discarded',
            uncertainty='Sample SD across three equally weighted independent training-seed means; no pooled-layout SD or silent zero-success seed removal',
            diagnostics='Saved training/gradient logs only, with exact captured-byte fingerprints. Input differentiability is not proof of physical direction accuracy',
            limitations='BC paired comparisons are descriptive without a new bounded zero-Q certificate; GRT only; no winner or promotion'),
        source_sha256={name:native.sha(ROOT/name) for name in SOURCE_NAMES},
        same_actual_noise_plan_all_streams=all(len(s['rows'])==COUNT for s in streams) and
            all(s['actual_noise_by_index']==streams[0]['actual_noise_by_index'] for s in streams),
        per_seed_streams=streams,independent_seed_aggregates=arms,matched_pairs=pairs,matched_contrasts=contrasts)
    (snapshot/'report.json').write_text(json.dumps(native.clean(report),indent=2,allow_nan=False)+'\n')
    native.write_csv(snapshot/'streams.csv',[{k:v for k,v in s.items() if k not in ('rows','training')} for s in streams])
    native.write_csv(snapshot/'seed_aggregates.csv',arms)
    native.write_csv(snapshot/'matched_pairs.csv',pairs)
    native.write_csv(snapshot/'matched_contrasts.csv',contrasts)
    (snapshot/'reporter_source.py').write_bytes(Path(__file__).read_bytes())
    artifacts={p.name:native.sha(p) for p in snapshot.iterdir() if p.is_file()}
    (snapshot/'completed.json').write_text(json.dumps(dict(status='completed',report_status=status,artifacts=artifacts,backend_calls=0),indent=2)+'\n')
    print(json.dumps(dict(status=status,snapshot=str(snapshot),backend_calls=0)),flush=True)
    return snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bounded-root','bc','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__':
    main()
