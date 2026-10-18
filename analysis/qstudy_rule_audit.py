"""Read-only TRAIN rule variation audit; no fitting, inference or backend calls.

Completed-layout continuous proxies inspired by EXPlace, not a reproduction of
its sequential mask/reward normalization. Rules and channel threshold are
fixed geometrically before reading any native associations. Validation labels
and the 48 development DRT evaluations never enter these computations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr
import torch

from analysis.placement_metrics import MacroHPWL, legality
from fql.qstudy_data import METRIC_NAMES, action_sha, bind, load_dataset, sha

ROOT = Path(__file__).resolve().parents[1]
DATASET_SHA = '0daecf483b64283af46e3aec101161912ebff2984a493d0778fba5b2ebba99e4'
DEFAULT = Path('/fact_data/bohanli/flowplace_fql/qstudy_v1/data_audit/bp_fe384/dataset.npz')
CONFIG = Path('/fact_data/bohanli/flowplace_fql/qstudy_v1/actors_v1/bc_frozen/config.json')
GRID = 224
RULES = dict(
    edge_clearance=dict(unit='fraction of core width/height', direction='lower means closer to some edge; no causal PPA direction',
        definition='mean_i min(left_i, right_i, bottom_i, top_i), negative gaps clamped to zero'),
    corner_clearance=dict(unit='fraction of core width/height', direction='lower means closer to a corner; no causal PPA direction',
        definition='mean_i [min(left_i,right_i)+min(bottom_i,top_i)], negative gaps clamped to zero'),
    cluster_bbox_fraction=dict(unit='core-area fraction, summed across non-singleton groups', direction='lower means tighter same-group packing',
        definition='sum_C area(bounding rectangle of all complete macro rectangles in C)/core_area'),
    dataflow_distance=dict(unit='weighted mean distance / core diagonal', direction='lower means connected dataflow nodes closer',
        definition='sum_{i!=j} max(F_ij,0)*Euclidean(center_i-center_j)/sum max(F_ij,0)/core_diagonal; no pruning'),
    port_blockage_fraction=dict(unit='fraction of macro area; overlapping blockages may double count', direction='lower means less obstruction of declared I/O keepouts',
        definition='sum_{i,b} area(macro_i intersect declared block_rectangle_b)/sum macro_area; cache rectangles treated as EXPlace grid224 coordinates'),
    narrow_channel_risk=dict(unit='dimensionless, summed core-normalized facing lengths', direction='lower means fewer close facing macro edges; not a monotonic global-spacing objective',
        definition='sum over facing nonoverlapping pairs [facing_length*(max(0,tau-gap)/tau)^2], tau=2/127 of each corresponding core axis; no pair whose orthogonal projections do not overlap'),
    macro_hpwl=dict(unit='cache coordinate units (not routed um)', direction='lower',
        definition='sum pin-offset macro-hypernet bounding-box half perimeters; no standard-cell graph'))


def read(path):
    return json.loads(Path(path).read_text())


def describe(values):
    x = np.asarray(values, dtype=float)
    finite = x[np.isfinite(x)]
    counts=np.unique(finite,return_counts=True)[1]
    return dict(count=len(x), finite=len(finite), missing=len(x)-len(finite),
        unique=int(len(np.unique(finite))), constant=bool(len(finite)>0 and np.ptp(finite)<=1e-12),
        most_frequent_exact_value_count=int(counts.max()) if len(counts) else 0,
        zero_count=int(np.count_nonzero(finite==0)),
        min=float(finite.min()) if len(finite) else None, max=float(finite.max()) if len(finite) else None,
        mean=float(finite.mean()) if len(finite) else None,
        std=float(finite.std(ddof=1)) if len(finite)>1 else None)


def rank_association(x, y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    keep=np.isfinite(x)&np.isfinite(y)
    x,y=x[keep],y[keep]
    ok=len(x)>=3 and np.ptp(x)>1e-12 and np.ptp(y)>1e-12
    return dict(n=len(x), spearman=float(spearmanr(x,y).statistic) if ok else None,
                not_identified=None if ok else 'insufficient finite variation')


def numeric_summary(value):
    """Bounded cache metadata summary, not a dump of arbitrary arrays."""
    if isinstance(value,dict):
        return dict(type='dict', count=len(value), keys=[str(k) for k in list(value)[:16]])
    try:
        x=np.asarray(value)
        info=dict(type=type(value).__name__,shape=list(x.shape),dtype=str(x.dtype))
        if np.issubdtype(x.dtype,np.number):
            good=x[np.isfinite(x)]
            info.update(nonzero=int(np.count_nonzero(x)),min=float(good.min()) if good.size else None,
                        max=float(good.max()) if good.size else None)
        return info
    except (TypeError,ValueError):
        return dict(type=type(value).__name__,count=len(value) if hasattr(value,'__len__') else None)


def structure(cache):
    names=list(cache['node_id_to_name'])
    canvas=np.array([cache['ratio_x']*GRID,cache['ratio_y']*GRID],float)
    sizes=np.array([[cache['node_info'][n]['x'],cache['node_info'][n]['y']] for n in names],float)/canvas
    if not np.isfinite(sizes).all() or (sizes<=0).any() or (canvas<=0).any():
        raise ValueError('Invalid cached macro rectangles')
    ids={n:i for i,n in enumerate(names)}
    clusters=[]
    for group in cache.get('macro_clusters',[]):
        group=list(dict.fromkeys(group))
        if any(n not in ids for n in group):
            raise ValueError('Cluster includes unknown macro')
        if len(group)>1:
            clusters.append([ids[n] for n in group])
    df=None
    flow=cache.get('dataflow_mat')
    mapping=cache.get('id2index',{})
    df_reason=None
    if flow is None:
        df_reason='dataflow matrix absent'
    elif not all(n in mapping for n in names):
        df_reason='id2index does not cover every modeled macro'
    else:
        flow=np.asarray(flow,float)
        index=np.asarray([mapping[n] for n in names],int)
        if flow.ndim!=2 or flow.shape[0]!=flow.shape[1] or index.min()<0 or index.max()>=len(flow):
            raise ValueError('Invalid dataflow index mapping')
        df=np.maximum(flow[np.ix_(index,index)],0)
        np.fill_diagonal(df,0)
        if not np.isfinite(df).all():
            raise ValueError('Nonfinite dataflow weights')
        if not df.sum():
            df_reason='no positive inter-macro flow weights'
    rects=np.asarray(cache.get('pin_blocking_rectangles',[]),float)
    if rects.size:
        if rects.ndim!=2 or rects.shape[1]!=4 or not np.isfinite(rects).all() or (rects[:,2:]<0).any():
            raise ValueError('Unknown I/O blocking rectangle shape/units')
        if (rects<0).any() or (rects[:,:2]+rects[:,2:]>GRID+1e-6).any():
            raise ValueError('I/O rectangles do not fit assumed EXPlace grid224 frame')
        rects=rects/GRID
    else:
        rects=np.empty((0,4))
    weights=[float(net.get('weight',1)) for net in cache['net_info'].values()]
    metadata=dict(names=names,node_count=len(names),canvas_cache_units=canvas.tolist(),
        macro_area_fraction=float(sizes.prod(1).sum()),
        unique_macro_sizes=int(len(np.unique(sizes,axis=0))),
        clusters=clusters,non_singleton_clusters=len(clusters),
        clusters_covering_all_macros=sum(len(g)==len(names) for g in clusters),
        dataflow=numeric_summary(flow),dataflow_effective_positive_entries=int((df>0).sum()) if df is not None else None,
        dataflow_missing_reason=df_reason, port_pos=numeric_summary(cache.get('port_pos')),
        blockage_rectangles=rects.tolist(),blockage_count=len(rects),
        port_unit_assumption='EXPlace consumer expects grid224 (x,y,width,height); normalized here by224; not learned from labels',
        port_source_limit='Pinned EXPlace env_utils.py verifies that code convention, not the historical cache construction; computed blockage statistics are conditional on that schema, not production reward admission.',
        expert_source='https://github.com/lamda-bbo/EXPlace/tree/d5d534a5fbdbc1ab4aa0bd450761277b437d0967/src',
        net_count=len(weights),net_weight_stats=describe(weights),
        nets_with_explicit_weight=sum('weight' in net for net in cache['net_info'].values()),
        node_field_names=sorted({str(k) for node in cache['node_info'].values() for k in node}),
        net_field_names=sorted({str(k) for net in cache['net_info'].values() for k in net}),
        explicit_port_graph_mask='FlowPlace openroad_utils._build_pyg_data sets every is_ports=0; this is not evidence port_pos is empty',
        timing_criticality='No per-net criticality/timing path is inferred from dataflow or generic net weight; these fields require separate proven provenance')
    return dict(names=names,canvas=canvas,sizes=sizes,clusters=clusters,df=df,
                df_reason=df_reason,rects=rects),metadata


def rules(centers, context, hpwl):
    c=(np.asarray(centers,float)+1)/2
    size=context['sizes']; lo,hi=c-size/2,c+size/2
    gaps=np.maximum(np.stack([lo[:,0],1-hi[:,0],lo[:,1],1-hi[:,1]],1),0)
    result=dict(edge_clearance=float(gaps.min(1).mean()),
        corner_clearance=float((gaps[:,:2].min(1)+gaps[:,2:].min(1)).mean()),
        cluster_bbox_fraction=float(sum(np.prod(hi[g].max(0)-lo[g].min(0)) for g in context['clusters'])),
        dataflow_distance=None, port_blockage_fraction=0., narrow_channel_risk=0., macro_hpwl=float(hpwl(centers)))
    df=context['df']
    if df is not None and df.sum()>0:
        dist=np.linalg.norm((c[:,None]-c[None,:])*context['canvas'],axis=-1)
        result['dataflow_distance']=float((df*dist).sum()/df.sum()/np.linalg.norm(context['canvas']))
    for rect in context['rects']:
        overlap=np.maximum(0,np.minimum(hi,rect[:2]+rect[2:])-np.maximum(lo,rect[:2]))
        result['port_blockage_fraction']+=float(overlap.prod(1).sum()/size.prod(1).sum())
    tau=2/127  # Two cells of the released final-legalizer lattice per core axis.
    for i in range(len(c)):
        for j in range(i):
            for axis in (0,1):
                other=1-axis
                facing=max(0,min(hi[i,other],hi[j,other])-max(lo[i,other],lo[j,other]))
                gap=max(lo[i,axis],lo[j,axis])-min(hi[i,axis],hi[j,axis])
                if facing>0 and gap>=-1e-12:
                    result['narrow_channel_risk']+=facing*(max(0,tau-max(0,gap))/tau)**2
    return result


def associations(values, metrics, mask, families):
    names=list(RULES)
    result={}
    for rule in names:
        x=np.asarray([np.nan if row[rule] is None else row[rule] for row in values],float)
        entries={}
        for k,name in enumerate(METRIC_NAMES):
            # All costs have lower-is-better direction, including violation magnitudes.
            y=np.where(mask[:,k],metrics[:,k],np.nan).copy()
            if k in (1,2):
                y=np.maximum(-y,0)
            groups=[]
            for family in np.unique(families):
                keep=(families==family)&np.isfinite(x)&np.isfinite(y)
                if keep.any():
                    groups.append((x[keep].mean(),y[keep].mean()))
            group=np.asarray(groups,float).reshape(-1,2)
            entries[name]=dict(row_level=rank_association(x,y),
                family_mean=rank_association(group[:,0],group[:,1]),
                target='negative slack magnitude (ns)' if k in (1,2) else name)
        result[rule]=entries
    return result


def checked_train(data,records):
    rows=[r for r in records if r['split']=='train']
    a=data['train']
    if len(rows)!=384 or len(a['raw_action'])!=384:
        raise ValueError('Wrong TRAIN denominator')
    for i,row in enumerate(rows):
        if (row['family']!=int(a['family'][i]) or row['variant']!=int(a['variant'][i]) or
                row['raw_action_sha256']!=action_sha(a['raw_action'][i]) or
                row['action_sha256']!=action_sha(a['executed_action'][i]) or row['status']!=a['status'][i]):
            raise ValueError('TRAIN raw/executed/native identity changed')
        truth=np.asarray([np.nan if row['metrics'][m] is None else row['metrics'][m] for m in METRIC_NAMES])
        if not np.array_equal(truth,a['metrics'][i],equal_nan=True):
            raise ValueError('TRAIN native metric correspondence changed')
    return a,rows


def restoration_inventory(rows):
    """Inspect exact source-side metadata, never decode/restore archives here."""
    result=[]
    for row in rows:
        original=row['bindings'].get('physical_manifest')
        if original is None:
            result.append(dict(family=row['family'],variant=row['variant'],status='no_successful_GRT_source'))
            continue
        binding=bind(original['path'],original['sha256'])
        run=Path(binding['path']).parent
        metadata=[]
        inventory=None
        for filename in ('grt_artifact_archive.json','legacy_archive_repack.json','grt_intermediates_inventory.json'):
            path=run/filename
            if path.is_file():
                value=read(path)
                if filename=='grt_intermediates_inventory.json':
                    inventory=value
                    metadata.append(dict(file=bind(path),entries=len(value),
                        bytes=sum(v['bytes'] for v in value.values())))
                else:
                    metadata.append(dict(file=bind(path),value=value))
        manifest=read(binding['path'])
        # The real config lives in bp_fe_top, but DESIGN_NICKNAME is bp_fe.
        # This audit admits only the pinned bp_fe384 dataset, not arbitrary
        # config-directory-to-ORFS-name inference.
        config_parent=Path(manifest['design_config']).parent.name
        if config_parent!='bp_fe_top':
            raise ValueError('Unexpected bp_fe TRAIN source design config')
        nickname='bp_fe'
        checkpoints=[]
        for filename in ('1_synth.v','1_synth.sdc','5_1_grt.odb','5_1_grt.sdc','route.guide'):
            key='orfs_work/results/nangate45/'+nickname+'/eval/'+filename
            path=run/key
            expected=inventory.get(key) if inventory is not None else None
            checkpoints.append(dict(path=str(path),present=path.is_file(),
                bytes=path.stat().st_size if path.is_file() else None,
                archive_expected=expected,actual_content_hash_verified_here=False))
        result.append(dict(family=row['family'],variant=row['variant'],collection_kind=row['collection_kind'],
            source_manifest=binding,run=str(run),run_resolved=str(run.resolve()),
            config_parent=config_parent,design_nickname=nickname,
            continuation_path_basis='Explicit audited DESIGN_NICKNAME=bp_fe; config parent bp_fe_top is not the results nickname',
            archive_metadata=metadata,source_status='metadata inventoried; DRT checkpoint restoration/verification NOT performed',
            continuation_files=checkpoints,
            all_continuation_files_present=all(v['present'] for v in checkpoints),
            all_missing_continuation_files_in_archive_inventory=all(v['present'] or v['archive_expected'] is not None for v in checkpoints),
            actual_backend_geometry=row.get('actual_backend_geometry')))
    return result


def run(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Rule and saved-data audit must run through CPU Slurm')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','1')))
    dataset=Path(args.dataset).resolve();cfg=read(CONFIG)
    dataset_binding=bind(dataset,DATASET_SHA)
    if cfg['dataset_sha256']!=DATASET_SHA or Path(cfg['dataset_path']).resolve()!=dataset:
        raise ValueError('Not the fixed measured TRAIN source')
    cache_path=ROOT/'baseline/runtime/benchmark_cache/bp_fe.pt'
    cache_binding=bind(cache_path,cfg['cache_sha256'])
    data,manifest=load_dataset(dataset)
    records=[json.loads(line) for line in dataset.with_name('rows.jsonl').read_text().splitlines() if line]
    a,rows=checked_train(data,records)
    cache=torch.load(cache_path,map_location='cpu',weights_only=False)
    context,metadata=structure(cache);hpwl=MacroHPWL(cache)
    source_files=['analysis/qstudy_rule_audit.py','analysis/qstudy_rule_audit.slurm',
                  'analysis/placement_metrics.py','fql/qstudy_data.py',
                  'upstream/flowplace/benchmark_utils/openroad_utils.py',
                  'fql/graph_o2o_archive.py','fql/graph_o2o_finish.py']
    report=dict(version='qstudy_rule_audit_v1',status='read_only_TRAIN_diagnostic_complete',
        inputs=dict(dataset=dataset_binding,cache=cache_binding,config=bind(CONFIG),
            manifest=bind(dataset.with_name('manifest.json')),rows=bind(dataset.with_name('rows.jsonl'))),
        source_sha256={n:sha(ROOT/n) for n in source_files},rules=RULES,cache=metadata,
        TRAIN=dict(rows=len(rows),families=len(set(a['family'].tolist())),
            status_counts=dict(Counter(a['status'].tolist())),source_counts=dict(Counter(r['collection_kind'] for r in rows)),
            native7_complete=bool(a['metric_mask'].all())),
        scopes=dict(validation_labels_used=False,DRT_evaluation_labels_used=False,normalizer_fitted=False,
            new_model_calls=0,new_backend_calls=0,policy_update=False),
        interpretations=['TRAIN associations are descriptive and source-dependent, not rule-causality or reward-selection evidence.',
            'All seven native outcomes are GRT; no DRT evaluation is relabeled for fitting.',
            'Nonoverlap/core bounds are hard validity checks, not helpful variable rewards after successful legalization.',
            'Dataflow weights and macro-net weights do not establish timing criticality.',
            'This is a complete-layout continuous adaptation; EXPlace sequential costs, pruning and reward scaling are not reproduced.'],
        layouts={},native_train_statistics={name:describe(a['metrics'][:,k]) for k,name in enumerate(METRIC_NAMES)})
    detailed=[]
    for coordinate_key in ('raw_action','executed_action'):
        coordinates=a[coordinate_key].reshape(len(rows),len(context['names']),2)
        values=[rules(x,context,hpwl) for x in coordinates]
        geometry=[legality(x,hpwl.sizes) for x in coordinates]
        report['layouts'][coordinate_key]=dict(
            rule_statistics={rule:describe([np.nan if v[rule] is None else v[rule] for v in values]) for rule in RULES},
            associations=associations(values,a['metrics'],a['metric_mask'],a['family']),
            legal_rows=sum(g['legal'] for g in geometry),overlap_rows=sum(g['overlap_pairs']>0 for g in geometry),
            out_of_bounds_rows=sum(g['out_of_bounds']>0 for g in geometry))
        for i,(v,g) in enumerate(zip(values,geometry)):
            detailed.append(dict(provenance_index=int(a['provenance_index'][i]),family=int(a['family'][i]),
                variant=int(a['variant'][i]),coordinates=coordinate_key,rules=v,geometry=g))
    report['DRT_TRAIN_restoration_inventory']=restoration_inventory(rows)
    report['DRT_TRAIN_proposal']=dict(status='coordinated_adapter_required_not_selected_or_submitted_here',
        sampling='Use the separately coordinated TRAIN64 adapter only:32 original TRAIN families stratified by TRAIN-only mean GRT WL quartiles,8 SHA-selected families each, both variants;64 total proposed requests. This audit creates no selection or dispatch.',
        evaluation_exclusion='Existing48 policy-evaluation DRT labels and historical16 validation excluded; source families cannot overlap them by relabeling.',
        fitting='Use the separately frozen four family folds; no fresh or policy-evaluation data for scale fitting.',
        physical='Resolve/check exact original GRT ODB/SDC/guide and source manifest first; restore archives through frozen helpers if needed; use unchanged8threads/cap20 and retain failed attempts.',
        caveat='Proposed64 labels are correlated base/perturb pairs, not64 independent designs; success not guaranteed and no work is authorized by this diagnostic.')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_job'+os.environ['SLURM_JOB_ID']
    output=Path(args.output).resolve()/stamp
    output.mkdir(parents=True,exist_ok=False)
    payload=output/'rows.jsonl';payload.write_text(''.join(json.dumps(v,allow_nan=False)+'\n' for v in detailed))
    report['rows']=bind(payload)
    target=output/'report.json';target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],report=str(target),sha256=sha(target),TRAIN_rows=len(rows),
        cache_node_count=len(context['names']),backend_calls=0,model_calls=0)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=DEFAULT)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__':
    main()
