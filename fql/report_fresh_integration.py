"""Summarize all predeclared variants; never select a favorable alpha/seed."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from .core import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',required=True);p.add_argument('--critic',required=True)
    p.add_argument('--backend');p.add_argument('--output',required=True);a=p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:raise RuntimeError('Use Slurm')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    r=json.loads((Path(a.pilot)/'report.json').read_text());c=json.loads((Path(a.critic)/'report.json').read_text())
    if r['status']!='completed' or c['status']!='completed':raise ValueError('Source experiment incomplete')
    results={}
    for seed in r['seeds']:
        for row in seed['results']:
            method=row['method'].split('/')[-1]
            results.setdefault(method,[]).append(row)
    summary={'status':'completed','stage':'macro-only and existing-label diagnostic','slurm_job_id':os.environ['SLURM_JOB_ID'],
        'macro_methods':{},'official_teacher_original':r['official_teacher_original'],
        'official_teacher_relegalized':r['official_teacher'],'best_observed_train':r['best_observed_train'],
        'real_grt_critic':{n:{'validation':m['validation'],'selected_penalty':m['selected_penalty'],
            'selected_train_cv':next(x['group_cv'] for x in m['train_cv'] if x['penalty']==m['selected_penalty'])}
            for n,m in c['models'].items()},'constant_grt_critic':c['constant_validation'],
        'heldout_evaluated':False,'backend':None}
    for name,rows in results.items():
        if len(rows)!=3:raise ValueError('Missing predeclared training seed')
        summary['macro_methods'][name]={'seed_macro_hpwl':[v['mean_macro_hpwl'] for v in rows],
            'mean_across_seed_means':float(np.mean([v['mean_macro_hpwl'] for v in rows])) if all(v['mean_macro_hpwl'] is not None for v in rows) else None,
            'legal_count':sum(v['legal_count'] for v in rows),'candidate_count':sum(v['num_candidates'] for v in rows),
            'seed_distillation_mse':[v.get('validation_fresh_teacher_distillation_mse') for v in rows],
            'seed_actor_selected_q_spearman':[v['actor_selected_critic']['spearman_r'] for v in rows]}
    if a.backend:
        b=json.loads((Path(a.backend)/'report.json').read_text())
        if b['status'] not in ['completed','completed_with_physical_failures']:raise ValueError('Backend aggregate incomplete')
        root=Path(__file__).resolve().parents[1]
        source_manifest=root/'datasets/bp_fe_grt_v1/manifest.json'
        qm=json.loads(source_manifest.read_text())
        for record in b['records']:
            if record['backend_consistency']!=qm['backend_consistency']:
                raise ValueError('New backend differs from the critic training evaluator')
        pairs=[]
        for seed in range(3):
            for noise in [192,193]:
                lookup={x['method'].split('/')[-1]:x for x in b['records'] if x['training_seed']==seed and x['index']==noise}
                if set(lookup)!={'bc_continued','fql_physical_alpha_1'}:raise ValueError('Missing paired method')
                bc,rl=lookup['bc_continued'],lookup['fql_physical_alpha_1']
                pair={'seed':seed,'noise':noise,'bc_status':bc['status'],'fql_status':rl['status']}
                if bc['status']==rl['status']=='success':
                    for metric in ['rWL_um','WNS_ns','TNS_ns','Power_W','StdCellArea_um2']:
                        pair[metric]={'bc':bc['values'][metric],'fql':rl['values'][metric],
                            'fql_minus_bc':rl['values'][metric]-bc['values'][metric]
                            if rl['values'][metric] is not None and bc['values'][metric] is not None else None}
                    pair['rwl_improvement_percent']=100*(bc['values']['rWL_um']-rl['values']['rWL_um'])/bc['values']['rWL_um']
                pairs.append(pair)
        summary['backend']={'status':b['status'],'successful_count':b['successful_count'],
            'physical_failure_count':b['physical_failure_count'],'total_count':b['total_count'],
            'paired_results':pairs,'quality':b['quality'],
            'stage':'global_route_only; not detailed-route Table 2 PPA'}
        summary['stage']='Fresh behavior-flow pilot plus paired real global-route evaluation'
        summary['backend']['training_evaluator_identity_verified']=True
        summary['backend']['critic_training_manifest_sha256']=hashlib.sha256(source_manifest.read_bytes()).hexdigest()
        summary['backend']['by_configuration']={}
        for method in ['bc_continued','fql_physical_alpha_1']:
            rows=[x for x in b['records'] if x['configuration']==method]
            result={'total':len(rows),'successful':sum(x['status']=='success' for x in rows),'seed_means':{}}
            for seed in range(3):
                selected=[x for x in rows if x['training_seed']==seed]
                if len(selected)!=2:raise ValueError('Incomplete predeclared seed denominator')
                result['seed_means'][str(seed)]={metric:float(np.mean([x['values'][metric] for x in selected]))
                    if all(x['status']=='success' and x['values'][metric] is not None for x in selected) else None
                    for metric in ['rWL_um','WNS_ns','TNS_ns','Power_W','StdCellArea_um2']}
            result['mean_across_seed_means']={metric:float(np.mean([x[metric] for x in result['seed_means'].values()]))
                if all(x[metric] is not None for x in result['seed_means'].values()) else None
                for metric in ['rWL_um','WNS_ns','TNS_ns','Power_W','StdCellArea_um2']}
            summary['backend']['by_configuration'][method]=result
        successful_pairs=[x for x in pairs if 'rwl_improvement_percent' in x]
        summary['backend']['paired_diagnostics']={
            'successful_pairs':len(successful_pairs),'predeclared_pairs':6,
            'rwl_improved_pairs':sum(x['rwl_improvement_percent']>0 for x in successful_pairs),
            'wns_improved_pairs':sum(x['WNS_ns']['fql_minus_bc']>0 for x in successful_pairs),
            'mean_pairwise_rwl_improvement_percent':float(np.mean([x['rwl_improvement_percent'] for x in successful_pairs])) if len(successful_pairs)==6 else None,
            'no_confidence_or_significance_claim':'Only 3 training seeds and 2 within-design noise families; no independent final heldout.'}
        fixed_path=root/'runs/bp_fe_grt_gate_v1/best_train_layout.json'
        fixed=json.loads(fixed_path.read_text());bound=fixed['backend_record']
        if fixed['selection_split']!='train_only' or bound['backend_consistency']!=qm['backend_consistency']:
            raise ValueError('Fixed training baseline scope/evaluator mismatch')
        metrics_path=Path(bound['metrics_path'])
        if hashlib.sha256(metrics_path.read_bytes()).hexdigest()!=bound['metrics_sha256']:
            raise ValueError('Fixed training reference metrics changed')
        reference=json.loads(metrics_path.read_text())
        summary['backend']['fixed_best_train_reference']={
            'path':str(fixed_path),'source_row_index':fixed['source_row_index'],
            'eligible_training_layout_count':fixed['eligible_training_layout_count'],
            'training_noise_families':fixed['training_noise_families'],
            'independent_measurements':1,'values':{k:reference[k] for k in ['rWL_um','WNS_ns','TNS_ns','Power_W','StdCellArea_um2']},
            'note':'Existing TRAIN-only minimum-rWL layout; not a new candidate or six replicated observations.'}
    write_json(out/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    names=list(results)
    means=[summary['macro_methods'][n]['mean_across_seed_means'] for n in names]
    axes[0].bar(range(len(names)),means)
    axes[0].axhline(r['official_teacher_original']['mean_macro_hpwl'],color='black',ls='--',label='Original FlowPlace')
    axes[0].set_xticks(range(len(names)),['Flow BC','Student BC','FQL a=1','FQL a=10','FQL a=100'],rotation=25,ha='right')
    axes[0].set_ylabel('Macro-only HPWL (cache units)');axes[0].legend()
    reps=list(c['models']);mae=[c['models'][n]['validation']['mae'] for n in reps]
    axes[1].bar(range(len(reps)),mae)
    axes[1].axhline(c['constant_validation']['mae'],color='black',ls='--',label='TRAIN-mean constant')
    axes[1].set_xticks(range(len(reps)),[n.replace('_','\n') for n in reps],fontsize=8)
    axes[1].set_ylabel('Existing GRT validation reward MAE');axes[1].legend()
    fig.suptitle('Exploratory single-circuit diagnostics; not PPA improvement evidence')
    fig.tight_layout();fig.savefig(out/'diagnostics.png',dpi=160);plt.close(fig)
    print(json.dumps({'output':str(out),'macro':summary['macro_methods'],'backend':summary['backend']},indent=2),flush=True)


if __name__=='__main__':main()
