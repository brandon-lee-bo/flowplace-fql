"""Lock hyperparameters using measured validation reward before heldout evaluation."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--training-dir',required=True)
    args=p.parse_args()
    root=Path(args.training_dir).resolve()
    src=root/'measured_validation_summary.json'
    data=json.loads(src.read_text())
    if data['phase']!='validation':raise ValueError('Requires validation-only measurements')
    dst=root/'selection.json'
    if dst.exists():raise FileExistsError('Selection already locked; do not silently replace')
    groups={}
    for name,summary in data['methods'].items():
        if not name.startswith('seed_'):continue
        seed,method=name.split('/',1)
        groups.setdefault(method,[]).append((name,summary['validation']))
    scores={}
    families={}
    for method,entries in groups.items():
        count=sum(s['count'] for _,s in entries)
        score=sum(s['reward']['mean']*s['count'] for _,s in entries)/count
        scores[method]={'mean_validation_reward':score,'count':count,
                        'legal_fraction':sum(s['legal_fraction']*s['count'] for _,s in entries)/count}
        # Tune alpha within each algorithm family, never select a favorable seed.
        family='coordinate_hpwl' if method.startswith('coordinate_hpwl_') else method.split('_alpha_')[0]
        families.setdefault(family,[]).append(method)
    for family,methods_in_family in families.items():
        expected_seeds={name.split('/')[0] for method in methods_in_family for name,_ in groups[method]}
        for method in methods_in_family:
            if {name.split('/')[0] for name,_ in groups[method]}!=expected_seeds:
                raise ValueError('Hyperparameters within a family must have identical training seeds')
    if 'distill_only' in groups:
        model_seeds={name.split('/')[0] for name,_ in groups['distill_only']}
        for method,entries in groups.items():
            if method.startswith(('fql_alpha_','analytic_hpwl_alpha_')) and {name.split('/')[0] for name,_ in entries}!=model_seeds:
                raise ValueError('Learned variants must match distillation training seeds')
    chosen={family:max(sorted(methods),key=lambda m:scores[m]['mean_validation_reward'])
            for family,methods in families.items()}
    methods=sorted(name for chosen_method in chosen.values() for name,_ in groups[chosen_method])
    dst.write_text(json.dumps({'training_dir':str(root),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
        'rule':'Maximum pooled measured validation reward within algorithm family; equal seeds; lexical tie break',
        'reward':'Dataset reward includes fixed illegal-layout penalty; no predicted Q used',
        'scores':scores,'chosen_by_family':chosen,'methods':methods,
        'coordinate_control':'Deterministic optimization, one seed_control group; not independent training replicates',
        'backend_samples':'First predeclared indices per heldout split; no best-of-K selection'},indent=2))
    print(dst)
    print(json.dumps(chosen,indent=2))


if __name__=='__main__':main()
