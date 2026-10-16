"""Headless equivalent of official parse_openroad.ipynb, without visualization."""
import json,pickle,os,hashlib
from api import ROOT,RUNTIME,DESIGNS,Args,PlaceDBInstanceNangate45
if 'SLURM_JOB_ID' not in os.environ:raise RuntimeError('Slurm required')
out=RUNTIME/'datasets/graph/openroad-s0';out.mkdir(parents=True,exist_ok=True)
manifest=[]
for i,name in enumerate(DESIGNS):
    problem=PlaceDBInstanceNangate45(Args(grid=224,n_grid=224),name,str(RUNTIME/'benchmark_cache'/f'{name}.pt'))
    for prefix,value in [('graph',problem.cond),('output',problem.x)]:
        with (out/f'{prefix}{i}.pickle').open('wb') as f:pickle.dump(value,f)
    manifest.append({'index':i,'design':name,'macros':len(problem.macro_names),'edges':int(problem.cond.edge_index.shape[1]),'graph_sha256':hashlib.sha256((out/f'graph{i}.pickle').read_bytes()).hexdigest()})
(out/'config.yaml').write_text('train_samples: 0\nval_samples: 6\nscale: 1\nchip_width: 2.0\nchip_height: 2.0\npreprocessed: true\n')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
