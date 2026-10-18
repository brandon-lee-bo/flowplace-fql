"""Generate one ICCAD layout with the released FlowPlace model and sampler."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'baseline'))
from api import UPSTREAM, CHECKPOINT, load_model, generate, legality, OmegaConf
from benchmark_utils.iccad2015_utils import ProblemInstance, write_def_from_tensor, get_inv_scaling_ratio
from audit_inputs import inspect_def, lef_blocks, sha
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--design',default='superblue18')
    p.add_argument('--seed',type=int,default=300);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    # Match released tensor operations; skip compilation of wrappers.
    torch.compile=lambda fn,*args,**kwargs:fn
    torch.manual_seed(a.seed)
    bench=ROOT/'iccad/assets/benchmarks/iccad2015.ot'
    meta=dict(design=a.design,seed=a.seed,steps=20,prior='uniform',hard_constraint=True,
        w_reg=0.1,w_dist=1.0,slurm_job=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),
        checkpoint_sha256=sha(CHECKPOINT),upstream_commit=(UPSTREAM/'SOURCE_COMMIT').read_text().strip(),
        torch=torch.__version__,cuda=torch.version.cuda,pythonhashseed=os.environ.get('PYTHONHASHSEED'),
        complete=False,compile_disabled=True,deterministic_algorithms=False,
        scope='New single-design seeded generation, not the paper eight-design RNG stream or Table 3 ablation')
    for name in ['generate_iccad.py','audit_inputs.py']:
        shutil.copy2(Path(__file__).with_name(name),a.output/name)
    shutil.copy2(ROOT/'baseline/api.py',a.output/'baseline_api.py')
    for ext in ['def','lef','v']:
        meta[f'input_{ext}_sha256']=sha(bench/a.design/f'{a.design}.{ext}')
    (a.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    try:
        args=SimpleNamespace(BENCHMARK_DIR=str(bench),grid_soft_coeff=10,grid=224,n_macro=512)
        instance=ProblemInstance(args,a.design)
        x,cond=instance._build_pyg_data()
        cond.is_ports=cond.is_ports.flatten().bool()
        if hasattr(cond,'chip_size'):del cond.chip_size
        problem=SimpleNamespace(x=x.to('cuda'),cond=cond.to('cuda'),macro_names=instance.node_id_to_name)
        model,cfg=load_model(problem,device='cuda',steps=20)
        cfg.task='iccad2015-s0'
        OmegaConf.save(cfg,a.output/'config.yaml')
        torch.cuda.synchronize();start=time.monotonic()
        centers,noise,traj=generate(model,problem,seed=a.seed,intermediate_every=1)
        torch.cuda.synchronize();meta['sampling_seconds_including_setup']=time.monotonic()-start
        meta['normalized_geometry']=legality(centers,cond.x)
        centers=centers.detach().cpu()
        torch.save(dict(centers=centers,noise=noise.cpu(),cond=cond.cpu(),
            original_x=x,macro_names=instance.node_id_to_name,
            trajectory=[t.cpu() for t in traj]),a.output/'sample.pt')
        positions={}
        for i,name in enumerate(instance.node_id_to_name):
            cx=(centers[i,0].item()+1)/2*instance.max_width
            cy=(centers[i,1].item()+1)/2*instance.max_height
            size=instance.node_info[name]
            positions[name]=(cx-size['x']/2,cy-size['y']/2,0,0)
        ix,iy=get_inv_scaling_ratio(instance.database)
        path=a.output/f'{a.design}.def'
        write_def_from_tensor(positions,instance.database,str(path),
            limited_x=round(instance.max_width*ix),limited_y=round(instance.max_height*iy),is_dataset=False)
        audit=inspect_def(path,lef_blocks(bench/a.design/f'{a.design}.lef'))
        (a.output/'def_geometry.json').write_text(json.dumps(audit,indent=2)+'\n')
        meta.update(output_def=str(path),output_sha256=sha(path),def_overlap_count=audit['overlap_count'],
                    def_out_of_die_count=len(audit['out_of_die']),fixed_count=audit['macro_count'])
        meta['complete']=True
        meta['legal']=meta['normalized_geometry']['is_legal'] and not audit['overlap_count'] and not audit['out_of_die']
        if not meta['legal']:raise RuntimeError('Generated sample/export is geometrically illegal; retained without repair')
    except Exception as e:
        meta['error']=repr(e)
        raise
    finally:
        (a.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
        print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':main()
