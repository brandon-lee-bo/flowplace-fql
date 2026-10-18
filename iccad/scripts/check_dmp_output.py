"""Reload final DEF and run DREAMPlace's all-cell legality checker; no moves."""
import argparse
import json
import os
from pathlib import Path
import sys
from audit_inputs import ROOT, sha

installed=ROOT/'iccad/assets/dmp_install_cpu'
sys.path[:0]=[str(installed),str(installed/'dreamplace')]
import numpy as np
import torch
from dreamplace import Params, PlaceDB
from dreamplace.ops.legality_check.legality_check import LegalityCheck

p=argparse.ArgumentParser();p.add_argument('run',type=Path)
p.add_argument('--dtype',choices=['float32','float64'],default='float32');a=p.parse_args()
assert os.environ.get('SLURM_JOB_ID')
torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
meta=json.loads((a.run/'manifest.json').read_text())
if not meta['complete']:
    verified=json.loads((a.run/'output_verification.json').read_text())
    assert verified['verified'] and verified['original_manifest_sha256']==sha(a.run/'manifest.json')
    meta=verified
path=Path(meta['output_def']);assert sha(path)==meta['output_sha256']
params=Params.Params();params.load(str(a.run/'config.json'))
params.def_input=str(path)
params.enable_fillers=0
params.dtype=a.dtype
params.num_threads=int(os.environ['SLURM_CPUS_PER_TASK'])
db=PlaceDB.PlaceDB();db(params)
t=lambda x:torch.from_numpy(np.asarray(x))
op=LegalityCheck(node_size_x=t(db.node_size_x),node_size_y=t(db.node_size_y),
    flat_region_boxes=t(db.flat_region_boxes),flat_region_boxes_start=t(db.flat_region_boxes_start),
    node2fence_region_map=t(db.node2fence_region_map),xl=db.xl,yl=db.yl,xh=db.xh,yh=db.yh,
    site_width=db.site_width,row_height=db.row_height,scale_factor=params.scale_factor,
    num_terminals=db.num_terminals,num_movable_nodes=db.num_movable_nodes)
legal=bool(op(t(np.concatenate([db.node_x,db.node_y]))))
out=dict(slurm_job=os.environ['SLURM_JOB_ID'],output_def=str(path),sha256=sha(path),
    checker_dtype=a.dtype,script_sha256=sha(__file__),
    legal=legal,scope='DREAMPlace all-cell row/site alignment, overlap, boundary and fence-region checker on reloaded final DEF; no routing/DRC signoff',
    movable_nodes=int(db.num_movable_nodes),fixed_nodes=int(db.num_terminals))
out['alignment_diagnostics']=dict(
    scale_factor=float(params.scale_factor),site_width=float(db.site_width),row_height=float(db.row_height),
    max_site_residual=float(np.max(np.abs((db.node_x[:db.num_movable_nodes]-db.xl)/db.site_width-np.rint((db.node_x[:db.num_movable_nodes]-db.xl)/db.site_width)))),
    max_row_residual=float(np.max(np.abs((db.node_y[:db.num_movable_nodes]-db.yl)/db.row_height-np.rint((db.node_y[:db.num_movable_nodes]-db.yl)/db.row_height)))))
filename='all_cell_legality.json' if a.dtype=='float32' else 'all_cell_legality_float64.json'
(a.run/filename).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2),flush=True)
if not legal:raise SystemExit(1)
