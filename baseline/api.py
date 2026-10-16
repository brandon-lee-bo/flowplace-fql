"""Pinned upstream FlowPlace adapter; no edits to upstream source.
All coordinates use normalized macro centers on [-1, 1]^2 unless stated.
"""
from pathlib import Path
import contextlib
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream/flowplace"
RUNTIME = ROOT / "baseline/runtime"
for path in (UPSTREAM, UPSTREAM / "flow_matching"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from benchmark_utils.openroad_utils import Args, PlaceDBInstanceNangate45
from flow_matching.models import FlowMatchingModel
from flow_matching.hard_constraint_legalizer import Legalizer

DESIGNS = ["ariane133", "ariane136", "bp_be", "bp_fe", "bp", "swerv_wrapper"]
CHECKPOINT = RUNTIME / "logs/debug/mask.61.train_mask_flowplace_uniform.61/best.ckpt"


def load_problem(design, device="cpu"):
    if design not in DESIGNS:
        raise ValueError(design)
    args = Args(grid=224, n_grid=224)
    problem = PlaceDBInstanceNangate45(args, design, str(RUNTIME / "benchmark_cache" / f"{design}.pt"))
    # Exactly the official load_and_parse_graph and preprocessed validation path.
    problem.cond.is_ports = problem.cond.is_ports.flatten().bool()
    if hasattr(problem.cond, "chip_size"):
        del problem.cond.chip_size
    problem.cond.to(device)
    problem.x = problem.x.to(device)
    return problem


def load_model(problem, device="cuda", steps=20, hard_constraint=True, w_reg=0.1, w_dist=1.0):
    with initialize_config_dir(version_base=None, config_dir=str(UPSTREAM / "flow_matching/configs")):
        cfg = compose(config_name="config_eval")
    with open_dict(cfg):
        cfg.task = "openroad-s0"
        cfg.method = "released_pretrained_adapter"
        cfg.from_checkpoint = str(CHECKPOINT)
        cfg.model.input_shape = list(problem.x.shape)
        cfg.model.device = device
        cfg.model.prior_noise_dist = "uniform"
        cfg.model.max_diffusion_steps = steps
        cfg.model.use_hard_constraint_in_sampling = hard_constraint
        cfg.model.hard_constraint_w_reg = w_reg
        cfg.model.hard_constraint_w_dist = w_dist
    model = FlowMatchingModel(**cfg.model).to(device).eval()
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    # Unlike upstream helper, do not silently swallow checkpoint-load errors.
    state = ckpt["model"]
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(str(result))
    return model, cfg


class _FixedNoise:
    def __init__(self, noise):
        self.noise = noise
    def sample(self, shape):
        if tuple(shape) != tuple(self.noise.shape):
            raise ValueError((tuple(shape), tuple(self.noise.shape)))
        return self.noise.unsqueeze(-1).clone()


@torch.no_grad()
def generate(model, problem, noise=None, seed=None, intermediate_every=0):
    """Run exact released sampler, accepting/saving explicit initial noise."""
    if noise is None:
        if seed is not None:
            generator = torch.Generator(device=model.device).manual_seed(seed)
            noise = 2 * torch.rand((1, len(problem.macro_names), 2), device=model.device, generator=generator) - 1
        else:
            noise = model._noise_dist.sample((1, len(problem.macro_names), 2)).squeeze(-1)
    noise = noise.to(model.device)
    if noise.ndim == 2:
        noise = noise.unsqueeze(0)
    old_dist = model._noise_dist
    model._noise_dist = _FixedNoise(noise)
    try:
        result, trajectory = model.reverse_samples(1, problem.x.unsqueeze(0), problem.cond, intermediate_every=intermediate_every)
    finally:
        model._noise_dist = old_dist
    return result.squeeze(0), noise.squeeze(0), trajectory


@torch.no_grad()
def legalize(raw_centers, problem, w_reg=0.1, w_dist=1.0, grid_res=128, padding=0.001):
    legalizer = Legalizer(problem.cond, grid_res=grid_res, scores={"w_reg":w_reg,"w_dist":w_dist}, device=raw_centers.device, dtype=raw_centers.dtype, padding=padding)
    return legalizer.run(raw_centers).squeeze(0)


def legality(centers, sizes, tolerance=1e-6):
    """Independent strict pairwise geometric verification; no proxy score."""
    c = centers.detach().cpu().double()
    s = sizes.detach().cpu().double()
    if c.ndim != 2 or c.shape[-1] != 2 or c.shape != s.shape or len(c) == 0:
        raise ValueError(f"Expected nonempty matching [V,2] centers/sizes; got {c.shape}, {s.shape}")
    nonfinite = int((~torch.isfinite(c)).sum() + (~torch.isfinite(s)).sum())
    nonpositive_sizes = int((s <= 0).sum())
    if nonfinite or nonpositive_sizes:
        return {"is_legal":False,"nonfinite_values":nonfinite,"nonpositive_size_values":nonpositive_sizes,"overlap_pairs":None,"overlap_area_normalized":None,"out_of_bounds_macros":None,"max_boundary_excess":None}
    lo, hi = c-s/2, c+s/2
    n = len(c)
    inter = (torch.minimum(hi[:,None,:], hi[None,:,:]) - torch.maximum(lo[:,None,:],lo[None,:,:])).clamp(min=0)
    area = inter[:,:,0]*inter[:,:,1]
    overlap = (inter[:,:,0]>tolerance)&(inter[:,:,1]>tolerance)
    overlap = overlap & torch.triu(torch.ones(n,n,dtype=torch.bool),diagonal=1)
    out = ((lo < -1-tolerance)|(hi > 1+tolerance)).any(dim=1)
    return {"is_legal":bool(not overlap.any() and not out.any()),"nonfinite_values":0,"nonpositive_size_values":0,"overlap_pairs":int(overlap.sum()),"overlap_area_normalized":float(area.triu(diagonal=1).sum()),"out_of_bounds_macros":int(out.sum()),"max_boundary_excess":float(torch.maximum((-1-lo).clamp(min=0),(hi-1).clamp(min=0)).max())}


def physical_positions(centers, problem):
    centers = centers.detach().cpu().numpy()
    canvas = np.array([problem.canvas_size_x,problem.canvas_size_y])
    result={}
    for i,name in enumerate(problem.macro_names):
        size=np.array([problem.node_info[name]["x"],problem.node_info[name]["y"]])
        ll=(centers[i]+1)*canvas/2-size/2
        result[name]=[float(ll[0]),float(ll[1]),float(size[0]),float(size[1])]
    return result


def macro_net_hpwl(centers, problem):
    """Actual cache macro hypernet HPWL in cache physical units (not routed WL)."""
    pos=physical_positions(centers,problem)
    value=0.0
    for net in problem.net_info.values():
        pins=[]
        for name,pin in net["nodes"].items():
            llx,lly,w,h=pos[name]
            # Cache offsets are from macro center, matching graph conversion.
            pins.append([llx+w/2+pin.get("x_offset",0),lly+h/2+pin.get("y_offset",0)])
        if len(pins)>1:
            p=np.asarray(pins)
            value+=float(np.ptp(p,axis=0).sum())
    return value


def save_sample(out_dir, centers, noise, problem, metadata):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    centers=centers.detach().cpu(); noise=noise.detach().cpu()
    torch.save({"centers":centers,"noise":noise,"sizes":problem.cond.x.detach().cpu(),"macro_names":problem.macro_names,"design":problem.design_name},out_dir/"sample.pt")
    physical=physical_positions(centers,problem)
    (out_dir/"macro_physical.json").write_text(json.dumps(physical,indent=2))
    (out_dir/"cache_geometry.json").write_text(json.dumps({"design":problem.design_name,"canvas_size_cache_units":[float(problem.canvas_size_x),float(problem.canvas_size_y)],"ratio_x":float(problem.ratio_x),"ratio_y":float(problem.ratio_y),"grid":224,"coordinates_rounded":False,"macro_id_type":"DREAMPlace node ID; verify against DEF component ordinal","physical_json_schema":"macro_ID -> [lower_left_x,lower_left_y,width,height] in cache units"},indent=2))
    # Official convert_to_openroad representation, with its one-grid-unit offset.
    grid={name:((p[0]-problem.ratio_x)/problem.ratio_x,(p[1]-problem.ratio_y)/problem.ratio_y,p[2]/problem.ratio_x,p[3]/problem.ratio_y) for name,p in physical.items()}
    torch.save(grid,out_dir/f"{problem.design_name}.pt")
    metadata={**metadata,**legality(centers,problem.cond.x),"macro_net_hpwl_cache_units":macro_net_hpwl(centers,problem),"num_macros":len(centers),"num_graph_edges":problem.cond.edge_index.shape[1],"coordinate_system":"centers normalized to [-1,1]; macro_physical.json lower-left in original benchmark-cache units; design.pt uses official grid conversion"}
    (out_dir/"metrics.json").write_text(json.dumps(metadata,indent=2))
    return metadata
