"""Real graph forward/backward, terminal-target and snapshot reload checks."""
import copy
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from analysis.placement_metrics import MacroHPWL, legality
from baseline.api import RUNTIME, load_problem, legalize
from .critic_gate import load_train_validation
from .graph_o2o_models import GraphPolicy, FeatureCritics, actor_objective


def main():
    if "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("Submit the GPU Slurm smoke")
    torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261018)
    problem = load_problem("bp_fe", "cuda")
    data, _, _ = load_train_validation("datasets/bp_fe_grt_v1/dataset.npz")
    x = torch.tensor(data["train"]["raw_action"], device="cuda").reshape(-1, len(problem.x), 2)
    y = torch.tensor(data["train"]["reward"], device="cuda")
    cache = torch.load(RUNTIME / "benchmark_cache/bp_fe.pt", map_location="cpu", weights_only=False)
    flow = GraphPolicy(problem)
    actor = copy.deepcopy(flow)
    critic = FeatureCritics(cache, x, y, actor.movable).cuda()
    z = 2 * torch.rand(4, len(problem.x), 2, device="cuda") - 1
    start = time.perf_counter()
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-5)
    before = {n: p.detach().clone() for n, p in actor.named_parameters()}
    loss, distill, q = actor_objective(actor, flow, critic, z, flow_steps=2)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad for p in actor.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert not any(p.grad is not None for p in critic.parameters())
    optimizer.step()
    changed = sum(not torch.equal(before[n], p) for n, p in actor.named_parameters())
    assert changed > 0
    fl = flow.flow_loss(x[:4], z, torch.rand(4, device="cuda"))
    fl.backward()
    assert any(p.grad is not None for p in flow.parameters())
    # Terminal target is measured reward; no fictitious flow-step transitions.
    qloss = (critic(x[:4]) - y[:4, None]).square().mean()
    qloss.backward()
    with torch.no_grad():
        raw = actor(z)
    assert torch.equal(raw[:, ~actor.movable], actor.fixed[~actor.movable].expand(4, -1, -1))
    metric = MacroHPWL(problem.benchmark_info)
    executed = legalize(raw[0], problem).cpu().numpy()
    geometric = legality(executed, metric.sizes)
    assert geometric["legal"], geometric
    out = Path("/fact_data/bohanli/flowplace_fql") / f"graph_o2o_smoke_{os.environ['SLURM_JOB_ID']}"
    out.mkdir(exist_ok=False)
    torch.save(actor.state_dict(), out / "actor.pt")
    clone = copy.deepcopy(actor)
    clone.load_state_dict(torch.load(out / "actor.pt", weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(actor(z), clone(z), atol=1e-4, rtol=1e-4)
    report = dict(status="passed", slurm_job_id=os.environ["SLURM_JOB_ID"],
                  nodes=len(problem.x), edges=problem.cond.edge_index.shape[1],
                  actor_parameters=sum(p.numel() for p in actor.parameters()),
                  changed_parameter_tensors=changed, loss=float(loss), distill=float(distill),
                  q=float(q), flow_loss=float(fl), critic_loss=float(qloss),
                  elapsed_seconds=time.perf_counter()-start, geometry=geometric)
    (out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
