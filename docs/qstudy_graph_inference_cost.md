# Graph inference cost protocol

The graph cost measurement uses the completed bp_fe main experiment, with three
independent training seeds and fixed round2 checkpoints. The labels are
`bc_r2_actor1`, `fql_frozen_r2_actor1`, and `common_behavior_euler10`. Both actors
have completed 1000+600+600 updates. The teacher is each seed's unchanged
`common['flow']`; its state is checked against both round2 checkpoints.

This does not measure the released FlowPlace20-hard sampler or reproduce a
paper speed claim. That sampler applies intermediate hard constraints and
clipping, whereas this teacher takes ten explicit Euler steps, restoring fixed
nodes after every step and clipping coordinates at the end. The one-step actor
uses its original clipping and fixed-node restoration.

Every method uses the same eight saved float32 CUDA-generated development noise
arrays from seed4600000. No random noise is generated, and fresh85000000 is not
used. Batch size is one. Each scope and method has ten untimed warmups and32
timed calls, cycling through all eight arrays four times. These are repeated
measurements of eight layouts, not32 independent layouts or training seeds.
Methods are interleaved, rotating their order deterministically on each
repetition; scopes run model-only, legalizer-only, full generation, then the
next seed. All inputs remain on the GPU throughout timing.

The three scopes have explicit boundaries:

- `model_only`: graph generation through the original coordinate clipping and
  fixed-node restoration, with no critic or external legalizer.
- `legalizer_only`: the corresponding method's generated raw coordinates
  through `baseline.api.legalize`, including its fresh Legalizer constructor
  on every call, with the original grid128, padding0.001, w_reg0.1 and w_dist1.
- `full_generation`: an actual consecutive model and legalizer call, ending at
  legal coordinates resident on the GPU. It is not computed by adding stage
  medians. D2H, geometry checks, hashing, DEF, file I/O and physical tools are
  excluded from all three scopes.

The clock is host `perf_counter`, with CUDA synchronization immediately before
and after each timed call. A real forward hook on `model._reverse_model` checks
1/10 NFE for actor/teacher, zero for the legalizer, and1/10 for full generation.
Hooks, output hashes, deterministic repeated-output checks and legality/fixed
node checks run outside timing. FlowMatchingModel overrides `__call__`, so a
hook placed on its outer object would not supply valid NFE evidence.

All eight output geometries, coordinate hashes and differences from the saved
export are reported. Invalid geometry is retained without selecting another
noise. The fixed-node count is explicit; count0 does not demonstrate handling
of a nonempty fixed-node set. The module verifies that model state is unchanged.
Loading costs, the optional diagnostic Q, compilation, noise sampling, transfer
and backend costs are excluded. No torch.compile is added or disabled.

Prepare runs in a bounded Slurm CPU allocation and writes one immutable plan
and source snapshots. It takes checkpoint digests from the existing completed
metadata, without loading or hashing bulk weights. The GPU run rehashes and
loads the six actual round2 checkpoints and three common checkpoints, verifies
their graph partition and complete binding, and strictly loads all policies
through the same GraphPolicy constructor. Python major.minor ABI and exact
PyTorch/NumPy/PyG/CUDA/executable identities are checked; prepare and measurement
both record their complete Python patch versions. All compared methods execute
in the same GPU allocation. The L20 GPU type, float32 precision, deterministic
settings, two PyTorch threads and actual device/driver details are recorded.
Model construction uses the pinned
public initialization checkpoint, then loads the selected state strictly.

After source review, the entry points are:

```bash
# Bounded CPU Slurm allocation, with no GPU needed:
.venv/bin/python -m analysis.qstudy_graph_inference_cost prepare --output PLAN_DIR

# GPU Slurm wrapper; use the producer's printed plan SHA:
sbatch analysis/qstudy_graph_inference_cost.slurm run \
  --plan PLAN_DIR/plan.json --plan-sha256 PLAN_SHA --output NEW_RESULT_DIR
```

The run writes per-call CSV, all output audits, and per-seed timing summaries.
The main mean and sample SD are calculated from the three seed means. Repeated
call variation within a seed is reported separately and is not uncertainty
over32 independent models. This measures inference cost, not placement quality.
