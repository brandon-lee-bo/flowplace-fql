# Frozen released FlowPlace teacher

The released teacher uses the public checkpoint with SHA256
`0fe61ef666c7ff34c32b1ea1b8fe23a1c8b45151955f0f16366a121fde5fd359`
and the original **20-step constrained sampler**. It stays in evaluation mode,
with gradients disabled, no FM warmup and no offline/online weight updates.

`ReleasedFlowPlaceTeacher` passes each student noise unchanged to
`baseline.api.generate`. Upstream performs intermediate hard legalization;
the wrapper adds no output clamp, extra final legalization or Euler substitute.
Fixed graph nodes remain unchanged. The public sampler supports B=1, so a
minibatch is sampled serially. Distillation can be expensive even though the
student needs one forward at inference.

## Configuration and integration

New released-teacher configs explicitly require:

```json
{
  "flow_policy": "released_flowplace",
  "flow_steps": 20,
  "flow_checkpoint_sha256": "0fe61ef666c7ff34c32b1ea1b8fe23a1c8b45151955f0f16366a121fde5fd359",
  "flow_warmup_steps": 0,
  "online_flow_update": false
}
```

The base, expanded-data and second-design graph trainers use the same factory.
Only actor and Q update in the released branch. `qstudy_config` defaults to the
released teacher, rejecting an online-flow-update request; it loads **student
weights only** from its audited common checkpoint and ignores the legacy common
flow weights. Its student initialization differs from a fresh V1.1 distillation
run, and those experiments must be distinguished.

The flow optimizer field remains in saved checkpoints for schema compatibility
but is inert. Resume rejects replacement teacher tensors, changed geometry,
wrong checkpoint SHA and incompatible settings. Untagged configs and explicit
`legacy_euler` retain historical semantics; their old results are not public20
results. Source-bound historical jobs still require their own archived code.

For a fresh graph experiment, preserving the source config:

```bash
python -m tools.prepare_released_flowplace \
  --source /path/to/original_graph_config.json \
  --output /path/to/new_experiment \
  --run-prefix NEW_UNIQUE_PREFIX
```

The helper writes JSON only and refuses to reuse an existing output directory.
The matched V1.0/V1.1 experiment has a separate entrypoint and
[example config](../examples/placement_versions/v11_matched.json); follow the
[parameter/reproduction guide](placement_versions_reproduction.md).

## Validation and quality boundary

Historical CPU job1830686 and GPU job1830674 verified actual public weights,
same-noise output equality (`atol=0, rtol=0`), fixed nodes, teacher freezing,
real actor/Q updates and checkpoint reload. Publication CPU job1849715 reran
the teacher, sequential and actor-orchestration suite: all57 tests passed,
including actual public-weight and conditional-prefix sampler tests.

V1.1 seed0 has since completed both matched BC/FQL training arms, with zero
teacher tensor changes. Its matched backend quality comparison remains
incomplete. Correct sampling and successful optimization do not imply a PPA
gain. The V2 conditional-prefix teacher is a separate adapter and does not
inherit the unchanged-sampler quality claim; see [the results](placement_versions.md).
