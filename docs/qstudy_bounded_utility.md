# Prospective bounded measured utility

`fql/qstudy_bounded_utility.py` is a target-definition helper for new experiments.
It changes no existing actor, critic, outcome, native reward, dataset or job.

The first five native metrics are routed WL, signed WNS, signed TNS, horizontal
overflow and vertical overflow. Convert slack to positive violation magnitude
`max(-slack, 0)`; the other active metrics must already be nonnegative. With
fixed nonnegative weights and positive common TRAIN reference scales, define:

```text
C = sum_j weight_j * positive_cost_j / common_TRAIN_scale_j
successful bounded cost B = 2*C/(1+C), utility U = -B
confirmed geometry/physical failure B = 4, U = -4
```

Every finite measured success outranks every confirmed failure, including
successes beyond the observed TRAIN support. Numerically very large successful
costs can saturate at B=2; they still outrank B=4. This preserves deterministic
successful-layout ordering and local metric tradeoff ratios, but changes tail
emphasis, gradient magnitude and expected utility under a stochastic policy.
It is a new objective version, not an equivalent rescaling of the old reward.
Predicted Q can still extrapolate incorrectly; the measured ordering guarantee
does not guarantee feasibility or correct ranking for an unseen proposal.
The ordinary calculation uses float64 intermediate arithmetic. Rows whose
weighted products/sums overflow or whose positive products/terms become
subnormal or zero use log-sum-exp and `2*sigmoid(log(C))`; an overflowing
intermediate must not force saturation when the final normalized C is finite.

```python
from fql.qstudy_bounded_utility import make_spec, derive_cost_utility

spec = make_spec(
    weights=[1, .15, .35, .10, .10],
    scales=frozen_common_train_scales,  # five supplied numbers, never fit here
    reference={
        "split": "train",
        "dataset_sha256": common_reference_dataset_sha256,
        "train_rows_sha256": ordered_reference_train_source_ids_sha256,
    },
)
targets = derive_cost_utility(native_metrics, statuses, spec, metric_mask)
# targets['cost'] is B; targets['utility'] is -B. Admit only targets['valid'].
```

Native inputs are `[rows,5]` or `[rows,7]`; power and area are ignored in this
version. Optional masks must be boolean with five or seven columns. A success
needs every active native metric finite, observed and valid. Missing active
labels, infrastructure failures, unknown statuses and unresolved outcomes stay
NaN/invalid. Inactive NaN labels do not matter. Only already-confirmed native
`geometry_failure` or `physical_failure` receives the failure target, even when
PPA is unavailable. The helper does not certify the status or its native
provenance; the caller must validate original request/outcome evidence first.

`validate_spec` rejects changed versions, constants, metric order or unsupported
fields. `scale_identity` binds the exact scales and TRAIN reference independently
of weights; `assert_common_scale` rejects differing references/scales in matched
comparisons. `utility_identity` additionally binds the complete weights. Reuse
one fixed reference across compared data sources; each grouped-CV fold must use
only its own admitted TRAIN reference, never held-out labels.

For a future scalar critic, fit measured B directly and extract Q as the
negative ensemble mean in these units (plus only a separately declared optional
uncertainty heuristic). Do not divide by mean TRAIN B again. Both success and
failure targets, actor extraction and measured candidate selection must share
this spec. Do not apply `2*C/(1+C)` to an unrestricted critic prediction:
predicted negative C is outside this target definition and has a pole at -1.
Existing linear-utility checkpoints and the old source-scale adapter are not
automatically compatible. Preserve original `outcome.reward` and native metrics;
store this derived utility/spec separately in any prospective experiment.
