# Fresh confirmation launch checklist — not a launch or selection

Use only after the root reviews the completed declared development matrices.
The default comparison is legacy frozen round2 versus BC frozen round2, with
current common flow and released public FlowPlace references. A negative result
is valid. This checklist creates no lock, noise, models or physical results.
See [the protocol](qstudy_fresh_confirmation_protocol.md) and the exact
[prospective rules](../fql/qstudy_confirmation_rules_v1.json).

## Required before the first builder invocation

- Review all declared main/source/bounded/flow-update results and failures; fix
  one candidate/round for all training seeds0,1,2. Do not select a best seed,
  checkpoint or output. Record why the default anchor is retained or replaced.
- Bind the complete inspected development inventory, including old noise
  900000/4300000/4500000/4600000/5300000, predictive/direction/selection studies,
  final native reports, BC equivalence evidence, and the rules JSON above.
- Verify every original selected checkpoint, completion/binding and replay
  parent chain. Default BC/FQL each has1000+600+600 actor updates and16 own
  collected labels per seed; frozen teacher/common initial states are matched.
  Generic chain verification does not certify a new method's special adapter:
  source/bounded/flow-updated choices need their original preparation, scale,
  source closure and matched-control limitations explicitly reviewed too.
- Prepare a standalone `common_train_utility_reference` JSON containing either
  one validated bounded utility spec or
  `{"version":"fresh_common_bounded_utilities_v1","utility_specs":{...}}`.
  Copy only already-saved full-TRAIN specs, for example
  `bounded_screen_v1/{wl,timing,congestion}/seed_0/summary.json` →
  `historical_validation.utility_spec`; verify their common384-TRAIN scales.
  The whole screen summary is **not** a valid utility-reference file. Fresh
  metrics must never fit scales. Physical WL remains the primary endpoint.
  The standalone inventory has now been prepared by Slurm job 1797614 at
  `fresh_common_utility_reference_v1/common_train_utilities.json` under
  `BASE`, SHA256
  `55de0bcbbc9129384d01879f7d42eed1c59366f8f6488604214ac9cf1cb8500d`.
  Its sibling `completed.json` binds all nine original summaries and the
  extraction script. This metadata preparation is independent of candidate
  selection; all three objective specs use the unchanged scales from 384
  training rows.
- Confirm quota/headroom for exports, atomic writes and retained evidence;
  scratch backend outputs require verified durable preservation. Lock actual
  GPU name/software/precision, graph/cache, DEF/LEF and physical-flow revision.
  Do not change frozen sources after locking; the builder also seals native
  and DRT source closures. It does not automatically bind the rules JSON:
  include that file explicitly in `development_inventory`.

Known default source configs, relative to
`BASE=/fact_data/bohanli/flowplace_fql/qstudy_v1`:

| Request field | Exact source; selected original round |
| --- | --- |
| `candidate.source_config` | `actors_v1/legacy_frozen/config.json`;2 |
| `bc.source_config` | `actors_v1/bc_frozen/config.json`;2 |
| `behavior_flow.source_config` | `flow_eval_v1/config.json`;common0 |
| `public_reference.source_config` | `public_flow_eval_v1/config.json`;public0 |
| `native_reference_config` | `actors_v1/legacy_frozen/config.json` |
| `candidate_is_legacy`, `legacy_anchor` | `true`, `null`; no duplicate anchor |

Every file entry is a canonical absolute `{path,sha256}` binding. Fill explicit
scientific reason, endpoints, failure rules, training-budget comparison,
inventory and `scientific_selection_approved_by_caller=true` in the builder
request. `template` supplies the schema; it is deliberately unapproved.

## Commands, only after the decision is reviewed

Run from `/fact_home/bohanli/RL_Placement/experiments/flowplace_fql`.
`WORK`, `REQUEST` and new canonical `FRESH` paths below are caller choices;
do not point them at development data or an existing declared evaluation.
Record every returned job ID and authoritative `sacct` exit status. Do not
resubmit these submission loops after an interruption without checking that
journal and immutable completion markers first.

```bash
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
BASE=/fact_data/bohanli/flowplace_fql/qstudy_v1
WORK="$ROOT/logs/fresh_confirmation_review_v1"
REQUEST="$WORK/reviewed_request.json"
FRESH="$BASE/fresh_confirmation_v1"
NATIVE="$BASE/fresh_native_finish_v1"
ROUTE="$BASE/fresh_route_finish_v1"

# Template prints JSON in the job log; fill/review REQUEST separately.
sbatch fql/qstudy_cpu_stage.slurm fql.qstudy_fresh_lock template

REQUEST_SHA=$(sha256sum "$REQUEST" | cut -d ' ' -f1)
sbatch fql/qstudy_cpu_stage.slurm fql.qstudy_fresh_lock build \
  --request "$REQUEST" --sha256 "$REQUEST_SHA"
```

After that build completes successfully, inspect the selection and its exact
source/config/checkpoint inventory. The builder only writes metadata. Then:

```bash
SELECTION="$FRESH/selection.json"
SELECTION_SHA=$(sha256sum "$SELECTION" | cut -d ' ' -f1)
VALIDATE=$(sbatch --parsable fql/qstudy_cpu_stage.slurm \
  fql.qstudy_fresh_eval --stage validate \
  --selection-lock "$SELECTION" --selection-sha256 "$SELECTION_SHA")
DECLARE=$(sbatch --parsable --dependency="afterok:$VALIDATE" \
  fql/qstudy_cpu_stage.slurm fql.qstudy_fresh_eval --stage declare \
  --selection-lock "$SELECTION" --selection-sha256 "$SELECTION_SHA")

# Generic module wrapper; explicit GPU overrides, maximum two export jobs.
# This is the DEFAULT four-stream plan only, not an automatic candidate picker.
lanes=("$DECLARE" "$DECLARE"); exports=(); i=0
for stream in candidate bc behavior_flow public_reference; do
  for seed in 0 1 2; do
    if [[ "$stream" == public_reference && "$seed" != 0 ]]; then continue; fi
    lane=$((i % 2))
    job=$(sbatch --parsable --dependency="afterok:${lanes[$lane]}" \
      --job-name="fresh_${stream}_s${seed}" \
      --partition=gpu-scavenger --qos=gpu-scavenger \
      --gres=gpu:nvidia_l20:1 --cpus-per-task=2 --mem=20G --time=00:30:00 \
      --exclude=natural-selection,enterprise,blue-space \
      --export=ALL,CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      fql/qstudy_cpu_stage.slurm fql.qstudy_fresh_eval --stage export \
      --selection-lock "$SELECTION" --selection-sha256 "$SELECTION_SHA" \
      --stream "$stream" --seed "$seed")
    exports+=("$job"); lanes[$lane]="$job"; i=$((i + 1))
  done
done
EXPORT_DEPS=$(IFS=:; printf '%s' "${exports[*]}")
sbatch --dependency="afterok:$EXPORT_DEPS" fql/qstudy_cpu_stage.slurm \
  fql.qstudy_fresh_native prepare --selection "$SELECTION" \
  --selection-sha256 "$SELECTION_SHA" --output "$NATIVE"
sbatch --dependency="afterok:$EXPORT_DEPS" fql/qstudy_cpu_stage.slurm \
  fql.qstudy_fresh_route prepare --selection "$SELECTION" \
  --selection-sha256 "$SELECTION_SHA" --output "$ROUTE"
```

Both preparations require **all ten** completed export inventories. They seal
80 GRT requests and40 DRT requests before physical measurement; DRT indices0--3
were already locked before noise export. No policy, critic or optimizer trains.
After both preparations pass, hash the sealed plans and submit existing workers:

```bash
NATIVE_SHA=$(sha256sum "$NATIVE/plan.json" | cut -d ' ' -f1)
ROUTE_SHA=$(sha256sum "$ROUTE/plan.json" | cut -d ' ' -f1)
GRT_JOB=$(sbatch --parsable fql/qstudy_fresh_native.slurm controller \
  --plan "$NATIVE/plan.json" --plan-sha256 "$NATIVE_SHA")
DRT_JOB=$(sbatch --parsable --dependency="afterok:$GRT_JOB" \
  fql/qstudy_fresh_route.slurm controller \
  --plan "$ROUTE/plan.json" --plan-sha256 "$ROUTE_SHA")
sbatch --dependency="afterany:$GRT_JOB" fql/qstudy_readonly_report.slurm \
  -m fql.qstudy_fresh_report --selection "$SELECTION" \
  --selection-sha256 "$SELECTION_SHA" --output "$BASE/fresh_grt_report_v1"
sbatch --dependency="afterany:$DRT_JOB" fql/qstudy_readonly_report.slurm \
  -m fql.qstudy_fresh_route_report --plan "$ROUTE/plan.json" \
  --plan-sha256 "$ROUTE_SHA" --output "$BASE/fresh_drt_report_v1"
```

| Stage | Existing resource contract |
| --- | --- |
| Metadata/hash validation/preparation | CPU2,8G,30min; Slurm |
| Exports | L20×1,CPU2,20G,30min; at most2 simultaneous above |
| GRT controller / workers | CPU1,3G,24h / CPU8,24G,2h; at most4 workers |
| DRT controller / workers | CPU1,3G,24h / CPU8,12G,2h30; at most2,cap20 |
| Read-only reports | CPU2,8G,20min; no inference or EDA |

## Completion and interpretation

The default has candidate24/12, BC24/12, common flow24/12, public8/4 GRT/DRT
logical requests: **80/40 total**, excluding real retry costs. Public is one
checkpoint with eight noise outputs, not three trained replicates. A new
candidate would require an additional three-seed legacy anchor and explicitly
revised104/52 budget before locking; do not silently reuse this default loop.

Verify all actual shared float32 CUDA85000000 noise hashes, raw/executed/DEF
bindings, seven GRT metrics and six final DRT metrics, failure/missing counts,
per-seed means/sample SD, paired differences and unique physical-call ledger.
Reports do not automatically apply the rules or select a winner. A successful
controller exit may account confirmed failures; it does not mean every layout
succeeded or DRC0. Unknown/infrastructure failures stay unresolved. Recovery
uses the same immutable selection/plan and missing-only genuine interruption
logic; never loosen guards, change metrics, add candidate-only repairs, replay
fresh labels, or regenerate a favorable output. Any later method change needs
a separately locked confirmation stream.

After both authoritative reports have completed, apply the already locked
rules with the saved-report-only reducer. Choose their exact immutable paths
from the recorded report jobs; never choose a report because its numbers are
favorable. `candidate` and `bc` below are the default locked stream IDs, not
new method selection. Keep the output outside the immutable export and DRT
roots.

```bash
GRT_REPORT=/absolute/path/from/completed/fresh_grt_report_job/report.json
DRT_REPORT=/absolute/path/from/completed/fresh_drt_report_job/report.json
RULES="$ROOT/fql/qstudy_confirmation_rules_v1.json"
sbatch fql/qstudy_readonly_report.slurm -m fql.qstudy_fresh_acceptance \
  --grt "$GRT_REPORT" --grt-sha256 "$(sha256sum "$GRT_REPORT" | cut -d ' ' -f1)" \
  --drt "$DRT_REPORT" --drt-sha256 "$(sha256sum "$DRT_REPORT" | cut -d ' ' -f1)" \
  --rules "$RULES" --rules-sha256 b6553120d9a4f410740d9f45ee3214dfa62b48f27b0a75a8c4cfe262f9b940ab \
  --candidate candidate --bc bc --output "$BASE/fresh_acceptance_report_v1"
```

The reducer rechecks the report, selection, declaration, noise and rules
bindings and recomputes the native metrics from individual requests. It has
no model, EDA, fitting or selection calls. It reports all locked streams'
success/failure/missing counts, complete candidate/BC seed means and paired
primary differences. Its operational labels are `inconclusive`,
`no_stable_gain`, `tradeoff` and `overall_improvement`; the full metric
reports remain necessary, including current-flow and public-reference
comparisons. Passing the BC gates is not a claim of superiority to FlowPlace.
