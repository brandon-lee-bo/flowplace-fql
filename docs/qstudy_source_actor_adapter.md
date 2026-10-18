# Reviewed source critics to actor configs

`fql/qstudy_source_actor_config.py` prepares immutable configs only, under a
Slurm CPU allocation. It neither selects a winner nor launches actor/backend
jobs. Supply a completed source-study `summary.json`, a matched base actor
config, and a dated JSON review note:

```json
{
  "date": "2026-10-18",
  "reason": "State the reviewed source/critic mechanism and prediction evidence",
  "source_report_sha256": "SHA256_OF_COMPLETED_SUMMARY",
  "reference_weights": [1, 0, 0, 0, 0],
  "selections": [
    {"source_arm": "O64", "variant": "legacy", "actor_arm": "o64_legacy", "run_prefix": "o64leg"},
    {"source_arm": "T64", "variant": "legacy", "actor_arm": "t64_legacy", "run_prefix": "t64leg"}
  ],
  "diagnostic_review_provenance": [
    {"role": "direction_probe", "path": "/absolute/path/to/reviewed/probe/report.json", "sha256": "SHA256"}
  ]
}
```

The adapter requires all36 source tasks and their four fits to be complete,
revalidates their bindings, and checks selected refit model buffers against
all and only that source's TRAIN rows. Initial TRAIN labels must be finite
and successful; failed rows cannot silently disappear at actor setup. The
original256 behavior targets and families must match exactly. T64/M64 still
share one seed0 teacher; three actor seeds do not create independent sources.

The reference physical utility uses O64 TRAIN scales. Legacy `q_weight` is
`S_source/S_O64`; multi-metric weights are `w_ref*S_source/S_O64` with
`q_weight=1`. Alpha remains10. Input/target scales and native reward semantics
are unchanged. No reward standard deviation is multiplied a second time.

All prepared legacy configs explicitly set `online_rounds=0`. The frozen actor
uses a failure target of-4 in source-normalized legacy Q units, so the corrected
actor weight would make its failure utility `-4*S_source/S_O64`. This generally
differs from common-utility-4. A future tested actor implementation must use
failure Q target `-4/q_weight` and preserve complete version bindings before
legacy source comparisons enter online replay. Multi-metric corrected weights
with `q_weight=1` retain the frozen-4 utility target. Config preparation alone
does not authorize or demonstrate online improvement.

```bash
sbatch --partition=cpu-scavenger --qos=cpu-scavenger \
  --exclude=natural-selection,enterprise --cpus-per-task=2 --mem=8G --time=00:10:00 \
  --wrap='.venv/bin/python -m fql.qstudy_source_actor_config \
    --study /absolute/source_study \
    --source-report /absolute/source_study/report/SNAPSHOT/summary.json \
    --base-config /absolute/matched_base/config.json \
    --review-note /absolute/review.json --output /absolute/new_config_root'
```

Run from the project directory, with temporary files directed to allocated
scratch. `preparation.json` is the final preparation gate. Source data,
checkpoints, configs, validation labels and direction probes are never rewritten.
