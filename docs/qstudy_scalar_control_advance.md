# Nonresident scalar-control advancement

This metadata helper binds the already sealed base `d86e2335`, continuation
`89404d72` and route `50d44c3a` manifests. It does not change their scientific
sources, requests, resources, noise streams, rewards, physical settings or
three-generation missing-only worker recovery. The fourteen phases are the
original three base phases, nine continuation phases and two route phases.

Each one-CPU/3GiB/ten-minute invocation checks the original receipts and
submission journals, calls the existing `stage_submit` for the first unfinished
phase, schedules one short callback, then exits. A live worker array is followed
by `afterany` on that exact job. Accounting delays use a two-minute deferred
callback, limited to six consecutive delays; there is no sleeping controller.
The complete callback inventory is bounded at 128 allocations. Failed scientific
workers and ambiguous submissions remain unresolved with their saved evidence;
they are never converted into physical failure labels or silently resubmitted.

The previously queued train1 submitter1800443 is checked after base completion.
A normal live submitter is awaited. Only its exact obsolete PENDING
DependencyNeverSatisfied state on failed original1800396 may be ignored; its
existing submission lock still prevents duplicate train1 arrays. No job is
cancelled by this helper. Base native0 recovery therefore remains possible even
when the old afterok barrier cannot run.

The exact original443 request/submission JSON files bind its command, manifest,
phase and allocated job. Live `scontrol` output is used only for fields it really
exposes. The sole scheduler compatibility adjustment is local: an exact
`squeue` invalid-job-id error falls back to the original `sacct` interpretation.
Private function namespaces supply it to every original metadata submitter;
no frozen module globals or sources are changed. Other scheduler errors fail.

Callback submission uses a lock, atomic journal, immutable intent before sbatch,
and exact unique-name adoption after an interruption. A rejected dependency
race can use a bounded delayed callback; an uncertain sbatch response cannot
authorize a second allocation. Callbacks may be requeued by Slurm, while each
worker retains its original policy. Inherited restart counts are removed from
new callback submission environments.

`prepare` writes the new source snapshots and admission only; it submits no job.
After review, launch the wrapper's `run --admission-sha256 <prepared SHA>` once,
without an initial dependency. The worker checks the admission, sources and all
three old manifests again. The journal is under the study's `advancement_v1`.
This helper supplies scheduling, not new evidence of model quality or PPA.
