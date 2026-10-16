# Measured deployment latency

Slurm job **1784344** completed in 45 seconds on an NVIDIA L20 with four CPU threads. Results are in `artifacts/deployed_latency_v1/report.json`; individual calls, output arrays, checkpoint hashes, configuration and startup source copies are preserved alongside it.

The benchmark starts with a float32 noise vector already on the GPU and ends with legal macro coordinates on the GPU. It includes reshaping, actual inference, clipping/masks, legalizer construction/execution where needed, and CUDA completion synchronization. The official hard teacher already includes its legalizer: no extra projection is added. Each one-step student includes one final projection.

| Method | Warm mean per layout |
|---|---:|
| Official hard FlowPlace teacher | 637.291 ms |
| BC, mean of three training-seed means | 9.884 ms |
| FQL alpha 1, mean of three training-seed means | 9.874 ms |

The same fixed heldout noises 240–255 were timed three times for each method, after five untimed warmup calls. Methods were interleaved in cyclic order. All three saved BC and FQL training seeds were included; neither methods nor inputs were selected from these timings. Repeated timing variation and variation between training-seed means are reported separately. The tiny BC/FQL timing difference is not evidence of an algorithmic speed difference.

Every timed final layout matches its previously saved deterministic evaluation output exactly, and passes independent overlap/boundary checks. Student batch-one raw outputs differ from the earlier batch-256 outputs by at most `5.662441253662109e-7`, but final legalizer results remain identical. These raw differences are recorded rather than hidden.

Teacher loading took 0.427383 seconds and its first call took 2.734719 seconds, recorded separately from the warm timings. First-call measurements use the existing environment and compiler caches; caches were not emptied, so this is not a clean-build compilation benchmark. Student first calls are also retained, including the initial 33.974 ms call versus roughly 9.7–9.8 ms for subsequent loaded actors.

Loading, warmup, noise creation, host transfers, geometry validation, DEF export, the physical backend, and data/training costs are excluded from the warm path. The result demonstrates faster placement generation under this narrow deployment scope. It does not demonstrate a comparable speedup of the entire EDA flow, which still performs the expensive backend evaluation. No weights, quality selection, or backend jobs were changed.
