# Frozen public DREAMPlace 4.1 Table 2 generation plan

Six designs, one fixed seed (1000), original per-design halo values, the same public BB/two-stage algorithm. `tasks.jsonl` records candidate generation commands; it is **not** a backend evaluation task file and is not consumed directly by `run_table2.py`. No jobs are submitted by creating this plan.

`bp_fe` uses the separately authorized v2 smoke as its predeclared result. Do not submit it twice. The other five remain pending. Maximum two concurrent generation jobs; each uses the existing Slurm wrapper, 8 CPUs, 24 GiB and four hours. The frozen source DEF hashes come from prior independent geometry audits and each run records freshly computed input hashes; compare these before accepting output. Config snapshots are under `author_configs/`.

A run is usable only after instance/master, DEF net/pin, both macro-classifier, finite/halo restoration, core/die/nonoverlap, stock orientation and reloaded pin-offset audits pass. Preserve all failure records. Do not tune based on PPA or replace the public optimizer/legalizer.

For each successful result, create a separate backend task with `method: DREAMPlace`, handoff DEF path/SHA, `generation_manifest` and its SHA, and `geometry_audit` pointing to `handoff_audit.json` (compatible `design/output_sha256/legal_no_halo` schema). Use the same final ORFS protocol as the six FlowPlace results. Final timing, routed wirelength, power and area come only from that full backend flow.

CPU outputs and timing cannot be assumed equal to the paper's GPU run. Missing original input and exact author settings remain reported limitations. Run storage is under `/fact_data`, scratch under `/scratch`; keep artifacts and check quota headroom before each backend submission. No selection by paper proximity, hidden cleanup, or silent failed-design exclusion.
