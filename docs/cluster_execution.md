# FactLab execution

User explicitly requires Slurm scheduling (2026-10-17 local time).
Inspected `/etc/slurm/slurm.conf`, `cli_filter.lua`, `task_prolog.sh`,
`scontrol show partition`, and user QOS association. We have not been given a
separate prose policy document, so do not claim to have audited one.

All training, inference, physical evaluation, and substantial build work must
run through `sbatch`/`srun`, with CPU, RAM, time and GPU requests explicit.
Use dev for short GPU development (maximum 12h), eda for EDA backend work,
or matching cpu-scavenger/gpu-scavenger partition+QOS for preemptible jobs.
Honor Slurm CUDA_VISIBLE_DEVICES and container containment variables; do not
override them to select unallocated GPUs. Set numerical library threads to
SLURM_CPUS_PER_TASK. Never cancel another project's jobs or touch its files.
Record job IDs, scripts, logs, exit codes and elapsed times for our jobs.

Lightweight source inspection, editing, downloads, package preparation and
scheduler queries are performed from the interactive session. Initial host
GPU query found all eight H20 GPUs busy; no inference/training was run there.
