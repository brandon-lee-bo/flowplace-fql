> 当前 Graph Flow-Q offline-to-online 研究见 [最新实验记录](graph_fql_offline_to_online.md) 与 [完整目标核验](completion_audit.md)。下文保留早期实验和任务快照，其中的“已完成”“尚未启动”只适用于对应历史阶段，不能作为当前目标或队列状态。

# Pipeline end state

2026-10-17, Hong Kong. All planned physical experiments are terminal; no
physical task remains pending. Computation used FactLab Slurm throughout.

- Released Table2 DEFs:5fullflow successes,1verified raw-DEF macro-overlap
  failure(swerv). bp recovered its unchanged cap4 flow after preemption.
  Summary1785253 intentionally exits1 and retains all6rows. This is not an
  exact numerical Table2 reproduction.
- RealGRT labels:96/96 independently verified. Critic gate1784156 failed0/3;
  the conditional GRT actor and48+24 backend expansion were not launched.
- Macro-proxy transfer:36outcomes,35success+1verifiedCTSfailure.
- Sampler transfer:11outcomes,10success+1fixed-zeroCTSfailure. One successful
  same-input infrastructure retry is separately recorded without enlarging
  the denominator. Aggregate1785491 and plots1785496 were independently checked.
- Detailed-route-cap diagnostics, generatedbp_fe/swerv fullflows,
  deterministic inference, teacher sensitivity, Euler BC learnability and
  generation-plus-legalization latency are complete and explicitly scoped.
- Actual result/audit binding regression1785504 completed successfully.
- Final Slurm allocation snapshot:1785509,analysis/artifacts/cost_final_v1.

Scientific outcomes and limitations are in results_report.md; proposals for
future experiments are in research_interpretation.md. These suggestions are
not queued jobs or unfinished requirements of this pilot. Do not automatically
launch the conditional actor after the failed critic gate.

Original attempts, physical failures, infrastructure interruptions and retry
maps remain intact. No unrelated project's jobs or files were changed.
