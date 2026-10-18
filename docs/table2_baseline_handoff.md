# Table 2 baseline：后续替换算法的接口

本页描述固定接口，不是实时进度。当前目标仅为六个 OpenROAD 电路上的
FlowPlace 与 DREAMPlace，后端是 OpenROAD fullflow；不包含 Innovus、Table 3
消融或新的 Flow-Q 实验。

## 新发现：原流程宏导入不具备完整合法性保证

2026-10-17，原生轨道对齐在原本合法的 DREAMPlace swerv 输入上引入一对
0.07 µm 重叠。原流程继续完成并保留实际结果，但尚不能当作宏几何全部合格的
后续 baseline。独立的统一精确坐标导入队列已为两方法/六设计冻结，见
`table2_exact_import_protocol.md`。它保留全部原作者宏命令，再用 OpenDB
恢复输入整数 DBU 坐标及原生 LOCKED 状态；不是旧工具不支持的 `-exact`。
两组结果必须分开，不能按 PPA 好坏挑选。后续接口需等该队列通过最终 DEF
逐 DBU 对应输入及无重叠/越界检查，再明确采用兼容修正版本。

## 当前冻结入口

| 部分 | 输入或记录 |
| --- | --- |
| FlowPlace 生成 | `baseline/artifacts/official_stream_seed300/run_manifest.json`；发布 checkpoint、全局 seed300、20步、原 hard-constraint sampler |
| FlowPlace 后端 | `backend/plans/table2_source_era_flowplace_v1/tasks.jsonl`；抢占重跑分别在 `backend/plans/table2_source_era_flowplace_retry1/`、`backend/plans/table2_source_era_flowplace_retry2/` |
| DREAMPlace 生成 | `/fact_data/bohanli/flowplace_fql_backend/dreamplace_table2/cohort_1788062/`；统一 H20、seed1000、public4.1.0 BB/两阶段与完整合法化 |
| DREAMPlace 后端 | `backend/plans/table2_source_era_dreamplace_v1/tasks.jsonl` |
| 共用后端 | OpenROAD `v2.0-17598-ga008522d8`、作者发布 flow/库/综合 seed/SDC、8线程、发布 GPL0/0 与 route cap4 |
| 比较与审计 | `analysis/run_table2_method_audit.py` → `analysis/summarize_table2_methods.py` |

DREAMPlace 有一项完整记录的单行 Abacus 浮点行索引修复。六个设计都使用同一
修复安装；之前成功的五个设计，其 handoff SHA 与修复前相同。作者的独立
DREAMPlace baseline 完整配置、原 seed 和 strong-legalizer 对应关系仍未匹配；
这些限制见 `table2_dreamplace_protocol.md`，不能省略。

补充的原生 OpenROAD 连接核查已完成（jobs1788111、1788116–1788120）：
六个发布 DEF 与各自缓存的 `1_synth.v`，逐一比较 leaf instance/master 和
完整信号网端点分组，全部相同；比较忽略 net 名字，不忽略引脚或顶层端口。
审计没有执行综合或优化。证据在
`/fact_data/bohanli/flowplace_fql_backend/table2_reports/connectivity_v1/summary.json`。
这是结构连接核查，不是功能形式等价证明；早期 manifest 中只完成 inventory
检查的历史描述保持原样。

## 算法与后端的边界

后续替换生成算法时，产出 **DEF 中合法的宏位置与方向**，并保留：

- 完整且一致的 macro instance/master 名称、尺寸、单位和连接；不能改变 netlist。
- 生成配置、seed、模型/源码标识、实际硬件与生成耗时的 manifest。
- 绑定该 DEF SHA256 的独立几何审计：宏无重叠、无越界。宏合法与 routed DRC
  是两种检查，不能相互代替。

一个后端 task 至少绑定 `method`、`design`、`run_id`、`def_path`、`def_sha256`，
并建议完整绑定 `generation_manifest` 与 `geometry_audit` 的路径和 SHA256。
参考现有 DREAMPlace task，不要重新手写另一套时钟、库或后端参数。

作者后端的 `run_mp` 从 DEF 读取宏位置，`run_wo_synth` 使用发布综合产物重做
标准单元放置和后续物理实现。因此生成器输出的标准单元位置不是最终布局。
修改生成器时应保持这个边界；否则不再是当前 baseline 的受控算法比较。

此外，发布流程通过原生 `place_macro` 执行导出的宏放置 Tcl，会进行轨道对齐；
输入 DEF 与最终 DEF 不保证逐 DBU 相同。补充核查同时记录严格输入位移、
最终位置是否保持原生宏放置结果，以及最终宏矩形合法性；见
`table2_macro_handoff.md`。不得为消除这类原生对齐而擅自改为 `-exact`。

## 运行与结果读取

新实验必须创建新 plan 和新 `run_id`，计算并冻结 plan SHA256；现有目录不会
被覆盖。物理作业使用 `backend/scripts/table2_fresh.slurm`，只通过 Slurm
scavenger 调度；节点、并发、内存和时限应根据实际资源另行确定并记录。
`FLOWPLACE_TABLE2_OPENROAD_EXE` 指向冻结的二进制；task 内的二进制和 image
SHA256 会在执行时校验，不能只凭版本名判断相同。

每个结果位于
`/fact_data/bohanli/flowplace_fql_backend/table2_results/<run_id>/`：

| 文件 | 用法 |
| --- | --- |
| `manifest.json` | 任务、源码、工具、执行命令、阶段退出码与最终验收状态 |
| `author_metrics.json` | 原作者 parser 的原始输出，保留文本显示精度 |
| `metrics.json` | 由最终原始 JSON 报告提取的统一全精度结果 |
| `metric_canonicalization.json` | 原 parser → 统一指标的逐项证据，禁止经验缩放 |
| `fullflow_audit.json` | 五项 PPA 与详细布线 DRC 的独立检查 |
| `provenance/`、`orfs_work/logs/`、`orfs_work/reports/` | 冻结输入与原始报告 |

以 rWL、WNS、TNS、power、standard-cell area 对照论文。不要用 macro HPWL、
global-route 线长或中间时序替代最终指标。完整 flow 成功也可能仍有 DRC；报告
必须保留该数量，不能表述为签核通过。

较大的中间数据库验收后无损归档到持久 `/fact_data`，原结果路径中的两个大型
目录可指向 scratch 缓存；报告、指标和配置仍在持久目录。恢复方法见
`table2_artifact_storage.md`。被抢占的尝试保留，只有哈希绑定的同配置基础设施
重跑可以替代主表中的未完成行，完整历史仍进入最终 JSON。
