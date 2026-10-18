> 当前 Graph Flow-Q offline-to-online 研究见 [最新实验记录](graph_fql_offline_to_online.md) 与 [完整目标核验](completion_audit.md)。下文保留早期实验和任务快照，其中的“已完成”“尚未启动”只适用于对应历史阶段，不能作为当前目标或队列状态。

## 2026-10-17 22:15 HKT：本轮 FQL 接入与真实后端核验全部完成

- 新 learned behavior flow + fresh-noise 一步策略已完成 3 seeds；12/12 配对 GRT 成功，独立原始报告复查通过，1 次相同输入抢占恢复。Table 2 仍暂停。
- 平均 rWL：BC 2,307,378.50 µm，FQL 2,286,268.67 µm（约 0.9% 改善）；WNS 变差、TNS 改善，三个 seed 收益不一致。FQL 未超过已知 TRAIN 最优 2,159,262 µm。
- Q 在实际 FQL 候选上的 Spearman -0.02857，不能由旧 validation 的 0.7265 推断策略优化可靠。没有新的 heldout 或多电路结论。
- 1788998 已完成原始数据复查，但最终汇总因计划保留字段重名失败。原计划/结果均保留；兼容适配只重命名文字说明键并绑定原计划，严格全成功汇总 **1789015**、完整配对报告 **1789016**、Q 校准 **1789017**、已有 teacher 参考核对 **1789018** 均 COMPLETED 0:0。
- 当前本轮没有未完成训练或物理任务。此前建议的后续研究没有自动启动。结果详见 `results_report.md` 最后一节；机器可读结果 `/fact_data/bohanli/flowplace_fql/fresh_flow_full_summary_v1/summary.json`。
- 恢复记录 `logs/fql_report_compatibility_recovery_v1.json` 与 `logs/fql_fresh_completed_v1.json`。以下保留早期快照，不能当成当前仍在排队。

## 2026-10-17 22:00 HKT：新 FQL 接入已训练，真实 GRT 对照运行中

Table 2 保持暂停。本轮重点是探索 FQL 接入，不恢复原复现任务。

- 新训练 **1788584 COMPLETED 0:0**：3 seeds 的 learned behavior flow、terminal critic 与 fresh-noise one-step student 已全部完成。BC 学生验证蒸馏 MSE 为 0.00574/0.00860/0.00525；宏 HPWL 的下降不能当作后端收益。初次 1788583 的导入命名冲突及失败日志保留。
- 物理特征 critic **1788582** 完成；raw physical 验证 MAE 0.02537，对照常数 0.03913。参数仅用 TRAIN 家族 CV 选定，但验证数据以前已被查看，结果为探索性，不是新的最终测试。
- GRT actor **1788587 COMPLETED 0:0**：3 seeds × 2 fixed validation noises × BC/FQL 两方法，固定 alpha=1、1500 步，不按 Q 选候选。12/12 通过独立宏几何与 DEF 舍入检查。
- 原后端数组 **1788588** 在计算前因容器配额根目录挂载错误退出；无物理结果产生。修正挂载的 **1788685** 使用相同 12 布局和同一物理工具配置。恢复记录 `logs/fql_physical_container_recovery_v1.json`。
- **1788685_0 PREEMPTED**，原文件保留；只对同一候选补跑一次 **1788997**，等待数组结束，维持后端并发不超过 2。完整原任务列表不变，恢复计划为 `fql/physical_preemption_plan_v1.json`。不是按质量重跑。
- 旧汇总 1788589/1788686 已取消；新 **1788998** 汇总全部 12 个逻辑候选，检查抢占证据和相同输入重试，独立重读原始物理报告。**1788999** 在核验成功后生成完整对照报告，目标 `/fact_data/bohanli/flowplace_fql/fresh_flow_full_summary_v1/`。提交不代表完成。
- 已启动任务的工具 SHA、design config 与 Q 训练标签后端一致；新数据是 global-route-only，不是 Table 2 详细布线结果。最终还要比较训练集已知最佳固定布局，不把它复制成多个随机样本。

## 2026-10-17 21:52 HKT：FQL 小规模接入实验已启动

- 用户进一步授权“仔细尝试，开始”。Table 2 的暂停保持；未解除其 hold，也未恢复任何复现控制器。新任务只服务 FQL 接入，不触及其他项目。
- 必要的软件检查 **1788581**（5项）、**1788586**（6项，增加物理特征 NumPy/Torch 一致性及有限差分梯度）均 COMPLETED 0:0。原新试验 **1788583** 在训练前因 upstream analysis.py 遮蔽本地包而失败，日志保留；只调整 import 顺序后同参数恢复为 **1788584**。
- **1788584**：从既有 TRAIN raw proposals 学习新的行为 flow，持续用新 Gaussian noise 蒸馏一步策略，同时训练终止奖励 critic。三 seeds、4000步、alpha 1/10/100及无Q对照。它补齐原版 FQL 的三网络机制，但仍是固定电路、终止动作、宏HPWL奖励的适配；不是多电路完整原版benchmark，也不能把收益写成PPA改善。首个seed已经有结果，其余仍在执行；完整记录位于 `/fact_data/bohanli/flowplace_fql/fresh_flow_pilot_1788584/`。
- **1788582 COMPLETED 0:0**：复用已核验的96个GRT标签，只用64个TRAIN布局做家族分组五折CV选择ridge惩罚，在原16个validation布局上诊断，不读取heldout预测。raw physical特征验证MAE 0.025367（TRAIN常数0.039135），Spearman 0.7265；TRAIN CV MAE 0.02527、Spearman 0.3200，信号有限。executed特征只适合先合法化再评分，不能假装对离散合法化有梯度。全六种输入表示都保留于 `/fact_data/bohanli/flowplace_fql/physical_critic_1788582/`。
- 后续 **1788587** 在宏试验与梯度检查成功后，从每个seed同一BC检查点出发，对比相同1500步的BC continuation和固定alpha=1的GRT-Q优化。原行为flow与raw physical ridge critic冻结，同噪声配对。不按预测Q挑候选。
- **1788588** 为固定十二个新GRT候选（3 seeds × first2 validation noises192/193 × 2 methods），并发2、每项8CPU/12GiB，使用与GRT标签一致的26Q4工具和公开物理流程。不是Table 2历史工具组；不改变标签对应工艺/约束来追求好数字。输出用fact_data bulk，暂停中的结果原样保留。导出失败则不启动数组。
- **1788589** 在数组结束后进行原始报告复查、终止失败分类和完整分母汇总；CTS失败不补零、不删除、不更改科学设置重试。既有gate与新试验为探索性结果，后续仍需新的测试数据及电路。计划与源哈希、全部任务ID见 `fql/physical_actor_plan_v1.json`、`logs/fql_fresh_submissions_v1.json`。

## 2026-10-17 21:39 HKT：用户暂停 Table 2，转向研究 FQL 接入

- 用户明确要求暂停复现，将主要精力转向理解 Flow-Q/FQL，判断如何接入 FlowPlace。本次只研究论文、官方源码及已有试验，没有启动新训练或后端扫描。
- 已取消自动监督/提交任务 **1788335、1788516、1788518、1788537**，停止仍在计算的 DREAMPlace/ariane136 **1788519_0**；已有文件和中间数据库保留，但不声称被停止的物理进程可以从内存原样续跑。尚未执行的 16 个相关物理、归档、报告及工具对照任务均已 hold，核查全部处于 PENDING 且 priority=0，无 Table 2 任务仍在执行；其中 1788520、1788517、1788399 在依赖被取消后显示 JobHeldAdmin，其余为 JobHeldUser。不影响其他项目任务。
- 原流程物理完成 **11/12**；统一坐标兼容组在暂停前已完成 **7/12**，FlowPlace/ariane133 及其归档已结束。保留所有既有结果、失败及工具来源证据，不能把未完成任务填成结果。
- 机器可读暂停标记为 `backend/plans/USER_PAUSED_TABLE2.json`，完整命令、队列前后快照和三份控制状态快照见 `backend/plans/user_pause_20261017_213950/`。原 state.json 只保留最后运行状态，不是当前 live 状态；轻量状态脚本会显示 user_paused。取消的控制任务不再自动重提，恢复时需要重新绑定依赖，不能仅解除所有 hold。
- 研究起点：原始 FQL 包含行为 flow、critic、一步 actor；现有 `fql/` 只是单电路、整布局单步终止、冻结 FlowPlace 教师及 critic 的适配。已有宏 HPWL 收益未转移到已测 GRT，真实 GRT critic 的验证 gate 为 0/3，因此不能把旧试验描述成成功替换，也不应不加诊断地扩大同一配置。先解释三部分各自输入输出，再讨论教师硬投影的蒸馏难度及后端 critic 的可靠性。既有证据见 `research_interpretation.md`、`results_report.md` 和 `fql_integration_design.md`。

## 2026-10-17 20:57 HKT：历史工具构建与作者输入导入通过

- **1788558 attempt_1 COMPLETED 0:0**：历史源码 `cd519bb5e861ec4d5d00931c85650c179a864e34` 已构建并发布，`-version`、ldd 和发布副本验证均通过；无算法源码修改。持久二进制：`/fact_data/bohanli/flowplace_fql_backend/tools/openroad_cd519_source_v1/build_1788558/attempt_1/openroad`，SHA256 `eadf0dd0baea50e510e77bf207e29ed1168ca473ddc1c6dd248a5a77573272cb`。原失败尝试没有被覆盖。
- **1788561 COMPLETED 0:0**：作者 bp_fe DEF 与匹配平台 LEF 实际读取成功，OpenDB 识别 33,199 个实例、11 个宏、DBU=2000。详细 manifest 位于 `tools/openroad_cd519_source_v1/import_1788561/manifest.json`。这是 CLI/输入兼容检查，不是 PPA 或合法性结论。
- 三项事先声明的固定作者 DEF 工具对照由 **1788559** 在两组完整主实验报告后提交；SDK 持久归档 **1788555** 同样等待两组报告腾出空间。当前主流程 11/12，兼容组 6/12；兼容 FlowPlace/ariane133 已通过 CTS，进入 GRT。其余兼容大任务由 1788537 存储准入控制器自动接续。
- 后续读取两组 `state.json` 和最终报告链，不要把本次工具准备成功当作 Table 2 数值复现完成；作者实际二进制仍未确认，已有 WNS/TNS 差距保留。

## 2026-10-17 20:55 HKT：已完成编译，恢复安装运行库路径

- 1788558 attempt_0 已编译并安装 executable，但 `-version` 返回 127：历史 `src/CMakeLists.txt` 将安装 RPATH 追加为现代容器的 `/opt/or-tools`，错误加载了不匹配的 OR-Tools/Abseil。原编译、安装、失败启动和退出码完整保留；不能把它记成可用二进制。
- 独立 Slurm 设置任务 **1788564** 仅在 CMake cache 的 `CMAKE_INSTALL_RPATH` 前置自建 OR-Tools 9.10.4067/lib，不改源码。1788558 已重新排队，后续需要重新安装、验证加载、发布二进制及其哈希。
- 新 CLI/LEF/DEF 导入检查为 **1788561**，仅用作者 bp_fe DEF 和发布平台对应五份 LEF 读取 11 个宏；不执行放置/布线、不声称 PPA 或合法性。规划器 **1788559** 现同时依赖构建、该导入检查及两组完整结果；SDK 归档 **1788555** 依赖构建及两组完整结果。所有依赖均已重绑，没有不可满足依赖遗留。证据见 `openroad_runtime_path_recovery_v1.json` 和 `table2_source_pin_diagnostic_v1/import_smoke_submission_v1.json`。

## 2026-10-17 20:45 HKT：历史依赖配置成功，源码正在编译

- 恢复下载 1788556 已完成两个官方压缩包：Boost 的原 MD5 通过，SHA256 为 `4b2136f98bdd1f5857f1c3dea9ac2018effe65286cf251534b6ae20cc45e1847`；OR-Tools SHA256 为 `ffa50a970557e4527dcb3e77d45467a15770b6e93ed3cf61c4b602a2566ce6cb`。Boost 静态库安装成功；全新 CMake 配置确认 Boost 1.80.0、OR-Tools 9.10.4067、SDK Eigen 3.4.0 与自建 spdlog 1.8.1。
- 该设置任务的最后 requeue 步骤遇到 Slurm 已清除失败任务 1788546 的 live job ID；所以其原 FAILED 状态保留，不能称整个设置任务成功退出。准备好的依赖和配置已核对后，以完全相同的构建脚本提交 **1788558**，当前正在编译。旧失败构建目录、每次日志、缓存与网络记录均保留。
- 取消尚未执行的旧规划任务 1788551；规划器增加显式 `--build-job` 参数，原文件已快照保存，新冻结哈希及恢复记录见 `table2_source_pin_diagnostic_v1/builder_resubmission_v1.json`。新规划器 **1788559**、SDK 归档 **1788555** 均正确等待 1788558、原完整报告 1788517 与兼容组完整报告 1788399；无 JobHeldAdmin/DependencyNeverSatisfied 遗留。
- 原流程与兼容组监督器正常，归档监督器 errors 为空。当前没有新源码二进制的物理结果；这些依赖/构建恢复不改变主表布局和后端参数。

## 2026-10-17 20:35 HKT：继续匹配历史构建依赖

- attempt_1 解决了 spdlog/fmt 冲突，但现代 Boost 1.87 移除的 Asio 接口又导致编译失败。原失败和缓存完整保留；未改动 OpenROAD 网络/分布式实现来绕过。
- 独立 Slurm 任务 **1788553** 按 pinned `DependencyInstaller.sh` 恢复 **Boost 1.80.0、OR-Tools 9.10.4067**，继续使用已构建的 spdlog 1.8.1。Boost 下载需通过原安装器给出的 MD5，同时记录两个下载的 SHA256；Boost 在 scratch 构建静态库。旧构建目录将移为 `build_make_modern_deps_failed_v1`，新配置排除现代 `/opt/or-tools`，成功后才接续 1788546 并重绑规划器依赖。当前是设置/编译恢复，尚无新工具的 PPA 结果。
- 下载实际速率导致预计时长超过原 600 秒请求上限；在触发从头重试前保留 42,426,368 字节，停止仅负责设置的 1788553，以 **1788556** 同流程断点续传（请求上限 1800 秒）。科学输入、工具目标版本和构建参数未变；旧日志和网络恢复记录均保留。
- 后续 SDK 持久化任务 **1788555** 依赖构建及两套完整报告，逐文件核对 OR-Tools 运行库归档，记录恢复路径。它不改变二进制、算法或布局，且等主实验归档后才使用持久空间。若原失败依赖未自动解除，依赖恢复必须同时核对该任务与 1788551 的调度状态。

## 2026-10-17 20:28 HKT：历史依赖恢复

- 原流程仍为 11/12，兼容导入组为 6/12。对 DREAMPlace/ariane136 的 Slurm 分配内只读进程检查表明，OpenROAD 仍在单线程计算（约 99.6% CPU），不是已完成或空闲；最后日志位于 CTS hold repair。没有修改该任务的时序修复参数。
- 初始依赖下载 1788539 因网络中断停止，原日志和不完整目录保留。恢复任务 1788549 已按根源码 gitlink 完成 OpenSTA `9afc487f07c02e009ca0906c5ab4c5d62fbcbc79` 和 ABC `ef5389d31526003c2ebd7e6d6d6fe3848a20f0a2` 的检出，使用 HTTP/1.1、partial clone 和排除测试/文档的 sparse checkout；递归 gitlink 状态一致且无算法补丁。
- 历史源码构建 1788546 的 attempt_0 因容器内 spdlog/fmt 过新而编译失败，失败日志保留。独立 Slurm 任务 1788550 已构建该历史 `DependencyInstaller.sh` 指定的 spdlog 1.8.1，并只修改独立构建缓存中的 `spdlog_DIR`。后续恢复检查因 CMake 把缓存类型记作 UNINITIALIZED 而非 PATH 错误退出；核对真实路径和编译 include flags 后才恢复原构建任务，attempt_1 不改算法源码。
- 临时暂停的后续规划任务在依赖重绑后出现 JobHeldAdmin，release 被拒；未更改优先级或调度限制。已取消这个尚未执行的自设暂停规划任务 1788548，以相同脚本、哈希和三项先验对照重新提交为 **1788551**，仍依赖构建成功、原流程完整报告 1788517 和兼容组完整报告 1788399。恢复证据见 `openroad_spdlog_recovery_v1.json` 与 `table2_source_pin_diagnostic_v1/planner_dependency_recovery_v1.json`。

## 2026-10-17 20:08 HKT：Table 2 继续执行

- 用户再次明确只复现 Table 2：FlowPlace 与 DREAMPlace，各六电路。原流程已完成 11/12 项；DREAMPlace/ariane136 的同输入恢复任务 1788519_0 仍在 CTS/hold repair。不能把尚未完成的行或与论文的时序差距说成已对齐。
- 原流程新的完整分母临时核对任务为 1788540，输出 `source_era_capacity_progress_v2`；未完成整组时预期返回非零，已完成行需逐项通过原始报告核对。
- 兼容导入组已完成六个小设计评估，六个大任务由存储准入控制器 1788537 接续。它计算 Ceph 实际配额、运行中任务的预留增长和归档余量，一次释放一个大任务；不修改输入、seed、工具或优化参数。已释放 FlowPlace/ariane133（1788317_4），其余随已核实归档释放空间后自动执行。状态及每次释放证据见 `storage_admission_controller_v1.json` 和 `storage_admission_hold_v1.json`。
- 历史工具构建可行性任务 1788539 正在获取源代码：选择版本 `cd519bb5e861ec4d5d00931c85650c179a864e34` 的依据是先前已核对的 ORFS 源码历史/gitlink，并非选择更接近论文 PPA 的版本。它仍不能证明作者实际使用该二进制；构建成功后才可进一步做固定作者 DEF 的工具差异控制。源码/构建缓存位于 `/scratch`，小型日志和版本记录保存在 `tools/openroad_cd519_source_v1/probe_1788539`。
- 独立源码构建任务为 1788546，使用已有容器的 Unix Makefiles、八 CPU、关闭 GUI/Python 接口和测试目标，不修改算法源文件。后续固定作者 DEF 对照规划器为 1788548，依赖构建成功以及两组十二行最终报告完成。预先声明三项：December-2024 工具/作者 bp、cd519 工具/作者 bp_fe、cd519 工具/作者 bp；统一发布 GPL0/0 与其他物理参数，不替换主表任何行。对照任务在同一 Slurm 数组中串行执行到无损归档结束，且启动前检查真实配额。声明见 `backend/plans/table2_source_pin_diagnostic_v1/declaration.json`。
- 旧 `metadata-base-ok.json` 中出现多个历史工具版本，不能用其中某个直接证明 Table 2 的作者二进制身份。独立归档核对任务 1788547 已验证六份元数据和六份综合网表头均确实存在于原始发布压缩包且一致；证据见 `backend/references/released_flow_embedded_versions_archive_audit_v1.json`。原先“没有版本记录”的过宽表述已在工具来源文档中纠正为“没有明确标识 Table 2 实际评估二进制的记录”。

## 2026-10-17 19:14 HKT：同输入时限恢复

- 原发布流程 9/12 已完成物理评估。DREAMPlace/ariane136 的原任务 1788077_3 在 9025 秒后 TIMEOUT，没有最终 metrics、audit 或 6_report.json；不是按 PPA 选重跑。
- 同输入/工具/线程/算法的 6 小时任务 1788519_0 已运行；原 TIMEOUT 和中间数据库独立保留、压缩归档，不能计作成功。绑定见 `backend/plans/table2_background_v1/t2auto_dreamplace_ariane136_1/retry_binding.json`。
- 原控制任务已正常停止，容量恢复控制器 1788516、最终报告依赖任务 1788517、归档控制器 1788518 接续原状态。只扩展资源时限与基础设施重试上限，不改布局或物理参数。转换快照见 `capacity_transition_v1.json`。
- 修正导入组已完成 5/12；已核对的前四项共 78 个宏均保持整数 DBU 坐标、方向与最终无重叠。新的 FlowPlace/bp_be 已完成，DREAMPlace/bp_be 正在详细布线。
- 修正组余下六个大任务因持久盘配额暂留存储准入 hold；完成归档、腾出真实空间后分批释放。不能永久留 hold，也不能把仍未完成的任务报告为复现成功。

## 2026-10-17 18:10 HKT 更新

原发布流程：8 行通过原完整核对，另 1 行（DREAMPlace/swerv_wrapper）物理实现完成并经单独 DRC 导出上限证明核对；剩余 bp、ariane133、ariane136 仍在 Slurm 运行。最新完整分母临时汇总为 `capaware_progress_v1`（9/12），不是完成报告。

390 个最终宏已核对；swerv 原生导入造成 0.07 µm 重叠，不能视为合格的新算法 baseline。独立统一坐标导入兼容组已提交全部 12 行，目前首对仍在详细布线。算法、seed 与后端指标设置不变，不混表。

最终后处理已分别更新为 1788398 / 1788399，保留原 strict 失败记录；源码、计数证明和归档说明见 [table2_drc_export_cap.md](table2_drc_export_cap.md)。物理任务没有被替换或更改。归档监督器 1788403 仅处理明确冻结绑定的中断尝试和有严格证明的 DRC 核对错误。

以下为较早快照，当前状态以两组 `state.json`、Slurm 和最终核对为准。

## Current status: Table 2 original flow plus uniform exact-import cohort (2026-10-17 17:40 HKT)

Original primary state: `backend/plans/table2_background_v1/state.json` (watcher1788095).
Seven of twelve physical runs completed; remaining jobs retain their original frozen inputs.
A genuine native-import geometry bug was found: DREAMPlace swerv macros are legal in the
input, but old `place_macro` track snapping introduces one 0.07 µm overlap. Original runs
and PPA stay intact. The original strict macro postprocess1788279 is expected to report
this failure, not qualify this cohort as a geometry-clean reusable baseline. Separate
controller1788349 will publish the full original table while retaining that failure.

Uniform corrected cohort: `backend/plans/table2_exact_import_v1/state.json`, physical
arrays1788317/1788319, watcher1788335, strict postprocess1788341. Both methods/all six
designs use the exact same original generator outputs and physical settings; the only
change is restoring integer-DBU macro coordinates after native import and keeping native
LOCKED status. The new cohort is explicitly diagnostic, never a row-wise replacement.
Native check1788342 passed both 28-macro swerv imports and detected deliberate one-DBU
negative controls. Final routed DEF preservation is still pending. See
`table2_exact_import_protocol.md`; these state JSONs supersede older chronology below.

# Pipeline end state

## Current execution pointers (2026-10-17)

The paragraphs below retain the chronological execution history. For the live
12-task selection, identical-input retry chains, supervisor health and terminal
audit job, read `backend/plans/table2_background_v1/state.json` rather than an
older progress paragraph. At 17:20 HKT, six primary rows are complete and the
other six are running; both placers have generated all six legal input layouts.

After the supervisor's full-cohort audit, postprocessor job 1788279 runs the
frozen final macro audit, plots, and Chinese report. It refuses partial cohorts
and stops on failed audits. Its submission is recorded in
`backend/plans/table2_background_v1/postprocess_submission_v1.json`; later
status/child IDs will be in `postprocess_v1.json` in that directory. Expected
outputs below are destinations, **not claims that they already exist**:

- `/fact_data/bohanli/flowplace_fql_backend/table2_reports/background_completed_v1/report/summary.json`
- `/fact_data/bohanli/flowplace_fql_backend/table2_reports/macro_preservation_completed_v1/summary.json`
- `/fact_data/bohanli/flowplace_fql_backend/table2_reports/reproduction_plots_completed_v1/`
- `/fact_data/bohanli/flowplace_fql_backend/table2_reports/completion_report_v1/report.md`

Current validated partial comparison:
`/fact_data/bohanli/flowplace_fql_backend/table2_reports/source_era_progress_v5/report/comparison.md`.
It is 6/12, not a completed-method mean. The additional old-tool GPL1/1 bp_fe
control (1788249) is separate from the primary rows. Storage operations and
recovery histories are documented in `table2_artifact_storage.md`.

## Active scope: Table 2 FlowPlace and DREAMPlace (2026-10-17)

The user clarified that the requested reproduction is **Table 2**, then added
the **DREAMPlace method** alongside FlowPlace, both on all six circuits, before
later algorithm substitution. Numerical agreement with the published results
is a priority, with differences investigated through source-supported controls.
This uses
the released Nangate45 **OpenROAD fullflow on FactLab via Slurm**. SCUT/Innovus
and Table 3 training-data ablations are outside this active scope; no real-design
Innovus run has been launched. Historical ICCAD work below remains preserved.

The complete six-design original-mode generation from job1783973 and its
verified DEF exports are frozen in `backend/plans/table2_flowplace_stream300_v2`.
The full six-design batch is deferred while the shared backend is checked on
small circuits. Job1787852 is a predeclared bp_fe diagnostic using the identical
released macro DEF and tool, changing only GPL timing/routability switches from
0 to1 to test the paper's explicit flow description. This is recorded separately
from the unmodified released recipe; no detailed-route cap, clock or density is
changed. DREAMPlace4.1 two-stage macro placement and its public legalization
implementation are being checked before preparing that second method.

The same predeclared combined-switch test on bp_be is job1787985, submitted
before the bp_fe final metrics were available. The first CPU DREAMPlace smoke,
job1787984, completed original optimization but failed an overly strict output
orientation guard: stock `place_io.apply` performs row-dependent N/FS flips.
Connectivity and macro geometry passed. Its failed manifest is preserved;
the next smoke validates stock flips and reloaded physical pin offsets.
An isolated GPU build and source-based dating of the released ORFS scripts
are being prepared. No six-design DREAMPlace PPA batch is submitted yet.

Update at 15:05 HKT: both GPL1/1 diagnostics finished and passed their raw-report
audits; see `table2_gpl_diagnostic.md`. They improve timing but do not account
for the area/power discrepancy. The flow scripts match upstream ORFS late-2024
sources. A binary selected by that historical evidence, OpenROAD
`v2.0-17598-ga008522d8`, is evaluating the same released bp_fe DEF with the
unchanged GPL0/0 recipe in job1787997. The exact author binary remains unknown.
CPU DREAMPlace smoke v3 passed geometry, orientation, and pin-connectivity
audits; its two earlier audit failures are preserved. GPU compilation and four
official tiny operator tests passed through jobs1787994/1787998/1788005, with
two official CUDA header compatibility fixes and a bool-mask test-fixture fix.
GPU macro-generation validation is next; the full six-design backend batches
remain deferred until the common tool/metric protocol is frozen.

Subsequent update: GPU DREAMPlace first cohort1788011 ended with five legal
handoffs and one native Abacus assertion on ariane133. Observation1788018
captured the exact operator input and confirmed a float32 row-index boundary
error in standard-cell legalization. A separately installed one-line floorDiv
compatibility fix is being validated before a uniform six-design rerun; the
original five-success/one-failure cohort stays intact. The old OpenROAD control
completed physical design; reconciliation1788017 independently verified its
text-only rounding discrepancy. Formal generated FlowPlace physical array
1788030 and dependent lossless archival array1788033 are now submitted. See
`table2_source_era_control.md` for the fixed-input numbers and frozen protocol.

Infrastructure update: `1788030_0` (bp_fe) was **PREEMPTED** after 160 seconds,
before final metrics. The interrupted directory and original six-task plan are
unchanged. Identical input/tool/seed/recipe retry `1788063_0` uses
`backend/plans/table2_source_era_flowplace_retry1`; only `run_id` differs.
Its `retry_binding.json` binds both task hashes and exact scheduler identity.
The impossible archive dependency `1788033_0` was cancelled; retry archival is
`1788064` (create) then `1788066` (verified commit). The initially submitted
commit `1788065` was cancelled before execution to correct its scratch root.
Other archive tasks retain their original dependencies.

`analysis/summarize_table2_methods.py` now accepts hash-bound infrastructure
retry bindings and keeps the interrupted manifest and scheduler cost in the
report. Replacement is restricted to PREEMPTED/NODE_FAIL/BOOT_FAIL with no
finished metrics, and task contents must be identical except for the run ID.
Interim audit `1788067` exercised the real binding and produced an intentionally
incomplete report (exit 1: no routed rows finished yet), not a parser failure.

The isolated one-line Abacus row-index correction passed regression `1788043`:
the original two-rectangle fixture aborts, the corrected fixture preserves
coordinates, and replay of the captured actual ariane133 operator input passes
stock full-layout legality with zero macro displacement. Uniform corrected
GPU generation cohort `1788062` is in progress; **all six outputs are new**, not
five original outputs mixed with one corrected output. No DMP physical PPA is
selected yet. Its backend must use the same source-era OpenROAD recipe.

Latest execution update (about 15:50 HKT): GPU cohort `1788062` completed
successfully, **6/6** independently audited handoffs. Physical DMP plan
`backend/plans/table2_source_era_dreamplace_v1` is frozen at SHA256
`e831aa44b494d02fe9e1665e28df4e8464766599e7562bd9710f48fb70360058`;
backend array `1788077` and archival array `1788078` are submitted. It uses
exactly the same source-era tool and physical recipe as generated FlowPlace.

Two further infrastructure interruptions occurred: original bp `1788030_1`
(499 seconds) and bp_fe retry `1788063_0` (170 seconds). Their directories,
metrics absence, source plans and scheduler identities remain preserved.
Identical tasks now run in `1788071`, plan
`backend/plans/table2_source_era_flowplace_retry2`, SHA256
`6e31ede2d26b598fdeb8231b626bd60ea6595952758880dd53634a263ce808ec`.
The corresponding impossible archive dependencies were cancelled; new
create/commit pairs are `1788072/1788073` and `1788074/1788075`.

To reduce GPU-development preemption, new/retry tasks require node enterprise
but remain in **cpu-scavenger**. Pending original tasks 4/5 received the same
node request. Pending-only requests use 16GiB (above the historical ~9.8GiB
fullflow peak) and 2h30 rather than 12h, fitting enterprise's backfill window
before a higher-priority EDA reservation. This changes scheduling only;
8 threads, all input hashes, algorithms and physical parameters are unchanged.
The scheduling records also preserve an initially rejected `16G` scontrol
syntax, immediately corrected to integer MiB; no held task was left behind.

DMP starts after bp_fe retry allocation exits, initially at concurrency 1 to
keep at most four physical allocations overall. Increase only when another
FlowPlace allocation finishes and storage headroom permits. Default physical
scripts/verifiers are unchanged while these tasks run.

Live audit `1788084` verified both infrastructure retry chains and all four
plans; its intentionally incomplete 12-row report is at
`/fact_data/bohanli/flowplace_fql_backend/table2_reports/source_era_matched_interim_v2/report`.
Final live-accounting audit `1788085` depends on all currently selected physical
arrays; it will retain all twelve outcomes and the superseded attempts.
If another infrastructure retry is needed, supersede this audit invocation
explicitly rather than silently changing its hash-bound source or task list.

## Background continuation after the token-saving request

The user requested submitting all tasks and conserving conversation tokens.
All twelve selected method/design tasks are submitted. Original FlowPlace
array throttle is now4; DREAMPlace throttle is2 and its artificial dependency
has been removed. Resource and partition constraints still govern startup.
The superseded fixed final audit1788085 was cancelled.

Persistent supervisor **1788095** is confirmed running in cpu-scavenger, with
1CPU/1GiB and automatic requeue. Its pending-only enterprise pin was cleared
to allow the lightweight supervisor to start on blue-space. Its state and
submission journal are in `backend/plans/table2_background_v1/`. It preserves
all prior plans and bindings, launches at most four additional same-input
retries per design for PREEMPTED/NODE_FAIL/BOOT_FAIL with no final reports, and
submits lossless archival plus the final live-accounting audit. Physical
retries still use enterprise/8CPU/16GiB/2h30 via cpu-scavenger. Application
failures are retained as failures rather than retried for favorable PPA.

bp_fe retry1788071_0 was preempted after600seconds; its original files are
intact. The supervisor successfully detected it and submitted identical-input
retry1788096 (`t2auto_flowplace_bp_fe_1`), proving the continuation is active.
Impossible archive dependencies1788072/1788073 were cancelled. At this update
five physical tasks are running and the other selected tasks are queued.
The supervisor's first accounting cycle has12 selected tasks and no error.

Final output will be
`/fact_data/bohanli/flowplace_fql_backend/table2_reports/background_completed_v1/report/`.
Read `summary.json`/`comparison.md` there when present; supervisor status
`audit_submitted` means the audit was queued, not that numerical reproduction
or all physical checks passed. No final PPA result is claimed at this point.

New fullflow evaluations will use bulk storage and an independently audited
final metric table. Algorithm outputs must share the same declared backend.
This supplements the previous released-DEF replay; it does not overwrite its
five successes or the raw swerv overlap failure. See `table2_protocol.md`.

## Subsequent ICCAD request (2026-10-17 daytime)

The user separately requested the paper's DREAMPlace4.1.0 + Innovus
EarlyGlobalRoute evaluation. This new work is tracked in `../iccad/README.md`.
It is not covered by the completed overnight Table2/FQL pilot below.
Innovus25.11 is installed on shared storage, but actual scheduled startup
fails with IMPLIC-4 for the configured shared license. All eight released inputs
were geometrically audited. The released superblue18 DEF and a fresh official
model sample both completed DREAMPlace4.1.0 CPU placement and passed independent
float64 reload legality checks, preserving512 fixed instances. Original false
verification failures from floating-point/serialization differences are retained
and explained in the ICCAD README. All submitted ICCAD jobs are terminal.
Later, the user's VPN/SSH forwarding enabled access to SCUT SMW8: Innovus19.12
successfully checked out its base license and completed a2-thread no-design Tcl
probe. The complete EGR evaluation draft still remains unexecuted. Further
SCUT scheduling checks and real-design import/EGR are deferred by the current
Table2-only scope. No new ICCAD routed PPA or Table3
reproduction is claimed. Detailed status: `../iccad/status_2026-10-17.md`.

## Completed original pilot

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
