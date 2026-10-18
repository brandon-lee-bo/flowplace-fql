# Table 2 精确坐标导入：独立兼容修正队列

2026-10-17。此队列因确定的导入几何错误而建立，与 PPA 是否接近论文无关。
原作者发布流程的 12 组主实验、失败与最终数字全部保留，不作替换或混选。

## 触发证据与原生小实验

独立检查 1788302 发现：DREAMPlace swerv_wrapper 输入宏合法，但原生
`place_macro` 对齐轨道后，一对恰好相接的宏发生 0.07 µm 重叠。
其余五个尚在跑的设计/方法组合导入后合法。具体实例、输入和导入后坐标见
`table2_macro_handoff.md` 及对应持久 JSON，检查容差始终为零 DBU。

原生 probe 的历史全部保留：

| Job | 观察与结果 |
| --- | --- |
| 1788310 | 旧 OpenROAD 不支持 `place_macro -exact`；exit 1，原生日志保留 |
| 1788312 | 不能直接移动已经 FIRM 的输入宏；exit 1，未进行布局优化 |
| 1788313 | 坐标恢复已执行；状态审计错误地预期 FIRM，实际原生输出为 LOCKED，严格断言失败 |
| 1788314 | 按原生 LOCKED 状态保留，28/28 宏坐标和方向严格对应输入，状态前后均为 LOCKED，无重叠/越界；exit 0 |

成功 probe：
`/fact_data/bohanli/flowplace_fql_backend/table2_reports/exact_import_probe_swerv_v4/summary.json`。
所有 probe 都只读入几何并测试导入，没有 GP、CTS、routing、reward 或 PPA 调参。

## 唯一的物理流程差异

保留作者 `run_mp` 与它生成的全部 `place_macro` 命令。在该任务自己的
`macro_out` 末尾追加原生 OpenDB 操作：暂时设为 PLACED、按冻结输入 DEF 的
整数 DBU lower-left 执行 `setLocation`、恢复原生 LOCKED 状态。逐宏确认方向
未变，并输出原生 bbox/master/orientation/status 供独立核查。

这不是新的坐标搜索或额外 placer；没有移动到另一个候选位置、改变方向、
重新生成样本或调整目标。它取消导入步骤意外引入的坐标位移，保留生成器已经
给出的合法布局。共享作者 flow 文件保持不变；原 Tcl 与执行 Tcl 都保存于
各自结果的 `provenance`，并绑定 SHA256。修正 runner 和 helper 也逐任务快照。

FlowPlace 与 DREAMPlace **全部六个电路统一使用同一修正**，不只修有问题的一行。
两方法仍使用原来冻结的六个输入 DEF、seed/生成 manifest、同一工具二进制、
作者库/综合网表/SDC、8 线程、GPL0/0、route cap4。新结果明确为
`diagnostic=true`、`not_release_recipe=true`，标签为
`released_flow_exact_macro_import_v1`，不能合入原发布流程主表。

## 执行与验收

冻结计划和源码哈希：`backend/plans/table2_exact_import_v1/protocol.json`。
数组 FlowPlace 1788317、DREAMPlace 1788319；各自最多并发一项，顺序固定为
swerv、bp_fe、bp_be、bp、ariane133、ariane136。该顺序让触发错误的配对案例
先执行，不能据其 PPA 决定是否改其他电路参数。

所有重任务通过 FactLab cpu-scavenger Slurm；每项 8 CPU、16 GiB、3h15。
源数据和结果持久化于 fact_data。归档数组 1788318、1788320 使用同一严格
完成/原始指标检查和无损归档协议，每项精确依赖对应成功物理任务。
基础设施重试 supervisor 为 1788335，状态在本计划目录的 `state.json`。
只能在 PREEMPTED/NODE_FAIL/BOOT_FAIL 且无最终指标时按相同输入重跑，不能为
更接近论文而重采样或重跑成功行。

需分别完成：12/12 原始 PPA/源码审计、统一后端比较、最终 DEF 逐 DBU 对应
输入与精确导入日志、682 个宏无重叠/越界。Std-cell/布线 DRC 仍如实报告；
宏合法不证明全芯片签核洁净。DREAMPlace 公开实现与作者完整 baseline 配置的
未匹配项同原队列，不能因修正导入而省略。

## 独立验收脚本与自动收尾

原生复核1788342已通过：两个 swerv 原生导入各28宏与输入整数DBU坐标完全一致，
故意改动1DBU的负例均被识别。该检查只覆盖导入时刻，不代表最终布线已验收。
证据：`/fact_data/bohanli/flowplace_fql_backend/table2_reports/exact_native_restoration_check_v1/summary.json`。

独立最终检查 `analysis/audit_table2_exact_import.py` 校验每任务的冻结 runner、
helper、原始/执行 Tcl、输入 DEF、后恢复 OpenDB 整数 bbox/方向/LOCKED 日志；
最终 DEF 必须严格保留输入坐标和 FIXED 状态，并用冻结 LEF 重算零容差几何。
自动收尾1788341绑定全部审计、作图及报告源码SHA，依赖watcher1788335：
`exact_import_completed_v1/report/summary.json` → `exact_import_macro_completed_v1`
与 `exact_import_plots_completed_v1` → `exact_import_report_v1/report.md`。
原流程报告与本对照组分别生成；diagnostic/非primary标记不会被报告脚本抹去。

## 2026-10-17 18:44：第一对完整最终结果

swerv_wrapper 的 FlowPlace 与 DREAMPlace 均完成所有物理阶段。第一对 cap-aware 汇总为 `table2_reports/exact_import_capaware_progress_v2/report/summary.json`，指标核对 job 1788485：2/12，完整分母尚未完成，返回非零是按设计的未完成状态。DREAMPlace 原运行 1788319_0 仍保留 FAILED，因为独立 DRC 文本计数假设错误；单独核对严格证明每类 10,000 条导出上限，其完整 DRC 26,921 原样保留。

最终宏检查 1788488 的 `exact_import_macro_progress_v2/summary.json`：**两行共 56 个宏，输入→最终 lower-left 和方向变化数均为零，全部对应原生日志，矩形无重叠且无越界**。检查整体仍为未完成（2/12），不能把这项局部证明写成全矩阵已合格。

| 方法 | rWL (百万 µm) | WNS ns | TNS ns | W | 标准单元面积 µm² | 完整 DRC |
|---|---:|---:|---:|---:|---:|---:|
| FlowPlace | 4.409639 | -0.859876 | -850.792 | 0.268968 | 227220 | 6639 |
| DREAMPlace | 4.700364 | -0.845724 | -617.219 | 0.267920 | 230173 | 26921 |

两方法的输入布局、seed、GPL0/0、线程数和路由 cap 均与原组冻结值相同，只统一修正宏导入坐标。上述不是按单项挑选的替代表行。DREAMPlace 原失败档案已由 1788479/1788480 按独立事后资格证据归档。

原归档数组 1788318/1788320 的提交缺少原 wrapper 必需的 SHA 参数，同时原 task driver 只接受 formal 任务，不适合本组明确 diagnostic 的任务。保留首个归档失败；只取消尚未运行的归档条目。新的严格兼容组归档数组 **1788451 / 1788452** 明确校验任务哈希、六设计名单、统一 policy、diagnostic 标签与完整成功资格。已完成首行，物理任务完全未改。
