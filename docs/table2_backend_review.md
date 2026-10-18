# Table 2 OpenROAD 后端复现审计

2026-10-17。只读审查现有脚本、配置、汇总及少量原始报告；本次没有提交作业、重新执行 EDA 工具或修改现有流程。下一次计算应先回答具体的流程差异问题，不应重新扫描参数以贴近论文表格。

现有证据已支持“作者发布 DEF 在可获得工具上完成了 5/6 个完整后端流程”，尚不支持“Table 2 数值复现成功”。`swerv_wrapper` 的失败已经有原始几何证据，不是仍待补跑的任务。

| 电路 | 当前结局 | rWL 相对论文 | 标准单元面积相对论文 | 详细布线 DRC |
|---|---|---:|---:|---:|
| ariane133 | 完成 | +3.19% | -13.86% | 1 |
| ariane136 | 完成 | +3.33% | -14.67% | 14 |
| bp | 抢占后由原 GRT 检查点恢复完成 | -0.46% | -11.26% | 45 |
| bp_be | 完成 | -1.79% | -15.94% | 247 |
| bp_fe | 完成 | -6.89% | -18.15% | 1904 |
| swerv_wrapper | 宏放置导入失败 | 缺失 | 缺失 | 缺失 |

数值与逐项差值见 [完整对照](../analysis/artifacts/table2_released_v1/comparison.md) 和 [CSV](../analysis/artifacts/table2_released_v1/metric_comparison.csv)。`bp` 恢复阶段时间只包含 DRT/fill/finish，不能当成从头运行时间。swerv 原 DEF 在 OpenROAD 吸附之前就有三处宏重叠；例如 DCCM bank1/bank3 水平方向重叠 3.42 µm。合法的生成 swerv 已单独完成过流程，但不能替换发布 DEF 的失败行。证据见 [几何审计](../backend/results/released_def_geometry/swerv_wrapper.json)。

当前后端入口真正执行的是作者 `make run_mp` 与 `make run_wo_synth`。前者读取 DEF 后只导出 macro placement；后者从发布包中已有 `1_synth.v` 和 `1_synth.sdc` 重建 floorplan、标准单元放置、CTS、GRT、DRT 和 finish。因而 DEF 中旧的标准单元位置不是当前布局输入；也不能只改变 Yosys 设置就期待面积改变，因为该入口不重跑综合。应检查实际复制的综合 seed，而不仅是配置中写着的 RTL/SDC 路径。源代码见 [作者 evaluator](../upstream/flowplace/benchmark_utils/openroad/openroad_ppa_evaluator.py)、[宏导出 Tcl](../backend/thirdparty/OpenROAD-flow-scripts/run_macroplace.tcl) 和 [Makefile](../backend/thirdparty/OpenROAD-flow-scripts/Makefile)。

最值得继续调查的是以下具体差异。

- **工具版本未匹配。** 当前容器为 `openroad/orfs:26Q4-151-g8b5e62fbb`，晚于论文，OpenROAD 自报版本为 `unknown`，已有镜像与二进制 SHA256 可以固定本次结果。作者精确 OpenROAD/OpenSTA/OpenRCX revision 未公开。布局、resizer、CTS、提取以及指标分类都可能因版本改变；这只是待检验原因，不能归因后直接结束。若找到作者版本或有明确发布日期依据的版本，应固定一个候选做成对比较，不能在多个版本中挑最接近论文者。
- **所有六个 eval 配置都关闭了 GPL 的 timing-driven 与 routability-driven。** 这与现有报告转述的论文流程不同。它是发布配置的事实，不是本地 wrapper 无意漏传的参数。应保留该发布基线，将按论文描述开启两项的结果单独标成诊断变体。
- **GRT 中打印“Repair setup and hold violations”，但真正的 `repair_timing_helper` 被注释。** 当前 [global_route.tcl](../backend/thirdparty/OpenROAD-flow-scripts/scripts/global_route.tcl) 执行 `repair_design`、详细放置和增量路由，并调用 `recover_power` helper；后者是否产生优化取决于 `RECOVER_POWER`，其值为零就直接返回。不能根据日志标题、`TNS_END_PERCENT=100` 或函数调用存在推断 GRT 做了 setup/hold 修复。CTS 阶段的 [cts.tcl](../backend/thirdparty/OpenROAD-flow-scripts/scripts/cts.tcl) 确实调用 `repair_timing_helper`，bp_fe 原日志也记录了 `repair_timing -verbose -repair_tns 100`；因此“整个流程没有 timing repair”同样不成立。较少面积伴随较差 TNS 与这些差异相容，但尚未建立因果关系。
- **发布脚本会吞掉部分物理失败。** `pdn.tcl` catch `pdngen` 错误后继续；CTS 和 GRT 对部分 `detailed_placement` 错误也没有终止；配置又设置 `SKIP_DP_CHECK=1`。bp_fe 原日志确有 `PDN-0113`。PPA 指标齐全不能代替 placement、PDN 或连通性检查。有效性核查应独立报告；若建立新的工程有效流程，需明确修复并传播这些错误，而不是把修改混入发布基线。
- **cap4 主要解释 DRC 残留，现有证据不支持用它解释主要 PPA 差距。** 发布平台显式设置 `-droute_end_iter 4`。已完成的 cap20 诊断中，生成 bp_fe 从 12 DRC 到 0，rWL 仅 +21 µm、TNS 约 -0.028 ns；发布 bp_fe 从 1904 到 192 DRC，rWL +199 µm、TNS -1.390 ns，面积与功耗不变。无需为了 Table 2 再重复相同实验。详见 [结果报告](results_report.md)。

面积、功耗和时序口径已经能够排除几个简单解释，但仍应增加可比报告。

- bp_fe 当前 `StdCellArea=59268.3 µm²`，宏面积 `160975 µm²`，非 filler 的总 instance area 为 `220243 µm²`。另有 filler 面积 `233601 µm²`、tap 面积 `578.816 µm²`；当前 stdcell 指标包含 tap。论文单元面积为 `72407 µm²`。简单加宏、加所有 filler 或加减 tap 都不能解释差距。应保留按 cell class 的数量/面积，核查逻辑、clock buffer、timing repair buffer 的阶段增量，而不是改列名或自行选择一个接近论文的面积定义。数据来自 [6_report.json](../backend/results/released_bp_fe_03/orfs_work/logs/nangate45/bp_fe/eval/6_report.json)。
- bp_fe 真实 JSON 总功耗为 `0.170899 W`，作者 parser 最后读文本得到 `0.171 W`。这种三位有效数字取整不能解释与 `0.21 W` 的差距。原功耗报告的 Macro 组约 `0.0805 W`，占 47.1%；因此 Liberty、时钟频率、输入活动假设与宏功耗模型都值得核对，不能只按标准单元面积比例推断功耗。当前 flow 没有本次观察到的向量激励输入；应保留报告的 internal/switching/leakage 和各 cell group，不人为缩放功耗。
- bp_fe 输入时钟为 1.4 ns，原 GRT 与 final SDC 都仍是 1.4 ns。`write_ref_sdc.tcl` 暂时生成 `updated_clks.sdc` 后恢复原周期，不能把那个辅助文件当成实际 final 约束。对其余设计也应核对实际加载的 SDC、时钟、IO delay、exceptions、约束 endpoint 数，而不是假定源配置路径保证完全一致。
- `bp` 的论文 TNS `-23.87 ns` 与现有 `-1732.43 ns` 差距尤其大，优先重新核对论文转录、实际约束及 failing endpoint 分布；不可仅凭 WNS 相近判断 timing 已复现。WNS 只代表最坏端点，TNS 还依赖负 slack 端点的数量和分布。
- 作者 parser 先读 JSON，但后续仍可能用文本覆盖 WNS/TNS/Power，DRT 日志的部分路径也使用首个匹配；这会对格式、重复布线和工具版本敏感。当前新版文本为 `tns max ...` / `wns max ...`，其旧正则不匹配，实际 WNS/TNS 保留 JSON；功耗则被文本覆盖。已有 [独立 audit](../backend/scripts/audit_fullflow.py) 对现有最终结果逐项与原 JSON、DRC 核对通过，故不能把当前大偏差归咎于未经证实的 parser 错误。

建议下一次工作按以下顺序进行。

1. **先补齐证据与报告，不重跑六电路。** 固定原 DEF、实际 seed netlist/SDC、LEF/Liberty、RCX rules、flow tree、容器和二进制哈希；保存 Make 展开后的有效变量，尤其 routing cap、GPL flags、repair flags、`RECOVER_POWER` 和时钟。当前 `parse_config_exports` 只正则读取顶层 `export X = ...`，不展开 include、`?=`、变量替换或命令行覆盖，不能代表完整实际配置。现有 `run_evaluation.py` 也没有完整 seed/flow tree 快照。
2. **首选 bp_fe 作为后端诊断电路。** 仅 11 个宏，已有 released 与 generated 输入、完整 checkpoints、原始数值 audit、GRT 对齐和 cap20 对照；当前完整发布输入约 13.6 分钟，已有 GRT 约 2.35 分钟。这些历史时间用于预算参考，不承诺新机器时间。基线已有结果，可先在独立报告作业中读取已有 placement/CTS/GRT/final checkpoints，增加 cell-class 面积、buffer 数量、endpoint slack 分布、功耗分项与 `check_placement` 结果。GRT 的寄生参数需在报告进程重新估计并标明；不能把中间阶段指标当作 Table 2。
3. **若开展一个新的物理流程对照，先确定研究的问题。** 有可靠作者版本依据时，优先“同一发布脚本/输入，仅工具版本变化”。否则可做“同一当前工具/输入，按论文叙述开启 GPL timing/routability”单独诊断。不要同时再打开被注释的 GRT repair、改变 route cap、密度、时钟或宏位置。若后续要测试 GRT repair，用同一保留 CTS checkpoint 独立对照，并单独记录该改动；先比较 GRT，再只对明确预定的配置完成 DRT/final。
4. **第二个确认电路选 bp_be。** 它已有约 14.3 分钟完整基线，面积/TNS 偏差方向与 bp_fe 相同，可检验发现是否仅限一个设计。`bp` 的异常 TNS 先做原报告/SDC 诊断，再决定是否值得大规模重跑；两例 Ariane 放在机制明确后的跨设计确认阶段。
5. **swerv 先解决输入来源问题。** 不再原样重跑已证实重叠的 released DEF，也不关闭 overlap 检查。应取得正确的作者 DEF/LEF 组合；若只能使用已有合法生成布局，独立报告其生成来源与 PPA，保留发布输入失败行。合法 swerv 控制已经跑过，无需为了证明工具能运行再次提交。
6. **最后才冻结整套六电路复跑。** 只在输入、工具、报告口径和所采用的流程版本确定后进行。成功、macro import failure、物理有效性缺陷、DRC 与基础设施中断分别记录；保留全部六行，不用成功子集均值替代全表结果。

对新 wrapper 的具体建议是：在运行前解析并验证唯一 seed 来源；把全部必需数值检查为有限值；让规范化结果始终以唯一最终阶段 JSON 为来源，同时保留作者原始结果；记录失败阶段和原始异常；将 `PPA_complete`、`placement_valid`、`PDN_valid`、`route_drc_clean` 分开。原作者 evaluator 与历史文件应保持原样。这些是来源与有效性修复，不会通过调整 PPA 参数制造表格一致。

本审查未修改脚本、参数、DEF、历史结果或作业状态。唯一新增文件是本文档。
