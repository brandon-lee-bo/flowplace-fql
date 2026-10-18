# Table 2 宏位置交接与轨道对齐

输入 DEF 的宏坐标不能直接假定与最终 DEF 逐 DBU 相同。作者流程的 `run_mp`
调用原生 `write_macro_placement`，生成 `place_macro ... -location ...
-orientation ...` Tcl；`config_eval.mk` 通过 `MACRO_PLACEMENT_TCL` 执行它。
生成的命令没有 `-exact`，原生宏放置会对齐布线轨道。本轮保留这套作者流程。

补充脚本 `analysis/audit_table2_macro_preservation.py` 分别记录两件事：

- **输入到最终位置的严格差异**：宏名称、master、方向、整数 DBU 坐标；不设
  位置容差，也不把轨道对齐造成的位移写成“完全没有移动”。
- **原生宏放置到最终位置的保持情况**：最终位置逐项对应
  `2_3_floorplan_macro.log` 的 `MPL-0035` bbox 与方向，同样零 DBU 容差。
  独立使用物理任务冻结的 LEF 尺寸重新检查最终宏矩形重叠与 core 越界。

脚本还核对最终 DEF 与已提交归档的 per-file inventory SHA256。原始日志、
报告和 manifest 保持在持久目录，最终 DEF 可从已验证的 scratch 缓存读取。
此检查不证明标准单元合法、布线 DRC 为零、功能等价或每一个中间步骤均未移动。

## 六项已完成结果的核查

来源是 `source_era_progress_v5/report/summary.json`，包含全部 12 项分母，其中
6 项已验收。其余 6 项明确标为未检查，不参与“通过”计数。

| 方法 | 电路 | 宏数量 | 与输入坐标不同 | 最终与原生宏放置一致 | 最终宏重叠/越界 |
| --- | --- | ---: | ---: | --- | --- |
| FlowPlace | bp | 24 | 24 | 是 | 0 / 0 |
| FlowPlace | bp_be | 10 | 10 | 是 | 0 / 0 |
| FlowPlace | bp_fe | 11 | 11 | 是 | 0 / 0 |
| FlowPlace | swerv_wrapper | 28 | 28 | 是 | 0 / 0 |
| DREAMPlace | bp_fe | 11 | 11 | 是 | 0 / 0 |
| DREAMPlace | bp_be | 10 | 10 | 是 | 0 / 0 |

首批四项的 56 个宏，其 x 坐标与方向不变，y 位移最大 210 DBU，即 0.105 µm。
后续 job 1788253 覆盖上述六项共 94 个宏，全部最终宏位置严格对应原生宏放置
阶段，且无宏重叠或 core 越界。完整差异记录在报告中；不能预先推广到尚未
核查的其他电路。

最初 job 1788226 因“输入到最终必须完全相同”的假设失败；job 1788228
保留全部严格差异，并证明四项最终宏几何仍合法。随后增加对原生宏放置阶段的
独立比较，job 1788230 完成且 exit 0；严格输入差异仍保留，没有放宽容差。

结果：
`/fact_data/bohanli/flowplace_fql_backend/table2_reports/macro_preservation_progress_v4/summary.json`。
该报告绑定 runner SHA256
`521fe0f12165be6590ca407b4bc52509cf6e014ce77a496e7011c664f3cc37e4`。

全部物理任务完成后，需要对完整 12 项再次运行此核查。后续替换生成算法时，
需明确原生对齐与独立精确导入两种协议，并检查所选协议的最终宏合法性；
两方法必须统一，不能混用。

## 大电路导入后的提前核查发现

在尚未结束的六组任务上，job 1788302 单独检查了原生宏放置日志的 588 个宏，
保持零 DBU 容差。五组合法；DREAMPlace swerv_wrapper 有一对宏出现
140 DBU（0.07 µm）纵向重叠，横向交叠 30400 DBU（15.2 µm）。该检查因真实
几何问题 exit 1，不能计为通过；最终 DEF 检查仍需完整 flow 结束后再做。

对应两个 `fakeram45_2048x39`：bank1 为 FS，输入 y=539.000 µm；bank6 为 N，
输入 y=758.800 µm。宏高度为 219.800 µm，因此输入恰好相接且合法。原生导入
分别变为 y=539.105 µm、758.835 µm，前者顶部为 758.905 µm，产生上述重叠。
这证明至少这个案例不能把“输入合法”推广为“原生导入后仍合法”。

证据：`/fact_data/bohanli/flowplace_fql_backend/table2_reports/native_macro_geometry_early_v1/summary.json`。
现有主实验不变。独立 probe 1788310 测试 `place_macro -exact`，但此旧工具没有
这个选项，原始错误日志已保留。随后 probe 1788312 单独测试原生宏导入后使用
OpenDB `setLocation` 精确保留输入坐标；它仅验证导入，不进行 GP/CTS/routing
或 PPA 优化，结果不能替代主表行。任何修正后的完整流程必须另行冻结并同时
用于两种方法，不能仅修正一个有问题或数值更差的结果。

probe1788313保留了实际LOCKED状态但旧断言错误；1788314修正审计预期后28/28
通过。统一修正队列及最终验收脚本见`table2_exact_import_protocol.md`。
独立检查1788342再次证实两方法共56宏的恢复结果，并确认1DBU负例可被识别。

大电路补充验收1788358已完成：原组8/12项、362个宏，最终位置全部逐DBU对应
原生导入日志且矩形合法；完整12项分母保留，其余4项仍未验收。
包含FlowPlace全部六设计，但不能据此预判尚未结束的DREAMPlace swerv。
证据：`/fact_data/bohanli/flowplace_fql_backend/table2_reports/macro_preservation_progress_v5/summary.json`。
