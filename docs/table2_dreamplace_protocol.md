# Table 2 DREAMPlace 4.1 公开实现协议审查

2026-10-17。本次只读检查公开论文、作者仓库及本地源码；没有运行 DREAMPlace/OpenROAD、提交作业或修改已有代码。本文是下一阶段实验协议建议，不是已经完成的复现结果。

可执行的近期路线是：固定 DREAMPlace 4.1.0 的公开 BB + 两阶段实现，使用同一 benchmark 的全连接 LEF/DEF 将宏明确改为可移动，完成宏合法化后，将宏坐标交给两方法共用的 OpenROAD 后端。必须标为“公开 4.1 两阶段重构基线”，不能把它直接称为已经复现论文所说的全部 strong macro legalization。

## 论文与公开实现之间的边界

[FlowPlace 论文](https://arxiv.org/pdf/2604.23658v1) §4 对 DREAMPlace baseline 指向文献 [4]：Chen、Wen、Liang、Lin，ICCAD 2023，*Stronger Mixed-Size Placement Backbone Considering Second-Order Information*。其两阶段和宏合法化描述不能仅因 Table 2 行标签引用 DREAMPlace 4.0 的文献 [18]，就解释为需要开启 OpenTimer timing-driven placement。论文另明确 OpenROAD 后端进行 timing/congestion-driven global placement。

[DREAMPlace 官方说明](https://github.com/limbo018/DREAMPlace#reference-results-for-macro-placement) 明确限制了 4.1 的公开范围：它只实现该 2023 工作的 BB step 与 two-stage flow。当前本地版本为 `iccad/assets/DREAMPlace-4.1.0`，固定 commit `5d13c9001a3bc900dca1e108e633d5dd45b00701`；现有 CPU 安装位于 `iccad/assets/dmp_install_cpu`，环境位于 `iccad/assets/dmp_env`。

源码中的 `macro_legalize` 是原有 Hannan-grid/Tetris 类合法化与 LP/min-cost-flow refinement，不能据此声称包含 Chen 2023 的完整新宏合法化算法。公开实现能够输出可验证的合法宏布局，但“结果合法”与“使用了论文指定的新算法”是两个独立事实。FlowPlace 的一句 baseline 描述没有进一步标识 strong legalizer 的文件、参数或实现，因此目前也不能反向断言作者必然用了未公开的完整 Chen legalizer；这项对应关系尚未确认。

[Chen 2023 作者公开论文](https://yibolin.com/publications/papers/PLACE_ICCAD2023_Chen.pdf) 的 Fig. 4 与 §III-C 给出更完整流程：从 GP 结果以 cutting-based overlap removal/sweep 构造 sequence pair；关系可行时转为 TCG 并用最小 L1 位移目标的对偶 min-cost flow 求解；不可行时按规模采用释放部分关系的 MILP（文中使用 Gurobi）或 parallel tempering 改变 sequence pair。之后固定宏、保留 cell 当前位置、降低 density weight 进入第二阶段。若自行补齐，只能标成按论文重构的独立变体，并记录 solver、终止规则和合法性测试，不能用随意 packing 替换后称为原算法。

独立公开源码核查还比较了 [AutoDMP 固定 revision](https://github.com/NVlabs/AutoDMP/blob/866a1cb286a0c037ddf27feeeea1e12b68c9161c/dreamplace/ops/macro_legalize/src/macro_legalize.cpp) 与 [MacroRegulator 固定 revision](https://github.com/lamda-bbo/macro-regulator/blob/fe475473c86121bd09e031b6329f18e2d8b9d5ce/DREAMPlace_source/dreamplace/ops/macro_legalize/src/macro_legalize.cpp)：二者的 `macro_legalize.cpp`、`hannan_legalize.h`、`lp_legalize.h` 三个核心文件均与本地 4.1.0 逐字相同，不能靠替换为这些仓库宣称补齐完整新 legalizer。目前未找到该完整实现，是此次有界搜索的结论，不是证明其绝不存在。

FlowPlace [公开 README](https://github.com/lamda-bbo/flowplace#evaluation) 将 `benchmark_utils/openroad` 中 DREAMPlace 配置用于将生成的宏位置转成 DEF。六个 `dmp_config/or_cases/*.json` 均设置 `macro_place_flag=0`；这些配置不是已经启用两阶段的 DREAMPlace baseline 配置。它们还含 `macro_criterion`、`macro_top_k_criterion_k`、`macros2place_each_step`、`remap_flag`、`l_overlap_penalty` 等 stock 4.1 无对应实现的字段。`Params.fromJson` 静默保存未知字段，不代表算法使用它们。公开转换脚本还调用 stock 4.1 不存在的 `_initialize_position()`；因此基线应直接使用 stock `dreamplace/Placer.py`，而非直接复用转换 wrapper。

## 输入与两阶段必须实际生效

1. 优先使用作者原始 OpenROAD benchmark 的未优化全连接 DEF/LEF；若缺失，明确从发布 DEF 派生输入，并逐项记录保留的 rows、IO、net connectivity、macro masters、orientation。宏和所有应参与 mixed-size placement 的标准单元必须可移动；IO 和真实固定障碍仍固定。不得把所有 `FIXED` 文本全局替换成 `PLACED`。只改审核通过的 instance 名单，并保存原始与派生 DEF 哈希及状态差异。
2. stock `PlaceDB.initialize` 的 movable macro mask 为 `area > 10 * mean_movable_area AND height > 2 * row_height`，而作者转换代码使用不同的 OR 判据。必须检查 stock mask 与 benchmark 真实宏名单一致。若没有 movable macro，stock 会静默将 `macro_place_flag=False`；把已经 FIXED 的 FlowPlace 宏 DEF 原样传进去将失去 baseline 的意义。
3. 本地发布 DEF 几何审计中的宏数量为 ariane133=132、ariane136=136、bp=24、bp_be=10、bp_fe=11、swerv_wrapper=28；已观察到的方向是 N/FS。逐一审计名称、master、尺寸、方向与 mask，不能仅验证总数量。C++ macro legalizer 又按高度大于五个 row 判断 movable macros，也应与同一名单核对。
4. 维持 `random_center_init_flag=1`。源码会重新随机初始化所有 movable nodes 的 x/y，而保留固定节点，所以在正确释放实例后，发布 DEF 的优化位置不会作为连续优化初值。初始方向、floorplan 和 IO 仍来自该模板；原始 benchmark 未发布时，这个来源限制要在结果中保留。最终方向遵循 stock 输出行为，不能强制恢复为模板方向；见下方 smoke 纠正。
5. 输入 DEF 若已有完整 NETS，使用其连接；若没有才配套提供已审计的 Verilog。必须核对输入 node/net/macro counts 和下游实际 `1_synth.v/.sdc` 来源，不能无说明地用一份不同综合 seed 替换 DREAMPlace 的逻辑图。两方法最终使用相同后端 seed，且 seed 与模型 benchmark 来源之间的差异要披露。

关键源码：[PlaceDB.py](../iccad/assets/DREAMPlace-4.1.0/dreamplace/PlaceDB.py)、[BasicPlace.py](../iccad/assets/DREAMPlace-4.1.0/dreamplace/BasicPlace.py)、[legalization_db.h](../iccad/assets/DREAMPlace-4.1.0/dreamplace/ops/utility/src/legalization_db.h)、[defs.h](../iccad/assets/DREAMPlace-4.1.0/dreamplace/ops/utility/src/defs.h)。

## 建议冻结的首轮配置

以作者每设计 JSON 中 stock 支持的字段为基础，使用绝对输入路径与独立结果目录，显式设置以下值。它是公开配置衍生的可审查方案，不能宣称作者 Table 2 原始命令就是如此。

| 字段 | 首轮建议值 / 理由 |
|---|---|
| `macro_place_flag` | `1`，启用公开两阶段流程 |
| `use_bb` | `1`，启用论文对应 BB step |
| `two_stage_density_scaler` | `1000`，公开 4.1 默认值 |
| `global_place_stages` | 保留作者单一 stage：1024×1024 bins、1000 iterations、learning rate 0.01、weighted_average、nesterov |
| `target_density` | `0.5`，作者六配置共有值；同时记录运行时实际值，stock 可将低于 cell utilization 的值调高 |
| `density_weight` / `gamma` | `8e-5` / `4.0`，作者配置 |
| `random_seed` | `1000`，作者配置与 stock 默认；首轮一设计一次，不按表格挑 seed |
| `random_center_init_flag` / `deterministic_flag` | `1` / `1`，作者配置 |
| `global_place_flag` / `legalize_flag` | `1` / `1`，执行完整公开 GP 与合法化 |
| `detailed_place_flag` | `0`，作者配置；宏结果之后交由共同 OpenROAD 后端 |
| `detailed_place_engine` | 空字符串，明确禁用外部 NTUplace 调用；stock `Placer.py` 按 engine 路径是否存在判断外部调用，并非只看 `detailed_place_flag` |
| `timing_opt_flag` / `routability_opt_flag` | 显式 `0` / `0`，不把下游 OpenROAD 优化误加入 DREAMPlace baseline |
| `gpu` / `num_threads` | 当前已编译 CPU 为 `0` / `8`；记录机器、线程和依赖。CPU 与作者 GPU 路径不保证数值一致，也不可比较运行时间加速比 |
| `ignore_net_degree` / `enable_fillers` / `gp_noise_ratio` / `stop_overflow` | `100` / `1` / `0.025` / `0.07`，作者配置 |

`NonLinearPlace` 在 `macro_place_flag=1` 时自动把第一 stage 插入一次，先做 mixed-size GP、宏合法化，再固定宏做第二阶段标准单元 GP。JSON 保留一个 stage 即可；手动复制成两个会变成三个。它还原地修改 `params.global_place_stages`，因此每个设计/seed 使用独立进程与新 Params，不能多次复用同一对象。

Halo 是一个未获原 baseline 配置证明的选择。若采用最小“作者配置衍生”方案，建议在计算前一次性冻结作者六配置已有值：ariane133/136 为 60，bp 为 40，bp_be 为 22，bp_fe/swerv 为 20，x/y 相同；同时注明这些值在公开转换配置的 `macro_place_flag=0` 下原本不生效，启用它们是一项重构假设。不要执行后再从 halo=0 与这些值中选择更接近论文者。

Halo 的单位是 DREAMPlace 归一化后的 site width，不能解释为微米。stock `PlaceDB` 将 site width 缩放为 1；`NonLinearPlace` 直接将 `2 * macro_halo_x/y` 加入宏宽/高。现有 Nangate45 site width=0.19 µm，所以 20、22、40、60 分别对应每边 3.8、4.18、7.6、11.4 µm。源码在调用第一阶段宏合法化前移除 halo，故它是 GP 中的空间缓冲，不是最终强制宏间距约束；不得在结果中声称它保证该间距。DMP halo 也不能与 ORFS `MACRO_PLACE_HALO` 混为一谈。

关键源码：[NonLinearPlace.py](../iccad/assets/DREAMPlace-4.1.0/dreamplace/NonLinearPlace.py)、[params.json](../iccad/assets/DREAMPlace-4.1.0/dreamplace/params.json)、[Placer.py](../iccad/assets/DREAMPlace-4.1.0/dreamplace/Placer.py)、[作者 bp_fe 配置](../upstream/flowplace/benchmark_utils/openroad/dmp_config/or_cases/bp_fe.json)、[Nangate45 tech LEF](../backend/thirdparty/OpenROAD-flow-scripts/platforms/nangate45/lef/NangateOpenCellLibrary.tech.lef)。

## 完成判据与共同后端

先选 bp_fe 单例验证导入和两阶段确实执行，再按同一冻结协议跑六电路；这一步先回答实现是否正确，不能据 bp_fe PPA 继续调整密度、halo、seed、时钟或工具版本。

保留完整运行配置（包括初始化后有效参数）、源码/安装文件哈希、随机种子、输入连接/几何摘要、各阶段日志和原始最终 DEF。验证最终数值有限、宏名单完整、无宏重叠、无越界、方向与单位正确；最好额外记录宏合法化前后与最终坐标，确认第二阶段固定逻辑没有遗漏宏。不要只依赖 Python 返回值：stock C++ macro legalizer 的合法性布尔值没有作为异常传播，greedy legalization 失败也可能记录日志后返回非法布局。

stock 完整 flow 在第一阶段后调用一次宏合法化，在最终 `legalize_flag=1` 的完整合法化中还会再次处理宏。保持这一公开行为，不再在输出后暗中添加另一个 placer/packer 修复；若发生失败则保存失败行，另行提出明示的算法变体。

现有作者 OpenROAD evaluator 的 `run_mp` 只导出宏位置，`run_wo_synth` 从发布综合 seed 重建标准单元布局。因此 DREAMPlace 标准单元 DEF 位置不是 Table 2 后端的最终布局输入。两方法应使用相同的 netlist/SDC、LEF/Liberty、tool image、flow tree、GPL flags、repair flags、route cap 和最终提取/报告点。若 root 冻结“按论文开启 GPL timing/routability”的新流程，应同时用于 FlowPlace 与 DREAMPlace，不能把一方法的新流程与另一方法的旧发布流程并列作算法比较。

保留六设计完整分母和失败状态，报告 rWL/WNS/TNS/power/area 的原始值、单位、DRC 与物理检查结果。无法获得作者原版本、DREAMPlace baseline configs/seed、完整 strong legalizer、原始输入方向/模板时，将这些逐项标成未匹配项；不能用“4.1.0 版本相同”代替端到端完全相同的证据。

## 新 runner 的独立只读复审

随后审查了 [run_dreamplace_table2.py](../backend/scripts/run_dreamplace_table2.py) 与 [Slurm wrapper](../backend/scripts/run_dreamplace_table2.slurm)，本审查未执行 DREAMPlace 或提交作业。已核对：所有 movable 坐标归中后重新随机初始化；两个宏分类 mask 与真实名单一致性 gate；原生 BB/两阶段和宏合法化调用；observer 不改变原函数输出或随机流；源 DEF→DMP→最终 DEF 的逐 net instance/pin multiset 校验；固定 IO、halo 恢复、坐标有限性与 row/core bounds 检查；handoff 只改变宏状态为 FIXED，保留 stock 最终坐标 token。Python AST 与 `bash -n` 检查通过。当前没有发现阻塞单个 bp_fe scheduled smoke 的代码问题；运行成功、峰值内存及实际输出仍待该实验验证。

本次复审绑定脚本 SHA256：Python `7740ae3aaa0f956b872b361e360d526eda3ecfe6020b96db2d8520ccf41ec2fa`，Slurm `c2331ba58937f0c931eac896bde1780dec458e765c9c1da6ebbe0bd2c5d82833`。DEF 与下游层次 Verilog 的检查仍只证明 instance/master inventory 一致，不是完整 pin-connectivity 电气等价证明；该限制不因 source DEF 与 DMP 的连接审计通过而消失。

## Smoke 后对方向与 pin 审计的纠正

以上是运行前的静态复审，未发现的两个审计错误在 scheduled bp_fe smoke 中暴露，必须保留失败记录。job `1787984` 完成 stock 两阶段优化后，11 个宏有 7 个 N↔FS 变化，矩形几何检查为零违规；v1 的“方向必须与模板相同”守卫过严。`place_io.cpp:60` 的 `apply` 对非 FIXED 节点按最终 y 所在 row 调整方向，`Node.h` 的 `setOrient` 只写方向字段，既不重跑优化也不修改此前 Python 的 pin 缓存。`PlaceDB.h:525` 的 row 索引除以 site/row height；审计需显式确认 DEF row pitch 与这一高度一致。发布 flow 的 `run_macroplace.tcl` 在 `read_def` 后调用原生 `write_macro_placement`，现存 bp_fe 的 `macro_out` 保留逐宏 R0/MX（对应 N/FS），下游方向不应被手工归 N。

v2（Python SHA256 `0159c3408c781949ec6777f54a73b668a29619bcae4add87a0e78753a108ee14`，job `1787991`）保留原生方向，但用最终 DEF 重读后的 **Python** pin offsets 直接验证 `(x,height-y)`，这个预期仍错误；独立复审也漏掉了 normalization。源码证据是 `PlaceDB.cpp:1731` 在原生数据库读入后先将 LEF N offsets 转为实际 DEF 方向，而 `PyPlaceDB.cpp:540/668` 再将复制到 Python 的数组转换回 N。因此初次与最终重读的 Python offsets 应在同一 N 归一化约定下比较，不能当成最终物理方向的直接证据。

可用现有只读绑定加强审计：从**最终 DEF 新读入**的 `rawdb`，调用 `node.pinX(pin)`/`pinY(pin)` 或 `rawdb.pinPos(pin,kX/kY)` 获取真实 DBU 物理 pin 坐标；按 `rawdb.nodeName(node)`、`rawdb.macroPin(pin).name()`、`rawdb.netName(net)` 对齐连接。初始 N 归一化 offset 除以 `params.scale_factor` 恢复 DBU，按最终 N→`(x,y)`、FS→`(x,height-y)` 转换，再加最终 DEF lower-left 与上述 raw pin 绝对坐标逐一核对。还应验证 raw node master、方向、尺寸、lower-left 与最终 DEF 一致。这样不修改任何 stock 位置或方向，也不靠放宽 tolerance 让错误预期通过。`Pin.offset()` 返回的 Point 与 `Pin.name()` 没有完整的 Python 绑定，优先使用上述已绑定标量接口，避免新增不必要的 C++ 修改。

v3（job `1787996`）按原始 raw DBU offsets 与最终重读 raw DBU offsets 验证前后方向变换，避免额外缩放误差；同时独立检查 Python N-normalized offsets 一致。执行 agent 报告 2,163 个宏 pin 连接均通过，Python offsets 差 0、实际方向 raw offsets 差 0；7/11 宏的方向变化符合 row 规则，handoff SHA256 `d18f962aab7a7280ca4e8c0cb6566aa2c5ddeb5fc5d94494c7957eb2ff9cda58`。该结果是 CPU 流程和输出审计成功；正式 GPU cohort 仍需 GPU 安装及单例 smoke，不能将 CPU 成功混作 GPU 结果。
