# 从规则奖励到真实最终后端奖励

2026-10-19。本文是新一轮 reward 设计依据，不是实验结果。当前 WL 质量加权 FM 只回答“无需 Q 的坐标梯度，利用已有 GRT 质量能否改善布局”；它不能替代专家规则、最终详细布线或 PPA 奖励实验。既有方法/诊断见 [文献笔记](qstudy_literature.md)，当前加权控制见 [协议](qstudy_quality_weighted_flow_protocol.md)。

## 已核实的机制

| 一手资料 | 实际机制及可迁移的思想 |
| --- | --- |
| [EXPlace，ICLR 2026，官方环境](https://github.com/lamda-bbo/EXPlace/blob/d5d534a5fbdbc1ab4aa0bd450761277b437d0967/src/place_env/place_env.py) | PPO 奖励是归一化 reg/hier/df/wire/port 代价的加权负和。reg 偏好边缘/角落；hier 惩罚同组宏包围盒的面积增量；df 是连接权重乘宏中心距离；port 惩罚宏与 I/O 预留矩形交叠。它们是工程代理目标，不是真实 STA 或最终布线标签。 |
| [EXPlace 配置](https://github.com/lamda-bbo/EXPlace/blob/d5d534a5fbdbc1ab4aa0bd450761277b437d0967/config/or.yaml)、[actor](https://github.com/lamda-bbo/EXPlace/blob/d5d534a5fbdbc1ab4aa0bd450761277b437d0967/src/model/actor.py) | OR 配置五项权重依次为 `.35/.30/.15/.15/.05`；物理 position mask 外，actor 还排除加权代价超过最小值加阈值的位置。研究规则 reward 时不能同时引入这种搜索范围限制，否则收益来源不清。 |
| [RTL-MP，ISPD 2022，§4–7](https://vlsicad.ucsd.edu/Publications/Conferences/389/c389.pdf) | 利用层次、连通与时序相关结构分组，同时考虑外框、边缘偏置、阻挡、位置指引、凹槽。式(8)的调参评价为 `0.1 WL + p_outline + p_bias + p_blockage + p_guidance + p_notch`。这不是 RL reward；文章也说明不同设计需要不同权衡，专家代价不等于普适最优。 |
| [COptiDICE，ICLR 2022，式(2)–(5)](https://arxiv.org/pdf/2204.08957)、[官方代码](https://github.com/google-deepmind/constrained_optidice) | 将任务收益和成本预算分开：`max E[r]，s.t. E[c_j]≤b_j`，并处理离线估计误差。可借鉴“线长目标＋时序/功耗成本约束”的定义；仅加入 hinge 惩罚不构成 COptiDICE 复现，也不保证每个布局都满足约束。 |
| [IPL，NeurIPS 2023，式(1)–(5)](https://proceedings.neurips.cc/paper_files/paper/2023/file/3be7859b36d9440372cae0a293f2e4cc-Paper-Conference.pdf) | 从成对偏好学习，用 `r=Q−γV(next)` 连接隐式奖励与 Q。终局一步任务退化为 `r=Q`；仍然需要外部偏好证据，不能以当前 Q 自己的排序证明自己正确。 |
| [Flow-to-Better，ICLR 2024，§3–4](https://proceedings.iclr.cc/paper_files/paper/2024/file/8951f484e8242b7f74817fdc390dd954-Paper-Conference.pdf) | 用偏好指导扩散模型产生更好的整条轨迹，再模仿；实验也使用由真实任务回报构造的 scripted preferences。它是 diffusion 方法，不能因名称中的 Flow 将其写成 Flow Matching。可借鉴整布局偏好，但不是自动获得真实 PPA 标签的方法。 |

## 规则的明确计算对象

以下是适合整布局终局任务的**自定义代理**，不声称逐字复现 EXPlace 的逐宏折扣回报。以执行布局的宏矩形 `R_i`、中心 `p_i`、芯片宽高 `W,H` 为输入：

- 边缘代价：`E_edge = mean_i min(l_i/W, (W−r_i)/W, b_i/H, (H−t_i)/H)`。矩形合法时非负；它鼓励留出中心区域，不能证明布线更好。
- 分组代价：`E_group = mean_g area(bbox({R_i:i∈g}))/(WH)`。仅使用来源已核实、固定的非平凡分组；不能将自动连通社区直接称为 RTL 语义模块。同一组包围盒缩小也可能造成局部拥塞。
- 连接代价：`E_flow = Σ_{i<j} F_ij · ||((p_ix−p_jx)/W,(p_iy−p_jy)/H)||₂ / Σ_{i<j} F_ij`。`F` 是结构连接代理，不能冒充 STA criticality。
- I/O 代价：`E_io = Σ_i Σ_k area(R_i∩K_k)/(WH)`，`K_k` 为固定的 I/O 预留矩形。先核实是否已经合并矩形、是否重复计数及实际非零覆盖。

EXPlace 的 [parser](https://github.com/lamda-bbo/EXPlace/blob/d5d534a5fbdbc1ab4aa0bd450761277b437d0967/src/problem_instance.py) 通过压缩后的结构图构造 dataflow 权重，默认最多两跳，每条有效路径按 `2^(−hop)` 累加并对称化；这不是由 slack 推得的权重。[环境构造](https://github.com/lamda-bbo/EXPlace/blob/d5d534a5fbdbc1ab4aa0bd450761277b437d0967/src/env_utils.py) 中 port 坐标和预留矩形使用网格单位，原始尺寸按各轴比例转换。不能直接将它们与归一化中心坐标相交。

| 本项目可获得什么 | 当前边界 |
| --- | --- |
| 宏尺寸、中心、连线与 pin offsets | [现有 HPWL 实现](../analysis/placement_metrics.py) 可计算宏超网 HPWL；它排除标准单元及显式 I/O 终端，不是全芯片 HPWL 或 routed WL。 |
| `macro_clusters/dataflow_mat/id2index/port_pos/pin_blocking_rectangles` | FlowPlace 的 `upstream/flowplace/benchmark_utils/openroad_utils.py` 读取这些缓存字段，但没有重新推导。**字段存在不等于实际有用**；须审计 bp_fe 的单位、ID 映射、非零权重及分组覆盖，再决定激活哪些项。 |
| 宏 pin 与静态网表连接 | 可构造连接强度或距离；当前图边没有真实 STA criticality。需要时从固定 TRAIN reference 的网表/SDC/liberty/STA 取路径或 net 权重，并绑定版本，不能使用评估布局的路径信息拟合 reward。 |
| 最终 DRT 线长、WNS、TNS、功耗、标准单元面积、DRC | 需要真实最终后端标签。布局可改变缓冲/尺寸调整与布线，宏总面积本身固定。功耗比较需保持活动假设、库、角点和分析阶段一致。 |

## 三个有区分力的目标假设

这些是候选实验定义，执行前另锁定一次权重、预算和有效项，不开启新网格。所有目标都同时保留未缩放的原生指标。

| 假设 | 建议目标与最小匹配对照 | 能回答什么 |
| --- | --- | --- |
| **A：规则是否提供线长遗漏的信息？** | `C_rule = HPWL/S_HPWL + λ·mean_{k∈active}(E_k/S_k)`；比较同数据/同学习器的 HPWL-only 与 rule-composite。固定同一合法化与可选动作，不加入专家 soft pruning；可先用质量加权 FM 避免把 Q 梯度误差混入规则检验。 | 规则是否在真实最终 PPA 上有增益，而非只让自己的规则分数变好。专家项恶化后端同样是有效结果。 |
| **B：真实最终反馈是否优于早期代理？** | `r=−DRT_WL/S_WL`，将 `max(−WNS,0)`、`max(−TNS,0)`、功耗、面积作为独立预算/guardrails；若实现标量惩罚，明确 `C=DRT_WL/S_WL + Σ λ_j·max(c_j−b_j,0)/S_j`。比较相同新 TRAIN 子集、相同学习器的 GRT-target 与 DRT-target。 | GRT→DRT 的目标错配是否限制改进。负 slack 不等于运行失败；TNS/WNS 同源相关，避免无解释地重复放大。 |
| **C：规则能否补充稀缺的最终标签？** | 在 B 的相同 DRT 数据上，比较 final-backend-only 与 `C_final+λE_verified/S_E`；只加入审计确认有变化的一项规则，先验固定选择，不能从旧评估48条中挑赢家。 | 专家先验是否在真实终局监督之上仍有额外价值，而不是来源/数据预算或后端阶段不同造成的差异。 |

行为模型的数据先验与最终合法化本身已有边缘偏好，新增边缘奖励可能只是重复已有偏置；所有对照保留相同 sampler、合法化权重及几何流程。是否需要修改这些流程是另一问题，不能与 reward 项一起改变后归因于 reward。

可选的目标表达是成对偏好：先根据预先规定的实际 PPA 可行性与主指标形成 `a⁺≻a⁻`，用 `−log σ((q(a⁺)−q(a⁻))/τ)` 学习。它避免强行解释很小的绝对分差，但阈值、平局和预算仍由 TRAIN 决定；规则生成的偏好只能验证规则模仿，真实后端才能检验工程质量。当前不需要另引入一套 IPL/FTB 网络。

## 数据、硬约束与判定

1. **新的最终后端 TRAIN 标签**：在原 TRAIN family 中预先固定新子集及选择规则，完整记录 raw→legalized→DEF/native 坐标链；对全部计划请求保留成功、物理失败、基础设施失败和缺失。既有48条 development final-DRT 是评估资料，不能回流为本轮 TRAIN 或据此挑权重；后续确认数据也不得参与。
2. **尺度仅从 TRAIN 拟合**：`S_k` 使用已声明、非零的 TRAIN 均值或物理参考；常数/空项显式停用，不除极小数制造梯度。fold-CV 各折独立拟合，部署用全 TRAIN 固定尺度。比较不同目标时另监测 actor 的蒸馏/Q 梯度压力，不能只匹配 loss 数值。
3. **硬约束与软偏好分开**：边界、宏重叠、固定节点和正式设计 blockage 按实际执行检查；边缘偏好、分组紧凑、经验 I/O 预留和凹槽代理默认是软项。没有具体几何规则，不能声称某个“规则 reward”保证 DRC=0。
4. **失败口径一致**：若延用有界成功成本，可使用 `B=2C/(1+C), r=−B`，已确认物理失败单独为 `−4`；这是本项目自定义变换，不是上述文献规定。未完成/工具故障不生成虚构 PPA 或失败 reward。DRC 非零的处理须预先声明，既不能偷偷删样本，也不能事后改作失败。
5. **最终证据**：同一来源、同一 TRAIN 标签预算、同三训练 seed 与相同生成噪声对照；分别报告 policy mean、失败/缺失分母、最终六项 PPA/DRC，以及 GRT 拥塞诊断。最终 DRT 没有同定义拥塞输出时保持 NA。规则分数、预测相关、rank weight 或更高 Q 都不能替代真实后端改进。

## 已覆盖的拥塞 reward 与当前范围

[已完成的 bounded 三臂对照](qstudy_bounded_objective_protocol.md) 已将真实
GRT H/V overflow 直接加入 reward，不能将当前工作概括为只有 channel 几何代理。
加入 H/V 后，相对 timing 臂，水平溢出均值从11.333降到4.083、垂直从2.000
降到1.167，但线长增加16,230.042µm；声明的拥塞 utility 仍差于 bounded WL。
这些是三训练 seed 的完整结果，说明存在取舍，没有证明综合目标已稳定改善。

这里的 H/V 是 OpenROAD GRT OpenDB marker 上正 demand−capacity 的边溢出总和，
不是详细布线 DRC，也不直接等同 Innovus 的 rO-H/rO-V。当前八项专家目标中的
channel 仅是几何代理；六项最终反馈目标没有同义最终拥塞标签。两套实验仍保留
GRT H/V 作为评估诊断，不能把缺失的最终拥塞记为0。

若完整结果支持进一步拆解，可另预先声明“仅 WL＋H/V 预算、关闭时序项”的
匹配对照，辨别与时序项的干扰；阈值和尺度只从 TRAIN 确定。它仍是软惩罚，
不构成硬约束保证或 COptiDICE 复现。目前不为覆盖要求新增此臂，也不修改已经
冻结的八项专家与六项最终反馈矩阵。
