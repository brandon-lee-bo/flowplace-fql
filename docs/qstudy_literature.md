# Q、奖励与 actor 损失：文献依据和可验证假设

查阅日期：2026-10-18。本文为新 Q 研究的设计依据，不是新增实验结果。已阅读本轮用户目标、`README.md`、`docs/cluster_execution.md` 和 `docs/graph_fql_offline_to_online.md`；下列建议须通过真实后端实验验证。旧 Table 2 任务保持暂停。

## 1. 先明确我们的问题

当前状态是固定电路上下文，动作是整张宏布局。原始动作经过合法化、DEF 量化及真实后端后得到一次终局反馈，终止 mask 为 0。因此当前 critic 的目标是 `Q(G,a) ≈ reward(backend(legalize(a)))`，没有未来状态的 Bellman 自举误差。我们的主要困难是：数据覆盖、后端指标与代理目标是否一致、原始坐标到执行坐标的非光滑变化，以及 actor 是否利用了预测误差。不能把长序列离线 RL 的所有问题直接套进来。

已有记录已经否定了几个简单推断：更多标签没有自动改善同提案排序；低训练 MAE 不代表可靠的新布局优化；更低 GRT 线长不代表更好时序；行为 flow 加入更多在线布局也可能导致退步。新实验应接续这些证据，而不是重新假设问题只是训练不够久。

## 2. Offline RL：原方法和我们的适配

| 方法与原始来源 | 已核对的机制 | 对本项目的作用与边界 |
| --- | --- | --- |
| [FQL，ICML 2025](https://proceedings.mlr.press/v267/park25f.html)：[论文 §3 / Algorithm 1](https://arxiv.org/html/2502.02538#S3)，[官方代码](https://github.com/seohongpark/fql/blob/e8cd16eb490332924dfa2492097219f181765933/agents/fql.py) | 行为 flow 只做 BC；独立一步 actor 使用同噪声蒸馏及 Q 最大化；无需穿过多步积分反传 Q 梯度。官方默认 twin-Q 聚合为 mean。 | 保留主体；当前全布局终局动作是我们的应用定义。官方 `normalize_q_loss=False`，虽然提供可选归一化，不能声称论文默认开启。 |
| [CQL，NeurIPS 2020](https://papers.neurips.cc/paper/2020/file/0d2b2061826a5df3221116a5085a6052-Paper.pdf)，[官方代码](https://github.com/aviralkumar2907/CQL) | 在 Bellman 拟合外，压低候选/OOD 动作的 Q，并相对抬高数据动作的 Q；常用实现含 log-sum-exp 和采样密度校正。 | 可以借鉴“抑制 actor 访问的虚假高分”，但这里没有 Bellman 自举。`mean(Q)-β·std(Q)` 不是 CQL，也不继承其下界保证。 |
| [Cal-QL，NeurIPS 2023](https://arxiv.org/html/2303.05479#S5)，[官方实现](https://github.com/nakamotoo/Cal-QL/blob/main/JaxCQL/conservative_sac.py) | 在 CQL 压低采样策略 Q 的项内，以参考策略的实测 return-to-go 作下界：`max(Q, reference_return)`，避免过度悲观妨碍在线适应。 | 单步任务的 return 就是实测 reward。若试验 Cal-QL 启发的正则，应保留其作用位置、参考值和候选分布；不能把所有未知布局的预测或真实 reward 直接截断成“至少比行为好”。 |
| [IQL，ICLR 2022](https://arxiv.org/abs/2110.06169)，[官方 critic](https://github.com/ikostrikov/implicit_q_learning/blob/master/critic.py)、[actor](https://github.com/ikostrikov/implicit_q_learning/blob/master/actor.py) | 用数据动作的 Q 对 V 做上 expectile 回归；策略通过优势加权似然学习提取，不需要对未观察动作做 actor 的 Q 梯度优化。官方 actor 的指数权重有上限。 | 可做实测 reward 加权的 flow/BC 对照，区分“挑好数据学习”与“Q 梯度创造更好动作”。终局适配不需要为了形式完整而虚构下一状态。它是 IQL 启发的控制，不是完整 IQL 复现。 |
| [TD3+BC，NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/a8166da05c5a094f7dc03724b41886e5-Abstract.html)，[官方更新](https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py) | 在 TD3 actor 加行为克隆，并做状态归一化。代码使用 `lambda = alpha / mean(abs(Q)).detach()`，actor 为 `−lambda·mean(Q)+MSE(actor,data_action)`，策略延迟更新。 | 适合借鉴 Q 项尺度和更新比例的消融；FQL 的蒸馏目标是同噪声 flow 输出，不能直接替换成随机一条数据动作后仍称同一基线。 |

上述正式方法均有其理论假设、网络与采样选择。我们的神经网络、有限样本和硬合法化适配不自动满足这些假设。尤其当前只有一个固定电路状态时，传统逐状态价值校准的作用不同于长序列任务。

官方 FQL 的精确可选实现是：先对两个 Q 取均值，`q_loss = -q.mean()`；开启 `normalize_q_loss` 时再乘 `stop_gradient(1 / abs(q).mean())`。它缩放的是 actor 的 Q 损失，**不是**重写实测 reward、归一化 critic target，或按梯度范数自动配平。我们建议记录两项梯度范数与夹角属于新增诊断；若之后按梯度配平，也须作为单独改动验证，不能归为原始 FQL 功能。

## 3. Placement：奖励、数据与后端反馈怎么用

| 工作 | 原论文/官方实现中与本轮直接相关的内容 | 可以借鉴什么 |
| --- | --- | --- |
| [FlowPlace，DAC 2026，§3](https://arxiv.org/html/2604.23658v1#S3)，[仓库](https://github.com/lamda-bbo/flowplace) | 以边界先验、合法 mask 合成布局，再按空间邻近造连接；GATv2＋全局 attention 的 flow 通过坐标速度监督训练；推理渐进施加硬约束。预训练没有用每条布局的真实后端奖励。 | 作为行为先验与电路表示来源，不等于已训练好的 PPA critic。区分公开模型的渐进约束采样与我们 actor 最后外接合法化的流程。 |
| [MaskPlace，NeurIPS 2022，§3](https://proceedings.neurips.cc/paper_files/paper/2022/file/97c8a8eb0e5231d107d0da51b79e09cb-Paper-Conference.pdf)，[代码](https://github.com/laiyao1/maskplace) | PPO 逐宏选网格位置；逐步奖励为 `HPWL_before−HPWL_after`。合法 mask 屏蔽重叠；论文的 congestion satisfaction block 在训练时忽略、测试时使用。 | 后端指标和训练目标应分开记录；不能把论文报告拥塞改善写成训练 reward 已包含拥塞。 |
| [EfficientPlace，ICML 2024](https://proceedings.mlr.press/v235/geng24b.html)，[环境代码](https://github.com/MIRALab-USTC/AI4EDA-EfficientPlace/blob/main/src/environment.py) | 结合 MCTS 全局探索与 RL。当前官方环境将 `wire_mask[position]·wire_mask_scale` 作为 HPWL 增量，reward 是其负值再缩放；非法动作另设惩罚。 | 可以学习它对行为多样性与分支探索的处理，不宜把它的 proxy 优化直接作为我们后端目标已解决的证据。 |
| [ChiPFormer，ICML 2023，§3](https://arxiv.org/html/2306.14744#S3)，[仓库](https://github.com/lvchenyangAI/chipformer) | 用跨电路专家轨迹预训练序列策略；论文明确因专家级数据而在离线阶段去掉 reward 项，之后在线用新轨迹微调。 | 需要与匹配的高质量数据模仿基线比较，不能认为“offline”字样本身意味着训练了后端 Q。 |
| [EXPlace，ICLR 2026](https://openreview.net/pdf?id=yqvNwfxRR6)，[环境代码](https://github.com/lamda-bbo/EXPlace/blob/main/src/place_env/place_env.py) | PPO 用归一化后 reg/hier/df/wire/port 代价的加权负和。论文另有时序驱动 DPO 微调；附录 D.5 对比有/无 DPO。 | 专家 proxy reward、时序评估及偏好微调是不同步骤。多目标后端反馈有研究依据，但不应声称公开 PPO 逐步 reward 已含 WNS/TNS，也不把论文 DPO 当作当前入口一定已接通。 |
| [LaMPlace，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/04c0399a47ee4107cd03b08f1f8c3eeb-Abstract-Conference.html)，[官方训练代码](https://github.com/MIRALab-USTC/AI4EDA-LaMPlace/blob/main/02_graph_train.py) | 从离线数据学跨阶段指标，再把预测器变成位置 mask。代码有四项 HPWL/congestion/−TNS/−WNS；先用缩放 MSE，再按阈值切换到成对排序损失，按验证 Kendall tau 保存。 | 直接支持“分指标预测＋排序监督”的实验。但排序不保证 actor 输入梯度正确；其公开脚本在切分前做标签归一化，我们应改为 TRAIN-only 冻结统计，不照搬这一口径。 |

“完整后端只在线下测一次、反复用于不同目标学习”确实是本轮可用的优势。在线方法同样能调用后端，所以新贡献需要是如何利用这些数据和反馈，而不只是宣称 offline 才能用后端。

## 4. 和我们的终局问题更接近的工作

[COMs，ICML 2021](https://proceedings.mlr.press/v139/trabucco21a.html)研究从固定“设计—实测目标”数据拟合 surrogate，再优化设计，避免优化器利用模型在 OOD 输入处的虚假高分。其训练将从数据出发、沿预测分数上升得到的对抗候选压低，同时拟合实测标签并相对保留数据分数；[官方实现](https://github.com/brandontrabucco/design-baselines/blob/master/design_baselines/coms_cleaned/trainers.py)生成候选时停止其梯度，另行更新模型。这个问题定义比多步控制更贴近我们的现有形式，但添加类似项只能称 COM-inspired，不能继承理论保证。

[RankNet，ICML 2005](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)提供成对逻辑损失的基础：预测分数差应与真实偏好一致。它约束样本之间的相对关系，不直接给出宏坐标变化的正确微分。[Sobolev Training，NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/758a06618c69880a6cee5314ee42d52f-Abstract.html)则使用真实导数监督。我们没有真实 EDA 导数，局部实测有限差分只能当有限尺度的变化监督；穿过硬合法化时尤其不能宣称它是真实解析梯度。

[CPO，ICML 2017](https://proceedings.mlr.press/v70/achiam17a.html)说明将成本约束单独建模的意义，但给 surrogate actor 加时序 hinge 并不等于实现了 CPO 的更新或保证。[多目标超体积优化，NeurIPS 2020](https://proceedings.nips.cc/paper_files/paper/2020/hash/6fec24eac8f18ed793f5eaad3dd7977c-Abstract.html)提供了关注 Pareto 取舍而非只看某个加权总分的依据；本轮不需要先引入整套 Bayesian optimization。

## 5. 六个可检验假设

以下公式和配置是根据文献与已有失败记录提出的实验设计，不是上述论文已证明在 placement 上有效的结论。具体预算、尺度和筛选规则应在执行计划中固定。

### H1：把多个真实指标分开学，再组合目标，比只拟合线长更可控

先预测成本向量 `c=[GRT_WL, max(0,−WNS), max(0,−TNS), congestion_H, congestion_V]`。功耗、面积先做数据审计；变化几乎为零或测量口径不清时不强行加入。每维用 TRAIN 统计单独归一化，固定变换并报告逆变换后的物理单位；一维训练标准差为零时不制造人为梯度。

成本向量是辅助预测目标，FQL actor 最终仍须优化一个明确的标量 utility，例如 `Qutility = −Σ wj·ĉj`；不能将向量直接称为标量 return。若以时序上限约束线长，写清 `−ĉWL−λ·max(0,ĉtiming−b)`，并在真实后端检查约束；预测约束通过不等于实测可行。

比较固定共享数据上的三种 utility：WL-only、WL＋timing、WL＋timing＋congestion。先用少量预定权重，并同时比较“直接标量回归”与“多头预测后组合”。时序可用 WNS 主目标、TNS 辅助监督，防止两者相关导致重复加权。拥塞标签若全为 0，应改用有区分能力、定义一致的资源利用或拥塞量，不能靠除一个极小尺度放大数值噪声。

判据：独立布局上的各指标误差、Pareto 支配关系和最终 PPA；不仅比较加权分数。若 scalar reward 更好但 TNS 更差，保留取舍。GRT 的时序、拥塞必须按 GRT 名称报告，不冒充详细布线 rO-H/rO-V 或最终 STA。

### H2：同布局的局部后端对照，比互不相关的全局样本更有助于 Q 的可用优化方向

数据以 base-family 为单位收集中心布局和正负扰动，记录 raw / legalized / executed 坐标及每项真实后端变化。邻域包含随机方向、选定宏移动及 critic 建议方向；训练/验证按完整 family 切分，不能把同一中心的近邻随机分到两边。

在相同标签预算下比较：回归；回归＋成对排序；回归＋局部差分拟合。排序只对真实差值超过预定容差的配对使用，避免迫使模型学习无意义的微小差异。候选公式：`L = L_reg + λrank·softplus(−sign(Δy)·ΔQ/τ) + λdelta·Huber(ΔQ−Δy)`；不必在第一批同时开启两个辅助项。

判据：新 family 的 pair accuracy、top-k regret、沿预测方向后的真实改进率，以及执行位移。必须有随机方向/反方向对照；“整体 Spearman 提高”不是 directional test 的替代。

**可预登记的最小局部/全局比较**：锁定相同训练布局、归一化、初始化与回归项，比较 R（仅回归）、G（回归＋全局成对排序）、L（回归＋同 family 局部排序）、D（回归＋同 family 差分回归）。G 与 L 每步采样相同数量的有效配对；L/D 只使用已有后端标签的同中心变体，重复执行坐标不增加样本数。配对真实差值容差、τ、辅助权重和步数由 TRAIN/内部验证决定，锁定后再看外部验证与方向探针。先分别比较 L、D，不预先只保留它们合并后的最好结果。报告全局相关、局部 pair accuracy、后端方向测试三类结果；按 family 聚合/重采样计算不确定性，不把同一 family 的许多 pair 当独立布局。

### H3：限制 actor 访问虚假高分区域，比单纯扩大 critic 更有效

最小对照为原 MSE critic；再加少量候选保守惩罚，例如 COM 启发的 `L_reg + β·(mean Q(a_adv)−mean Q(a_data))`。`a_adv` 从 TRAIN 动作出发，进行少量受限上升，停止候选路径梯度，再更新 critic；候选生成过程和半径全部记录。若候选来自 actor、随机布局、局部扰动，应分别标记，不能混成同一种覆盖假设。

另可比较小 ensemble 的 `mean−βstd` 评分，但应称启发式悲观评分，验证 ensemble 分歧是否与真实误差相关。它与训练期 CQL/COM 正则不是同一项；twin-Q 也不是已校准的置信区间。

判据：actor 新提案的过估计、真实 PPA、离线/在线早期退步和实际离开数据的程度。若保守项让所有预测都降低、却没有提升选择或真实效果，应判为过度悲观或无效。

### H4：蒸馏与 Q 的相对梯度，而非损失数值本身，决定 actor 是否有效利用 Q

首先冻结 reward 归一化和行为 teacher，记录蒸馏项、Q 项、两项各自对动作及 actor 参数的梯度范数和夹角。用同一批数据、训练步数及噪声比较少量 `α`（例如 1、10、100），再只对有希望的一档比较 Q 项归一化及 critic:actor 更新比例。起始网格是待检验工程选择，不是论文推荐 placement 值。

`Q/mean(abs(Q))` 对 reward 常数平移并不不变，因此它不是所有目标组合的通用尺度修复。若换成零均值 target，其分母可能很小，应有明确 epsilon、固定/动态尺度选择及日志，不能悄悄放大 Q 梯度。

判据：actor 偏离 teacher 的程度、Q 预测收益与真实收益差距、多 seed 后端表现。不能仅以 actor loss 更低选配置。

### H5：在新的 Q 实验中固定行为 flow，能减少在线分布漂移的干扰

已有 16/48 在线数据路径控制表明行为数据也可能造成退步，但“旧数据仍训练”并非“冻结 flow”。从相同离线 checkpoint 出发，比较 flow 固定与 flow 更新；两者 actor 和 Q 都按各自实际计划训练，保持新增后端预算一致。匹配 BC 也应有相同的 flow 设置。

判据：BC/FQL 的配对变化、flow 输出分布变化及在线各轮真实效果。只在某个 flow 设置下 FQL 优于 BC时，结论应限定于该设置，不能把共同的适应收益全部归为 Q。

### H6：实测质量加权学习是重要控制，能检验 Q 梯度是否有独立价值

用 TRAIN 的真实 utility 定义截断的优势权重，比较 uniform BC、filtered BC、reward-weighted flow/BC，以及相同 teacher 下的 FQL。权重由训练数据统计构造；温度和截断上限在验证集选定。若训练 flow 的加权 velocity loss，只能称 IQL/advantage-weighting 启发的适配，不是 IQL 的似然目标逐字复现。

公平条件：相同标签、相同 teacher 采样预算、相同在线反馈量。保留同预算 FlowPlace best-of-N / 已测布局选择参考。若加权模仿与 FQL 一样好，应先承认收益来自利用高质量数据，继续研究 Q 是否能在数据外产生真实改善；不要以名称将其包装成已证明 RL 优势。

## 6. 立即可执行的实验顺序

1. **固定输入审计和指标重标记**：复用已有原生后端标签，检查单位、缺失、合法性、失败、拥塞分布和 TRAIN 家族划分；从同一成本向量构造多种 utility。资源/工具参数固定，新增重计算全部走 Slurm。
2. **critic 层筛选**：先 H1/H2 的小矩阵，再按 OOD 误差考虑 H3；相同标签预算比较容量（原 2×128、小一档、较大一档）和损失，早停/模型选择只使用验证集。
3. **小规模真实方向探针**：验证最有希望的 critic 在新 family 的局部执行方向上是否有作用。若只会排序，先用于候选筛选并如实命名，不直接宣称可指导 actor。
4. **接入 actor**：使用 H4/H5 控制和 H6 对照，保留原 FQL 作为锚点。配置锁定后再做三个训练 seed、新生成噪声及 online 更新，不反复用最终评估数据调参。
5. **最终 PPA 与规模扩展**：跟踪真实线长、时序、拥塞、功耗、面积、DRC、宏合法性及调用成本；在同预算基线和更多电路上验证。

重采样、训练失败、job 抢占和物理失败应分开。抢占/未知结果不是差布局，不能标成负 reward。有效物理失败应预定义处理，并保留失败率；成功但不利的 PPA 不删除。多 seed 是独立训练，生成噪声重复不是新增独立训练。

## 7. 新贡献需要满足什么

直接把 FQL 的网络换成 FlowPlace 骨干、再添加加权 PPA reward，单独看贡献较弱。更有意义的候选是：面向硬合法化后端的局部成对数据与可用 critic；可解释的多指标/约束优化；可验证地抑制 actor 利用代理误差的方法。每一项都需要对照证明实际后端收益，并报告代价、规模与失效条件。目前本文只提供这些可检验研究方向，不宣称已获得创新或改进。

## 8. 2026-10-18：已冻结首轮筛选的解释边界及后续必要对照

首轮已固定为 7 个 critic 变体 × 3 个训练 seed × 2 个数据集，共 42 项任务；每项包含 TRAIN 内三折 family 分组验证及全 TRAIN 重拟合。这一轮是机制筛选，不因下列建议更改已提交配置。固定步数、按 family 分折、每折仅用该折 TRAIN 拟合输入/目标尺度和 HPWL 残差，均能避免相邻变体跨折及预处理标签泄漏。最后 16 条历史 validation 已被观察过，只能作为调参与诊断资料。

本轮结果必须保留以下限制，后续验证时再补齐相应控制：

1. **不能把 `huber_local` 优于 `huber_rank` 解释成“局部监督优于全局监督”。** 前者是同 family 差分平方损失，后者是全局排序逻辑损失，同时改变了配对范围与损失形式。应按 H2 补 R/G/L/D 对照，特别是同 family 排序 L；先比较相同排序损失下 G 与 L，再比较局部排序 L 与局部差分 D。
2. **相同训练步数不等于相同有效配对监督。** 当前随机逐行 minibatch 中，同 family 的两条记录只有同时被抽到才产生 D 监督；数据量、batch 大小和重复抽样都会改变有效配对数。当前结果可作探索，不能宣称已经匹配局部/全局配对预算。后续预先抽取相同数量的有效 G/L 配对，记录唯一 family、唯一执行布局、有效非平局配对和实际辅助项梯度。归一化后的平均 pair loss 不消除配对覆盖与梯度方差的差异。
3. **多头目标必须和直接标量回归分开比较。** 当前 legacy 只回归线长；多头模型预测五项成本后才能组合 timing/congestion utility。因此首轮能比较 WL 预测或探索多指标可行性，不能证明多头结构优于同一 utility 的标量 critic。后续直接用 TRAIN 构造的相同标量 utility 训练同容量 scalar critic，并使用相同训练数据、初始化规则和优化预算；分别报告 utility 与原生 PPA。
4. **训练配对阈值与诊断阈值需要统一解释。** 当前排序忽略小于 `0.01 × TRAIN target_std` 的真实差值，而 family pair accuracy 仅排除绝对差不超过 `1e-9` 的配对。近乎平局的配对可能进入诊断却没有排序监督。后续同时给出全部非平局结果和超过预定物理容差的结果；阈值由 TRAIN 的量纲及可复测性确定，不能看验证结果后移动。pair 仍按 family 聚合，不能当独立样本重复计算置信区间。
5. **分折时 utility 尺度来自各折 TRAIN。** 这是无泄漏做法，但不同折的标量 utility 不完全相同。跨折汇总应先报告各折归一化分数和原生指标，再按预定方法汇总；不能直接把不同尺度的预测拼接成一个全局排序。全 TRAIN 重拟合后固定一套尺度，供后续 actor、在线更新和实测目标共同使用。
6. **42 项筛选不自动给出唯一最佳配置。** 在新方向探针之前，应另行冻结主要 utility、最多推进的配置数、family-CV 选择指标、平局处理和 ensemble β；完整保留所有尝试。16 条历史验证及多个 utility/β 中的最佳单点不能作为独立改进证据。可保留 legacy 锚点及少量不同机制代表；反方向与等位移随机方向必须同时测量，不能只测所选 Q 的正方向。
7. **常数或缺失指标不能提供有证据的优化梯度。** 当前审计的数据五项成本都有观测和变化，但未来某折或新电路可能没有。应显式标记常数/缺失及有效数量；不能把常数目标拟合残差当作可用物理方向。完全缺失的验证指标应保留 `n=0` 和空误差，避免在归一化 MAE 时对空值做除法。部分物理失败保留在审计与失败率中，不以虚构的零 PPA 补入回归。

以上属于对已冻结筛选的科学解释与下一阶段要求，不表示这些后续对照、方向探针或 actor 改进已经完成。排序、相关性和回归误差是代理诊断；真实后端方向及独立策略布局才是可优化性的检验。

## 9. 训练期保守 critic：固定的一项 COM 启发对照

既有 `mean(Q)−β·std(Q)` 筛选未得到稳定改善；它没有改变 critic 的训练。
因此另检验 H3 的训练期机制，不再扩展该后处理 β 网格。原版 COMs 使用
MSE、对偶系数及上升候选；下面是适应当前有界成本、Huber 回归、双模型及
坐标限制的单独实验，不是官方 COMs/CQL 复现，也无价值下界保证。

- 数据、三折 family 划分、初始化、minibatch、优化器及2000步预算与已完成的
  bounded WL 控制相同。仅新增3个 seed ×（3折＋全 TRAIN refit）＝12次拟合；
  原12次 WL 控制通过原验证器核实并绑定，不重训。
- 监督项仍是标准化 B 的 Huber 损失；额外损失为
  `0.1 × mean(z_B(data)−z_B(adversary))`。它同时对两个外层预测求模型梯度。
  `z_B` 使用对应折 TRAIN 的均值、标准差，actor 若使用该模型仍取
  `Q=−mean(predicted_B)`，没有再除以 TRAIN 平均 B。
- 候选只从本折 TRAIN 的原始坐标产生。可移动中心已在[-1,1]内的行参与保守项；
  越界原始行仍完整参与监督项，并明确报告其覆盖率。固定节点不移动。
- 从每个合格锚点沿 `−mean(z_B)` 上升3步，每步将可移动坐标方向归一为 RMS1，
  步长0.01；先截断到中心范围，再将位移缩回 RMS 半径0.02。它是保持可行的
  截断，不宣称是两个约束交集的精确投影。保留锚点及三步访问点中预测 Q 最高者，
  平分取最早。每步及最终候选都停止梯度，不对候选路径反传。
- 每100步记录监督项、保守项、两项参数梯度及比例、参与行数、位移和约束检查。
  报告保留七项原生标签，但模型只预测有界 WL 成本，不能称为七头预测器。
- 按折计算 MAE、排序、局部 pair accuracy、top regret 和常数对照，再按 seed 汇总；
  16条历史验证另报。所有3个 seed、越界预测、输入梯度及覆盖率均保留。
  不因一个历史验证指标漂亮而直接推进 actor；先检查整组结果能否支持所述机制，
  再决定是否值得做真实后端方向检验。负结果关闭这项固定对照，不自动扩参。

实现为 `fql/qstudy_com_screen.py`。独立审查及11项 CPU 测试在1797001通过，
包括 β=0 与冻结控制的模型、Adam、随机数状态、minibatch 和预测逐项一致，
以及真实保守项的中断恢复一致性。正式计划将绑定源代码、原控制产物及训练数据；
这项实验不新增布局标签、actor 训练或后端调用。

## 10. 2026-10-18：其他 offline RL 怎样使用 reward

用户追加要求：查其他 offline RL 的 reward，再判断怎样用于 placement。
本节是新的文献核查与实验建议；不改写前面已完成矩阵，不表示下面的适配
已经训练，也不选择 fresh 候选。原来的开发结果仍为开发证据。

先区分任务 reward、算法损失和评价指标。FQL/IQL/CQL 通常接收环境或数据集
给出的 reward；行为正则、保守 Q 项、优势权重不等于额外的物理质量标签。
我们当前一次输出整张布局、测一次终局结果，是终局 contextual bandit；
因此 `Q(s,a)` 学的是该布局的期望终局 utility，不需要虚构宏块序列或 Bellman
未来回报。flow 的积分时间也不是环境的决策步数。

| 方法与一手来源 | 实际 reward / 使用方式 | 适合借鉴的内容 |
| --- | --- | --- |
| [FQL，附录 E.1](https://arxiv.org/html/2502.02538)，[OGBench v1.1.0 标签代码](https://github.com/seohongpark/ogbench/blob/v1.1.0/ogbench/relabel_utils.py) | 主要 OGBench 任务的 reward 是“尚未完成子目标数量”的负值；迷宫只有一个目标，未达成为−1，达成为0。算法另用 BC flow 蒸馏约束 one-step actor。 | 明确任务完成标准；不能把蒸馏误差当作布局质量。placement 连续指标仍有价值，不必只保留通过/失败。 |
| [IQL 官方预处理](https://github.com/ikostrikov/implicit_q_learning/blob/master/train_offline.py)，[actor](https://github.com/ikostrikov/implicit_q_learning/blob/master/actor.py) | 沿用任务 reward；AntMaze 做 `r−1`，locomotion 按数据集轨迹回报范围缩放。actor 用截断的指数优势权重学习数据里的动作。 | 用实测质量对布局学习加权，作为与 FQL 匹配的数据利用控制；加权 flow-matching 是思想适配，不是原 IQL 似然目标。 |
| [AWAC，附录 A-C](https://arxiv.org/html/2006.09359#A1.SS3)，[官方 actor 更新](https://github.com/rail-berkeley/rlkit/blob/master/rlkit/torch/sac/awac_trainer.py) | 使用任务原有连续或稀疏奖励；附录开门任务为成功指示量减1。用优势的指数权重加权数据动作的似然。 | 质量权重属于策略学习损失，不是新的环境 reward；应区分数据利用方式与物理目标改变。 |
| [TD3+BC 官方实现](https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py) | 任务 reward 和 BC 正则分开；actor 的 `alpha / mean(abs(Q))` 用于平衡 Q 与模仿项。 | reward 单位或尺度变化会影响固定蒸馏系数的实际作用。继续记录两项梯度；不把动态 Q 缩放写成环境 reward。 |
| [COptiDICE](https://arxiv.org/html/2204.08957) | 将任务 reward 和约束成本分开；RWRL 实验中违反指定安全约束产生成本1，策略优化受平均成本上限约束。 | 将线长作为主要目标，时序/拥塞作为独立有预算的成本，检验它是否减少“线长改善却明显恶化时序”的取舍。预测约束不保证每张布局实际通过。 |
| [ChiPFormer，§3.4](https://proceedings.mlr.press/v202/lai23c/lai23c.pdf) | 专家离线预训练去掉 reward；在线阶段按轨迹回报的指数权重回归，并保留探索。 | 后端标签可以给已测布局分配学习权重，先检查不使用动作 Q 梯度能否改善；这同时检验 FQL 的 Q 梯度是否提供独立收益。 |
| [LaMPlace](https://proceedings.iclr.cc/paper_files/paper/2025/hash/04c0399a47ee4107cd03b08f1f8c3eeb-Abstract-Conference.html)，[四指标训练代码](https://github.com/MIRALab-USTC/AI4EDA-LaMPlace/blob/main/02_graph_train.py) | 它是离线监督预测与搜索，不能叫 offline actor-critic。公开代码学习 HPWL、congestion、−TNS、−WNS，结合回归和排序。 | 后端各指标可以分别监督。需要标明标签生成阶段，不能将其全部叫最终 DRT PPA；排序好也不保证坐标梯度有用。 |

另一个直接相关的非 RL 来源是 [PPAPlace，2026-08 预印本，§4](https://arxiv.org/html/2608.13790v1#S4)。
它用同一批布局比较不同阶段标签与最终布线时序的排序关系，支持先验证标签
保真度再决定训练阶段。该论文报告的 GRT/DRT 关系来自其布局来源和流程，
不能外推成我们的相关系数。GRT 不应被预先判为无效，DRT 标签也不自动修复
critic 对未见布局或坐标变化方向的错误。

### 结合本轮证据，接下来怎样利用

1. **先分清标签误差与 Q 误差。** 对已完成的同布局 GRT/DRT 配对，按电路、
   指标和策略分别检查排序、改进方向是否一致。这48个 DRT 结果已经用于
   开发比较；可以继续做开发诊断，不能再叫新的独立测试。若要训练最终布线
   目标，另采 TRAIN 布局并提前保留 family 验证和最终测试，不直接把旧测试
   标签塞入 replay 后继续沿用“未见测试”的说法。
2. **优先补实测质量加权控制。** 在相同 TRAIN 标签、flow 初始权重和预算下，
   比较 uniform BC、按实测 utility 截断加权的 BC/flow、原 FQL。对同一电路
   用冻结的训练参考构造相对质量；温度、上限仅用 TRAIN/验证确定。它不要求
   actor 沿可能错误的 Q 坐标梯度前进；能否改善仍须真实后端验证。
3. **约束式目标作为独立假设。** 初始主要目标可以是
   `r=(WL_ref−WL)/WL_ref`；时序成本取负 slack 幅度超出预定上限的正部，
   拥塞也单独设置容许预算。参考值、尺度和预算事先固定，各成本单独预测与
   报告。先与相同数据上的加权和 FQL 比较，再决定是否值得引入对偶更新；
   这不是完整 COptiDICE 复现，也不继承其保证。DRC/宏合法性继续单独检查，
   机器/许可证/超时等基础设施失败不能标成差布局 reward。
4. **保持 FQL 主体和对照可解释。** 继续保留 flow BC、独立 one-step actor、
   Q 引导及蒸馏正则。先固定 teacher，匹配三训练 seed、相同标签和后端预算。
   新目标须同时报告各项原生 PPA、失败率及 Q 预测改进与实测改进的差距。
   当前 timing/congestion Q 在自己提案上的排序已较差；这支持先核查质量
   学习与输入覆盖，不支持仅增强 Q 权重或声称换 reward 就会解决。

以上是来源支持的下一步假设，尚无新的多 seed 收益或创新结论。原确认规则
保持不变；如推进新分支，应先记录新的开发范围，再锁定独立确认配置。

对单步任务，`−WL/S` 改成 `1−WL/S` 在理想模型下不改变同电路动作排序或
动作梯度。按固定参考写相对改善便于解释和跨电路尺度处理，单独不构成新
优化目标或贡献。reward 缩放会改变固定蒸馏系数下 Q 项的相对影响，需与
真正增加约束、改变目标取舍的实验分开。

## 11. 2026-10-19：固定 expert8 / finalPPA6 的可解释性审计

只审计[专家八目标](qstudy_rule_reward_protocol.md)和
[最终反馈六目标](qstudy_final_feedback_protocol.md)，不改变冻结矩阵。
`B(C)=2C/(1+C)` 严格递增、导数为 `2/(1+C)^2`，单布局排序不变，但大成本
差异被压缩。它是凹变换，随机策略优化 `E[B(C)]` 不等价于 `E[C]`，不能称为
纯单位归一化。`mean(C)=1` 既不证明所有样本接近1，也不证明已经严重饱和。
当前 critic **直接回归 B**，actor 取 `-mean(predicted B)`，没有再穿过该解析式；
不能说该公式直接截断 actor 梯度。TRAIN 标准化后输出恢复 B 单位，同 alpha
或 B 范围仍不等于同动作/参数梯度。模型不截断预测，负 B 等乐观外推仍可能
被 actor 利用；失败 B4 的标签次序不是预测器学会安全边界的保证。

保存审计1798614与96-fit报告1798710显示：四个激活规则在384条 executed TRAIN
均无精确零值，不能统称稀疏 reward；但 group 只有19个值，351/384（91.4%）
等于同一最大值，完全常数检查不能排除此类退化。dataflow 有382个不同值，
channel 有370个，但其阈值/几何分支仍可能局部平坦。seed0全TRAIN refit 的 B
均值约 .998–1.000；除 edge 外，标准化 buffer 中的 target_std 约 .03，受
3%mean 下限约束，不能把该 buffer 值当作标签的真实标准差，也不能据此
认定大 C 尾部已经饱和。dataflow 全局 CV rho 约 .706，而
family-pair accuracy 仅 .4876±.0409（三 critic seeds），不能据前者推断梯度有效。
原始报告路径/SHA见[进度证据](qstudy_progress.md)。所有384条 raw 布局都被执行
流程移动（mean movable RMS .08070），raw 规则导数不是执行规则或后端的精确导数。

| 一手机制 | 真正可迁移的思想与边界 |
| --- | --- |
| [IQL 预处理](https://github.com/ikostrikov/implicit_q_learning/blob/master/train_offline.py)、[actor](https://github.com/ikostrikov/implicit_q_learning/blob/master/actor.py) | 任务 reward 与截断指数优势加权分开；不能照搬 locomotion 的1000步轨迹缩放，质量加权 FM 也不是完整 IQL。 |
| [CQL critic](https://github.com/aviralkumar2907/CQL/blob/master/d4rl/rlkit/torch/sac/cql.py) | 候选/数据 Q 比较是训练时保守正则，不是发明物理 reward；reward 尺度改变时须审查其系数/温度，ensemble 标准差惩罚不继承其保证。 |
| [Cal-QL 校准](https://github.com/nakamotoo/Cal-QL/blob/main/JaxCQL/conservative_sac.py) | 在保守估计内部用数据 Monte Carlo return 校准候选 Q，不是裁剪全部 reward；终局 return 就是同一变换下的实测 utility，不能混用旧线性 return 与新 B。 |
| [TD3+BC actor](https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py)、[FQL 固定版本](https://github.com/seohongpark/fql/blob/e8cd16eb490332924dfa2492097219f181765933/agents/fql.py) | `alpha/mean(abs(Q))` 或 FQL 可选 Q-loss 缩放是 actor 正则平衡，不是 reward 或梯度范数配平。FQL 默认不开此归一化；本实现增大 alpha 是加强模仿。 |

本应用是整布局单步 terminal bandit，监督目标无 Bellman bootstrap/discount。
Euler10 是生成积分，不是十个环境动作；没有将终局反馈分摊到积分步，也不
借此声称解决逐宏 RL 的长期信用分配。等原生结果完整后，至多考虑两项条件提案：

1. **已知规则作为 critic 先验。** 若 dataflow 有有效 TRAIN 变化，但原复合 critic
   的局部分辨/物理输出仍弱，固定 `hpwl_dataflow` 一个目标，对比全学习与
   `predB=b0(raw)+delta_theta(raw)`；其中
   `b0=B(.8*HPWL(raw)/S_H+.2*dataflow(raw)/S_E)`，残差纠正到同一 executed B。
   保持原标签、尺度、folds、网络/actor预算；不能在 B 外直接加规则而称目标不变。
   这只是 raw 函数先验，不是 hard legalizer 的 exact gradient；不用近常数 group
   或非光滑 channel，不从八臂挑有利规则。先做3seed×(3CV+refit)=12fits，若值得
   原生检验再固定3actor×8noise=24GRT。[EXPlace 原环境](https://github.com/lamda-bbo/EXPlace/blob/d5d534a5fbdbc1ab4aa0bd450761277b437d0967/src/place_env/place_env.py)支持规则来源，分解是我们的候选适配。
2. **只检验过强 Q 压力。** 仅在同目标保存梯度及实测共同显示乐观外推/破坏
   teacher 时，固定一个目标比较 alpha30 与原 alpha10，同 critic/teacher/3seed/
   1000updates/8noise，最多新增3actor/24GRT。它检验正则强度，不验证 B 饱和
   或目标优劣；若缺乏正确方向，不能以降低 alpha 强推 Q。两提案均尚未执行，
   只用原 TRAIN，不把已有 development/final evaluation 标签回流训练。
