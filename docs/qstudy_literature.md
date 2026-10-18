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
