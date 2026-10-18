# Q-study：行为 flow 数据来源对照，2026-10-18

本补充接受新的研究方向：**直接让当前行为 flow 生成布局，再用同一 EDA 后端打标签，供 Q 学习。** 这属于新的离线数据采集；只有之后策略用自身新增反馈更新参数，才称在线 RL。本文件规定来源对照，不改动已经冻结的 42 项 critic 筛选矩阵，也不把计划写成已完成结果。

## 1. 已核实的旧来源与当前 teacher

| 项目 | 历史公开 FlowPlace 数据 | 本次当前 FQL teacher 数据 |
| --- | --- | --- |
| 权重 | 公开 `best.ckpt`，SHA `0fe61ef666c7ff34c32b1ea1b8fe23a1c8b45151955f0f16366a121fde5fd359` | 对应设计的 `common.pt` 中 `flow` 状态；绑定具体路径、文件 SHA 和来源训练 seed |
| 训练状态 | 公开预训练模型 | 从公开模型继续共同预热：flow 300 步；随后的一步 actor 800 步预热不更新这个 flow |
| 噪声 | uniform `[-1,1]`，历史采集 seed `20261017 + family` | 同一 uniform prior，新的 seed 基数 `5300000`；保存实际噪声和 SHA |
| 采样函数 | `baseline.api.generate → FlowMatchingModel.reverse_samples`，20 步 | `GraphPolicy.sample_flow(noise, steps=10)` |
| 中间更新 | 先预测终点 `u1=x+(1−t)v`，对 `u1` 做硬合法化，再用 `t_next` 将当前状态与合法终点混合；每步恢复固定节点并 clamp 到 `[-2,2]` | 标准 Euler：`x_next=x+v/10`；每步恢复固定节点，不做中间硬投影；末端 clamp 到 `[-1,1]` |
| 输出后的执行 | 原始提案再通过统一外部 `legalize`，导出 DEF，执行固定后端 | 同一个外部 `legalize`、DEF 导出和固定后端 |
| 成对变体 | 历史保留 variant 0 和 1：原布局及逐坐标高斯 `σ=.04` 扰动后 clip 的布局 | 原布局及一个预定随机方向扰动，目标为 movable 坐标 RMS `.02`；记录 clip 后实际位移 |

这不是只把积分步数 20 改成 10。**权重、采样更新式、逐步合法化和扰动构造都可能改变数据分布。** 当前 FQL teacher 的目标是其 Euler 末端原始坐标；它不是每一步外部合法化后的坐标，也不是一步 actor 输出。

已检查的实现入口：

- [collect_dataset.py](../collect_dataset.py)：64–106 行加载公开模型、按噪声生成 teacher、构造 `.04/.15` 扰动，并外部合法化。历史后端子集仅取 variant 0/1，没有取 `.15` 的 variant 2。
- [baseline/api.py](../baseline/api.py)：44–99 行的默认 `steps=20`、`hard_constraint=True` 和原始采样器调用；外部合法化使用 grid 128、padding `.001`、`w_reg=.1`、`w_dist=1`。
- [upstream FlowMatchingModel](../upstream/flowplace/flow_matching/models.py)：194–246 行的终点预测、硬投影和插值；[时间表](../upstream/flowplace/flow_matching/schedulers.py)为含首尾端点的均匀时间表。
- [graph_o2o_models.py](../fql/graph_o2o_models.py)：`GraphPolicy.restore/velocity/sample_flow` 明确定义当前 10 步 Euler teacher。
- [graph_o2o_train.py](../fql/graph_o2o_train.py)：共同预热和后续每步 flow 更新分开；[qstudy_config.py](../fql/qstudy_config.py)中的 `common_checkpoint()` 选择已审计的共同预热父权重。
- [qstudy_data.py](../fql/qstudy_data.py)：历史 Q 数据导入、TRAIN/validation 边界，以及固定的 256 个行为目标。

旧数据最初采集时只有宏级 HPWL proxy；本次使用的是随后补充且已审计的真实 GRT 标签，不能把原 proxy 数组当作后端 PPA。bp_fe 的原始 192 个 TRAIN teacher 布局和 64 个已标注原始提案构成现有 256 个行为训练目标；其中 192 个并不因此拥有后端标签。

## 2. 新采集的固定边界

计划使用 `fql/qstudy_teacher.py`，采集每个已声明设计、指定 common teacher 的 40 个新噪声家族，每家族两条记录。导出完成与后端标签完成分开登记。

| 用途 | 家族 | 布局请求 | 能否训练 Q |
| --- | ---: | ---: | --- |
| teacher TRAIN | local family 0–31 | 64：32 个 base＋32 个扰动 | 可以，使用成功布局的真实后端指标 |
| teacher validation | local family 32–39 | 16：8 个 base＋8 个扰动 | 不可以 |

生成与后端请求应固定并保存：设计/cache SHA、common checkpoint SHA、实际噪声及 SHA、生成代码 SHA、采样步数、uniform prior、固定节点 mask、扰动随机种子、目标/实际 RMS、原始坐标、合法化坐标、DEF SHA、后端原生结果与实际几何。所有生成和后端工作通过 Slurm。后端保持当前八线程、工具/工艺/时钟/密度/布线设置，标签阶段是 GRT，不称最终 Table 2 PPA。

请求分为 `data_train` 和 `data_validation`。只有前者标为 `offline_train_eligible=True`；两者均不自动进入 actor 的在线 replay。先保留请求用途，再决定各对照的数据集合，不能因为看见某个结果好就改 split。

**家族身份必须带来源。** 使用 `(design, source_namespace, source_checkpoint_sha, local_family)`；旧来源可命名 `public20hard`，新来源命名 `common0_euler10_noise5300000`。新导出器可用 `5300000 + local_family` 作为兼容整数 ID，同时保存完整命名空间。旧 family 0 与新 local family 0 既不是同一家族，也不是同噪声配对；不能因为编号相似而合并。跨来源或跨 split 的相同原始/执行布局要单独查重并记录，不当额外独立样本。

这里的 teacher 来源 seed 0 选择了一套固定权重。之后用 Q 训练 seed 0/1/2，是在同一个数据池上独立训练三个 critic，**不是三套 teacher 数据分布**。若后续 actor 改用其他 common seed 的 flow，需检查相应来源差异，或者单独采集并明确增加预算，不能声称 seed 0 的 teacher 标签已经覆盖所有 actor teacher。

## 3. 主要对照：同样 64 个真实 TRAIN 标签

先锁定同一 critic 架构、输入、损失、训练步数、优化器、三组训练 seed 和后端设置；来源是主要研究因素。若同时研究两种 critic，按“来源 × critic”完整报告，不让每个来源各自挑选最有利模型。

| 对照 | 历史来源 TRAIN | 当前 teacher TRAIN | TRAIN 总量 | 测试问题 |
| --- | ---: | ---: | ---: | --- |
| O64 | 64：32 家族×2 | 0 | 64 | 旧数据的参考表现 |
| T64 | 0 | 64：32 家族×2 | 64 | 换成当前 teacher 数据是否改善目标分布上的 Q |
| M64 | 32：16 家族×2 | 32：16 家族×2 | 64 | 相同标签量下混合来源是否兼顾两种分布 |

M64 的 16＋16 家族在查看新后端标签之前固定。本次实现使用 NumPy `default_rng(74241)`，依次打乱排序后的旧 TRAIN 家族和预先声明的新 TRAIN 家族，各取前 16 个，将实际名单写入不可变 `plan.json`；必须保留每家族两个变体，不能逐行随机抽样或按质量挑选。三个对照均评估相同的旧 validation 16 条和新 teacher validation 16 条，分别列结果，不把两侧混成单一验证均值后隐藏分布差异。

**扰动构造的必要补充。** 目前 O64 与 T64 的扰动半径不匹配，因此首轮只能归因为“数据来源与邻域构造的整体改变”。不增加后端调用即可补充 base-only 控制：O32-base、T32-base、M32-base（16 旧 base＋16 新 base）。它们分别使用同样 32 个原布局标签，去掉 `.04` 高斯与 RMS `.02` 的扰动差别；验证也另报两个来源各自的 8 个 base。不能用 32 标签的 base-only 数字直接宣称优于 64 标签方案。

第一轮只改变 Q 的标签来源。现有 256 个 `behavior_action`、共同 flow/actor 起点和 teacher 采样器保持一致；不同时把新样本加入行为 flow 的训练目标。之后如果希望让新布局同时训练 flow，把它作为独立因素，与“仅改 Q 标签”做匹配对照。

## 4. 验证、尺度和数据排除

- **每个对照做 TRAIN 内 family 分组交叉验证。** mixed 对照按来源分层分家族，两个变体始终同折；每折仅由其训练部分拟合特征/目标尺度、残差基线及损失参数。不要把同家族两个变体随机拆开。
- **两侧验证全部固定。** 旧 16 条已经被观察过，是历史调参资料；新 16 条起初是新 teacher 来源验证，第一次用于选配置后也成为开发验证，不能反复称全新最终测试。最终策略确认另用未参与决策的新噪声与实测后端。
- **原生单位结果优先。** 报告 WL、带符号 WNS/TNS、H/V overflow，以及可用功耗/面积；critic 内部可用对应正成本。按各来源 TRAIN 拟合的 utility 尺度可能不同，不能直接比较数值大小；跨对照比较 utility 时使用事先固定、仅来自旧 offline TRAIN 的共同报告尺度，并同步报告原生指标。共同报告尺度不是给 T64 拟合网络或选择损失的标签。
- **先全 TRAIN 重拟合，再固定参数评估两侧验证。** 报告 MAE、Spearman/Kendall、家族内 pair accuracy/有效配对数、固定候选池 top-choice regret；补充 base-only 和按家族平均的结果，避免只报一个最好的相关系数。多 seed 的均值/样本 SD 先在各 seed 内汇总，家族/变体/fold 不当独立训练 seed。
- **验证、原 heldout、历史在线反馈及已观察诊断提案不进入新离线 TRAIN。** Q 的输入仍是后端运行前可得的原始提案和电路/物理先验。实测后端 PPA 作为监督目标，不作为该样本的 critic 输入；执行数据库仅用于审计，不作为前置特征；模型预测不得冒充实测标签。
- **保留失败和重复。** 基础设施故障恢复同一请求，不赋负 PPA；确认的物理失败保留状态、失败率和预定 utility 语义，不造出零线长/零时序。若有失败，报告请求预算与实际成功标签数，不能把“请求 64”写成“64 个有效 PPA 标签”，也不能为了凑满高质量成功样本静默替换家族。相同执行布局的重复请求仍占实际采集预算，统计上不增加独立证据。

## 5. 较大预算与历史结果的使用范围

`O384 + T64` 是 **448 个 TRAIN 请求/标签的扩充实验**，按成功数量另行核算；它不能与 O64/T64/M64 当作等标签预算比较。应同时报告 O384 原参考，并明确新增的 64 个 teacher TRAIN 和 16 个 teacher validation 请求的成本。每项新物理评估计算一次，之后可供多个损失对照重复使用，不能为每个 critic 重复跑相同布局。

已有四个 10 步行为 flow 的真实 GRT 结果可以保留为历史描述：[graph_o2o_flow_probe.py](../fql/graph_o2o_flow_probe.py)使用的是早期 BC/FQL **离线训练完成后的共享 flow**，读取四个早已固定的 evaluation 噪声，并标记不用于训练。它不是本次 `common.pt` 中仅预热 300 步的 flow，也不是 40 个新家族。已有记录的四布局均值为 2.411552 m；它和匹配一步 BC 的比较只说明那个小诊断，不能加入 T64、充当新的 teacher validation，或重新命名成新确认实验。

## 6. 什么结果支持什么结论

若 T64 在新 teacher validation 和新家族方向测试上更可靠，而旧 validation 没变好，支持“旧 critic 与当前策略的提案分布不匹配”，不等于通用预测能力提升。若 M64 同时改善两侧，支持来源覆盖有帮助；如果只有 base-only 控制成立，应解释扰动/合法化的作用。若仅回归误差或整体排序变好，而家族内排序、实际正负 Q 方向仍近似随机，优先作为候选评分器，不能直接宣布 actor 优化方向已可靠。

接入 FQL 后，保持匹配 BC、相同 teacher、训练/采样预算和新评估噪声；Q 建议方向要同时有反方向及等位移随机方向对照。只有真实后端的独立策略布局稳定改善，才支持当前数据来源控制对 placement 有实际价值。以上均是待验证假设；新数据导出完成、EDA 标签完成、critic 完成、策略完成与确认评估完成必须分别记录。

## 7. 独立实施入口

`fql/qstudy_source.py` 与 `.slurm` 负责 `plan → check → assemble → train → report`；不修改冻结的 critic 模块。主要目录为 `/fact_data/bohanli/flowplace_fql/qstudy_v1/source_controls_v1`。旧来源是已审计的 `data_audit/bp_fe64/dataset.npz`，新来源是 `teacher_source_v1/bp_fe_common0/dataset/dataset.npz`。

- `plan` 在新标签产生前仅按家族元数据和固定 seed 冻结六组名单，保存输入文件 SHA 和 teacher common 权重 SHA。
- `check` 检查旧来源真实文件绑定、256 行行为目标，以及未来六组分组 CV 的可重复、互斥、完整和来源平衡性质；不运行模型或后端。
- `assemble` 等新数据完成后验证每条原始监督的坐标、原生指标和文件 SHA，保留失败与 NaN mask；同一物理配置、工具版本和八线程要求不变。六组均保留 32 条验证数据及来源注释，并原样复制同一套 256 个行为目标。
- `train` 是 36 项矩阵：六来源设置 × `legacy`/`huber_residual` × critic seed 0/1/2。每项 2000 步、3 折 CV 与全 TRAIN refit，复用未改动的 `qstudy_screen.fit()`；只在调度层按来源独立分组家族。每来源用 `41307 + 1009×来源序号` 打乱家族后均分三折，两个变体同折。初始化、优化器、步数和 minibatch seed `810000+critic_seed` 沿用筛选实现。
- `report` 逐项核实模型 checkpoint、绑定和保存的 CV 行号后，输出原始成本单位的来源分层指标、三 seed 均值/样本 SD、旧/新验证矩阵和相同候选上的配对差值。OOF 的 O64 和 T64 没有相同来源目标，不能凭共同 critic seed 冒充配对样本；OOF 配对只用 `(source,family,variant)` 真正交集。旧/新验证目标在所有设置之间相同。缺失的 legacy 时序/拥塞预测标为 NA，不填零；没有自动挑选或推进 actor。

断点恢复只接受完全相同的数据、配置、实现和分折；完成的数据目录和训练绑定不可覆盖。每次报告写独立快照。报告中的 WNS/TNS 预测采用正违例成本 `max(-slack,0)`，单位仍为 ns；不能把正成本误读为原生带符号 slack。

各来源拟合出的 TRAIN 均值和 objective scale 可不同；这里比较的是还原物理单位后的预测。之后接入 actor，若各自直接用 `Q=-cost/TRAIN_mean` 配同一个 α，实际蒸馏与 Q 梯度权重会同时改变。来源 actor 对照应另声明共同 utility 尺度或匹配的 α 处理，不能把这个隐含变化归因于 Q 数据来源。
