# Q-study：离线数据及后端标签审计

这次研究先复用已有真实后端反馈，保持电路、工艺、时钟和物理流程不变。训练数据、历史在线反馈、调参验证及最终评估分开保存；旧 Table 2 任务继续暂停。

## 固定输入与数据接口

数据由 `fql/qstudy_data.py` 在 Slurm 中组装，输出到：

`/fact_data/bohanli/flowplace_fql/qstudy_v1/data_audit/`

| 子目录 | 原始离线 TRAIN | 历史验证 | 家族边界 | 行为 flow 目标 |
| --- | ---: | ---: | --- | ---: |
| `bp_fe64` | 64 | 16 | TRAIN 0–31；validation 192–199 | 256 |
| `bp_fe384` | 384 | 16 | TRAIN 0–191；validation 192–199 | 256 |
| `bp_be64` | 64 | 16 | TRAIN 0–31；validation 192–199 | 256 |

每个家族含原布局和一个固定扰动变体。TRAIN/validation 按噪声家族分离，避免同一布局的相近变体进入两侧。`bp_fe384` 的额外 320 条标签来自原 TRAIN 家族 32–191，不是在线采集；行为 flow 目标保持原来的 64 个 TRAIN 原始提案加 192 个 TRAIN teacher 布局。这个选择用于保持原实验对照，不表示它必然最优。

每个目录保存 `dataset.npz`、`rows.jsonl` 和 `manifest.json`。公开接口：

```python
from fql.qstudy_data import load_dataset, METRIC_NAMES

data, manifest = load_dataset(
    '/fact_data/bohanli/flowplace_fql/qstudy_v1/data_audit/bp_fe384/dataset.npz')
train = data['train']
validation = data['validation']
```

每个 split 的字段：

| 字段 | 形状或类型 | 含义 |
| --- | --- | --- |
| `raw_action` | float32，`N × D` | 模型原始坐标提案；bp_fe 的 D=22，bp_be 的 D=20 |
| `executed_action` | float32，`N × D` | 原合法化器输出，仍未经历 DEF 取整和后端轨道吸附 |
| `metrics` | float64，`N × 7` | 原生带符号的 GRT 实测指标 |
| `metric_mask` | bool，`N × 7` | 是否有真实测量；缺失值保留 NaN |
| `family`、`variant`、`source_row` | int64，`N` | 原噪声家族、扰动变体和原数据行号 |
| `status` | 字符串，`N` | success 或已确认的物理/几何失败 |
| `provenance_index` | int64，`N` | 对应 `rows.jsonl` 的行号 |

顶层另有 `data['behavior_action']` 和 `data['behavior_family']`，供行为 flow 使用。任何 cost 变换、特征缩放、指标归一化必须只由 TRAIN 拟合；`load_dataset` 不做损失权重选择。

七个指标的固定顺序为：

1. `rWL_um`：GRT 连线长度，µm。
2. `WNS_ns`：ns，保留原生带符号值。
3. `TNS_ns`：ns，保留原生带符号值。
4. `horizontal_overflow_sum`：最终 GRT 边上正的需求减容量之和。
5. `vertical_overflow_sum`：同上，垂直方向。
6. `Power_W`：W。
7. `StdCellArea_um2`：标准单元面积，µm²。

H/V overflow 不是百分比，也不是详细布线 DRC；本次 GRT 指标不称为最终详细布线 PPA。失败样本不赋造出的物理指标，保留缺失及失败类别。

## 可追溯性与执行动作

组装器核验原始坐标与执行坐标、行为源行号、action SHA、request/outcome SHA、DEF SHA、原生 metrics 和物理 manifest。`rows.jsonl` 保留来源路径及摘要、工具/工艺/设计配置摘要、物理流程版本。模型原始动作和合法化坐标都保留，不能因为两者接近就覆盖其一。

`executed_action` 这个历史字段表示合法化坐标，不等于实际后端数据库中的精确坐标。已有 `backend_macro_audit.json` 时，原生宏包围盒、DBU、输入到实际坐标的变化及几何合法性另存入行级 `actual_backend_geometry`。缺少这项原生审计时明确标记缺失，不推测最终坐标。

## 历史数据与污染边界

`historical_online/rows.jsonl` 单独保存过去最终模型实际使用的反馈：每个策略每个训练 seed 的两轮、每轮 8 条 own-collect 数据。不并入新的初始离线 TRAIN；所有 role=evaluation 样本排除。它们可用于历史机制分析或明确声明的扩展数据实验，不能算成本轮新采集的在线证据。

过去固定的 16 条验证布局已经用于诊断，所以现在仍可用于调参，但不能称为全新最终测试。固定 evaluation noise seed 900000、40 个提案的 Q 迁移检查、局部梯度探针和已公开的 DRT 结果也都被观察过。确认最终方案时需要新的、预先固定的噪声与测量，不能把旧样本重新命名为独立验证。

原 bp_fe 数据文件另含 16 条 heldout 标签。组装器只读取它们的 split 标识以排除，不导出指标，不读取相应的原生后端文件。噪声 family 和 split 元数据不作为标签泄漏的理由去合并数据。

较大设计的现有缓存包括 `bp`、`swerv_wrapper`、`ariane133`、`ariane136`。目前已知的是稀疏历史 Table 2 后端评估，不是可直接训练 Q 的同配置 GRT 数据集。其阶段、导入方式和修复预算可能不同，不直接混入本轮 TRAIN。是否收集新的较大设计标签，由本轮实验计划单独冻结。

## 作业与结果

执行入口：`sbatch fql/qstudy_audit.slurm`。申请 cpu-scavenger 的 2 CPU、6 GiB、20 分钟，排除 enterprise 和 natural-selection，不触碰旧 Table 2 hold 或其他项目任务。

- `1795653`：导入阶段遇到 final print 的语法错误，1 秒内退出；未完成数据审计或后端计算。
- `1795656`：COMPLETED，退出码 0，24 秒；三个数据集和 160 条历史在线反馈均核验保存。执行时的源码另存为 `data_audit/qstudy_data_job1795656.py`，与审计中的源码 SHA 对应。
- `1795659`：COMPLETED，退出码 0，1 秒；只读核验历史详细布线和重复动作，没有重新执行 EDA 后端。结果见 `historical_diagnostics.json`。

完成后 `audit.json` 包含七项指标的缺失数量、均值/方差/范围、零值数量、Pearson/Spearman 相关性、同家族扰动差分、唯一动作数、合法化位移、实际后端几何覆盖率及较大设计缓存清单。所有统计分别计算 TRAIN 和历史 validation，不使用 heldout 标签。

## 数据支持哪些实验假设

三个数据集的七项指标均无缺失，所有导出的初始离线请求均成功，并且都有独立的实际后端宏几何证据。bp_fe64 的 64 个 TRAIN 布局、bp_fe384 的 384 个 TRAIN 布局、bp_be64 的 64 个 TRAIN 布局不是三个互相独立的数据池：前 64 条 bp_fe 数据包含在 384 条中。

| TRAIN 数据 | GRT WL 均值 ± 样本标准差（m） | WNS 范围（ns） | TNS 范围（ns） | H overflow 范围 | V overflow 范围 |
| --- | --- | --- | --- | --- | --- |
| bp_fe64 | 2.3411 ± 0.0757 | −0.4936 至 −0.2776 | −318.74 至 −73.72 | 0–727 | 0–210 |
| bp_fe384 | 2.3689 ± 0.0884 | −0.5616 至 −0.2776 | −318.74 至 −73.72 | 0–857 | 0–604 |
| bp_be64 | 3.5377 ± 0.1173 | −1.5144 至 −1.0114 | −1021.90 至 −447.46 | 58–1325 | 1–1009 |

bp_fe384 的 TRAIN 线长与带符号 WNS、TNS 的 Spearman 相关分别仅为 −0.091、−0.181。它支持“不能靠优化线长自动获得时序收益”这个实验动机，但不能据此认定某种时序权重一定有效。

拥塞既有零值，也有长尾：bp_fe384 的 H/V 分别有 50/91 个零，其余可达 857/604；bp_be 则所有 TRAIN 样本均有正溢出。可以对比原值及 log1p 变换，并只用 TRAIN 定尺度，避免不同电路或指标的数值量级支配 loss。

功耗和面积有变化，但相对范围较窄。bp_fe384 的功耗均值 0.168985 W、样本标准差 0.001481 W；面积均值 60052.2 µm²、样本标准差 655.8 µm²。先作为完整报告和诊断项，再检验是否有必要加入策略目标。

已有 192 对 bp_fe TRAIN 家族内变体可用于成对排序监督，无须先新增后端标签。其平均绝对差分别为线长 28543 µm、WNS 0.0401 ns、TNS 43.03 ns；这些差异不代表所有指标同向改善。

原始提案到合法化坐标的 RMS 位移均值为 bp_fe384 的 0.0807、bp_be64 的 0.0901（归一化坐标），最大分别为 0.2013、0.1767。不能假设 actor 上的小连续变化必然对应实际后端中的小变化。

bp_fe384 的 TRAIN 有 382 种执行布局，bp_be64 有 63 种；bp_be 的验证集 16 行只有 14 种执行布局。保留原采集预算和逻辑行数，同时报告重复执行布局，不能把相同后端布局当作额外独立样本。

重复执行动作全部发生在同一家族内：bp_fe TRAIN 家族 62、147，bp_be TRAIN 家族 15、validation 家族 196、198；没有跨家族或跨 TRAIN/validation 的重复执行坐标。

五份标准研究详细布线汇总共有 107 个逻辑请求、105 个不同物理运行，原生 manifest 和 metrics 摘要均核验通过。额外两项 best400 参考不在这五份标准 summary 中，因此这个附加清单不能写成“覆盖全部 109 项”。这些历史详细布线结果均没有并入初始 TRAIN，也不作为新测试。

## 新实验的后端入口

`fql/qstudy_backend.py` 和 `fql/qstudy_backend.slurm` 提供新的 bp_fe/bp_be GRT worker，复用已有物理 runner、实际几何审计、压缩归档和两个电路各自的固定物理版本。

```bash
sbatch --array=0-N%4 fql/qstudy_backend.slurm \
  --experiment /PATH/TO/EXPERIMENT --seed 0 --round 0 --storage scratch
```

`N` 应为该轮冻结请求数减 1；提交前由控制器检查资源和预算。不要直接复制未替换占位符的命令。

实验 `config.json` 必须包含 `design`、`dataset_path`、`physical_failure_reward=-4`；可显式冻结 `dataset_sha256`、`reward_scale`、`backend_threads=8`、`backend_storage`。bp_be 默认使用原固定 verification plan，也可显式指定 `backend_verification_plan` 与其 SHA。worker 核对源数据物理版本，bp_fe 为 `9250cb...`，bp_be 为 `d9b318...`，并核验工具、输入 DEF 和八线程设置。

请求沿用原 `tasks.jsonl`／`completed.json`／逐样本 `request.json` 格式，增加原始动作和噪声的 SHA；`actions.npz` 同时保存噪声、原始提案和合法化坐标。请求的配置摘要、checkpoint 摘要、DEF 和导出审计、终局 mask=0，以及 `role=collect` 才能进入反馈的语义均强制检查。

worker 返回七项原始指标。兼容字段 `reward` 仍表示负的归一化 GRT 线长，新的多目标 utility 由训练端根据这些指标计算，不能把它误认为已加入时序或拥塞的 reward。

已经完成的物理测量在归档前写入 `measurement.json`；重试优先核验并恢复已完成运行，不重复 EDA。仍存活或状态不明的 attempt 不重复启动。未知基础设施／解析错误不会获得伪造 PPA 或失败 reward；已独立确认的几何、后端宏重叠或特定物理失败保留缺失指标和失败类别。

Slurm `1795682` 在 2 CPU／4 GiB 的只读 preflight 中完成了 5 项测试：两个真实设计的固定后端 context；原始坐标／噪声摘要被修改时拒绝；已完成运行恢复且禁止重新调用物理 subprocess；已存在 outcome 的重入核验；存活运行避免重复和未知基础设施错误不产生数值 reward。没有运行新的物理评估或归档。结果为 `qstudy_v1/preflight/backend_preflight.json`，测试源码为 `fql/test_qstudy_backend.py`。
