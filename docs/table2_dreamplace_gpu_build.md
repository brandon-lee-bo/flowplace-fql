# Table 2 DREAMPlace 4.1 独立 GPU 构建候选

独立 H20 GPU 安装现已完成，四个小型 CPU/GPU 算子测试通过，成功验证 job 为 `1788005`。安装根目录是 `/fact_data/bohanli/flowplace_fql_backend/dreamplace_gpu_build/job_1787994/install`，对应解释器为同级 `env/bin/python`；正式六电路仍须先通过 GPU bp_fe smoke。构建复用现有依赖，固定公开 4.1.0 源码，不覆盖 CPU 安装，也不下载或升级软件。下面保留完整失败与修复链，不能把它描述为最初未修复组合直接成功。

## 已确认环境与兼容性边界

| 项目 | 本地观察 / 含义 |
|---|---|
| DREAMPlace | commit `5d13c9001a3bc900dca1e108e633d5dd45b00701`；沿用现有 OpenTimer 两个 doctest 测试目标的 `DOCTEST_CONFIG_NO_POSIX_SIGNALS` 编译定义修复，保存完整嵌套 diff。未改优化算法 |
| Python / PyTorch | Python 3.9；现有 `.venv` 的 `torch==2.2.1+cu121`，编译 CUDA 12.1，C++ ABI=0 |
| Toolkit | `/usr/local/cuda-12.8/bin/nvcc`，12.8.93；`/usr/local/cuda` 指向 12.8。另有 13.3，此候选明确不用。对常见安装位置及 modules 的有界检查未找到 12.1 toolkit |
| 主机编译器 | GCC 11.5.0，CMake 3.31.8；沿用 CPU 构建的 Python 头文件、共享库和 Boost staging |
| GPU 分区 | `gpu-scavenger` 同时提供 L20/H20，最长 24 小时；`dev` 提供 L20，最长 12 小时；`ai` 提供 H20。脚本缺省 L20，实际首次运行由 root 显式 override 为单张 H20，见文末 |

`torch.utils.cpp_extension._check_cuda_version` 的本地 2.2.1 源码对 CUDA **主版本不同报错、同主版本的小版本差异警告**。因此 12.8 与 cu121 并非已知的主版本硬冲突。但 DREAMPlace 的 [TorchExtension.cmake](../iccad/assets/DREAMPlace-4.1.0/cmake/TorchExtension.cmake) 自己链接 Torch/CUDA，并没有调用这个检查；上述行为不证明扩展可以编译或运行。README 的已测 Torch 列表止于 2.0，当前 2.2.1 / GCC 11 也不在列出的已测组合中。

[NVIDIA CUDA 小版本兼容文档](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) 讨论的是 driver/toolkit 边界并带有 PTX/API 限制，不能据此直接证明 PyTorch C++ 扩展的跨 toolkit 小版本兼容。此候选先编译并链接真实 weighted-average-wirelength CUDA 目标；成功后完整构建，再运行三个小型官方 CPU/GPU 数值单测：HPWL、weighted-average-wirelength、electric-potential。它们覆盖基础路由长度、梯度和密度算子，但不等于完整六电路验收。

stock CMake 按 `torch.cuda.is_available()` 选择是否编译 CUDA。在没有可见 GPU 的 CPU allocation 上配置，即使 nvcc 存在，也会静默选择 CPU。因此此候选必须申请 GPU，在配置前后分别检查可见 GPU 与 `CUDA_FOUND=TRUE`；构建架构从获配 GPU 实际 capability 读取，采用此项目 legacy FindCUDA 所需的 `8.9` 一类点号写法，不用现代 CMake 的 `89` 写法，也不把多个无关架构全部编译。

## 资源、位置和执行入口

默认申请 **1 张 L20、8 CPU、32 GiB RAM、2 小时**，关闭自动 requeue。这是首次有界兼容性尝试的资源上限，不是已测构建时间。源代码与构建目录在 `/scratch/bohanli/flowplace_table2/dmp_gpu_build_<jobid>/`；安装、独立 venv、日志和哈希在 `/fact_data/bohanli/flowplace_fql_backend/dreamplace_gpu_build/job_<jobid>/`。已有 CPU source/build/install 约 353/416/60 MiB，可作数量级参考；GPU 安装和构建可能更大，尚未实测。脚本要求 bulk 至少 4 GiB quota/free headroom；当前约 37 GiB headroom 是准备时状态，不是以后提交时保证。

初始候选提交命令：

```sh
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
sbatch backend/scripts/build_dreamplace_table2_gpu.slurm
```

独立 venv 只创建解释器环境，通过明确 `PYTHONPATH` **只读复用**现有 `.venv` 的 Torch 等依赖和 `dmp_env` 的少量附加包；它不是可脱离原路径搬迁的完整依赖拷贝。保留已有 CPU env/install 的路径及内容。源码先复制到 scratch，因为 OpenTimer 的一些构建输出写入源目录。每次作业使用新目录，拒绝覆盖既有尝试。

脚本保存 source/submodule revision、工作树 diff、复制后源码逐文件 SHA256、软件版本、GPU/driver/capability、CMake cache、安装文件哈希、CUDA 模块实际导入路径以及已载入 CUDA 共享库路径。Torch wheel 的 RPATH 可能先加载其打包的 CUDA 12.1 runtime，因此脚本不通过全局 `LD_LIBRARY_PATH` 强迫混换库。错误会保留分阶段日志和退出码；仅出现 `BUILD_AND_OPERATOR_TESTS_PASSED` 才表示此脚本的安装检查全部通过。

如果 12.8/cu121 构建或小算子失败，先根据首个具体错误判断是否只是构建接口兼容问题；不要自动升级 Torch、改 stock 算法、轮试多个版本并按 PPA 挑选。若确需匹配的 12.1 toolkit，应另列下载体积、来源与构建计划再推进，此候选不做这个下载。

## 接入正式实验的条件

构建作业不执行任何 Table 2 电路。成功后仍需使用新 GPU 安装，显式 `gpu=1`，对 bp_fe 做一次与 CPU smoke 同协议的两阶段/合法性/重读 pin 审计，再冻结六设计共用的 GPU 路径、GPU 类型、参数和共同 OpenROAD 后端。CPU smoke 保留为流程对照，不因其数字更接近论文而混进 GPU 正式 cohort。GPU 路径也无法补足作者原始 baseline 配置、完整 strong-legalizer 对应关系或原硬件未公开的限制；参见 [协议审查](table2_dreamplace_protocol.md)。

## 已提交尝试

- root 提交 `1787992`（L20），因 GPU reservation / 节点不可用预计等待至次日上午；在开始前取消，未产生构建或实验结果。
- root 用同一脚本执行 `sbatch --gres=gpu:nvidia_h20:1 backend/scripts/build_dreamplace_table2_gpu.slurm`，得到 `1787994`。只覆盖 GPU 型号，仍是 8 CPU /32 GiB /2 小时。这是资源可用性选择，发生在任何 GPU PPA 结果产生之前；正式六设计随后应统一固定 H20，不以结果混选硬件。
- `1787994` 在 `gravitation` 启动；`environment.json` 实测 NVIDIA H20、capability 9.0、Torch 2.2.1+cu121、nvcc 12.8。配置确认 `CUDA_FOUND=TRUE`，首个 `weighted_average_wirelength_cuda` 的 CUDA 编译和共享模块链接成功，开始完整构建。尚待完整构建、安装及算子检查结果，不能由首个目标成功推断全部可用。
- 脚本 SHA256：`c7115e46ef9ea5a5b0402967058bbdbc7b6e5a54b3ea8fb4bf83b9357f5269d8`。所有实际配置和日志位于 `/fact_data/bohanli/flowplace_fql_backend/dreamplace_gpu_build/job_1787994/`，实际 source/build 位于 `/scratch/bohanli/flowplace_table2/dmp_gpu_build_1787994/`。

`1787994` 随后在 `pin_pos_cuda_segment` 编译失败，完整日志保留，退出码 2。首错是系统 CUDA 12.8 CUB 中未加全局限定的 `cuda::proclaim_return_type` 与 `cuda::discard_memory` 被解析为项目的 `DreamPlace::cuda`。这两处问题已有 NVIDIA 官方修复：[PR 2940](https://github.com/NVIDIA/cccl/pull/2940/files)、[PR 3733](https://github.com/NVIDIA/cccl/pull/3733/files)，均只为函数引用添加全局 `::`。

经 root 授权，[resume_dreamplace_table2_gpu.slurm](../backend/scripts/resume_dreamplace_table2_gpu.slurm) 已提交为 `1787998`，仍使用 1 H20 /8 CPU /32 GiB，复用原有 source/build/env/install 路径。两个原系统 header 复制到 scratch 的 `cccl_namespace_overlay` 后精确应用上述修复，通过 CUDA include path 使用；系统 toolkit 和 DREAMPlace 源码保持不变。保留原失败记录，另在 `job_1787994/retries/job_1787998/` 保存原/改 header SHA256、官方 URL、精确 patch、CMake cache 前后版本、增量构建与测试日志。新增 include 参数要求相关 CUDA 对象重编，已经完成且不受影响的 CPU 目标被复用。

retry 的首个 `pin_pos_cuda_segment` 目标已编译链接通过；完整构建和小算子检查仍待完成。除原三个算子外，增加 pin-pos 的官方小型 GPU 前向/梯度检查。成功时根目录写 `build_completion.json` 指向成功 retry，并生成 `BUILD_AND_OPERATOR_TESTS_PASSED`；原始 `exit_code.txt=2` 不覆盖，因此验收不能仅看第一尝试的退出码，也不能仅看一个 `.so` 文件存在。

`1787998` 随后完成全量构建与安装，HPWL 的 CPU/GPU 测试通过；在 weighted-average 单测的 CPU backward 阶段因测试 fixture 使用 `uint8` pin mask 而被 Torch 2.2 的 `masked_fill_` 拒绝。实际 stock `BasicPlace.py:182` 用比较表达式生成 bool pin mask，因此不需要修改优化代码。该失败与退出码 1 原样保留。

验证续作业 [validate_dreamplace_table2_gpu.slurm](../backend/scripts/validate_dreamplace_table2_gpu.slurm) 为 `1788005`，H20 /8 CPU /32 GiB，8 秒内以退出码 0 完成。只在一个独立单测副本将 pin-mask fixture 的 dtype 从 `np.uint8` 改为 `np.bool_`，保留 net mask 类型、全部数值期待及容差；安装和优化代码均未改变。成功的编译、安装与 HPWL 证据复用，只继续 weighted-average、electric-potential 和 pin-pos。四份日志均有 `Ran 1 test`、`OK` 及 CUDA 执行输出。`test_fixture.patch`、原/改 SHA256、`validation_continuation.json` 记录这项测试兼容修正与复用来源。

最终成功证据在 `job_1787994/retries/job_1788005/`，根目录 `build_completion.json` 显式记录编译 retry `1787998` 和验证 retry `1788005`，`BUILD_AND_OPERATOR_TESTS_PASSED` 已生成。实际加载的 `libcudart.so.12` 来自现有 Torch wheel 的 `nvidia/cuda_runtime/lib`；未改 `LD_LIBRARY_PATH`。实测 install 约 68 MiB、scratch build 约 524 MiB。安装已经交给执行 agent 做新 bp_fe GPU smoke；这份构建验收本身没有运行任何 Table 2 placement 或 OpenROAD PPA。
