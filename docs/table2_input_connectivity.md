# Table 2 输入连接核查

六个电路全部通过。作者发布的布局 DEF 与后端缓存的 `1_synth.v`，具有相同的 leaf instance/master 和信号网端点分组。比较忽略信号网名称，保留全部单元引脚和顶层端口。

| 电路 | 单元数 | 非空信号网数 | 连接端点数 | Slurm job |
| --- | ---: | ---: | ---: | --- |
| bp_fe | 33,199 | 39,512 | 121,637 | 1788111 |
| bp | 306,819 | 348,080 | 1,028,799 | 1788116 |
| ariane133 | 167,805 | 197,372 | 642,734 | 1788117 |
| ariane136 | 171,230 | 201,175 | 657,411 | 1788118 |
| bp_be | 51,393 | 62,229 | 193,523 | 1788119 |
| swerv_wrapper | 98,107 | 113,622 | 379,158 | 1788120 |

使用同一冻结的 OpenROAD `v2.0-17598-ga008522d8`，在独立进程中分别读 DEF 和读 Verilog/link，不执行综合、优化或放置。六项均由 Slurm cpu-scavenger 运行，合计分配 104 CPU 秒。

原始报告、两侧导出 Tcl、工具日志、输入/输出 SHA256、脚本快照与调度记录保存在 [完整证据](/fact_data/bohanli/flowplace_fql_backend/table2_reports/connectivity_v1/summary.json)。

这是按工具解析结果进行的结构连接核查，不是功能形式等价证明；它也不证明原论文未公开的初始 DEF、宏方向历史或 baseline 参数已恢复。历史 generation manifest 保持原样。
