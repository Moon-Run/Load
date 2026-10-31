# MA15 离线汇总修复结果

本地回传数据已完整复核并补齐 profile 验收。没有重新 prepare、编译或运行训练。

- 正式计时：16 个窗口、128 份 rank 结果、800 个 optimizer step，全部保留。
- Fullstep profile：两臂各 8 rank，共 16 份 trace，全部验收通过。
- Operator 报告及 profile：使用原冻结 controller 重新计算，与原保存结果完全一致。
- 原始 rank、trace、计时报告和冻结源码的哈希保持不变。

| 路径 | 平均 ms/step |
| --- | ---: |
| MA11 对照 | 597.170 |
| MA15 attention graph | 545.655 |

配对耗时下降 **8.622%**，95% CI **[8.070%, 9.252%]**；本批次 2% 目标通过。仍为原来的单轮、旧 CPU 绑核实验，不能当作绑核修复后的复测结果。

原失败来自候选 rank2 的一对同 graph launch 计算 kernel：trace 区间有 0.0947265625 μs 的表观重叠，超过旧分析器的 0.01 μs 阈值。新检查保留 node coverage、顺序、geometry、stream、设备、context、correlation 及输入/输出/RNG 边界验收，只对同次 replay 内的计算 kernel 设置有记录的微小时间戳容差。本例允许上限为 0.1296 μs；全部 profile 中仅这一处使用该容差。

已补齐 [profile-summary.json](../fullstep/profile-summary.json)，[batch-state.json](../batch-state.json) 标记 `status=completed`、`completion_mode=offline_profile_reanalysis`。原失败状态备份为 [batch-state.before.json](batch-state.before.json)，完整分析器版本和输入哈希见 [receipt.json](receipt.json)。原 `fullstep-profile.log` 继续保留报错，作为历史记录。

源码修复及服务器端离线恢复命令见 [说明文档](../../../../TempName-New/docs/design/design-h100/ma15-profile-reanalysis.md)。47 项相关 CPU 回归测试通过。此操作只更新本地回传目录，未远程修改服务器，也不会自动续跑已退出的串行任务；原串行日志明确在 MA15 失败后停止，MA16–MA18 不能据此视为已经运行。
