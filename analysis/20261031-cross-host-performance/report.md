# MA11 / FLUX＋基础量化跨服务器性能诊断（2026-10-31）

后续现场信息补充：用户提供的新服务器为双路 Xeon Platinum 8468，允许 CPU 0–191；GPU 0–3 属 NUMA 0（CPU 0–47、96–143），GPU 4–7 属 NUMA 1（CPU 48–95、144–191）。旧公式会将 rank 2、3、4、5 固定到 GPU 的远端 NUMA，已确认存在错误绑定来源。可见 cgroup 无 quota、无节流记录；驱动包版本为 595.71.05。`TempName-New` 已着手统一移除内部 singleton 绑核、保留外部 CPU 集并记录 affinity，详见 [修复与复测说明](../../../TempName-New/docs/design/design-h100/cpu-affinity.md)。这不证明历史全部约 30% 差值由绑核造成，修复后的服务器性能仍待复测。以下为基于原始历史日志的诊断记录。

现有证据强烈支持：性能退化主要发生在新服务器的主机提交、线程调度和跨 rank 等待路径，未发现新仓库把 MA11 或基础量化计算路径迁移错的证据。GPU 计算核本身没有普遍变慢。尚不能把根因进一步定为某个 CPU 型号、CPU 配额、绑核冲突或具体驱动版本；这些现场信息没有保存进历史日志。

当前分析主机是 `ln01`，没有 `nvidia-smi`，不是两次实验服务器。本次完成了历史数据复算、冻结源码核对及只读采集工具；没有在实验服务器复测，也没有修改训练、绑核或驱动设置。

| 正式计时，相同算法路径 | 旧 gxn75，ms/step | 新 10-118-122-131，ms/step | 耗时增长 |
| --- | ---: | ---: | ---: |
| MA11：F4-cache-zcount-csoftmax | 449.182 | 595.856 | +32.653% |
| FLUX＋基础量化 v2：basic，全 H8 | 582.318 | 765.361 | +31.434% |

MA11 使用旧运行候选臂与新 MA14-v2 的对照臂比较；新 MA14 候选的 597.173 ms 不是本表的 MA11。基础量化使用旧 `basic-quant-receiver` v2 与 MAV2 实验中的 `basic` 比较，不采用不同后端、不同数值轨迹的旧 `a-baseline-fusion` v1。两类算法一起退化约三成，不能只归因于 MA14 或 MAV2 新机制。

原始正式报告：

- [旧 MA11](../../../TempName/logs/h100/phase6/ma11-v1-causal-softmax-1x-50steps-20261027-202038-900556/fullstep/report.md)，第 5 行。
- [新 MA11，MA14-v2 对照](../../logs/ma14-v2-epoch-reuse-1x-50steps-20261031-030311-454459/fullstep/report.md)，第 5 行。
- [旧基础量化 v2](../../../TempName/logs/h100/phase6/basic-quant-receiver-1x-50steps-20261025-090410-798870/compare1/report.md)，第 38 行。
- [新基础量化，MAV2 对照](../../logs/mav2-then-ma14-ma18-20261031-014137-056456/mav2/compare1/report.md)，第 63 行。

比较的四组都是每窗口 10 步预热、50 步正式计时。MA11 每端各有 8 个该臂窗口，基础量化每端各有 12 个该臂窗口。新旧模型、初始参数／RNG／token 指纹一致；基础量化平均 loss 都是 3.821911，末步平均 loss 都是 0.019168687984347343。不要把 plan 内嵌历史 selection 的 20 步当成本次正式配置。跨服务器差值是描述性比较，不能沿用同机候选对照的置信区间解释为严格的跨机因果实验。

修复前的源码迁移核对给出的证据如下，范围和文件清单另存 [code_identity.json](code_identity.json)。后续 CPU affinity 修复会改变这里列举的当前源码，其前后差异不应误认为原迁移差异：

- 当前新旧仓库共有的 phase6 源码 503 个、`src` 259 个、`include` 40 个、`python` 44 个逐字节相同；共有构建入口一致。
- 历史 MA11 与新 MA14 的 MA11 对照臂，归一化构建根目录后，358 项编译源码 SHA256 全部相同，30 个共有 builder 输入相同；冻结 Megatron 的 1383 个文件相同。
- MA11 新旧 worker 的 8 个训练函数 AST 和完整 timing 分支 AST 相同。基础量化对应的 7 个训练函数和 timing 分支同样一致。没有新增审计或 profile 进入正式 timing。
- 基础量化 FLUX GEMM 的 349 项源码哈希相同，基础 codec 的 6 项上游源码哈希相同，nvcc 二进制 SHA256 相同，均使用 CUDA 12.8 编译。
- MAV2 包装层确实新增了 policy 分发，但 `basic` 仍调用原 H8 编码、基础解码和同一 `all_to_all_single`。没有观察到被误换为选择性变换路径。

这些结果不是“所有二进制完全相同”的证明。重新编译的动态库哈希不同；新 codec 显式添加了 `--ftz=false` 和 `-ccbin /usr/bin/g++-11`。但原始 trace 的计算核耗时持平，不支持它们造成整体约 30% GPU 计算退化。

原始 rank0 trace 的直接证据如下。API 行为每次调用的中位微秒；GPU kernel 行为单次 profile 内该类事件耗时之和，不能与正式 ms/step 混用。

| 测量项 | 旧 | 新 | 含义 |
| --- | ---: | ---: | --- |
| MA11 CPU 算子数 / GPU kernel 数 | 50552 / 8990 | 50552 / 8990 | 没有增加计算工作量 |
| MA11 cudaLaunchKernel 中位，μs；6127 次 | 5.960 | 7.805 | 每次 CPU 启动调用约慢 31% |
| MA11 cudaLaunchKernelExC 中位，μs；2863 次 | 8.803 | 11.340 | 第二类启动 API 同样变慢 |
| MA11 FLUX GEMM，ms；256 次 | 22.771 | 22.802 | GPU GEMM 基本不变 |
| MA11 softmax 前向 / 后向，ms | 14.415 / 5.224 | 14.386 / 5.205 | 优化核本身基本不变 |
| 基础量化 CPU 算子数 / GPU kernel 数 | 70588 / 12725 | 70588 / 12725 | 同样工作量 |
| 基础量化 cudaLaunchKernel 中位，μs；9094 次 | 5.857 | 7.660 | 每次 CPU 启动调用约慢 31% |
| 基础量化 GPU GEMM，ms；256 次 | 8.318 | 8.262 | 基本不变 |
| 基础量化 H128 编码，ms；256 次 | 16.442 | 16.279 | 基本不变 |
| 基础量化解码，ms；256 次 | 7.074 | 6.957 | 基本不变 |

将基础量化 trace 的 GPU kernel、memcpy、memset 区间按 GPU 合并，避免并发事件重复计时：

| rank0 单步 profile 时间线，ms | 旧 | 新 | 增量 |
| --- | ---: | ---: | ---: |
| 首个 GPU 事件到最后一个 GPU 事件的跨度 | 984.963 | 1320.665 | +335.701 |
| GPU 活动区间并集 | 909.196 | 927.420 | +18.223 |
| 本进程未记录 GPU 活动的区间 | 75.767 | 393.245 | +317.478 |

跨度增量中约 **94.57% 表现为空隙增加**。这不是对全卡利用率的测量：trace 不覆盖其他进程，也不能单凭空隙区分 CPU 执行、调度抢占、驱动阻塞等原因。它说明主要新增时间没有花在这些计算核内部，与启动 API 全面变慢一致。

MA11 的 NCCL AllGather 总时长从 232.955 增至 308.154 ms，IPC barrier 从 111.758 增至 157.336 ms；这些事件包含等待其他 rank 到达的时间，不能直接当成链路带宽下降。基础量化的 SendRecv 总时长反而从 154.892 降到 113.930 ms，也不支持“所有通信链路统一慢了三成”的解释。CPU 注释和算子存在嵌套，所有 CPU 项不能相加来分摊正式 step。

最值得优先检查的代码与机器交互点是**单逻辑 CPU 绑核**。四份 worker 都在导入 torch 之前执行：

```python
allowed_cpus = sorted(os.sched_getaffinity(0))
os.sched_setaffinity(0, {allowed_cpus[local_rank * len(allowed_cpus) // WORLD]})
```

见 [实际新 MA14 worker](../../logs/ma14-v2-epoch-reuse-1x-50steps-20261031-030311-454459/fullstep/scripts/worker.py) 第 17–19 行，以及生成模板 [tools/a800/phase5/e2e/worker.py](../../../TempName-New/tools/a800/phase5/e2e/worker.py) 第 15–16 行。Linux affinity 作用于调用线程，随后创建的线程通常继承创建者的 CPU mask；库也可能自行改绑。因此需要采集实际各线程 affinity，不能直接声称整个进程所有线程永远只有一个核。

该选择规则只看逻辑编号，不看 GPU 所属 NUMA，也不区分 SMT 同胞。例如允许集合为 `0-127` 时选 `[0,16,32,48,64,80,96,112]`；若机器恰好把 `i` 和 `i+64` 编为同一物理核的 SMT 线程，八个 rank 实际仅覆盖四个物理核。**这只是条件示例，尚未证实新服务器如此编号。** 旧服务器能快并不能证明此策略在新拓扑上合适。手工在外层使用 `taskset` 也不等于解除 worker 内部的 singleton 绑核。

正式 worker 用 `time.perf_counter()` 包住训练循环并在末尾 `torch.cuda.synchronize()`（新 MA14 worker 第 238–244 行）。因此计时包括 Python、CPU/CUDA 提交和等待，不是纯 GPU kernel 执行时间。一个更慢的 CPU 核、被限制的 CPU 时间、SMT 争用或驱动提交开销都可能让相同 H100 的 fullstep 变慢。

已知环境与缺项：

- 两端均为 8 张 `NVIDIA H100 80GB HBM3`、132 SM、CC 9.0；不是已观察到 GPU 型号档次下降。
- 实际 PyTorch / CUDA runtime / NCCL 均为 `2.6.0+cu124 / 12.4 / 2.21.5`。`NCCL_NVLS_ENABLE=0`、`CUDA_DEVICE_MAX_CONNECTIONS=1` 一致。
- 新环境字符串 `NCCL_VERSION=2.25.1-1` 不等于实际加载 NCCL；不能据此认定 NCCL 升级导致。
- trace 的 CUDA driver API 数值为 `13000 → 13020`，证明记录的驱动接口版本不同，**不是**具体驱动包版本，也不足以认定驱动回归。
- 两边都出现过 fused-softmax constraints / NVSHMEM 加载警告；它们不是新仓库独有回退证据，且 MA11 自定义 softmax 已由 trace 验证。
- 缺少该次运行的 CPU 型号与频率、每 rank/线程绑核、cgroup 配额与节流增量、NVLink 拓扑、GPU 功耗/时钟、同期其他负载。启动前 GPU 空闲快照不能证明全程独占。

现场处理顺序：

1. 在新服务器的**原实验容器及原启动环境**采集 CPU、cpuset、拓扑和驱动；若训练正在运行，追加各 worker PID。先确认实际 CPU 映射，不凭逻辑编号直接设置新绑核。
2. 若发现 CPU quota 或节流，比较负载前后同一 cgroup 的 `cpu.stat`，确认 `nr_throttled` / `throttled_usec` 是否增长。累计历史计数非零本身不是本次瓶颈证明。容器看不到祖先限制时需宿主侧补采集。
3. 若发现 SMT 冲突或 NUMA 错配，做新独立会话 A/B：原 singleton 策略对照每 rank 独立物理核、贴近 GPU 的 CPU 集，给辅助线程留下可调度 CPU。两臂共同改变策略，固定模型、输入、库、warmup 与 timing；不直接编辑历史冻结 worker/manifest，也不把不同绑核设置的结果混在同一正式报告。
4. 若仍有差异，在**同一新服务器、同一 Python 与依赖环境**顺序运行旧仓库和新仓库的 MA11 入口，使用新输出目录、本机编译。若二者都约 596 ms，支持环境原因；若旧入口稳定约 449 ms 而新入口约 596 ms，重新追查包装层、实际动态库、导入路径与构建配置。不要并行跑 A/B。
5. CPU 侧没有异常后，再做控制变量的驱动/通信排查；需要同步采集的纯通信基准才可把 rank 到达等待与链路吞吐分开。当前证据不支持直接降级驱动、打开 NVLS 或重写 CUDA 核。

已新增只读采集器 [collect_host_performance.py](../../../TempName-New/tools/h100/diagnostics/collect_host_performance.py)。在新服务器同步该文件后，在仓库根目录执行（输出目录必须尚不存在）：

```bash
python3 tools/h100/diagnostics/collect_host_performance.py \
  --output /tmp/h100-host-check-20261031-01
```

有运行中 worker 时，可在同一命令追加 `--pid 真实PID`，每个 rank 重复一次。采集器保存每线程 affinity、可见 cgroup 祖先限制、逻辑核与物理核/SMT/NUMA 映射和只读 NVIDIA 状态，不导入 torch、不运行 GPU 负载、不修改设置。单次采集不证明整段训练的频率或独占性。

复算入口和验证：

```bash
python3 Load/analysis/20261031-cross-host-performance/summarize_traces.py \
  --output /tmp/cross-host-performance-recomputed.json
```

[metrics.json](metrics.json) 保存四份 trace 的来源 SHA256、完整分类/名称统计、正式窗口数据与 GPU 区间并集。复算脚本已实际运行；采集脚本已验证帮助、无 GPU 环境、无效 PID 容错及拒绝覆盖已有输出目录。以上验证不代表已经在新服务器完成修复或性能验收。

对 API/排队/kernel 三类时长的区分参考 [NVIDIA Nsight Systems 分析指南](https://docs.nvidia.com/nsight-systems/AnalysisGuide/)；CPU wrapper 与 launch 延迟的区分参考 [NVIDIA 关于启动开销的说明](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/)。本报告数值来自本地日志，不来自产品理论性能。
