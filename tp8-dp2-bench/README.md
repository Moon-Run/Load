# TP8＋DP2 独立测试包

本目录用于 gxn74＋gxn75 的 16 张 H100：每节点 TP8、跨节点 DP2，只比较 **MA11、Flux＋基础量化 v2、Megatron＋CUDA量化**。两节点都放在 **`/workspace/tp8-dp2-bench`**，不覆盖 `/workspace/TempName`，无需拉取其 Git 历史或实验数据。

包内包含通信检测、正式训练入口、三组冻结的 worker、MA11 与 Flux GemmOnly 编译源码、两个 CUDA codec、量化选择计划，以及固定版本的 Megatron／CUTLASS／NCCL 源码。不包含旧日志、profile、模型 checkpoint、语料或 Conda 环境。使用现有 Python 环境和 CUDA 12.8，在各节点首次运行时编译；构建不联网下载依赖。

Megatron 固定为 `3ea68ad6042cc1204386ae9364358f7c4de1bc37`。包中 `package.json` 记录每个文件的 SHA256，两节点必须使用同一包。无需依赖两边旧仓库的 HEAD。`provenance.json` 中原工作区路径仅供追溯，不作为运行依赖。

## 放置与校验

把 `tp8-dp2-bench.tar.gz` 和 `tp8-dp2-bench.tar.gz.sha256` 传到 **两个容器的 `/workspace/`**，分别执行：

```bash
cd /workspace
sha256sum -c tp8-dp2-bench.tar.gz.sha256
tar -xzf tp8-dp2-bench.tar.gz
cd /workspace/tp8-dp2-bench
sha256sum -c SHA256SUMS > package-check.log
```

目标目录应为新目录；若已经运行过旧包，先将旧目录改名保留，再解压本次包。只移动本包即可，74 不需要更新旧的 TempName 仓库。75 也使用同一个包执行，避免代码版本不一致。

## 单独检测通信

以下两条命令分别在对应容器运行，启动间隔不超过 3 分钟。默认控制网络 `bond0`、地址 `10.18.133.74`，数据网络使用两节点都 ACTIVE 的 `mlx5_1:1,mlx5_4:1,mlx5_6:1,mlx5_8:1`，关闭 NVLS。会在本进程提高 memlock，并检查 GPU 空闲；不终止其他用户的任务。

74：

```bash
cd /workspace/tp8-dp2-bench
bash run_link.sh --execute --node-rank 0 \
  --python /opt/conda/envs/flux-h100/bin/python \
  --run-id tp8-dp2-link-check
```

75：

```bash
cd /workspace/tp8-dp2-bench
bash run_link.sh --execute --node-rank 1 \
  --python /workspace/conda-envs/flux-h100-ablation3-20261017/bin/python \
  --run-id tp8-dp2-link-check
```

检查输出目录的 `state.json`：两端都必须为 `completed`。实际执行 TP AllGather／ReduceScatter、8 组跨节点 DP AllReduce，并从 NCCL 数据通道日志验证 IB，拒绝 Socket 回退。该阶段不训练。

## 正式三组训练

下面入口会自动再次执行通信检测，再编译、做三组 smoke、运行完整计时，最后为每组另启独立进程采集 profile。无需先手动执行上面的通信命令。两节点必须使用**同一个 `--run-id`**，且输出目录必须不存在；重试请两边一起换新 ID。

74：

```bash
cd /workspace/tp8-dp2-bench
nohup bash run_training.sh --execute --node-rank 0 \
  --python /opt/conda/envs/flux-h100/bin/python \
  --run-id tp8-dp2-01 > tp8-dp2-01-node0.log 2>&1 &
```

75：

```bash
cd /workspace/tp8-dp2-bench
nohup bash run_training.sh --execute --node-rank 1 \
  --python /workspace/conda-envs/flux-h100-ablation3-20261017/bin/python \
  --run-id tp8-dp2-01 > tp8-dp2-01-node1.log 2>&1 &
```

两条命令应尽快依次启动。默认协调端口 **29580**、torchrun／自动通信检测端口 **29581**；单独 `run_link.sh` 使用 **29574**。不需要 SSH 免密或共享文件系统。控制器等待两端同阶段就绪，任何构建、正确性或执行路径错误均记录失败并通知另一端；**性能无收益不会中止剩余对照**。不带 `--execute` 仅校验文件并打印计划，不启动 GPU 或网络通信。

查看进度：

```bash
tail -f tp8-dp2-01-node0.log  # 74；75 改为 node1
```

首次编译进度还可查看 `build/node0-*/compile.log`（75 为 `node1-*`）。成功构建按包 ID 缓存；失败构建不会当作可用缓存。若需重新编译，使用 `--build-root /workspace/tp8-dp2-build-retry-node0`（75 对应 node1）指定新目录。

## 口径与结果

- 固定 32 层、hidden 4096、FFN 16384、32 heads、sequence 2048、BF16、MBS=1、**GBS=4**。每节点 **2 个 micro-batch**，全局每 optimizer step 8192 tokens。两份 DP 数据不同，同一 TP 组数据相同。
- rank 0–7 是 74 的 TP 组，rank 8–15 是 75 的 TP 组；DP 组为 `[0,8]` 至 `[7,15]`。量化只用于原有 TP 通信，跨节点 DP 梯度同步保留 Megatron 原实现。
- Flux 基础组保留 **receiver fusion v2**：Flux GemmOnly＋独立 CUDA 编码＋NCCL All-to-All＋接收端解码/BF16 求和融合。Megatron CUDA 组保留原生 GEMM＋独立 CUDA 编解码；MA11 保留 CR3＋MA8-v2＋MA9＋MA11。
- 每组 smoke 为 2 步预热＋3 步检查：核对全部可训练参数的 DP 梯度与更新指纹、量化数值，以及 MA11 原生算术自检。三个对照使用相同初始化和全局输入；量化策略不同，不要求三组 loss／梯度轨迹逐位相同。
- 默认 **12 个配对块×3 组＝36 个计时窗口**，每窗口新进程、10 步预热＋50 步完整 optimizer step。六种组内顺序各出现两次，共 1800 个正式 optimizer step、576 份 rank 计时记录。所有窗口保留；跨 16 rank 的最慢耗时定义窗口 ms/step。
- 每组另开独立 profile，检查实际 launch 归属、MA8/MA9/MA11 路径及真实 DP2 AllReduce。profile 不计入性能样本。MA8／MA11 调用数按 2 个本地 micro-batch 校验；DP 新增的梯度缩放不套用旧单节点全局 `aten::mul` 总数。
- 报告给出三组算术平均／中位 ms/step，以及 MA11 相对两基线的逐块 log 耗时比 bootstrap 95% CI。默认一轮属探索性结果；可两端都指定 `--trials 2`，两轮分别报告，不混用历史单节点耗时。

两端结果各位于 `runs/tp8-dp2-01/node0/`、`runs/tp8-dp2-01/node1/`。完成需 **两端 `state.json.status=completed`**，且有 `report.json`。`timing-progress.json` 只是未完成的计时进度，不能代替最终通过标记。`report.json` 汇总全部 16 rank；原始日志、trace 留在各自节点，分析时请拉取两端结果目录。

包的本地生成与 CPU 验证不能证明双节点 CUDA/NCCL 已成功，也不预设性能收益。仍需在 74／75 上实际执行上述检测与训练。此测试沿用固定合成全局 batch，用于端到端性能比较，不代表真实语料收敛验证。
