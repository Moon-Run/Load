# Load
aaa
root@gxn75:/workspace/TempName# bash tools/h100/phase6/probe_tp8_dp2.sh --label gxn75
环境探测输出：/workspace/TempName/outputs/h100/phase6/tp8-dp2-probe-gxn75-20261027-235454-030368
本地系统信息已采集；正在检查训练 Python。
TP8 DP2 环境探测：gxn75（容器 hostname=gxn75）
仓库：/workspace/TempName
Megatron：/workspace/Megatron-LM
仓库 commit：ec0b9b159ef465e372a420e499a7ec90f056b1df
Megatron commit：3ea68ad6042cc1204386ae9364358f7c4de1bc37
训练 Python：/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python
PyTorch=2.6.0+cu124 CUDA=12.4 NCCL=[2, 21, 5]
本地 GPU 检查：{"pinned_torch_260": true, "cuda_runtime_124": true, "eight_full_h100": true, "all_local_peer_pairs": true}
当前 GPU：
0, GPU-1706d57e-5c35-8a8e-3e54-162cd2813cd1, NVIDIA H100 80GB HBM3, 81559, 0, 0, 580.65.06
1, GPU-f5a032d7-653f-36c5-d45f-6b7fee264e5a, NVIDIA H100 80GB HBM3, 81559, 0, 0, 580.65.06
2, GPU-aaa779c1-00d2-231c-584b-73505b36ca09, NVIDIA H100 80GB HBM3, 81559, 0, 0, 580.65.06
3, GPU-3d4c9ce1-d3f4-deb1-2583-54edf8a7828a, NVIDIA H100 80GB HBM3, 81559, 0, 0, 580.65.06
4, GPU-47f842a7-68bd-1e86-5dc3-65c9db5ec042, NVIDIA H100 80GB HBM3, 81559, 0, 0, 580.65.06
5, GPU-0f4d8136-f3b3-6fb0-e8f4-9491251b829e, NVIDIA H100 80GB HBM3, 81559, 0, 0, 580.65.06
6, GPU-93b43235-1c0c-7b47-4c95-f3f677600bd8, NVIDIA H100 80GB HBM3, 81559, 0, 0, 580.65.06
7, GPU-d45e5e56-238c-7cf7-c09f-45e539028049, NVIDIA H100 80GB HBM3, 81559, 0, 0, 580.65.06
当前 GPU 进程（快照）：
未列出计算进程
候选网卡 IP（尚未验证互通）：
  ibp24s0: 192.168.20.75/24 [up]
  ibp92s0: 192.168.30.75/24 [up]
  ibp140s0: 192.168.50.75/24 [up]
  bond0: 10.18.133.75/16 [up]
  ibp164s0: 192.168.70.75/24 [up]
  docker0: 172.17.0.1/16 [down]
CUDA 编译器：[{"path": "/usr/local/cuda-12.8/bin/nvcc", "status": "ok", "release": ["12.8"]}, {"path": "/usr/local/cuda-12.4/bin/nvcc", "status": "ok", "release": ["12.4"]}]
RDMA 网卡：[{"name": "mlx5_0", "ports": [{"state": "1: DOWN", "phys_state": "2: Polling", "rate": "10 Gb/sec (4X SDR)", "link_layer": "InfiniBand", "port": "1"}], "netdevs": ["ibp21s0"]}, {"name": "mlx5_1", "ports": [{"state": "4: ACTIVE", "phys_state": "5: LinkUp", "rate": "400 Gb/sec (4X NDR)", "link_layer": "InfiniBand", "port": "1"}], "netdevs": ["ibp24s0"]}, {"name": "mlx5_2", "ports": [{"state": "1: DOWN", "phys_state": "3: Disabled", "rate": "40 Gb/sec (4X QDR)", "link_layer": "Ethernet", "port": "1"}], "netdevs": ["enp34s0f0np0"]}, {"name": "mlx5_3", "ports": [{"state": "1: DOWN", "phys_state": "3: Disabled", "rate": "40 Gb/sec (4X QDR)", "link_layer": "Ethernet", "port": "1"}], "netdevs": ["enp34s0f1np1"]}, {"name": "mlx5_4", "ports": [{"state": "4: ACTIVE", "phys_state": "5: LinkUp", "rate": "400 Gb/sec (4X NDR)", "link_layer": "InfiniBand", "port": "1"}], "netdevs": ["ibp92s0"]}, {"name": "mlx5_5", "ports": [{"state": "1: DOWN", "phys_state": "2: Polling", "rate": "10 Gb/sec (4X SDR)", "link_layer": "InfiniBand", "port": "1"}], "netdevs": ["ibp95s0"]}, {"name": "mlx5_6", "ports": [{"state": "4: ACTIVE", "phys_state": "5: LinkUp", "rate": "400 Gb/sec (4X NDR)", "link_layer": "InfiniBand", "port": "1"}], "netdevs": ["ibp140s0"]}, {"name": "mlx5_7", "ports": [{"state": "1: DOWN", "phys_state": "2: Polling", "rate": "10 Gb/sec (4X SDR)", "link_layer": "InfiniBand", "port": "1"}], "netdevs": ["ibp143s0"]}, {"name": "mlx5_8", "ports": [{"state": "4: ACTIVE", "phys_state": "5: LinkUp", "rate": "400 Gb/sec (4X NDR)", "link_layer": "InfiniBand", "port": "1"}], "netdevs": ["ibp164s0"]}, {"name": "mlx5_9", "ports": [{"state": "1: DOWN", "phys_state": "2: Polling", "rate": "10 Gb/sec (4X SDR)", "link_layer": "InfiniBand", "port": "1"}], "netdevs": ["ibp167s0"]}]
容器 RDMA 设备：[{"path": "/dev/infiniband/issm0", "readable": true, "writable": true}, {"path": "/dev/infiniband/issm1", "readable": true, "writable": true}, {"path": "/dev/infiniband/issm4", "readable": true, "writable": true}, {"path": "/dev/infiniband/issm5", "readable": true, "writable": true}, {"path": "/dev/infiniband/issm6", "readable": true, "writable": true}, {"path": "/dev/infiniband/issm7", "readable": true, "writable": true}, {"path": "/dev/infiniband/issm8", "readable": true, "writable": true}, {"path": "/dev/infiniband/issm9", "readable": true, "writable": true}, {"path": "/dev/infiniband/rdma_cm", "readable": true, "writable": true}, {"path": "/dev/infiniband/umad0", "readable": true, "writable": true}, {"path": "/dev/infiniband/umad1", "readable": true, "writable": true}, {"path": "/dev/infiniband/umad2", "readable": true, "writable": true}, {"path": "/dev/infiniband/umad3", "readable": true, "writable": true}, {"path": "/dev/infiniband/umad4", "readable": true, "writable": true}, {"path": "/dev/infiniband/umad5", "readable": true, "writable": true}, {"path": "/dev/infiniband/umad6", "readable": true, "writable": true}, {"path": "/dev/infiniband/umad7", "readable": true, "writable": true}, {"path": "/dev/infiniband/umad8", "readable": true, "writable": true}, {"path": "/dev/infiniband/umad9", "readable": true, "writable": true}, {"path": "/dev/infiniband/uverbs0", "readable": true, "writable": true}, {"path": "/dev/infiniband/uverbs1", "readable": true, "writable": true}, {"path": "/dev/infiniband/uverbs2", "readable": true, "writable": true}, {"path": "/dev/infiniband/uverbs3", "readable": true, "writable": true}, {"path": "/dev/infiniband/uverbs4", "readable": true, "writable": true}, {"path": "/dev/infiniband/uverbs5", "readable": true, "writable": true}, {"path": "/dev/infiniband/uverbs6", "readable": true, "writable": true}, {"path": "/dev/infiniband/uverbs7", "readable": true, "writable": true}, {"path": "/dev/infiniband/uverbs8", "readable": true, "writable": true}, {"path": "/dev/infiniband/uverbs9", "readable": true, "writable": true}]
memlock soft/hard：[65536, 65536]
本次未进行双节点连接或 NCCL 通信测试，未启动训练。
正式比较范围：MA11、Flux＋基础量化 v2、Megatron＋CUDA量化；TP8 DP2，global batch=4，micro batch=1。
请返回两台的 summary.txt；详细信息保存在 report.json。
root@gxn75:/workspace/TempName# ulimit -l unlimited
root@gxn75:/workspace/TempName# ulimit -l
unlimited
root@gxn75:/workspace/TempName# 