# 训练优化与 cuDNN attention

本文保留训练配置、实现约束和历史验证结论。详细实验脚本及排障过程可从本分支
`72ace2a` 及更早的提交中查阅；下述性能数字来自历史实验，并非每台机器上的保证。

## 生产配置

Pi0.5 的优化示例见
[`pi05_base_bi_flexiv_earbuds_case_insertion_teleop_rtc_0904_h100.yaml`](../configs/_examples/pi05_base_bi_flexiv_earbuds_case_insertion_teleop_rtc_0904_h100.yaml)。

```yaml
model:
  type: Pi0Config
  pi05: true
  max_token_len: 128
  batch_image_views: true
  use_cudnn_attention: true
  cudnn_attention_dtype: float16
data:
  base_config:
    tactile: false
strict_batch_order: false
```

`max_token_len: 128` 只针对上述耳机盒数据集做过验证：历史采样的 4037 个状态
（含各维极值样本）最长为 116 tokens。更换数据集、prompt 或状态维度后需重新检查
完整 transform 链的 token 长度；tokenizer 超长会警告并截断。不要直接套用 128。
训练前也需重新计算 norm stats，并检查 `lr_schedule.decay_steps` 是否符合训练计划。

## 保留的优化

- DataLoader worker 返回 CPU Torch Tensor，通过共享内存传输 batch；主进程以
  `tensor.numpy()` 的零拷贝视图构造 JAX array，减少大批量 NumPy 数据的进程间复制。
- 训练结束显式关闭 DataLoader，回收 worker。
- 梯度和参数范数仅在 `log_interval` 步计算，减少全模型归约；loss 每步计算。
- `batch_image_views` 将多路相机按样本交错合批，只调用一次 SigLIP，然后还原各视角。
- cuDNN FP16 fused attention 使用动态 loss scaling 和 custom VJP，复用前向残差。

main 已包含的跳过未使用触觉视频、乱序 batch 交付和无限 sampler 继续保留。
`strict_batch_order: false` 可避免队头阻塞，但交付顺序不保证跨运行一致；对比训练数值时设为 `true`。
Gemma 与 SigLIP 保持默认的全量 rematerialization；历史 8×H100、batch 256 实验中，
关闭 remat 或保存更多残差均超出显存容量。

## cuDNN 数值约束

全层 BF16 cuDNN attention 曾在约 1000 步后偏离显式基线并发散，不能依据短跑有限就用于生产。
当前显式路径仍是默认回退方案：`use_cudnn_attention: false`。
启用 cuDNN 时使用 `cudnn_attention_dtype: float16`。

排障中发现两个独立问题：

- 全空 attention mask 行可能产生 NaN query 梯度。实现将这些行的 query 梯度置零，保留原 mask。
- peaked attention 行的 backward 对存储输出的舍入误差敏感。FP16 提供更多尾数位；
  反向将余切按 2 的幂缩放后送入内核，再以 FP32 还原梯度，缓解 FP16 指数范围限制。

当前 VJP 直接调用固定 JAX 0.5.3 的私有 cuDNN forward/backward rule。
升级 JAX 或 CUDA/cuDNN 时必须复核接口、分片约束和训练收敛性。
q/k/v 与 mask/bias 显式使用一致的 batch 分片，避免 cuDNN partitioner 编译失败。
通过 `policy_config.create_trained_policy` 加载 JAX 推理策略时会关闭 cuDNN attention；
带 KV cache 的 attention 也继续使用显式实现。

## 历史验证与性能

2026-09-04，8×H100、全局 batch 256、固定 seed 和严格 batch 顺序，从 pi05_base step 0
起跑的 FP16 VJP 初版通过 3000 步验收。历史 BF16 分叉点附近，FP16 loss 与显式基线差小于
0.5%，记录的梯度范数未超过比较基线同点的 1.3 倍。

当前直接调用 forward/backward rule 的版本另做过真实激活等价性检查与 400 步对照：
前向、dK/dV 逐位相同，dQ 差异与内核自身跨运行的原子累加差异同阶，loss 与初版逐点一致。
这与初版的 3000 步训练验证是不同的验证范围。

同日 H100 端到端实验中，FP16 基线为 1.328 s/step；token 长度 200→128 后为
1.246 s/step；单独加入下述 XLA 标志为 1.281 s/step；二者组合为 1.193 s/step，
相对基线减少约 10.2%。口径为 400 步运行中 step 100 到 step 300 的日志墙钟差除以 200。
这些实验使用同一配置和严格 batch 顺序，不能与其他机器、任务或 batch 的提速比例直接相乘。

新的 attention 后端或 VJP 改动，应从 step 0 固定数据、seed、batch 顺序及超参跑满
3000 次更新，与显式基线对比 loss 和梯度趋势；确认参数与 optimizer state 有限，
梯度范数不超过基线同点的 3 倍后，再进入有 checkpoint 的受监控长跑。
单次前后向、吞吐短跑或单步梯度距离不能替代收敛验证。

## 环境与启动

保持 pip CUDA 12.8 库来源一致，避免将 `$CONDA_PREFIX/lib` 加到 `LD_LIBRARY_PATH` 前面。
当前依赖配置使用 PyTorch 2.11 对应的 cuDNN 9.19；训练启动时记录 JAX cuDNN runtime 版本。
运行 `check_cuda_stack.py` 检查实际加载的库来源,以及生产 shape 下 BF16 内核与 FP16 custom VJP
的前后向(含全空 mask 行的 dQ)。该检查只覆盖单次前后向数值,不替代 FP16 收敛验证。

```bash
env -u LD_LIBRARY_PATH python scripts/check_cuda_stack.py

env -u LD_LIBRARY_PATH \
  XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true \
    --xla_gpu_all_gather_combine_threshold_bytes=1073741824 \
    --xla_gpu_reduce_scatter_combine_threshold_bytes=1073741824 \
    --xla_gpu_all_reduce_combine_threshold_bytes=1073741824 \
    --xla_gpu_enable_pipelined_all_gather=true \
    --xla_gpu_enable_pipelined_reduce_scatter=true \
    --xla_gpu_enable_while_loop_double_buffering=true" \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  python scripts/train.py <config> --exp-name=<name>
```

这些 XLA 标志作为整体做过验证，未逐项测量贡献。继续已有实验使用 `--resume`；
仅在确实需要覆盖目标实验目录时使用 `--overwrite`。
