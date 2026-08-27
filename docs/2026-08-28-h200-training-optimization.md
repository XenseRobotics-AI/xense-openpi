# H200 训练性能优化记录

**日期**：2026-08-27 至 2026-08-28

**机器**：8×H200

**分支**：`optimize-training-duration`

**模型配置**：`pi05_base_bi_flexiv_earbud_case_insertion_0826_h200`

**起始 Checkpoint**：step 49000

本文记录一次从数据管线拆分计时到 GPU trace 的训练性能优化过程。目标不是只看某个局部计时，而是在相同 Checkpoint、相同训练配置下逐项修改，并用稳态端到端 step 时间判断收益。

更早的 `in_order=False`、无限 sampler 和跳过未使用触觉视频解码等工作见 [data-pipeline-stalls.md](data-pipeline-stalls.md)。本文从 multiprocessing queue、JAX 数组构造和模型计算继续向下分析。

## 1. 测量口径

所有短跑都从原始 step 49000 Checkpoint 只读恢复到独立实验目录，关闭 W&B、周期保存和最终 Checkpoint 保存，避免覆盖训练结果。每个性能变体运行 40 step，丢弃前 10 step 的编译和预热数据，统计 step 49011–49040 的 30 个稳态样本。

JAX dispatch 是异步的，所以不能把 `dispatch` 单独当成 step 时间。本文使用以下和作为主要指标：

```text
measured_step = dispatch + compute_sync + next_batch_total
```

数据管线 profiling 进一步拆成：

- `main_queue_wait`：主进程等待 DataLoader 返回 batch；
- `worker_getitem`：worker 内所有样本 `__getitem__` 时间之和，不是单个 step 的墙钟时间；
- `worker_collate`：worker 内 collate 时间；
- `jax_array_construct`：主进程构造分片 JAX array 的时间；
- `h2d_wait`：显式等待 Host→Device 完成的时间。

profiling 会增加计时和同步，因此只用于短跑，不应在生产训练中默认开启。

## 2. 第一步：共享内存 Tensor 和零拷贝 NumPy 视图

### 问题

worker 原先返回 NumPy batch。大 batch 通过 multiprocessing queue 时需要 pickle/复制，主进程还要再次把它包装成 JAX array。此前已经消除了严格顺序交付和 epoch 边界停顿，但这里仍有一段稳定的数据传输开销。

### 实现

collate 改为返回 CPU Torch Tensor。PyTorch multiprocessing reducer 会通过共享内存传递 tensor storage；主进程用 `tensor.numpy()` 得到同一块 CPU storage 的零拷贝视图，再交给 `jax.make_array_from_process_local_data`。

同时增加显式 `DataLoader.close()`：训练退出时主动调用 multiprocessing iterator 的 shutdown，等待 worker，超时后终止残留进程，避免把 native library 的析构异常留到解释器退出阶段。

### 结果

- 修改前：`3.275728 s/step`
- 共享 Tensor 后：`2.828845 s/step`
- step 时间减少 `13.64%`，等价吞吐提升 `15.80%`

共享 Tensor 后的稳态数据侧均值为：

- `main_queue_wait = 0.005030 s`
- `worker_collate = 0.081114 s`
- `jax_array_construct = 0.009046 s`
- `h2d_wait = 0.006171 s`
- `next_batch_total = 0.046491 s`

这说明主进程队列等待、JAX array 构造和 H2D 已经不是主要瓶颈。`worker_getitem` 约 130 秒是 512 个样本耗时之和，由 64 个 worker 并行完成，不能直接加到 step 墙钟时间上。

## 3. 第二步：20-step JAX/XProf trace

训练脚本增加可控的 XProf 开关：指定绝对起始 step、trace step 数和输出目录。在 trace 起止位置显式 `block_until_ready`，保证采集边界对应完整训练迭代。

本次采集 step 49011–49030，共 20 个完整 step。按每张 GPU、每个 step 汇总 trace 中记录的 kernel 时间：

- GEMM：约 `390.449 ms`
- Fusion：约 `183.930 ms`
- NCCL：约 `45.990 ms`
- 其他 kernel：约 `2.562 ms`
- Memcpy：约 `1.509 ms`
- Memset：约 `0.465 ms`

在这些已记录的 kernel 时间中，GEMM 与 Fusion 约占 `92.6%`，NCCL 约占 `7.4%`。因此当前优先级应是减少或融合模型计算，而不是继续优化 Host→Device 或 NCCL。

trace 中 kernel 可能重叠，不能把 kernel duration 简单相加后当作 GPU 利用率；同样，也不能仅凭 host dispatch 区间判断真实计算时间。

## 4. 第三步：只在日志 step 计算范数

### 问题

训练原先每个 step 都计算整个 trainable pytree 的 `grad_norm` 和大部分 kernel 参数的 `param_norm`，但这些指标只在 `log_interval` step 被记录。

### 实现

`train_step` 增加静态布尔参数 `compute_metrics`：

- 日志 step 计算真实 `grad_norm` 和 `param_norm`；
- 非日志 step 返回同结构的 NaN 标量，保持 JIT 输出 pytree 稳定；
- host 聚合使用 `nanmean`，忽略非日志 step 的占位值；
- 该布尔值作为静态位置参数传给 JIT，分别编译计算指标和跳过指标的两个变体。

使用位置参数是必要的：当前 JAX 在显式设置 `in_shardings` 时不接受 pjit 关键字实参。

### 结果

- 共享 Tensor 基线：`2.828845 s/step`
- 仅日志 step 计算范数：`2.814683 s/step`
- step 时间减少 `0.50%`，等价吞吐提升 `0.50%`

收益存在但很小，说明范数归约不是当前主要瓶颈。这个改动仍值得保留，因为它与日志语义一致，且不会牺牲已记录指标。

## 5. 第四步：三路相机合并为一次 SigLIP 调用

### 问题

`Pi0.embed_prefix` 原先对 head、left wrist、right wrist 三路相机分别调用一次相同的 SigLIP image encoder。这会重复进入相同计算图，并重复触发参数使用和 kernel launch。

### 实现

新增可选配置 `batch_image_views`。开启后：

1. 将三路图像堆叠为 `(B, V, H, W, C)`；
2. reshape 为 `(B*V, H, W, C)`，只调用一次 image encoder；
3. 将 token reshape 回 `(B, V, ...)`，按原相机名拆分。

布局使用 sample-major interleave，同一样本的三路视图连续排列。这样在数据并行/FSDP shard 中，各设备仍拿到完整样本的所有视图，不需要跨设备重排。

该开关默认关闭。生产训练可通过以下参数启用：

```bash
--model.batch-image-views
```

### 结果

- 只做范数优化：`2.814683 s/step`
- 再合并三路相机：`2.773743 s/step`
- 相对上一阶段 step 时间减少 `1.45%`，等价吞吐提升 `1.48%`
- 相对共享 Tensor 基线累计吞吐提升 `1.99%`
- 相对最初 NumPy queue 版本累计 step 时间减少 `15.32%`，等价吞吐提升 `18.10%`

相机合并后，`dispatch` 从约 `2.657 s` 降到 `2.236 s`，但 `compute_sync` 从约 `0.115 s` 增到 `0.492 s`。这是异步工作在 dispatch 与同步边界之间重新分布，真实收益必须看两者与 `next_batch_total` 的总和，不能宣称 dispatch 单项显示的约 16% 提升。

## 6. 正确性和运行验证

- 增加单元测试，对三路相机分别编码与合并编码的 token 做逐元素相等比较；
- 模型与 YAML 相关回归测试：`23 passed`；
- 前一阶段数据管线与显式 shutdown 回归：`37 passed`；
- 范数优化和相机合并各完成一次 40-step、8×H200 实际训练；
- 两次运行都成功恢复 step 49000 Checkpoint，没有 OOM，没有保存新 Checkpoint；
- `compute_metrics=True/False` 两个 JIT 分支都在真实训练中执行；
- 两次运行结束后显式 shutdown 成功，worker 无残留，8 张 GPU 显存全部释放；
- 本地与训练服务器上的修改文件 SHA256 一致。

## 7. 当前结论与下一步

当前稳态数据交付只占约 `0.04–0.05 s/step`，H2D 只有数毫秒；XProf 又显示已记录 kernel 时间主要集中在 GEMM/Fusion。因此，继续微调 queue、collate 或 H2D 不太可能带来显著收益。

下一轮建议按以下顺序推进：

1. **单独优化冷启动。** 64 个 worker 拉起约 320 秒，首批数据约 119 秒，总首批约 450 秒。它不影响长训练稳态吞吐，但严重影响恢复训练、调参和短 profiling。应检查 worker 是否串行重复初始化 JAX、数据集元数据或视频 backend，并评估持久 worker 池、延迟初始化和元数据预缓存。
2. **继续拆最大的 GEMM/Fusion。** 优先检查 trace 中最大的 SigLIP/Gemma GEMM 和 `loop_transpose_fusion_4`，验证是否存在多余 transpose、重复 materialization 或可合并的前向计算。
3. **做更长的相机合并 A/B。** 当前 30 个稳态 step 已确认约 1.5% 增量收益；在默认开启前，建议再做至少 200–500 个稳态 step，确认不同 batch 和训练阶段下收益稳定。
4. **降低 checkpoint 干扰。** 长训练若不需要每 1000 step 恢复点，可提高 `save_interval`；此前生产日志已经确认 checkpoint 会造成下一 step 的数据等待尖峰。

最终判断：共享内存 Tensor 是本轮最大且已验证的稳态收益；日志 step 范数是低风险小收益；三路相机合并方向正确但收益约 1.5%，应在更长 A/B 后再默认开启。后续优化应从数据搬运转向模型计算图，同时把约 450 秒冷启动作为独立问题处理。
