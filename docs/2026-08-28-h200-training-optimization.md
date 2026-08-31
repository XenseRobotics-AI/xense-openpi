# H200 训练性能优化记录

> [!CAUTION]
> 2026-08-30 更新：**`use_cudnn_attention` 目前不得用于生产训练。** 全空 mask 修复只消除了瞬时 NaN；带该修复的训练仍会在 step 1000–1200 之间偏离已验证基线并缓慢发散至 NaN。单变量 A/B、已排除的候选和新的验收要求见 [2026-08-30-cudnn-attention-divergence.md](2026-08-30-cudnn-attention-divergence.md)。
>
> 本文以下关于启动命令、cuDNN 版本确认和 A/B 吞吐的内容写于该问题定位之前，其中 `LD_LIBRARY_PATH` 与「确认输出 91400」两条**已经过时**，见下方修订。

> [!WARNING]
> 2026-08-29 生产长跑发现：cuDNN Attention 遇到 padding 产生的全空 mask 行时，前向有限但 Q 梯度为 NaN。修复、影响范围和 checkpoint 恢复要求见 [2026-08-29-cudnn-attention-nan-incident.md](2026-08-29-cudnn-attention-nan-incident.md)。未包含该修复的实现不得用于有效训练。

> **跨机器生产启动命令**（2026-08-30 修订）：**不要再导出 `LD_LIBRARY_PATH="$CONDA_PREFIX/lib"`。** conda cuDNN 已移除，环境统一到 pip cu128 栈；继续导出该目录会让其中残留的 conda-forge `libcublas`/`libcudart`/`libnccl` 遮蔽 pip 的同名库，重演同一类版本错配。

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  python scripts/train.py my_task \
    --exp-name=run_0520 \
    --overwrite
```

<details><summary>原命令（已过时，仅供追溯）</summary>

```bash
LD_LIBRARY_PATH="$CONDA_PREFIX/lib" \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  python scripts/train.py my_task --exp-name=run_0520 --overwrite
```

</details>

> **2026-08-29 长跑事故（已修复）**：原实现从 step 16100 起记录到 NaN；受污染 checkpoint 已删除，并从保留的 step 15000 恢复。最终修复保留原始 attention mask，只对全空 query 行停止 Q 梯度。逐 step 有限性保护已按生产要求移除，训练继续按原逻辑记录 loss；详见事故文档。

`my_task` 的模型配置需要包含 `use_cudnn_attention: true`；如果 YAML 中没有设置，则在命令末尾增加 `--model.use-cudnn-attention`。`--overwrite` 会删除同名实验的原 Checkpoint 目录；恢复已有训练应改用 `--resume`。

训练启动后会在初始化日志中直接打印：

```text
JAX cuDNN runtime version: 91400
```

该值来自 `cuda_versions.cudnn_get_version()`。

> [!IMPORTANT]
> 2026-08-30 修订：**这个数字不足以证明 cuDNN 栈是干净的。** 它由 cuDNN 的引擎子库返回，因此一个 9.10.2 的 dispatcher stub 驱动 9.14 引擎库时，它照样打印 `91400`——事故期间的实际情况正是如此。请改用：
>
> ```bash
> python scripts/check_cuda_stack.py
> ```
>
> 该脚本遍历 `/proc/self/maps`，在 `libcudnn*` 来自多个目录或文件名版本与运行时报告不一致时判 FAIL，并在真实训练形状上跑一次前向+反向与 fp32 参考解对比。详见 [2026-08-30-cudnn-attention-divergence.md](2026-08-30-cudnn-attention-divergence.md)。

**日期**：2026-08-27 至 2026-08-28

**机器**：8×H200

**分支**：`optimize-training-duration`

**模型配置**：`pi05_base_bi_flexiv_earbud_case_insertion_0826_h200`

**起始 Checkpoint**：step 49000（前四项优化）；step 16000（cuDNN Attention A/B）

本文记录一次从数据管线拆分计时到 GPU trace 的训练性能优化过程。目标不是只看某个局部计时，而是在相同 Checkpoint、相同训练配置下逐项修改，并用稳态端到端 step 时间判断收益。

更早的 `in_order=False`、无限 sampler 和跳过未使用触觉视频解码等工作见 [data-pipeline-stalls.md](data-pipeline-stalls.md)。本文从 multiprocessing queue、JAX 数组构造和模型计算继续向下分析。

## 1. 测量口径

前四项优化都从原始 step 49000 Checkpoint 只读恢复；cuDNN Attention A/B 从当前生产 step 16000 Checkpoint 只读恢复。所有短跑都使用独立实验目录，关闭 W&B、周期保存和最终 Checkpoint 保存，避免覆盖训练结果。每个变体运行到起始 step + 40，丢弃起始 step 到起始 step + 10 的编译、日志和预热数据，统计后续 30 个稳态样本。例如 step 49011–49040 或 step 16011–16040。

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

## 6. 第五步：cuDNN fused Attention

### 问题

Gemma Attention 原先显式执行 QK einsum、mask、softmax 和 PV einsum。XProf 已显示 GEMM/Fusion 占已记录 kernel 时间的 92.6%，因此训练 Attention 是比继续压缩数据搬运更有价值的候选。当前模型使用 BF16 GQA 和 block attention mask，必须验证 cuDNN 是否支持完整的真实训练形状，不能只测试一个小型无 mask MHA。

### 实现

新增可选配置 `use_cudnn_attention`。开启后，训练时调用：

```python
jax.nn.dot_product_attention(
    q,
    k,
    v,
    mask=attn_mask,
    scale=1.0,
    implementation="cudnn",
)
```

`q` 在调用前已经按 head dimension 缩放，因此这里显式使用 `scale=1.0`，避免重复缩放。该路径只在 `kv_cache is None` 的训练路径启用；带 KV cache 的推理继续使用原实现。开关默认关闭，可通过 `--model.use-cudnn-attention` 启用。

### cuDNN 版本和动态库选择

当前环境同时存在两套可被动态链接器找到的 cuDNN：

- `$CONDA_PREFIX/lib` 中的 cuDNN 9.14；
- Python `site-packages/nvidia/cudnn/lib` 路径，以及元数据版本为 `9.10.2.21` 的 `nvidia-cudnn-cu12` wheel。

cuDNN 不是由 `torch.backends.cudnn.version()` 统一替整个 Python 环境选择的。PyTorch 和 JAX 都要由 Linux 动态链接器根据当前进程的 `LD_LIBRARY_PATH`、wheel 的 RPATH/RUNPATH、已经加载的同 SONAME 库和导入顺序解析 `libcudnn.so.9` 及其组件。因此必须在实际启动训练的同一个 shell 中检查 JAX，而不能用 PyTorch 的版本输出推断 JAX。

当前已激活 shell 的实际状态是：

```text
LD_LIBRARY_PATH=$CONDA_PREFIX/lib
PyTorch cuDNN runtime = 91400
JAX cuDNN runtime     = 91400
JAX build cuDNN       = 90101
```

如果显式移除 `LD_LIBRARY_PATH`，PyTorch 和 JAX 都会解析到 9.10.2：

```bash
env -u LD_LIBRARY_PATH python -c \
  'import jax; from jax._src.lib import cuda_versions; print(cuda_versions.cudnn_get_version())'
# 91002

env -u LD_LIBRARY_PATH python -c \
  'import torch; print(torch.backends.cudnn.version())'
# 91002
```

所以“PyTorch 是 9.14、JAX 默认是 9.10”不是同一个进程环境下两个框架固定选择不同版本；准确说法是：**当前 shell 有 `$CONDA_PREFIX/lib` 时两者都是 9.14，去掉该搜索路径时两者都是 9.10.2。** 之前观察到的 JAX 9.10.2 是在 `env -u LD_LIBRARY_PATH` 的诊断进程中得到的。

强制 cuDNN Attention 时，9.10.2 对本模型测试形状报错：

```text
XlaRuntimeError: INTERNAL: No valid engine configs for Matmul_MUL...
```

使用环境中已经安装的 9.14 后，小型 MHA、真实 BF16 GQA 形状、block mask 前向和反向全部通过。因此不需要下载新包，但生产启动前必须保证 JAX 实际显示 `91400`：

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"

python -c \
  'from jax._src.lib import cuda_versions; print(cuda_versions.cudnn_get_version())'
# 必须输出 91400

python scripts/train.py pi05_base_bi_flexiv_earbud_case_insertion_0826_h200 \
  --resume \
  --model.use-cudnn-attention
```

### A/B 结果

测试条件为 8×H200、全局 batch 512、`batch_image_views=true`、相同 step 16000 Checkpoint 和相同数据配置。两组均使用 cuDNN 9.14 运行时，唯一变量是是否启用 fused Attention。统计 step 16011–16040 的 30 个稳态样本，不包含数据 worker 启动、首批解码、Checkpoint 恢复和 JIT/cuDNN 编译。

- 原显式 Attention：均值 `2.795537 s/step`，中位数 `2.797350 s/step`，标准差 `0.007647 s`；
- cuDNN Attention：均值 `1.978580 s/step`，中位数 `1.979450 s/step`，标准差 `0.011704 s`；
- step 时间减少 `29.22%`；
- 等价吞吐从 `183.15 samples/s` 提升到 `258.77 samples/s`，提升 `41.29%`；
- 从生产 step 16000 跑到 step 60000，按纯稳态 step 时间估算可节省约 `9.98 小时`。

分项均值也显示收益来自模型计算，而不是数据波动：

- 原实现：`dispatch=2.237247 s`，`compute_sync=0.484967 s`，`next_batch_total=0.073323 s`；
- cuDNN：`dispatch=1.516877 s`，`compute_sync=0.394600 s`，`next_batch_total=0.067103 s`。

数值对齐测试中，BF16 masked attention 前向最大绝对误差为 `0.00390625`、平均绝对误差为 `0.00026494`；Q/K/V 梯度最大绝对误差分别为 `1.91e-6`、`4.77e-7` 和 `2.38e-7`，属于正常 BF16 数值差异。

### 2026-08-29 生产长跑验证

生产训练从原 step 16000 Checkpoint 恢复，使用 8×H200、全局 batch 512、96 个 DataLoader worker、`batch_image_views=true`、`use_cudnn_attention=true`、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`。启动日志确认 JAX 实际加载 cuDNN runtime `91400`。截至检查时，tmux 训练进程仍在运行，进度条约为 step 55100/60000；日志中没有 OOM、`No valid engine configs` 或 XLA runtime exception。

速度统计使用日志中已经完整刷新的 step 16150–51650 记录。普通非日志、非保存 step 共 356 个样本：

- **当前一个普通稳态 step：均值 `1.910590 s`，中位数 `1.910000 s`；**
- 标准差 `0.010744 s`，范围 `1.86–1.95 s`；
- 分项均值为 `dispatch=1.867331 s`、`next_batch=0.042612 s`；
- 按 batch 512 折算吞吐约 `267.98 samples/s`；
- 每 100 step 的日志 step 均值 `2.360375 s`，其中日志约 `0.436000 s`；
- 每 1000 step 的 Checkpoint step 均值 `4.656286 s`，其中保存约 `2.288286 s`；
- 按上述频率摊销日志和保存开销后，长期约为 **`1.917238 s/step`**。

因此，生产长跑的速度与 40-step A/B 中的 `1.978580 s/step` 一致并略快；从纯吞吐看，cuDNN fused Attention 的约 1.9 秒/step 收益稳定存在。

但是训练数值已经失效：

```text
Step 16100: grad_norm=nan, loss=0.1185, param_norm=nan
Step 16200: grad_norm=nan, loss=nan,    param_norm=nan
Step 16300 及之后: grad_norm=nan, loss=nan, param_norm=nan
```

step 16100 是第一个写入指标的生产日志点，因此只能确定 NaN 在 step 16000–16100 之间产生，不能从当前日志定位到更精确的单步。已完整刷新的日志一直到 step 51600 都没有再次出现有限 loss。周期保存又已经清理原 step 16000；当前保留的最后一个切换前 Checkpoint 是 step 15000，而 step 20000–55000 均是在 NaN 出现后保存，不能作为有效训练结果使用。

结论是：**1.91 秒/step 是真实、稳定的硬件吞吐，但来自数值无效的训练，不能作为启用 cuDNN Attention 的生产验收。** 在定位并修复 NaN 前，应关闭 `use_cudnn_attention`；恢复有效训练时只能从仍保留的切换前 Checkpoint 重新开始。后续诊断需要缩短日志间隔并逐 step 检查 loss、梯度和参数有限性，重点覆盖真实 block mask 下的完整优化器更新，而不只是独立 Attention 前向/反向。

## 7. 正确性和运行验证

- 增加单元测试，对三路相机分别编码与合并编码的 token 做逐元素相等比较；
- 模型与 YAML 相关回归测试：`23 passed`；
- 前一阶段数据管线与显式 shutdown 回归：`37 passed`；
- 范数优化和相机合并各完成一次 40-step、8×H200 实际训练；
- 两次运行都成功恢复 step 49000 Checkpoint，没有 OOM，没有保存新 Checkpoint；
- `compute_metrics=True/False` 两个 JIT 分支都在真实训练中执行；
- 两次运行结束后显式 shutdown 成功，worker 无残留，8 张 GPU 显存全部释放；
- 本地与训练服务器上的修改文件 SHA256 一致。
- cuDNN Attention 小型与真实形状的前向/反向 smoke test 通过；
- cuDNN Attention 相关模型回归：`7 passed`；
- 原 Attention 和 cuDNN Attention 均完成一次 step 16000–16040 的 8×H200 短跑，没有 OOM、cuDNN engine fallback 或训练异常，也没有保存新 Checkpoint；
- cuDNN 短跑退出时个别 DataLoader worker 在清理阶段打印 `killed by signal: Aborted`，发生在训练完成和计时结束之后，主训练进程与 GPU 资源均正常退出；这不是 Attention 计算错误。
- 后续生产长跑推翻了短跑的数值有效性结论：step 16100 首次记录到 NaN 梯度，step 16200 起 loss 持续为 NaN；这说明现有 smoke test 和 40-step 吞吐短跑不足以作为生产正确性验证。
- 最小 H200 复现确认全空 mask 行会让原 cuDNN 路径产生 16384 个 Q 梯度 NaN；修复后 Q/K/V 梯度全部有限，且有效 query 输出不变；
- 从干净 step 15000 checkpoint 做 8×H200、batch 512 的 60-step 实际短跑，loss、梯度、参数和更新均保持有限，`update_is_finite=1`。

## 8. 当前结论与下一步

当前稳态数据交付只占约 `0.04–0.05 s/step`，H2D 只有数毫秒；XProf 又显示已记录 kernel 时间主要集中在 GEMM/Fusion。因此，继续微调 queue、collate 或 H2D 不太可能带来显著收益。

cuDNN fused Attention 的旧长跑普通 step 为 `1.910590 s`，但该区间从最早日志点起梯度、参数已经 NaN，不能作为有效训练的吞吐基线。修复后连续有限训练的普通 step 稳定约为 `2.31 s`；移除逐 step 有限性保护和把安全 mask 计算移出扫描层都没有改变该结果。最终实现保留原 mask，仅停止全空 query 的 Q 梯度，并保留原有 loss/grad norm/param norm 日志。确认 JAX runtime 为 91400 仍只能证明动态库选择正确；生产验收需要持续观察日志。

下一轮稳态优化建议按以下顺序推进：

1. **完成修复后的生产长跑验收。** 从干净 checkpoint 恢复并持续观察 loss、grad norm 和 param norm；当前策略不会自动拒绝或停止非有限更新，发现异常时需人工停止并回退到前一个干净 checkpoint。
2. **缩短有效 token 长度。** 全数据扫描得到 Pi0.5 tokenized state 最大长度 165；可单独 A/B `max_token_len=168` 与当前 200，减少 Gemma attention/MLP 的序列计算，同时保留 3 token 余量。
3. **继续拆最大的 GEMM/Fusion。** 在生产正确性验收后重新采集 trace，再确认剩余最大的 SigLIP/Gemma GEMM 和 fusion。
4. **降低 checkpoint 干扰。** 长训练若不需要每 1000 step 恢复点，可提高 `save_interval`；此前生产日志已经确认 checkpoint 会造成下一 step 的数据等待尖峰。

最终判断：共享内存 Tensor 消除了主要数据交付开销；日志 step 范数和三路相机合并是低风险增量收益；cuDNN fused Attention 的全空 mask 梯度 NaN 已定位并修复，最小复现和 60-step 真实训练通过。当前最高优先级是让从 step 15000 恢复的训练完成更长的有限性验收。
