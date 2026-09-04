# openpi 训练提速与 cuDNN Attention 数值事故：完整记录

**覆盖时间**：2026-08-19 → 2026-09-04
**分支**：`optimize-training-duration`（性能主线 commit `ec72f34` → `cf07bbe`；其中
`ec72f34` 已进入本分支基线 `main`，分支新增优化从 `9aef52d` 开始）
**机器**：8×H200、8×H100 80GB（ebcloud）
**模型**：Pi0.5（`pi05_base` 微调，BF16 GQA，18 层 Gemma）

本文按时间线合并了原先分散的 5 份记录（数据管线、H200 优化、cuDNN NaN 事故、D 项根因分析、
H100 A/B），并与 commit 对齐。每个阶段给出：动机 → 实现（关键代码）→ 实测 → 结论。
被推翻的中间结论保留在「弯路」小节里，避免后来人重走。

---

## 0. 总账

| 阶段 | 日期 | commit | 改动 | 端到端收益 |
| --- | --- | --- | --- | --- |
| 一 | 08-19 ~ 08-27 | `ec72f34` | 跳过触觉解码 / `in_order=False` / 无限 sampler | 消除周期性加载器停顿，生产 **≈1.14×** |
| 二 | 08-27 ~ 08-28 | `9aef52d` `8ccf99c` | 共享内存 Tensor / XProf / 只在日志步算范数 / 三路相机合批 | step 时间 **−15.3%** |
| 三 | 08-29 ~ 08-31 | `476babe`…`895811c` | cuDNN fused attention（收益 −29%，但**训练发散**） | 净收益 0，转为诊断 |
| 四 | 09-03 ~ 09-04 | `df9755f` `3ea2b1c` | 定位根因 + fp16 内核（3000 步验收通过） | 显式 1.60 → fp16 **1.33 s/step** |
| 五 | 09-04 | `cf07bbe` | `max_token_len` 200→128、XLA 集合通信标志 | 再 **−10.2%**，1.328 → 1.193 s/step |

**当前生产配置**（`configs/_examples/pi05_base_bi_flexiv_earbuds_case_insertion_teleop_rtc_0904_h100.yaml`）：

```yaml
model:
  max_token_len: 128          # 阶段五，按数据集实测收紧
  batch_image_views: true     # 阶段二
  use_cudnn_attention: true   # 阶段三
  cudnn_attention_dtype: float16   # 阶段四：必须 float16，bfloat16 会发散
data:
  base_config:
    tactile: false            # 阶段一
strict_batch_order: false     # 阶段一
```

配合 `XLA_FLAGS`（见 §5.2）。上表的测试跨越不同机器、batch 和任务，**不能把各阶段百分比直接连乘成
一个受控的总提速数字**；可复现的最终同配置结论是阶段五的 1.328 → 1.193 s/step。

性能主线的 commit 对应关系如下。`3b2795e`、`3c5aac6` 只是夹在分支中的 Nova5 配置改动，
与本文提速结论无关，故不纳入阶段收益。

| 日期 | commit | 作用 |
| --- | --- | --- |
| 08-27 | `ec72f34` | 合入跳过触觉解码、乱序交付与无限 sampler；同时把 PyTorch 下限提高到支持 `in_order` 的 2.6 |
| 08-27 | `9aef52d` | 共享内存 Tensor、零拷贝 NumPy 视图、显式关闭 DataLoader 与数据管线 profiling |
| 08-28 | `8ccf99c` | XProf 开关、只在日志步计算范数、三路相机合批及其测试 |
| 08-28 | `d0aec27` | 更新 H100 RTC 训练配置；没有单独的性能 A/B |
| 08-29 | `476babe` `f42992e` | 引入 cuDNN fused attention，并记录最初的短跑/生产验证 |
| 08-29 | `c1b70b2` `6305750` `51d9af3` | 处理全空 mask NaN；撤回逐步有限性保护；用 stop-gradient 保留快速原始 mask |
| 08-31 | `ad08973` `c9e77cc` `895811c` | 建立 strict-order/分层诊断，定位长期发散，并第一次合并事故文档 |
| 09-03 | `df9755f` | 增加 attention 精度与 Adam 更新空间诊断 |
| 09-04 | `3ea2b1c` | 定位 D 项舍入机制，实现并验证 fp16 custom VJP |
| 09-04 | `cf07bbe` | token 长度、XLA 集合通信和 remat A/B，形成 H100 最终配置 |

---

## 1. 阶段一：数据管线（2026-08-19 实现，08-25 生产复核，commit `ec72f34`）

起点是「训练每隔几十步卡 6 秒」。三个改动，其中只有两个真正兑现。

### 1.1 改动 A：跳过未使用的触觉视频解码（收益 0）

`LeRobotDataset.__getitem__` 无条件解码 `meta.video_keys` 里的**全部** 7 路视频，而 repack 只取
head / left_wrist / right_wrist——4 路触觉流解完直接丢。H.264 随机寻址解码只有顺序解码的 1/200，纯浪费。

实现是覆盖 `_get_query_timestamps`，让 `_query_videos` 永远不被要求解它们
（`src/openpi/training/data_loader.py:203`）：

```python
class SelectiveVideoLeRobotDataset(lerobot_dataset.LeRobotDataset):
    decode_video_keys: frozenset[str] | None = None   # 必须是普通属性：dataset 要 pickle 给 spawn 的 worker

    @override
    def _get_query_timestamps(self, current_ts, query_indices=None):
        query_timestamps = super()._get_query_timestamps(current_ts, query_indices)
        if self.decode_video_keys is None:
            return query_timestamps        # None = 不过滤，忘了设也退化成原行为
        return {key: ts for key, ts in query_timestamps.items() if key in self.decode_video_keys}
```

白名单由 `_resolve_decode_video_keys` 按子串 `tactile` 匹配（两代命名 `_tactile_{0,1}` 和
`_tactile_{left,right}` 都能吃下），并且**启动时校验**：如果 repack 引用了被禁用的流直接 `ValueError`，
不会静默拿到黑图。没有改 lerobot，没有转换数据。

**结果**：单样本 `__getitem__` 262.4 ms → 153.3 ms（**1.71×**），全 transform 链输出 20 个样本**逐位相同**。
**但 400 步端到端 A/B 完全没变**（768.4 s vs 770.8 s，停顿 20 次 vs 20 次）。唯一实测收益是启动快 37.3 s。

原因：这台机 batch 256 + 192 核，加载器本来就跟得上 GPU；省下的 CPU 余量 GPU 用不上。更关键的是
§1.4 的天花板与解码量无关。**保留它**（零风险、每样本少烧 40% CPU、epoch 边界省 37 s），
但别指望它让当前配置变快。

### 1.2 改动 B1：队头阻塞 → `in_order=False`

诊断装置是关键：真数据 + 真 transform + 真 DataLoader，用 `sleep(1.55)` 替代 GPU 步，同时抓 torch
迭代器的两个私有状态：

```
outstanding = _send_idx - _rcvd_idx     已派给 worker 但还没回来的任务数
buffered    = _task_info 里已完成的结果  算完了、但还轮不到它交付的 batch 数
```

**这两个数把「算不出来」和「算出来了但过不去」区分开——是整件事里最值钱的一步。**

| batch | wait | outstanding | buffered |
| --- | --- | --- | --- |
| 64 | 3.02 s | 1024 | 10 |
| 66 | 9.84 s | 1024 | **41** |
| 130 | 9.66 s | 1024 | **41** |

`outstanding` 恒为 1024（`prefetch_factor 16 × 64 workers`，管线一直是满的），停顿时 `buffered` 高达 41
——**41 个 batch 已经算完躺在那儿，训练循环却在等**。torch 严格按 sampler 顺序交付，任务 *j* 派给
worker `j mod 64`，这一轮最慢的那个 worker 堵住排在它后面的所有 batch，每 64 步撞一次。
调深 `prefetch_factor` 没用——问题不是没算，是不让过。

修复：`DataLoader(in_order=False)`（PyTorch ≥2.6 为这个场景加的官方参数），谁先算完谁先交付。
sampler 本来就是随机打散的，排序毫无意义。已验证一个 pass 内每个 index 仍恰好出现一次，不重不漏。

### 1.3 改动 B2：epoch 边界重建迭代器 → `InfiniteSampler`

`TorchDataLoader.__iter__` 在 sampler 耗尽时 `break` 出去重新 `iter()`，预取管线被整个拆掉重建，
训练循环要等**一个 worker 独自造完一整个 batch**：

```
STALL batch 80: wait=62.96s outstanding=79 buffered=60
```

**63.0 秒** = 256 样本 × 246 ms/样本。修复（`data_loader.py:529`）：

```python
class InfiniteSampler(torch.utils.data.Sampler[int]):
    """永不停止、每轮重新打散，StopIteration 再也不会触发。"""
    def __iter__(self):
        epoch = 0
        while True:
            if self._shuffle:
                generator = torch.Generator().manual_seed(self._seed + epoch)
                yield from torch.randperm(self._num_samples, generator=generator).tolist()
            else:
                yield from range(self._num_samples)
            epoch += 1
```

代价为零：openpi 本来就不 checkpoint 数据加载器状态（`restore_state` 直接 `del data_loader`），
batch 序列从来就不可恢复。

### 1.4 结果与残留

受控 400 步真实训练 A/B（同机同配置，唯一变量是 B1+B2）：

| 指标 | 修复前 | 修复后 | 倍数 |
| --- | --- | --- | --- |
| 400 步总时长 | 770.8 s | **715.7 s** | **1.08×** |
| 停顿步数（>0.5 s） | **20** | **0** | — |
| 最大 `next_batch` | 7.83 s | **0.43 s** | 18× |
| 中位 `next_batch` | 0.030 s | 0.290 s | — |

> 中位数从 0.030 涨到 0.290 s **不是退步**：同样的产能缺口，以前攒成每 64 步一次的 6 秒尖峰，
> 现在均摊到每一步，总等待反而少 16.8 s。

**生产复核**（08-25，`unscrew-the-bottle-cap-0821`，64,951 步 / 30.60 h）：数据停顿从归一化后的
每万步 334.7 次 / 2464 s 降到 **5.5 次 / 20 s**。该数据集一个 epoch 才 329 步，65,000 步里有 197 次
epoch 边界，按修复前每次 63 s 算光边界就该吃掉 3.45 h。实际停顿损失 0.04 h，**约 1.14×**。

残留的 36 次 `next_batch > 3 s` **全部落在 checkpoint 保存的下一步**（步号 1001、2001、3001…）。
`save_interval` 从 1000 提到 5000 即可，已写进后续所有生产 yaml。

修完之后 `buffered` **恒为 0**，加载器满负荷跑，稳态产能约 1.84 s/batch。折算每 worker 460 ms/样本，
而单进程只要 153 ms——并发到 64 个 worker 时每样本成本涨了 3 倍。这个天花板不在解码上
（直接证据：tactile 开/关对等待时间毫无影响）。阶段五复测显示等数据只剩 0.03–0.04 s/step，
数据侧已不是瓶颈，这条线到此为止。

### 1.5 开关

```yaml
data:
  base_config:
    tactile: false          # 默认；true = 解码触觉流（Pi05Tactile 等模型才需要）
strict_batch_order: false   # 默认；true = 恢复严格顺序交付，队头阻塞会回来
```

两个开关的默认值就是优化后的行为。`strict_batch_order: false` 意味着 batch 的**交付顺序**不再逐次可复现
（每个 index 每轮仍恰好出现一次，样本内容不变）。诊断用的单变量 A/B 必须设 `true`——阶段三/四的所有
3000 步验收都是这么跑的。

---

## 2. 阶段二：主机传输与计算侧（2026-08-27 ~ 08-28，commit `9aef52d` `8ccf99c`）

**测量口径**（后续所有短跑沿用）：从同一 checkpoint 只读恢复，独立实验目录，关 W&B 和所有 checkpoint 保存；
跑到起始 step + 40，丢弃前 10 步的编译/预热，统计后 30 个稳态样本。JAX dispatch 是异步的，
所以主指标是三者之和，不能单看 `dispatch`：

```
measured_step = dispatch + compute_sync + next_batch_total
```

### 2.1 共享内存 Tensor（commit `9aef52d`，−13.6%）

worker 原先返回 NumPy batch，过 multiprocessing queue 要 pickle/复制约 900 MiB，主进程再包装成 JAX array。
collate 改为返回 **CPU Torch Tensor**——PyTorch 的 multiprocessing reducer 会通过共享内存传 storage：

```python
def _collate_fn(items, *, profile=False):
    """Collate batch elements into CPU tensors so workers use shared-memory IPC."""
    batch = jax.tree.map(
        lambda *xs: torch.from_numpy(np.stack([np.asarray(x) for x in xs], axis=0)),
        *items,
    )
```

主进程再用 `tensor.numpy()` 拿同一块 storage 的**零拷贝视图**交给 `jax.make_array_from_process_local_data`。
同时加了显式 `DataLoader.close()`，训练退出时主动 shutdown worker 并超时终止残留进程。

`3.275728 → 2.828845 s/step`（**−13.64%**）。之后数据侧分项均值：`main_queue_wait 0.005 s`、
`jax_array_construct 0.009 s`、`h2d_wait 0.006 s`、`next_batch_total 0.046 s`——队列、array 构造和 H2D
都已经不是瓶颈。

### 2.2 XProf trace 定方向（commit `8ccf99c`）

训练脚本加了可控的 trace 开关（绝对起始 step + step 数 + 输出目录，起止显式 `block_until_ready`）。
采集 step 49011–49030 后按每卡每步汇总 kernel 时间：

```
GEMM     390.4 ms      Fusion   183.9 ms      NCCL   46.0 ms
其他       2.6 ms      Memcpy     1.5 ms      Memset  0.5 ms
```

GEMM + Fusion 占已记录 kernel 时间的 **92.6%**，NCCL 7.4%。**继续优化 H2D 或 NCCL 没有意义，
下一步必须动模型计算。** 这一步直接决定了阶段三去碰 attention。

（注意：trace 里 kernel 可能重叠，不能把 duration 相加当 GPU 利用率。）

### 2.3 只在日志步计算范数（commit `8ccf99c`，−0.5%）

`grad_norm` / `param_norm` 是整个 trainable pytree 的归约，但只在 `log_interval` 步被记录。
`train_step` 加静态布尔 `compute_metrics`（`scripts/train.py:168`）：

```python
    if compute_metrics:
        grad_norm = optax.global_norm(grads)
        param_norm = optax.global_norm(kernel_params)
    else:
        # 保持两个 JIT 变体的输出 pytree 一致；host 端用 nanmean 忽略占位值
        grad_norm = jnp.asarray(jnp.nan, dtype=loss.dtype)
        param_norm = jnp.asarray(jnp.nan, dtype=loss.dtype)
```

必须用**位置参数**传给 JIT：当前 JAX 在显式设置 `in_shardings` 时不接受 pjit 关键字实参。
收益很小但语义正确、零代价，保留。

### 2.4 三路相机合并为一次 SigLIP 调用（commit `8ccf99c`，−1.45%）

`Pi0.embed_prefix` 原先对 head / left_wrist / right_wrist 分别调用同一个 image encoder 三次
（`src/openpi/models/pi0.py:19`）：

```python
def _encode_image_views_as_batch(image_encoder, images):
    """Encode camera views as a sample-major interleaved batch."""
    names = tuple(images)
    stacked = jnp.stack([images[name] for name in names], axis=1)   # (B, V, H, W, C)
    batch_size, num_views = stacked.shape[:2]
    flat = stacked.reshape((batch_size * num_views, *stacked.shape[2:]))
    flat_tokens, _ = image_encoder(flat, train=False)               # 只调一次
    tokens = flat_tokens.reshape((batch_size, num_views, *flat_tokens.shape[1:]))
    return {name: tokens[:, index] for index, name in enumerate(names)}
```

布局是 **sample-major interleave**：同一样本的三路视图连续排列，所以 FSDP/DP 分片时每张卡仍拿到
完整样本的所有视图，不需要跨设备重排。开关 `model.batch_image_views`（默认关）。
单元测试逐元素比较了分别编码与合并编码的 token。

累计：`3.275728 → 2.773743 s/step`，**−15.32%**（等价吞吐 +18.10%）。

随后 `d0aec27` 只更新了两份 H100 RTC 配置的名称和参数，没有改变上述实现，也没有独立 A/B，
因此不把它计算为新的提速阶段。

> 一个容易被误读的现象：相机合并后 `dispatch` 从 2.657 s 降到 2.236 s，但 `compute_sync` 从 0.115 s
> 涨到 0.492 s。这是异步工作在 dispatch 与同步边界之间重新分布，**不能宣称 dispatch 单项的 16% 提升**。

---

## 3. 阶段三：cuDNN fused Attention 与两类失效（08-29 ~ 08-31）

> **一句话**：吞吐收益是真的（−29%），但全 18 层 cuDNN bf16 会让训练在约 1000 步后缓慢发散至 NaN。
> 这个阶段的产出是**两类失效的严格区分**和一份验收标准，真正的修复在阶段四。

### 3.1 实现与收益（commit `476babe`）

真实训练形状：`3×256 图像 token + 200 文本 token + 50 action token = 1018`，
`num_heads=8, num_kv_heads=1, head_dim=256, bf16`，block mask。

```python
jax.nn.dot_product_attention(q, k, v, mask=attn_mask, scale=1.0, implementation="cudnn")
```

`q` 在调用前已按 head dimension 缩放，所以必须 `scale=1.0` 避免重复缩放。只在 `kv_cache is None`
的训练路径启用，带 KV cache 的推理保持原实现。

8×H200 / batch 512 / step 16000 恢复的 A/B：`2.795537 → 1.978580 s/step`（**−29.22%**），
等价吞吐 183.15 → 258.77 samples/s。数值对齐测试当时也过了：BF16 前向最大绝对误差 0.0039，
Q/K/V 梯度最大绝对误差 1.9e-6 / 4.8e-7 / 2.4e-7。

**这些都是真的。问题在于它们全都不足以作为正确性验收。**

### 3.2 第一类失效：全空 mask 行的瞬时 NaN（08-29，commit `c1b70b2` `51d9af3`）

prompt 被补齐到 `max_token_len=200`，padding token 既不是有效 query 也不是有效 key，
所以 `make_attn_mask` 会生成整行全 `False` 的 query 行。切到 cuDNN 后从 clean step 16000 恢复：

```
Step 16000（切换前，显式）: grad_norm=0.6721, loss=0.1429, param_norm=1806.9792
Step 16100（切换后，cuDNN）: grad_norm=nan,    loss=0.1185, param_norm=nan
Step 16200 及以后:          grad_norm=nan,    loss=nan,    param_norm=nan
```

最小复现（BF16 GQA + 16 个全空 query 行）：**前向有限，Q 梯度 16384 个 NaN，K/V 梯度有限**。
真实生产形状是 71680 个 Q 梯度 NaN。

> **一个隐蔽的坑**：日志窗口对 loss 用 `nanmean`，第一个窗口里唯一有限的首步 loss 让聚合结果显示
> `0.1185`，掩盖了窗口其余全是 NaN。**「窗口 loss 有限」不能证明窗口内每个更新都有限。**

最终修复保留原始 mask，只对全空行停 Q 梯度（`src/openpi/models/gemma.py` 中的
`_stop_gradient_for_fully_masked_queries`）：

```python
def _stop_gradient_for_fully_masked_queries(q, attn_mask):
    query_has_key = jnp.any(attn_mask, axis=-1)[:, 0, :, None, None]
    return jnp.where(query_has_key, q, jax.lax.stop_gradient(q))
```

语义：Q 前向值逐元素不变；有效 query 的输出和梯度不变；无效 query 的 Q 梯度固定为 0；block mask 完整保留。

另一个方案是给每个空行开一个 dummy key——在有效行上给出**逐位相同**的 dQ/dK/dV，但每层要重建
`(B, 1, T, S)` mask，batch 256 下 1.38 vs 1.05 s/step，所以选了 stop-gradient。

提交演进要注意：`c1b70b2` 最初还加入了逐 step 的有限性检查，并采用 dummy-key mask；
`6305750` 恢复原训练更新语义、移除这项逐步保护，`51d9af3` 再把 dummy-key 换成上面的 stop-gradient，
这才是后续诊断和当前代码使用的实现。

checkpoint 处置：原 step 16000 已被保留策略删除，step 20000–55000 全部受污染并删除，
只能从 step 15000 重来（重跑约 1000 个有效 step）。**含 NaN 的参数或 Adam mu/nu 不能通过更换
attention 实现恢复。**

### 3.3 第二类失效：全层 cuDNN 在约 1000 步后缓慢发散（08-30 ~ 08-31，commit `ad08973` `895811c`）

修完全空 mask 后，全 18 层 cuDNN 前约 1000 步与显式基线贴合，随后分叉：

```
step    显式基线          cuDNN 9.14        cuDNN 9.19
 200    0.9244 /   2.89   0.9333 /   3.58   0.9348 /   3.53
1000    0.4220 /   2.77   0.4306 /   2.50   0.4319 /   3.16
1200    0.3742 /   2.12   0.6484 /  31.40   0.6600 /  35.49     ← 分叉点
2500    0.2457 /   1.33   5.7692 / 675.08   4.7055 / 759.18
5900    0.1763 /   1.36   nan               已停止
```

单元格为 `loss / grad_norm`。**因果起点是 step 1000–1200，不是最终 NaN 的 step 5900。**
单变量归因：同机同数据同超参同 seed，`git diff d0aec27 HEAD -- src/ scripts/` 的 72 行增删全部属于
attention 开关本身。

**但 cuDNN 的单步结果并不明显错误**：同权重同 batch 的完整模型梯度 A/B，梯度范数比 0.9937–1.0038，
相对梯度差 1.7e-2–2.1e-2；单层对 fp32 参考的相对误差 cuDNN 5.4e-3 vs 显式 bf16 3.2e-3，同阶。
cuDNN 同 batch 重复 backward 的随机差异（原子累加）就有 0.9%。**它给的是另一个合法的 BF16 近似。**

### 3.4 机制：Adam 逐坐标放大结构性差异

从 step-1000 checkpoint 恢复真实 Adam 状态，对同一 batch 比较实际参数更新：

```
                        相对更新差异   更新 cosine   符号翻转坐标
真实全层 cuDNN 差分         3.3521%      0.999440      0.7885%
1.5% 人造乘性噪声           0.7313%      --            0.1459%
```

梯度 L2 只差 1.5%，但经 Adam 逐坐标归一化后变成 3.35% 的更新差异，0.79% 的坐标符号翻转。
而按 `|g|` 分配能量的 1.5% 独立随机噪声几乎不扰动这些坐标，**从 step 0 稳定跑满 3000 步**
（step 2900 `loss=0.2411 / grad_norm=1.6389`）——证明训练配方并非对任意 1.5% 梯度噪声都没裕度，
问题在**误差的结构**而非幅度。

分层实验显示位置比数量更重要：

```
cuDNN 层段      相对更新差异   符号翻转坐标
全 18 层           3.3118%       0.7769%
最前 3 层          3.3016%       0.7720%     ← 只融合前 3 层几乎等同全 18 层
中间 3 层          1.9894%       0.4634%
最后 3 层          0.8470%       0.1926%
稳定噪声参考       0.7313%       0.1459%
```

早层的微小数值差异会穿过后续 15 层前向和反向放大。

### 3.5 当时的折中：只融合最后 3 层

`configs/diag_cudnn_last3.yaml`，8×H100 / batch 256 / strict-order，从 step 0 跑满 3000 步，
曲线与显式基线一致（step 1200 `0.3757 / 2.39`，step 2900 `0.2424 / 1.48`）。
但 `~1.5 s/step` 对显式的 1.57–1.64，只有 4–9% 加速，远低于全层的潜在 36%。

层选择开关（`Pi0Config`，阶段四后已不需要，保留作诊断用）：

```yaml
model:
  use_cudnn_attention: true
  cudnn_attention_layer_start: 15
  cudnn_attention_num_layers: 3
```

### 3.6 CUDA/cuDNN 环境事故

事故期间 `cuda_versions.cudnn_get_version()` 返回 `91400`，但进程实际加载的是
**9.10.2 的 dispatcher stub + 9.14 的 engine 库**——`cudnnGetVersion()` 的返回值来自 engine，
所以**只看这个数字发现不了混装**。

```
site-packages/nvidia/cudnn/lib/libcudnn.so.9        <- 9.10.2 dispatcher
$CONDA_PREFIX/lib/libcudnn_graph.so.9.14.0          <- 9.14 engine
```

当前环境已统一（CUDA 12.8 / PyTorch 2.11.0+cu128 / nvidia-cudnn-cu12 9.19.0.56 / JAX 0.5.3）。
**不要再导出 `LD_LIBRARY_PATH="$CONDA_PREFIX/lib"`**，残留的 conda-forge `libcublas`/`libcudart`/`libnccl`
会遮蔽 pip cu128 库。生产启动前在同一 shell 跑：

```bash
env -u LD_LIBRARY_PATH python scripts/check_cuda_stack.py
```

该脚本遍历 `/proc/self/maps`，在 `libcudnn*` 来自多个目录或版本不一致时判 FAIL，
并在真实训练形状上做前向+反向与 fp32 参考对比。

> **PASS 只证明环境单源、单步内核有限。第二类缓慢发散曾通过所有这类 smoke test。**

---

## 4. 阶段四：根因与 fp16 修复（09-03 ~ 09-04，commit `df9755f` `3ea2b1c`）

阶段三证明了「BF16 差分被 Adam 放大」，但没回答**差分从哪来、为什么它不像随机噪声**。这一阶段用真实激活回答了。

### 4.1 机制：flash 反向的 D 项用了 bf16 舍入后的 O

cuDNN fused attention 的反向按 flash-attention 方式算 `dS = P ⊙ (dP − D)`，其中
`D_i = rowsum(dO_i ⊙ O_i)` 用的是**前向存下的、已舍入到 bf16 的 O**。
显式路径用同一份舍入后的 dP 算 `D_i = Σ_j P_ij dP_ij`，抵消是自洽的。

两者之差是一个每行标量：

```
Δ_i = dO_i · (bf16(O_i) − O_i)
```

进入梯度后方向为 `−Δ_i · Σ_j P_ij K_j`，**破坏了 softmax 梯度「行和为零」的不变量**。

对 **peaked 行**（`maxP > 0.99`，即注意力 sink）杀伤最大：那里 `dP` 与 `rowsum(dO·O)` 几乎完全抵消，
真实 dQ 是 ε 级量，比普通行小两个数量级，而 H1 误差不随 ε 缩小——所以相对误差达 O(1)。

### 4.2 证据

两个探针：`scripts/attention_dterm_probe.py`（GPU，monkeypatch `gemma.Block.__call__`，
用 `jax.debug.callback` 在一次真实训练步里抓下 18 层的 q/k/v/mask 和反向余切 dO，缓存 1.5 GB）；
`scripts/attention_dterm_cpu_analysis.py`（CPU float64，对 peaked 行逐行重建精确反向）。

> 实现细节：模拟 bf16 舍入必须用 `jax.lax.reduce_precision`，`astype` 往返会被 XLA 的
> `allow_excess_precision` 删掉，预测会恒为零。

单层实测（`cos` 是误差向量与 H1 预测方向的 cos）：

```
层   变体           dQ rel    cos(all)   peaked rel   cos(peaked)
 0   explicit_bf16  2.46e-3   -0.02      2.8e-3       -0.04
     cudnn_bf16     3.58e-3   +0.49      3.5e-1       +0.83
     cudnn_fp16     1.71e-3   -0.02      3.9e-2       +0.13
 8   explicit_bf16  2.88e-3   +0.00      3.5e-3       -0.10
     cudnn_bf16     4.28e-3   +0.51      5.1e-1       +0.96
     cudnn_fp16     1.78e-3   +0.03      7.2e-2       +0.57

18 层几何平均：explicit_bf16 2.33e-3 / cudnn_bf16 3.72e-3 / cudnn_fp16 1.76e-3
18 层平均 cos(all)：explicit 0.006 / cudnn_bf16 0.418 / cudnn_fp16 0.006
```

- cuDNN bf16 的误差在**每一层**都与 H1 预测显著同向；显式和 fp16 都不同向。
- H1 预测的 peaked 行误差量级与实测吻合（0.41/0.35、0.63/0.51、0.86/1.21）。cos 不到 1 是因为
  cuDNN 前向的 PV 乘也把 P 舍入到 bf16。
- 候选机制 H2（dP 先舍入再减 D）的 cos 为 0 或负，排除。
- dV 没有 D 项，cuDNN bf16 的 dV 误差（2.5e-3）与显式（1.9e-3）同阶——干净的对照。

**这解释了阶段三的全部现象**：H1 分量按行沿 `Σ_j P_ij K_j` 方向、符号随舍入随机，投到参数上落在
`x ⊗ K_argmax` 这类**低维子空间**，不与 `|g|` 成比例。sink 机制处于平衡态时这些坐标的真实梯度和
二阶矩都很小，Adam 归一化后每步都在那里走满 lr 的随机步，约 1000 步后 sink 结构失稳。
乘性噪声在小 `|g|` 坐标上也小，Adam 看不见。跨 batch delta 两两 cos ±0.25 在 30 亿维空间里不是白噪声，
正是「同一子空间、随机符号」。只融合前 3 层≈全 18 层，因为 sink/peaked 结构在早层形成。
改 mask、seq_lengths、cuDNN 版本、LR、b1/b2/eps 都没用，**因为它们都不改变 O 的舍入**。

### 4.3 修复：fp16 内核 + 2 的幂动态缩放

fp16 比 bf16 多 3 位尾数，H1 误差降约 8 倍；代价是指数范围小得多，所以余切要做动态 loss scaling。
真实激活的范围 `max|q| ≤ 3.5`、`|k| ≤ 30`、`|v| ≤ 84`、`|dO| 1e-4–1e-2`，都在 fp16 范围内。

关键实现（`src/openpi/models/gemma.py:201`，`_cudnn_attention_in_dtype`）：

```python
    def attention_fwd(q, k, v, attn_mask):
        qc, kc, vc = (x.astype(compute_dtype) for x in (q, k, v))
        bias = jnp.where(attn_mask, jnp.asarray(0, compute_dtype),
                         _cudnn_fa.get_large_negative_number(compute_dtype))
        # cuDNN 的 custom partitioner 要求 q/k/v/bias 分片一致
        qc, kc, vc, bias = sharding.activation_sharding_constraint((qc, kc, vc, bias))
        out, res = _cudnn_fa._dot_product_attention_fwd_rule(...)
        return out.astype(out_dtype), (res, attn_mask)

    def attention_bwd(residuals, d_out):
        res, attn_mask = residuals
        d_out32 = d_out.astype(jnp.float32)
        max_abs = jnp.max(jnp.abs(d_out32))
        # 2 的幂缩放，使 max|d_out * scale| 落在 [0.25, 0.5)：尾数上精确，
        # 给内核里的 dP = dO @ V^T 和 dQ/dK/dV 留约 17 个 binade 余量
        exponent = jnp.floor(jnp.log2(jnp.maximum(max_abs, tiny)))
        scale = jnp.where(max_abs > 0, jnp.exp2(-(exponent + 2.0)), 1.0)
        grads = _cudnn_fa._dot_product_attention_bwd_rule(..., (d_out32 * scale).astype(compute_dtype))
        dq, dk, dv = grads[:3]
        # 与 _stop_gradient_for_fully_masked_queries 同语义：全 mask 行的 dQ 置零而非 NaN
        query_has_key = jnp.any(attn_mask, axis=-1)[:, 0, :, None, None]
        dq = jnp.where(query_has_key, dq, jnp.zeros_like(dq))
        ...
```

开关：`model.cudnn_attention_dtype: float16`（默认 `bfloat16`，逐位不变）。

**两版实现**：

- **v1**（3000 步验收用的）：custom VJP 的反向里用 `jax.vjp` 重走一遍 `jax.nn.dot_product_attention`
  前向再取反向。配合 `nothing_saveable` remat，每层反向比 bf16 路径多一次 attention 前向。1.37 s/step。
- **v2**（当前代码，commit `3ea2b1c`）：直接调 jax 0.5.3 私有的
  `jax._src.cudnn.fused_attention_stablehlo._dot_product_attention_fwd_rule / _bwd_rule`
  ——就是 `jax.nn.dot_product_attention(implementation="cudnn")` 自己 custom_vjp 里接的那两个函数。
  softmax stats 和输出当残差保存，反向只跑一次 fused backward。1.31–1.35 s/step。

  `attn_mask` 必须作为**显式参数**而非闭包传入，否则 flax remat 单独 trace 反向时会捕获泄漏的 tracer。

  `scripts/cudnn_fp16_vjp_equivalence.py` 在真实激活（层 0/4/8/14/17）上比较 v2 与 v1：
  前向逐位相同，dK/dV 逐位相同，dQ 相对差 5e-6–1.2e-5，与同一内核自身的 run-to-run 差
  （1.6e-6–1.4e-5，原子累加）同阶。

### 4.4 3000 步 strict-order 验收：通过

`configs/diag_cudnn_fp16.yaml`，与 `diag_cudnn_strict_order` 同数据/seed/batch 顺序/超参，
全 18 层 cuDNN fp16，从 pi05_base step 0 起，8×H100 batch 256：

```
step    fp16 loss / grad_norm    显式基线           last3 基线        历史 cuDNN bf16
   0    6.4030 /  65.59          6.4023 / 65.63     6.4029 / 65.52    6.4023 / 65.63
1000    0.4231 /   2.21          0.4220 /  2.77     0.4240 /  2.38    0.4319 /  3.16
1200    0.3759 /   2.36          0.3742 /  2.12     0.3757 /  2.39    0.6600 / 35.49   ← 历史分叉点
2000    0.2792 /   1.93                             0.2793 /  1.83    3.7753 / 150.5
2900    0.2427 /   1.59                             0.2424 /  1.48    6.2039 / 603.5
```

30 个 log 点全部有限，`grad_norm` 最大 4.75（step 100），之后 1.4–2.8，**没有一个点超过基线同点的
1.3 倍**（验收上限 3 倍）。历史分叉点处 loss 与显式基线差 <0.5%。
**把 cuDNN 内核精度从 bf16 换成 fp16 就消除了发散，验证了 §4.1 的机制。**

### 4.5 步时：attention 内核本身不解释差距

```
路径                                   s/step      来源
显式 attention                          1.57–1.64   8-31 报告 / 今日复测 1.60–1.63
cuDNN bf16（发散）                       1.05        8-31 报告
cuDNN fp16 v1（反向重算前向）             1.34–1.37   3000 步验收 run
cuDNN fp16 v2（直接调 fwd/bwd 规则）      1.31–1.35   400 步（loss 逐点与 v1 一致）
cuDNN bf16，今日同 config 复测            1.32–1.35   400 步
```

去掉反向重算只省约 2%。单卡微基准（`scripts/cudnn_attention_microbench.py`，层 0 真实激活，
T=S=1018，含 `nothing_saveable` remat）说明原因：

```
变体                fwd      fwd+bwd(remat)   ×18 层
cudnn_bf16         1.35 ms   5.24 ms          0.094 s/step
cudnn_fp16_v2      1.54 ms   5.53 ms          0.099 s/step
explicit_bf16      4.46 ms   8.49 ms          0.153 s/step
```

18 层 attention 前向+反向合计只占每步约 **0.1 s**；fp16 内核与 bf16 同速，动态缩放开销可忽略，
显式路径也只多 0.06 s。**所以 attention 内核解释不了 1.33 vs 1.05 的差距，也解释不了显式的 1.6 s**
——差距在整图层面（XLA 的 fusion/remat 决策、显存压力）。今日同 config 的 bf16 复测已是 1.32–1.35，
与 fp16 持平，说明 8-31 那个 1.05 s 不是同一编译环境下的可比数字。

### 4.6 一个必须记住的负面结论：单步指标不区分好坏

`scripts/attention_precision_update_probe.py`，锚点是 fp16 验收 run 的 step-1000 train state
（参数 + Adam 矩），以 fp32 显式 attention 为真值：

```
                   原始梯度 rel / sign_flips     Adam 更新 rel / cos / sign_flips
explicit_bf16      3.37e-2 / 1.53e-2            3.13e-2 / 0.99951 / 7.47e-3
cudnn_bf16         3.18e-2 / 1.43e-2            2.93e-2 / 0.99957 / 6.97e-3     ← 会发散的那个
cudnn_fp16         3.01e-2 / 1.38e-2            2.81e-2 / 0.99961 / 6.73e-3
```

**会发散的 cudnn_bf16 在这个指标上与两条稳定路径不可区分，甚至略好于 explicit_bf16。**
D 项误差在范数上只占 ~1e-3，危害来自方向的结构性（同一低维子空间、跨 step 累积）。
判定 fused attention 后端好坏要看 §4.2 的 `cos(err, H1)` 与 peaked 行误差，最终靠 3000 步 strict-order run。
**不要再用单步梯度/更新距离做验收指标。**

---

## 5. 阶段五：H100 端到端 A/B（09-04，commit `cf07bbe`）

**配置**：`pi05_base_bi_flexiv_earbuds_case_insertion_teleop_rtc_0904_h100.yaml`
（pi05、cuDNN fp16、`batch_image_views`、全局 batch 256、64 workers、fsdp 8）
**数据**：`Xense/earbud_case_insertion_teleop_0515_left_right`，776 集 / 1,145,613 帧
**口径**：每个变体从 pi05_base 起跑 400 步，`strict_batch_order: true`，关 checkpoint 和 W&B；
主指标是日志 `Step 100` 到 `Step 300` 的时间戳之差除以 200（含日志开销的端到端墙钟）；
`loss@300` 用来确认数值没变。

| 变体 | s/step | 相对基线 | loss@300 | 结论 |
| --- | --- | --- | --- | --- |
| E0 基线 | 1.328 | — | 0.7356 | 等数据 0.03 s，数据侧不是瓶颈 |
| E1 `max_token_len` 200→128 | 1.246 | **−6.2%** | 0.7356 | 采纳 |
| E2 关闭 remat | OOM | — | — | XLA 试图分配 280 GiB |
| E2b gemma remat `dots_with_no_batch_dims_saveable` | 编译失败 | — | — | cuDNN 分片器报 q/k/v 分片不一致 |
| E2c 同上 + q/k/v 显式分片约束 | OOM | — | — | 编译过了，单块残差 84.6 GB |
| E3 XLA 集合通信标志 | 1.281 | **−3.5%** | 0.7357 | 采纳 |
| E4 `fsdp_devices` 8→4 | 1.335 | +0.5% | 0.7354 | 无收益 |
| **E5 = E1 + E3** | **1.193** | **−10.2%** | 0.7356 | 生产采用；192.8 → 214.6 samples/s |

60000 步按稳态估算：22.1 h → 19.9 h，省约 2.2 h。

### 5.1 E1：token 长度按数据集实测收紧

`scripts/token_len_scan.py` 走真实 transform 链（repack → BiFlexivInputs → Normalize → TokenizePrompt），
对 4037 个状态（含每一维的最小/最大值样本）做 tokenize：min 110、p50 113、p99 115、**max 116**。
原来 200 里有 84 个是纯 padding，但 attention/MLP 照算。128 留 12 个余量。

```bash
JAX_PLATFORMS=cpu uv run scripts/token_len_scan.py \
  --config-name pi05_base_bi_flexiv_earbuds_case_insertion_teleop_rtc_0904_h100
```

> **换数据集或改 prompt 时必须重扫**：tokenizer 超长只会 warning 并截断，不会报错。

### 5.2 E3：XLA 标志

```
--xla_gpu_enable_latency_hiding_scheduler=true
--xla_gpu_all_gather_combine_threshold_bytes=1073741824
--xla_gpu_reduce_scatter_combine_threshold_bytes=1073741824
--xla_gpu_all_reduce_combine_threshold_bytes=1073741824
--xla_gpu_enable_pipelined_all_gather=true
--xla_gpu_enable_pipelined_reduce_scatter=true
--xla_gpu_enable_while_loop_double_buffering=true
```

FSDP 每步的参数 all-gather / 梯度 reduce-scatter 合并成大块并与计算流水。
**没有逐项拆分测哪一个起作用。** 通过 `XLA_FLAGS` 环境变量传入，已写在 0904 yaml 头部注释里。

### 5.3 E2：remat 在 80 GB 上没有空间

`gemma.py` / `siglip.py` 每层用 `nothing_saveable` 全量重算。本次给 `Pi0Config` 加了
`gemma_remat_policy` / `siglip_remat_policy`（默认不变），并在 cuDNN 调用前给 q/k/v/mask 加了统一的
batch 分片约束——否则非默认 remat 策略下编译失败：

```python
def _cudnn_attention_call(q, k, v, attn_mask):
    q = _stop_gradient_for_fully_masked_queries(q, attn_mask)
    # cuDNN 的 custom partitioner 要求 q/k/v 分片一致。显式钉到 batch 轴，
    # 否则 XLA 分片传播可能逐 operand 不同（非默认 remat 策略下出现过），编译报 "should have same sharding"。
    q, k, v, attn_mask = sharding.activation_sharding_constraint((q, k, v, attn_mask))
    return jax.nn.dot_product_attention(q, k, v, mask=attn_mask, scale=1.0, implementation="cudnn")
```

但 batch 256（每卡 32）时：完全不 remat 要 280 GiB 单块，只保存无 batch 维 matmul 也要 84.6 GB，都放不下。
**在 H200 141 GB 或每卡 batch 更小时这两个开关才可能兑现**，理论上限约为一次前向的重算（15–25%）。

### 5.4 其他观察

- worker 启动一次约 330 s（spawn 上下文，64 进程各自 import jax/torch/lerobot 并反序列化 1.1M 行数据集），
  是每次启动的一次性开销。
- `ResizeImages` 实际调用的是 `xense_client.image_tools`（PIL），worker 里**不跑 JAX resize**
  ——阶段一怀疑的头号嫌疑人被排除。
- fsdp 4 相对 8 没有收益：参数 all-gather 的量只差 12.5%，而每卡持有的优化器状态翻倍。
- 短跑结束后偶有 DataLoader worker 成为孤儿进程（ppid 1，每个约 3 GB RSS，一次残留 39 个）；
  批量实验后用 `ps -eo pid,ppid,cmd | grep spawn_main` 检查。
- 该配置 `num_train_steps: 60000` 但 `lr_schedule.decay_steps` 仍是默认 30000，后半程一直在底线 LR。
  **这是训练质量问题，本次未动。**

---

## 6. 新 attention 后端的验收标准

任何新的全层 fused/flash attention backend、custom VJP 或稳定化方案必须：

1. 从 **step 0** 开始；
2. `strict_batch_order: true`，固定 seed、数据集、batch 和超参；
3. 跑满 **3000 个更新**，不得只跑 40、60 或 500 步；
4. 与显式基线逐点比较 step 200/500/1000/1500/2000/2900；
5. 全程 loss、梯度、参数和 optimizer state 有限；
6. `grad_norm` 不得超过基线同点的 3 倍；
7. 通过后再进入有 checkpoint 的受监控长跑。

**以下均不是有效验收**：单次前向/反向有限；40-step 吞吐短跑；从已收敛 checkpoint 短跑；
200–500 个连续有限 step；只检查最终是否 NaN；单步梯度/Adam 更新距离（§4.6）；
跨不同 batch 顺序的 run 直接比较单点 grad norm。

**建议增加发散护栏**：当前只在日志步算 `grad_norm`。在 host 端维护滚动中位数，
连续多个窗口超过固定倍数时中止——按 §3.3 的曲线可在 step 1200–1300 停下，而不是污染数万个 step。

**checkpoint 恢复规则**：一旦参数或 Adam `mu`/`nu` 出现 NaN，后续 checkpoint 全部视为污染，
不得通过切换 attention、降 LR 或改 optimizer 继续；必须回退到最后一个确定干净的 checkpoint，
用独立实验名验证（关 W&B 和所有保存）后才能恢复生产保存。

---

## 7. 不要重复的弯路

**测量方法**

1. **先看停顿的周期再动手。** 周期 = `num_workers` 是队头阻塞，周期 = 一个 epoch 的步数是边界重建，
   周期 = `save_interval` 是 checkpoint——都不是「解码太慢」。
2. **`buffered` 千金难买。** 停顿时它是 0 说明真的算不出来，是 41 说明算出来了过不去，两种修法完全相反。
3. **别把别的机器上的预估当结论。** 「跳过触觉解码约 1.8×」在 h200 箱 batch 512 下是对的，
   搬到 batch 256 + 192 核就一点不剩。
4. **别拿两个不同任务的日志比绝对时长。** 要比就比归一化指标，或者跑受控 A/B。
5. **别单看 `dispatch`。** JAX 异步，收益可能只是从 dispatch 挪到 compute_sync。
6. **`nanmean` 会掩盖 NaN。** 窗口 loss 有限不证明窗口内每个更新有限。

**cuDNN attention**

7. 不要再把全空 mask 的瞬时 NaN 与全层缓慢发散当成同一问题。
8. 不要再把 cuDNN 9.15.1 的非 128 倍序列长度缺陷或 `T=1024` 当作本任务的修复
   （实测 T=1018 与 T=1024 的 dQ 误差同为 3.2e-3，NaN=0）。
9. 不要再尝试传完整物理长度的 `query_seq_lengths`/`key_value_seq_lengths`
   （编译确认进了 custom call，但数值指纹 `3.311% / 0.776%` 与已知发散路径完全一致，3000 步照样发散）。
10. 不要再改 dummy-key / stop-gradient 的 mask 语义来解决发散——两者在有效行上数值等价。
11. 不要只调 `peak_lr`（2.5e-5→1.5e-5 只推迟约 300 步）、Adam `b1`（0.9→0.99 推迟到约 1400 步）、
    `b2`（0.95→0.99 照样发散）或 `eps`（更新差异几乎不变）。
12. 不要用独立随机噪声代替真实 cuDNN 差分做实验——1.5% 乘性噪声能稳定跑满 3000 步。
13. 不要试「V 减去 sink 键的 V」：peaked 行大多不在看 BOS，而是在看 action/text/image 的其他键。
14. 不要只看 `cuda_versions.cudnn_get_version()` 判断动态库是否干净（它由 engine 返回，看不见 dispatcher 混装）。
15. 用 `jax.lax.reduce_precision` 模拟低精度舍入，`astype` 往返会被 XLA 删掉。

---

## 8. 脚本与配置索引

**诊断脚本**

| 脚本 | 用途 |
| --- | --- |
| `scripts/check_cuda_stack.py` | 枚举 `/proc/self/maps` 里实际加载的 cuDNN/cuBLAS/NCCL，+ 真实形状数值 smoke test |
| `scripts/token_len_scan.py` | 走真实 transform 链扫数据集的 tokenized prompt 长度分布 |
| `scripts/attention_dterm_probe.py` | 抓真实激活 + 单层五变体误差 + H1/H2 预测（阶段四主力） |
| `scripts/attention_dterm_cpu_analysis.py` | peaked 行的 float64 逐行解剖 |
| `scripts/cudnn_fp16_vjp_equivalence.py` | fp16 VJP v2（直接调规则）与 v1（反向重算）的等价性 |
| `scripts/cudnn_attention_microbench.py` | 单层 attention 前向/反向微基准 |
| `scripts/attention_precision_update_probe.py` | 真实 Adam 状态上的更新空间对比（**注意 §4.6：该指标不区分好坏**） |
| `scripts/grad_ab_probe.py` | 完整模型显式/cuDNN 梯度 A/B |
| `scripts/grad_ab_direction_probe.py` | 多 batch 差分方向和随机分量 |
| `scripts/hybrid_update_ab_probe.py` | 不同 cuDNN 层段的更新空间误差 |
| `scripts/optimizer_update_ab_probe.py` | 从真实 Adam 状态比较参数更新误差 |

**诊断配置**：`configs/diag_cudnn_strict_order.yaml`（全层 bf16，会发散）、
`configs/diag_explicit_strict_order.yaml`（基线）、`configs/diag_cudnn_last3.yaml`（后 3 层折中）、
`configs/diag_cudnn_fp16.yaml`（fp16 验收）、`configs/diag_cudnn_b1_099.yaml`。

**代码位置**

| 文件 | 相关改动 |
| --- | --- |
| `src/openpi/training/data_loader.py` | `SelectiveVideoLeRobotDataset`、`_resolve_decode_video_keys`、`InfiniteSampler`、`in_order` 接线、Torch Tensor collate、`close()` |
| `src/openpi/training/config.py` | `DataConfig.tactile`、`TrainConfig.strict_batch_order` |
| `src/openpi/models/pi0_config.py` | `batch_image_views`、`use_cudnn_attention`、`cudnn_attention_{layer_start,num_layers,dtype}`、`gemma_remat_policy`、`siglip_remat_policy` |
| `src/openpi/models/pi0.py` | `_encode_image_views_as_batch` |
| `src/openpi/models/gemma.py` | `_stop_gradient_for_fully_masked_queries`、`_cudnn_attention_call`、`_cudnn_attention_in_dtype`、`_cudnn_static_args`、逐层 cuDNN 开关 |
| `scripts/train.py` | `compute_metrics` 静态分支、XProf trace 开关、数据管线 profiling |

---

## 9. 生产启动

```bash
env -u LD_LIBRARY_PATH python scripts/check_cuda_stack.py    # 先确认环境单源

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

换数据集时记得：重扫 `max_token_len`（§5.1）、重算 norm stats、检查 `lr_schedule.decay_steps`
是否匹配 `num_train_steps`（§5.4）。
