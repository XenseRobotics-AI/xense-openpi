# cuDNN fused Attention NaN 与训练发散：统一事故报告

**覆盖日期**：2026-08-29 至 2026-08-31

**涉及机器**：8×H200、8×H100 80GB

**涉及实现**：JAX `dot_product_attention(..., implementation="cudnn")`、BF16 GQA

**最终状态**：全空 mask 的瞬时 NaN 已修复；全 18 层 cuDNN 的缓慢训练发散仍未解决。生产训练不得启用全 18 层 cuDNN。

> [!CAUTION]
> 本次调查包含两种不同的失效，不能混为一个 cuDNN bug：
>
> 1. **瞬时 NaN**：全空 attention mask query 行使旧环境中的 cuDNN backward 产生 NaN Q 梯度。`_stop_gradient_for_fully_masked_queries` 已修复这一问题。
> 2. **缓慢发散**：修复全空行后，全 18 层 cuDNN 仍在 step 1000–1200 与显式 attention 基线分叉，随后梯度爆炸并最终变为 NaN。根因是 BF16 实现差异的坐标结构被 Adam 放大，而不是非 128 倍序列长度、cuDNN 版本或 mask 修复。

> [!IMPORTANT]
> 截至本文最后一次验证：
>
> - **稳妥生产方案**：`use_cudnn_attention: false`；
> - **已通过 3000 步短程验收的折中方案**：只在 Gemma 最后 3 层启用 cuDNN；
> - **禁止用于生产**：全 18 层 cuDNN，即 `use_cudnn_attention: true` 且 `cudnn_attention_num_layers: null`；
> - 已经出现 NaN 的参数和 Adam 状态不可修复，必须从确定干净的 checkpoint 恢复。

## 1. 背景与真实训练形状

Pi0.5 训练时的 Gemma attention 序列由三部分组成：

```text
3 × 256 个图像 token + 200 个文本 token + 50 个 action token = 1018
num_heads=8, num_kv_heads=1, head_dim=256, dtype=bf16
```

prompt 被补齐到 `max_token_len=200`。padding token 既不是有效 query，也不是有效 key，因此 `make_attn_mask` 会生成整行全 `False` 的 query 行。训练使用 block mask；只有 action token 上的 cotangent 非零。

显式 attention 路径执行 QK einsum、有限 `big_neg` mask、softmax 和 PV einsum。cuDNN 路径调用：

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

`q` 已在调用前按 head dimension 缩放，因此必须使用 `scale=1.0`，避免重复缩放。

## 2. 第一类失效：全空 mask 行导致瞬时 NaN

### 2.1 事故范围

**日期**：2026-08-29

**机器**：8×H200

**配置**：`pi05_base_bi_flexiv_earbud_case_insertion_0826_h200`

**触发方式**：从 clean step 16000 切换到 `use_cudnn_attention=true` 后恢复生产训练。

切换前显式 attention 指标正常：

```text
Step 16000: grad_norm=0.6721, loss=0.1429, param_norm=1806.9792
Step 16100: grad_norm=1.1132, loss=0.1456, param_norm=1806.9926
Step 16200: grad_norm=0.8198, loss=0.1402, param_norm=1807.0051
Step 16300: grad_norm=0.9735, loss=0.1410, param_norm=1807.0171
```

从 checkpoint 切到 cuDNN 后：

```text
Step 16100: grad_norm=nan, loss=0.1185, param_norm=nan
Step 16200: grad_norm=nan, loss=nan,    param_norm=nan
Step 16300 及以后: grad_norm=nan, loss=nan, param_norm=nan
```

checkpoint 目录名记录循环 step，而保存的 `train_state.step` 已经加一。恢复后的第一个前向 loss 仍有限，但同一步反向产生 NaN 并污染参数，之后所有前向均为 NaN。

日志窗口对 loss 使用 `nanmean`，第一个窗口里唯一有限的首步 loss 使聚合结果仍显示 `0.1185`，掩盖了该窗口其余 NaN。因此“窗口 loss 有限”不能证明窗口内每个更新都有限。

### 2.2 最小复现与根因

在当时实际加载 cuDNN runtime `91400` 的环境中，使用 BF16 GQA 和含 16 个全空 query 行的 mask：

```text
forward output finite: true
Q/K/V gradient finite: [false, true, true]
Q gradient NaN count: 16384
```

为全空 query 行临时开放一个 dummy key 后：

```text
Q/K/V gradient finite: [true, true, true]
Q/K/V gradient NaN count: [0, 0, 0]
valid query rows unchanged: true
```

完整生产形状也复现了相同现象：原始 backward 产生 71680 个 Q 梯度 NaN，K/V 梯度有限。

### 2.3 实际修复

最终实现保留原始 attention mask，不为全空行打开 dummy key。调用 cuDNN 前找出至少存在一个有效 key 的 query 行，并只停止全空 query 行的 Q 梯度：

```python
query_has_key = jnp.any(attn_mask, axis=-1)[:, 0, :, None, None]
q = jnp.where(query_has_key, q, jax.lax.stop_gradient(q))
```

该变换具有以下语义：

- Q 的前向值逐元素不变；
- 有效 query 的输出和梯度不变；
- 无效 query 的 Q 梯度固定为 0；
- 原始 block mask 完整保留。

dummy-key 版本在有效行上与 stop-gradient 版本给出逐位相同的 dQ/dK/dV，但每层重建 `(B, 1, T, S)` mask，batch 256 下约为 `1.38 s/step`；stop-gradient 版本约为 `1.05 s/step`，因此保留后者。

### 2.4 修复验证及其局限

当时完成的验证包括：

- `src/openpi/models/pi0_test.py`：`8 passed`；
- `scripts/train_test.py`：`2 passed`；
- H200/cuDNN BF16 GQA 真实形状 Q/K/V 梯度全部有限；
- 从 clean step 15000 独立恢复，连续执行 step 15001–15060；
- 每 10 步的 loss、grad norm、param norm 和 `update_is_finite` 均有限。

这足以证明第一类“第一步反向立即 NaN”已被修复，但不足以证明训练长期收敛。后续事实表明，60 步、200–500 步、已收敛 checkpoint 短跑和单次有限性 smoke test 都会漏掉第二类失效。

### 2.5 checkpoint 处置

H200 事故中最后一个确定干净且仍保留的 checkpoint 是 step 15000。原 step 16000 已被保留策略删除，step 20000–55000 受到 NaN 污染并已删除。从 step 15000 恢复意味着重跑约 1000 个有效 step。

已经包含 NaN 的参数、Adam 一阶矩或二阶矩不能通过更换 attention 实现恢复；不得继续使用污染 checkpoint。

## 3. 第二类失效：全层 cuDNN 在约 1000 步后缓慢发散

### 3.1 现象

全空 mask 修复后，全 18 层 cuDNN 在前约 1000 步与显式 attention 基线贴合，随后在 step 1000–1200 分叉，梯度持续放大，最终变为 NaN。

```text
step    显式基线             cuDNN 9.14            cuDNN 9.19
 200    0.9244 /   2.89      0.9333 /   3.58       0.9348 /   3.53
1000    0.4220 /   2.77      0.4306 /   2.50       0.4319 /   3.16
1100    0.4077 /   1.79      0.4784 /   3.16       0.4723 /   3.37
1200    0.3742 /   2.12      0.6484 /  31.40       0.6600 /  35.49
2500    0.2457 /   1.33      5.7692 / 675.08       4.7055 / 759.18
5900    0.1763 /   1.36      nan                    已停止
```

每个单元格为 `loss / grad_norm`。

因果起点是 step 1000–1200，而不是最终出现 NaN 的 step 5900。NaN 只是长期发散后的溢出结果。

### 3.2 单变量责任归属

同机、同数据集、同超参、同 seed 的 A/B 中，唯一变量是 `use_cudnn_attention`：

```text
cve5hiyt（0828）  commit d0aec27  显式 attention  step 59900 loss=0.0589
orbnh1b7（0829）  commit 51d9af3  全层 cuDNN     step 5900  loss=nan
```

`git diff d0aec27 HEAD -- src/ scripts/` 的 72 行增删全部属于 attention 开关本身，排除了学习率、数据集和训练配置变化。

### 3.3 cuDNN 单步结果并非明显错误

同一权重、真实 batch 和 rng 的完整模型梯度 A/B：

```text
batch 0  loss 显式=5.859448  cuDNN=5.851340  |grad| 104.31 vs 103.65  ratio=0.9937
batch 1                                      |grad|  67.19 vs  67.45  ratio=1.0038
batch 2                                      |grad|  70.71 vs  70.61  ratio=0.9986

相对梯度差：1.75e-02 / 1.66e-02 / 2.12e-02
```

梯度范数比值接近 1，没有单个模块出现异常放大。单层真实形状中，cuDNN 对 fp32 参考的相对误差约 `5.4e-03`，显式 BF16 路径约 `3.2e-03`，两者同阶。

同一 batch 重复 cuDNN backward 的随机差异约 `0.9%`，来自 atomic 累加；cuDNN 与显式路径之间的确定性分量约 `1.5%`。因此 cuDNN 给出的是另一个合法的 BF16 近似，而不是单步明显错误的梯度。

## 4. 根因机制：Adam 放大坐标级差异

### 4.1 差分没有固定跨 batch 方向

对 8 个真实 batch 分别计算显式梯度和 3 次 cuDNN backward 均值：

```text
跨 batch delta 两两 cosine：-0.004043 ± 0.126301，范围 [-0.247516, +0.226915]
||delta|| / ||g_explicit||： 1.556024e-02 ± 2.76e-03
cuDNN 同 batch 随机分量：   8.752676e-03
cos(delta, gradient)：      -0.00113 ± 0.225
cos(delta, parameter)：     +0.000016
```

差分没有稳定的跨 batch 方向，也不沿当前梯度或参数方向累积。

### 4.2 Adam 更新空间暴露了真正差异

从 strict-order 全层 cuDNN run 的 step 1000 checkpoint 恢复真实 Adam 状态，对同一真实 batch 和 rng 比较实际参数更新：

```text
                         相对更新差异   更新 cosine   符号翻转坐标
真实全层 cuDNN 差分          3.3521%      0.999440       0.7885%
1.5% 人造乘性噪声            0.7313%      --             0.1459%
```

cuDNN 原始梯度 L2 差异只有约 1.5%，但其误差分布会扰动小梯度、小二阶矩坐标，经过 Adam 逐坐标归一化后变成约 3.35% 的更新差异，并使约 0.79% 的非零更新坐标翻转符号。

相比之下，按 `|g|` 分配能量的 1.5% 独立随机噪声几乎不扰动这些敏感坐标。该人造噪声从 step 0 稳定跑满 3000 步，step 2900 为 `loss=0.2411 / grad_norm=1.6389`，证明训练配方并非对任意 1.5% 梯度噪声都没有裕度。

### 4.3 早层误差会穿过后续网络放大

在同一个 step-1000 Adam 状态上，仅让不同连续层段使用 cuDNN：

```text
cuDNN 层段       相对更新差异   更新 cosine   符号翻转坐标
全 18 层            3.3118%       0.999451       0.7769%
最前 6 层           3.2061%       0.999486       0.7548%
中间 6 层           2.2169%       0.999754       0.5141%
最后 6 层           1.1633%       0.999932       0.2720%
最前 3 层           3.3016%       0.999455       0.7720%
中间 3 层           1.9894%       0.999802       0.4634%
最后 3 层           0.8470%       0.999964       0.1926%
稳定噪声参考         0.7313%       --             0.1459%
```

只融合最前 3 层几乎等同全 18 层。层的位置比数量更重要：早层的微小数值差异会穿过后续 15 层前向和反向传播，显著放大。

## 5. 已否定的修复假设

### 5.1 升级 cuDNN 不能解决缓慢发散

cuDNN 9.14 与 9.19 的发散曲线在 step 1000–1200 的同一位置分叉。当前统一环境实际加载 `nvidia-cudnn-cu12==9.19.0.56`，全层训练仍发散。

### 5.2 非 128 倍序列长度不是本任务根因

cuDNN 9.15.1 release note 提到旧版 SDPA backward 在静态序列长度不是 128 倍数时可能产生错误结果或 NaN，因此曾怀疑 `T=1018` 命中该缺陷。真实形状对照结果：

```text
T=1018 (%128=122)  dQ rel-L2: 显式 3.23e-03  cuDNN 3.21e-03  NaN=0
T=1024 (%128=0)    dQ rel-L2: 显式 3.23e-03  cuDNN 3.21e-03  NaN=0
```

把 `max_token_len` 从 200 临时改为 206 不能改变数值行为。该 NVIDIA 缺陷与本任务的缓慢发散不是同一问题。

参考：

- [NVIDIA/cudnn-frontend #160](https://github.com/NVIDIA/cudnn-frontend/issues/160)
- [cuDNN 9.15.1 Release Notes](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.15.1/release-notes.html)

### 5.3 原生 seq_lengths / padding kernel 不能解决发散

2026-08-31 对旧 summary 建议的方案进行了最终验证：向 cuDNN 调用同时传入完整物理长度的 `query_seq_lengths` 和 `key_value_seq_lengths`，值均为 1018；实际 padding 语义仍由原 block mask 负责。

编译结果确认 seq lengths 已作为两个 `s32[B]` operand 进入 `__cudnn$fmhaScaleBiasSoftmax` 和 backward custom call，但未切换到数值上不同的算法。

真实单层形状、三次重复 backward：

```text
NaN count，无 seq_lengths： [0, 0, 0]
NaN count，有 seq_lengths： [0, 0, 0]
相对 fp32 梯度误差，无 seq_lengths：约 5.66924e-03
相对 fp32 梯度误差，有 seq_lengths：约 5.66927e-03
两路径相对梯度差：5.3e-06 至 3.1e-05
```

两路径之差与同一 cuDNN backward 的 atomic 随机波动同量级。

在 step-1000 真实 Adam 状态上的全模型对照：

```text
显式 loss：                 0.31219399
padding-cuDNN loss：        0.31204814
Adam update 相对差：        3.310997%
update cosine：             0.99945178
符号翻转坐标：              0.77606%
```

这与已知会发散的全 18 层 cuDNN 指纹 `3.3118% / 0.7769%` 几乎完全一致。把 Adam `eps` 提高到 `1e-6` 或 `1e-5` 后，update 相对差仍分别为 `3.1655%` 和 `3.0275%`，符号翻转仍约 `0.776%`。

最后又完成了 3000-step、8×H100、batch 256、`strict_batch_order=true` 的全 18 层 padding-cuDNN 验收。W&B、周期 checkpoint 和最终 checkpoint 均关闭：

```text
step    loss      grad_norm    param_norm
   0    6.4023       65.6324    1802.3865
 500    0.5719        2.5464    1802.4017
1000    0.4298        2.4472    1802.5095
1100    0.4591        2.2270    1802.5511
1200    0.5902       13.5020    1802.5952
1300    0.9563      108.9634    1802.6366
1500    2.2053       45.7198    1802.7318
2000    3.7753      150.5048    1802.9735
2400    4.7852     6469.9019    1803.1691
2500    5.7166      609.2015    1803.2141
2900    6.2039      603.4764    1803.3545
```

step 1200 的显式基线约为 `0.3742 / 2.12`，padding-cuDNN 梯度已达到约 6.4 倍，超过 3 倍验收上限。该方案跑满 3000 步时尚未直接变成 NaN，但已沿已知轨迹持续发散，不能作为解决办法。

### 5.4 其他已排除项

- **两种 mask 修复**：dummy key 与 Q stop-gradient 在有效行上数值等价，两者都会出现第二类发散；
- **动态库混装**：统一到单源 pip cu128/cuDNN 9.19 后仍发散；
- **logit 幅度**：std 1–80 扫描中 cuDNN 与显式误差同阶；
- **head_dim=256**：与 head_dim=128 表现一致；
- **cotangent 稠密度**：只在 action token 非零和所有有效 token 稠密时结果一致；
- **dBias 路径**：实际 batch 256 不满足 `bias.shape[0]==1`，未进入该路径；
- **FSDP 分片**：attention 只按 batch 分片，每卡拥有完整序列；
- **batch 到达顺序**：`strict_batch_order=true` 仍在 step 1000–1200 分叉；
- **降低 peak LR**：`2.5e-5 -> 1.5e-5` 只把分叉推迟约 300 步；
- **提高 Adam b2**：`0.95 -> 0.99` 从 step 0 跑满 3000 步仍发散；
- **提高 Adam b1**：`0.9 -> 0.99` 只把明显分叉推迟到约 step 1400；
- **提高 Adam eps**：update 差异和符号翻转几乎不变；
- **1.5% 独立随机梯度噪声**：显式 attention 稳定跑满 3000 步，没有复现发散。

## 6. 已验证的稳定折中：只融合最后 3 层

配置 `configs/diag_cudnn_last3.yaml` 使用：

```yaml
model:
  use_cudnn_attention: true
  cudnn_attention_layer_start: 15
  cudnn_attention_num_layers: 3
```

8×H100、batch 256、strict-order、原始 Adam，从 pi05_base step 0 开始，完整执行 3000 个更新：

```text
step    loss      grad_norm    param_norm
   0    6.4029      65.5204     1802.3865
 500    0.5719       2.4526     1802.4014
1000    0.4240       2.3789     1802.5061
1100    0.4050       2.1847     1802.5416
1200    0.3757       2.3869     1802.5775
1600    0.3238       2.6505     1802.7225
2000    0.2793       1.8254     1802.8666
2500    0.2463       2.0116     1803.0460
2900    0.2424       1.4769     1803.1906
```

该曲线与显式基线一致，满足 3000 步验收。普通 step 约 `1.5 s/step`，相比显式路径约 `1.57–1.64 s/step` 只有约 4–9% 加速，明显低于全层 cuDNN 的约 36% 潜在收益。

最后 3 层方案只是已通过短程验收的保守折中，尚未完成 60000 步生产验收，不能等同于完整生产结论。

## 7. CUDA/cuDNN 环境事故与当前要求

### 7.1 历史混装问题

事故期间 `cuda_versions.cudnn_get_version()` 返回 `91400`，但进程实际加载了 9.10.2 dispatcher stub 和 9.14 engine 库：

```text
site-packages/nvidia/cudnn/lib/libcudnn.so.9          <- 9.10.2 dispatcher
$CONDA_PREFIX/lib/libcudnn_graph.so.9.14.0            <- 9.14 engine
$CONDA_PREFIX/lib/libcudnn_engines_precompiled.so.9.14.0
$CONDA_PREFIX/lib/libcudnn_ops.so.9.14.0
```

`cudnnGetVersion()` 的返回值来自 engine，因此只看 `91400` 无法发现 dispatcher/engine 混装。

### 7.2 当前统一环境

环境已统一为：

```text
CUDA Toolkit          12.8
PyTorch               2.11.0+cu128
torchvision           0.26.0+cu128
torchcodec             0.11.1+cu128
nvidia-cudnn-cu12      9.19.0.56
JAX                    0.5.3
```

启动训练时不要导出 `LD_LIBRARY_PATH="$CONDA_PREFIX/lib"`。该目录中残留的 conda-forge `libcublas`、`libcudart` 或 `libnccl` 可能遮蔽 pip cu128 库。

生产启动前在相同 shell 中运行：

```bash
env -u LD_LIBRARY_PATH python scripts/check_cuda_stack.py
```

脚本会枚举 `/proc/self/maps` 中实际加载的 cuDNN/cuBLAS/NCCL 路径，并在真实 attention 形状上做前向、反向和 fp32 参考对比。

> [!NOTE]
> `check_cuda_stack.py` PASS 只证明环境单源且单步内核有限、误差在阈值内。第二类缓慢发散曾通过所有此类 smoke test，因此 PASS 不能替代 3000 步训练验收。

## 8. 生产处置与恢复规则

### 8.1 当前可选配置

需要绝对稳妥时：

```yaml
model:
  use_cudnn_attention: false
```

需要先获取小幅吞吐收益并接受受监控长跑时：

```yaml
model:
  use_cudnn_attention: true
  cudnn_attention_layer_start: 15
  cudnn_attention_num_layers: 3
```

不得用于生产：

```yaml
model:
  use_cudnn_attention: true
  cudnn_attention_num_layers: null  # 全 18 层
```

### 8.2 checkpoint 恢复

- 一旦参数、Adam `mu` 或 `nu` 出现 NaN，后续 checkpoint 均视为污染；
- 不得通过切换 attention、降低 LR 或修改 optimizer 后继续污染 checkpoint；
- 必须回退到最后一个确定干净的 checkpoint；
- 恢复前检查 checkpoint 内参数、optimizer state 和 step；
- 使用独立实验名验证，关闭 W&B、周期保存和最终保存，避免覆盖生产目录；
- 只有通过完整验收后才能恢复生产保存。

### 8.3 建议增加发散护栏

当前训练只在日志 step 计算 `grad_norm`。建议在 host 端维护滚动中位数，并在连续多个日志窗口超过固定倍数时中止。按本次曲线，该护栏可在 step 1200–1300 停止，而不是等到 loss 变成 NaN 或污染数万个 step。

护栏只能减少损失，不能让全层 cuDNN 变得收敛。

## 9. 新 attention 实现的统一验收标准

任何新的全层 fused/flash attention backend、custom VJP 或稳定化方案必须：

1. 从 **step 0** 开始；
2. 使用 `strict_batch_order=true`，固定 seed、数据集、batch 和超参；
3. 跑满 **3000 个更新**，不得只跑 40、60 或 500 步；
4. 与显式 attention 基线逐点比较 step 200/500/1000/1500/2000/2900；
5. 全程 loss、梯度、参数和 optimizer state 有限；
6. `grad_norm` 不得超过基线同点的 3 倍；
7. 通过后再进入有 checkpoint 的受监控长跑；
8. 生产长跑仍需持续观察 loss 和 grad norm。

以下均不是有效正确性验收：

- 单次前向/反向有限；
- 40-step 吞吐短跑；
- 从已经收敛的 checkpoint 短跑；
- 200–500 个连续有限 step；
- 只检查最终是否出现 NaN；
- 跨不同 batch 顺序的 run 直接比较单点 grad norm。

## 10. 不要重复的弯路

- 不要再次把全空 mask 的瞬时 NaN 与全层缓慢发散当成同一问题；
- 不要再把 cuDNN 9.15.1 或 `T=1024` 当作本任务的修复；
- 不要再尝试完整物理长度的 `query_seq_lengths/key_value_seq_lengths`；
- 不要再修改 dummy-key/stop-gradient mask 语义来解决第二类发散；
- 不要只调低 `peak_lr` 或提高 Adam `b1`、`b2`、`eps`；
- 不要用 1.5% 独立随机噪声代替真实 cuDNN 差分；
- 不要只看 `cuda_versions.cudnn_get_version()` 判断动态库是否干净；
- 不要把最后 3 层的 3000 步通过误写成全 18 层或 60000 步生产通过。

## 11. 相关脚本与配置

- `scripts/check_cuda_stack.py`：检查实际映射的 CUDA/cuDNN 库并做真实形状数值 smoke test；
- `scripts/grad_ab_probe.py`：完整模型显式/cuDNN 梯度 A/B；
- `scripts/grad_ab_direction_probe.py`：多 batch 差分方向和随机分量；
- `scripts/optimizer_update_ab_probe.py`：从真实 Adam 状态比较参数更新误差；
- `scripts/hybrid_update_ab_probe.py`：比较不同 cuDNN 层段的更新空间误差；
- `configs/diag_cudnn_strict_order.yaml`：全层 cuDNN strict-order 诊断；
- `configs/diag_cudnn_last3.yaml`：最后 3 层 cuDNN 稳定折中诊断。

## 12. 文档整合说明

本文整合并取代以下三份调查记录作为当前权威结论：

- `2026-08-29-cudnn-attention-nan-incident.md`：全空 mask 的瞬时 NaN 事故；
- `2026-08-30-cudnn-attention-divergence.md`：全层 cuDNN 缓慢发散、Adam 机制和最后 3 层方案；
- 原 `2026-08-31-cudnn-sdpa-nan-summary.md`：关于非 128 倍长度、升级 cuDNN 和 padding kernel 的待验证假设。

整合完成后，前两份独立文件已删除，历史内容仍可通过 Git 提交记录追溯。后续调查和生产决策以本文为准。
