# 全层 cuDNN 发散的根因：flash 反向 D 项用了 bf16 舍入后的 O

**日期**：2026-09-03

**机器**：8×H100 80GB（ebcloud，hb-openpi 环境：jax 0.5.3、nvidia-cudnn-cu12 9.19.0.56，`check_cuda_stack.py` PASS）

**前置报告**：`docs/2026-08-31-cudnn-sdpa-nan-summary.md`。那份报告证明了全 18 层 cuDNN 的缓慢发散是"cuDNN 与显式路径的 BF16 差分被 Adam 逐坐标放大"，但没有回答差分从哪里来、为什么它不像随机噪声。本文用真实激活回答这两个问题，并给出一个不改内核的缓解手段及其验收结果。

> [!IMPORTANT]
> 结论摘要：
>
> 1. cuDNN fused attention 的反向按 flash-attention 方式算 `dS = P ⊙ (dP − D)`，其中 `D_i = rowsum(dO_i ⊙ O_i)` 用的是**前向存下的、已舍入到 bf16 的 O**。显式路径用同一份舍入后的 dP 算 `D_i = Σ_j P_ij dP_ij`，抵消是自洽的。两者的差是一个每行标量 `Δ_i = dO_i · (bf16(O_i) − O_i)`，进入梯度后方向为 `−Δ_i · Σ_j P_ij K_j`，破坏了 softmax 梯度"行和为零"的不变量。
> 2. 该分量在真实激活上可被逐层预测：cuDNN bf16 的 dQ 误差与预测方向的 cos 在全部行上为 0.36–0.55，在 peaked 行（maxP>0.99）上为 0.66–0.99；显式 bf16 与 cuDNN fp16 的 cos 均 ≈ 0。
> 3. peaked 行的真实 dQ 是 ε 级抵消量，所以那里 cuDNN bf16 的相对误差达 7%–140%（显式路径 0.2%–0.5%）。它们绝对量小，但方向沿该行 argmax 键、且跨 batch 落在同一低维子空间，正是 Adam 会放大的结构性误差。
> 4. 把 cuDNN 内核改为 fp16（多 3 位尾数，配合 2 的幂动态缩放余切）后，peaked 行误差降约 8 倍，全局 dQ 误差 1.76e-3 低于显式 bf16 的 2.33e-3，结构性分量消失（cos ≈ 0）。
> 5. "把 V 减去 sink 键的 V 再送内核"无效，因为 peaked 行大多不在看 BOS sink，而是在看 action/text/image 的其他键；BOS 的 |V| 本来就只有 1–5。

## 1. 方法

两个探针，都用 `configs/diag_cudnn_strict_order.yaml` 的模型与数据、pi05_base 权重、一个真实 batch（B=8，单卡）：

- `scripts/attention_dterm_probe.py`（GPU）：monkeypatch `gemma.Block.__call__` 与 `gemma._cudnn_attention_call`，在一次真实训练步里用 `jax.debug.callback` 抓下 18 层的 q/k/v/mask 和反向余切 dO（缓存到 `dterm_capture.npz`，1.5 GB）。然后在单层上以 fp32 显式 attention 为真值，比较五个变体的 dQ/dK/dV：`explicit_bf16`（生产路径）、`cudnn_bf16`（历史路径）、`cudnn_fp16`（候选）、`cudnn_bf16_shift`/`cudnn_fp16_shift`（V 减 V_sink）。同时按上面的公式算 H1 预测误差，用 `jax.lax.reduce_precision` 模拟 bf16 舍入（`astype` 往返会被 XLA 的 `allow_excess_precision` 删掉，预测会恒为零）。
- `scripts/attention_dterm_cpu_analysis.py`（CPU，float64）：读同一份缓存，对 peaked 行逐行重建精确反向，回答"哪些行 peaked、真实 dQ 有多小、H1 误差占多少"。

## 2. 事实

### 2.1 sink 在哪里

18 层、8 个样本，注意力质量最大的键全部是位置 768 = 文本第一个 token（BOS），占有效注意力质量 8%–59%（第 14–15 层最高）。BOS 键的 |K| 8–13（其他键中位数 17–29），**|V| 只有 0.8–5（其他键中位数 10–58）**，第 0/16/17 层例外（9–17）。

### 2.2 哪些行是 peaked（maxP > 0.99）

peaked 行占有效行 0.1%–7%，且**大多不在看 BOS**：

```text
层   peaked 行   其中看 BOS   其余在看
 0     3978        1270       img 1108, txt 1600
 4      131           0       act 131
 8       64           0       act 48, txt 16
 9       98           0       txt 90, img 8
14     1885        1802       txt 74, img 9
15      323         168       img 107, txt 48
17      561          47       txt 357, img 153（该层非 action 行 dO ≡ 0）
```

### 2.3 peaked 行上的误差解剖（CPU，float64）

每行几何平均。`|dQ_true|` 是精确梯度范数，`errH1` 是 H1 预测误差，`shift` 表示 V 减 V_768：

```text
层  行类别                  行数   ε=1−maxP  |dQ_true|   |errH1|/|dQ_true|  cos(errH1, K_argmax)
 0  看 BOS, P>0.99          1270   1.0e-3    3.8e-06     0.33 (shift: 9e-4)  1.00
 0  看其他键, P>0.99        2708   5.3e-4    1.1e-06     0.64                1.00
 4  看其他键(act), P>0.99    131   8.2e-4    1.3e-06     1.09                1.00
 8  看其他键, P>0.99          64   4.7e-3    2.0e-05     0.80                1.00
11  看其他键, P>0.99          13   4.4e-3    3.1e-05     1.04                1.00
14  看 BOS, P>0.99          1802   6.0e-3    5.6e-08     0.020 (shift: 5e-4) 1.00
14  看其他键, P>0.99          83   2.0e-3    8.7e-07     0.58                1.00
16  看其他键, P>0.99           4   4.7e-3    2.3e-10     1.47                1.00
 *  普通行 (P≤0.9)          3000   ~0.7      0.5–3e-04   1.2e-3 – 2.2e-3     0.80–0.88
```

- peaked 行的真实 dQ 比普通行小两个数量级（ε 级抵消），H1 误差不随 ε 缩小，所以相对误差达 O(1)。
- 看 BOS 的行因为 |V_768| 小、O 本身小，H1 只有 2%；shift 能再压 40 倍但没必要。
- 看其他键的 peaked 行 |O| ≈ |V_argmax| = 13–115，H1 误差 60%–150%，shift（减的是 V_768）对它们完全无效。
- 普通行上 H1 分量约 1.5e-3 相对、方向沿 argmax 键（cos 0.8）：绝对量占主导，但它是结构性的。

### 2.4 GPU 单层实测：误差与 H1 预测的一致性

`dQ rel` 是全部行的相对 L2 误差，`peaked rel` 只算 maxP>0.99 的行，`cos` 是误差向量与 H1 预测的 cos：

```text
层   变体           dQ rel    cos(all)   peaked rel   cos(peaked)   H1 预测 peaked rel
 0   explicit_bf16  2.46e-3   -0.02      2.8e-3       -0.04
     cudnn_bf16     3.58e-3   +0.49      3.5e-1       +0.83         0.41
     cudnn_fp16     1.71e-3   -0.02      3.9e-2       +0.13
 4   explicit_bf16  2.39e-3   +0.02      2.8e-3       -0.04
     cudnn_bf16     4.12e-3   +0.44      5.3e-2       +0.47         0.16
     cudnn_fp16     1.77e-3   +0.03      1.6e-2       +0.67
 8   explicit_bf16  2.88e-3   +0.00      3.5e-3       -0.10
     cudnn_bf16     4.28e-3   +0.51      5.1e-1       +0.96         0.63
     cudnn_fp16     1.78e-3   +0.03      7.2e-2       +0.57
10   explicit_bf16  2.27e-3   -0.01      2.4e-3       +0.56
     cudnn_bf16     4.36e-3   +0.55      5.0e-1       +0.91         0.53
     cudnn_fp16     1.76e-3   -0.01      8.3e-2       +0.31
11   cudnn_bf16     4.70e-3   +0.51      1.21         +0.76         0.86
     cudnn_fp16     1.77e-3   -0.01      1.3e-1       -0.70
14   explicit_bf16  2.22e-3   +0.05      2.3e-3       -0.03
     cudnn_bf16     2.86e-3   +0.12      4.3e-1       +0.63         0.49
     cudnn_fp16     1.72e-3   +0.04      3.5e-2       +0.13
16   cudnn_bf16     2.46e-3   +0.39      1.41         +1.00         2.2
     cudnn_fp16     1.71e-3   +0.01      1.69         -0.92
18 层几何平均：explicit_bf16 2.33e-3 / cudnn_bf16 3.72e-3 / cudnn_fp16 1.76e-3
18 层平均 cos(all)：explicit 0.006 / cudnn_bf16 0.418 / cudnn_fp16 0.006
```

- cuDNN bf16 的误差在**每一层**都与 H1 预测显著同向；显式与 fp16 都不同向。
- H1 预测的 peaked 行误差量级与实测吻合（0.41/0.35、0.63/0.51、0.53/0.50、0.86/1.21、0.49/0.43、2.2/1.41）。cos 不到 1 是因为 cuDNN 前向的 PV 矩阵乘也把 P 舍入到 bf16，其内部 fp32 O 与本文的 fp32 O 差约 2⁻⁹，两者的舍入决策只是部分相关。
- 第二个候选机制 H2（dP 先舍入到 bf16 再减 D）的 cos 为 0 或负值，排除。
- dV 没有 D 项，cuDNN bf16 的 dV 误差（2.5e-3）与显式（1.9e-3）同阶，是对照。

### 2.5 为什么这解释了 8-31 报告的全部现象

- **1.5% 随机乘性噪声无害、cuDNN 差分有害**：H1 分量按行沿 `Σ_j P_ij K_j` 方向、符号随舍入随机，投到参数上落在 `x ⊗ K_argmax` 这类低维子空间，不与 |g| 成比例；sink 机制处于平衡态时这些坐标的真实梯度和二阶矩都很小，Adam 归一化后每步都在那里走满 lr 的随机步，累积到约 1000 步后 sink 结构失稳。乘性噪声在小 |g| 坐标上也小，Adam 看不见。
- **跨 batch delta 两两 cos ±0.25**：在 30 亿维空间里这不是白噪声，正是"同一子空间、随机符号"。
- **只融合前 3 层 ≈ 全 18 层**：sink/peaked 结构在早层形成，早层 q/k 投影的系统性扰动穿过后面 15 层。
- **改 mask、seq_lengths、cuDNN 版本、LR、b1/b2/eps 都没用**：它们都不改变 O 的舍入。

## 3. 缓解手段：fp16 内核 + 动态缩放

`src/openpi/models/gemma.py` 的 `_cudnn_attention_in_dtype`：q/k/v 转 fp16（多 3 位尾数，H1 误差降约 8 倍），custom VJP 里把余切按 2 的幂缩放到 max|dO·s| ∈ [0.25, 0.5) 再送 fp16 反向，fp32 反缩放。真实激活的范围 max|q| ≤ 3.5、|k| ≤ 30、|v| ≤ 84、|dO| 1e-4–1e-2，均在 fp16 范围内。开关：`model.cudnn_attention_dtype: float16`（默认 `bfloat16`，逐位不变）。

单层效果见 2.4：peaked 行误差 0.35 → 0.039（层 0）、0.51 → 0.072（层 8），全局误差低于显式 bf16。

实现有两版：

- **v1（验收 run 用的）**：custom VJP 的反向里用 `jax.vjp` 重新走一遍 `jax.nn.dot_product_attention` 前向再取反向；配合 `nothing_saveable` remat，每层反向比 bf16 路径多一次 attention 前向。实测 8×H100 batch 256 为 1.37 s/step（显式 1.57–1.64，cuDNN bf16 1.05）。
- **v2（当前代码）**：直接调用 jax 0.5.3 私有的 `jax._src.cudnn.fused_attention_stablehlo._dot_product_attention_fwd_rule / _bwd_rule`（就是 `jax.nn.dot_product_attention(implementation="cudnn")` 自己 custom_vjp 里接的那两个函数），把 softmax stats 和输出当残差保存，反向只跑一次 fused backward。mask→bias（`where(mask, 0, -2<<14)`）、BTNH 布局、静态参数与公共路径一致；全 mask 的 query 行在反向里显式置零 dQ，等价于 bf16 路径的 `_stop_gradient_for_fully_masked_queries`。
  `scripts/cudnn_fp16_vjp_equivalence.py` 在真实激活（层 0/4/8/14/17）上比较 v2 与 v1：前向逐位相同，dK/dV 逐位相同，dQ 相对差 5e-6–1.2e-5，与同一内核自身的 run-to-run 差（1.6e-6–1.4e-5，原子累加）同阶；全 mask 行 dQ 为 0，无非有限值。步时见第 4 节末。

## 4. 3000 步 strict-order 验收（`configs/diag_cudnn_fp16.yaml`）：通过

与 `diag_cudnn_strict_order` 同数据、seed、batch 顺序、超参，全 18 层 cuDNN fp16（v1 实现），从 pi05_base step 0 起，8×H100，batch 256。2026-09-03 21:00 启动、22:09 结束，3000 步全程有限。显式基线取自 8-31 报告（"显式基线"列是显式 strict-order run；"last3"列是只融合后 3 层、已判定与显式一致的 run，点更全）。

```text
step    fp16 loss / grad_norm    显式基线           last3 基线         历史 cuDNN bf16
   0    6.4030 /  65.59          6.4023 / 65.63     6.4029 / 65.52     6.4023 / 65.63
 200    0.9355 /   3.23          0.9244 /  2.89                        0.9348 /  3.53
 500    0.5717 /   2.57                             0.5719 /  2.45     0.5719 /  2.55
1000    0.4231 /   2.21          0.4220 /  2.77     0.4240 /  2.38     0.4319 /  3.16
1100    0.4071 /   2.30          0.4077 /  1.79     0.4050 /  2.18     0.4723 /  3.37
1200    0.3759 /   2.36          0.3742 /  2.12     0.3757 /  2.39     0.6600 / 35.49
1600    0.3209 /   2.77                             0.3238 /  2.65
2000    0.2792 /   1.93                             0.2793 /  1.83     3.7753 / 150.5
2500    0.2464 /   1.70          0.2457 /  1.33     0.2463 /  2.01     4.7055 / 759.2
2900    0.2427 /   1.59                             0.2424 /  1.48     6.2039 / 603.5
```

- 全部 30 个 log 点有限，grad_norm 最大值 4.75（step 100），之后 1.4–2.8，没有一个点超过基线同点的 1.3 倍（验收上限 3 倍）。
- step 1000–1500 历史分叉点处 loss 与显式基线差 <0.5%，与 last3 基线差 <0.3%；step 2500/2900 的 loss 0.2464/0.2427 对基线 0.2457–0.2463/0.2424。历史 cuDNN bf16 在同一点是 4.7/6.2。
- 3000 步全程 param_norm 1802.39 → 1803.19，与 last3 基线的 1803.19 一致。
- step 1000 的 checkpoint 保存在远端 `checkpoints/diag_cudnn_fp16/diag_cudnn_fp16/1000`，可作 `attention_precision_update_probe.py` 的锚点。

结论：把 cuDNN 内核的计算精度从 bf16 换成 fp16 就消除了全层 cuDNN 的发散，验证了第 2 节的机制。

### 4.1 步时

8×H100，batch 256，`[TIMING]` 行的 total（不含 log 的步）以及相邻 100 步 log 的墙钟：

```text
路径                                      s/step       来源
显式 attention                             1.57–1.64    8-31 报告
cuDNN bf16（发散）                          1.05         8-31 报告（mask 修复时的短跑）
cuDNN fp16 v1（反向重算前向）                1.34–1.37    本次 3000 步验收 run
cuDNN fp16 v2（直接调 fwd/bwd 规则）         1.31–1.35    `diag_cudnn_fp16_speed` 400 步（loss 逐点与 v1 一致：step 300 0.7243 对 0.7245）
cuDNN bf16，今日同 config 复测              1.32–1.35    `diag_cudnn_bf16_speed` 400 步（step 300 loss 0.7250）
显式 attention，今日同 config 复测          1.60–1.63    `diag_explicit_speed` 400 步（step 300 loss 0.7244）
```

去掉反向重算只省了约 2%。单卡微基准（`scripts/cudnn_attention_microbench.py`，层 0 真实激活平铺到每卡 batch 32，T=S=1018，8 q 头 / 1 kv 头，含 `nothing_saveable` remat）说明原因：

```text
变体                fwd      fwd+bwd(remat)   ×18 层
cudnn_bf16         1.35 ms   5.24 ms          0.094 s/step
cudnn_fp16_plain   1.53 ms   5.43 ms          0.098 s/step   （fp16 内核、jax 自带 VJP、无缩放）
cudnn_fp16_v2      1.54 ms   5.53 ms          0.099 s/step
explicit_bf16      4.46 ms   8.49 ms          0.153 s/step
```

18 层 attention 的前向加反向合计只占每步约 0.1 s；fp16 内核与 bf16 内核同速，动态缩放和直接调规则的开销可忽略；显式路径也只多 0.06 s。所以 attention 内核本身解释不了 1.33 s 对 1.05 s 的差距，也解释不了显式路径 1.6 s 对 cuDNN 的差距——差距来自整图层面（XLA 在显式路径下的 fusion/remat 决策、显存压力），要用同 config 的 bf16 复测和整步 profile 来定位，见上表最后一行。

### 4.2 更新空间探针：单步指标不区分好坏

`scripts/attention_precision_update_probe.py diag_cudnn_fp16 diag_cudnn_fp16 1000`，锚点是 fp16 验收 run 的 step-1000 train state（参数 + Adam 矩），batch 32，同一真实 batch 和 rng，以 fp32 显式 attention 为真值：

```text
batch 0            原始梯度 vs fp32：rel / sign_flips     Adam 更新 vs fp32：rel / cos / sign_flips
explicit_bf16      3.37e-2 / 1.53e-2                      3.13e-2 / 0.99951 / 7.47e-3
cudnn_bf16         3.18e-2 / 1.43e-2                      2.93e-2 / 0.99957 / 6.97e-3
cudnn_fp16         3.01e-2 / 1.38e-2                      2.81e-2 / 0.99961 / 6.73e-3
```

会发散的 cudnn_bf16 与两条稳定路径在这个指标上不可区分（甚至略好于 explicit_bf16）。这与第 2 节一致：D 项误差在范数上只占 ~1e-3，它的危害来自方向的结构性（同一低维子空间、跨 step 累积），单步的 rel/cos/sign_flips 看不见。判定 fused attention 后端好坏要看 2.4 节的 cos(err, H1) 与 peaked 行误差，最终靠 3000 步 strict-order run；不要再用单步梯度/更新距离做验收指标。

## 5. 备选方案（fp16 已通过验收，仅作记录）

1. **前 k 层显式 + 其余 fp16 cuDNN**：8-31 报告的分段实验表明前 3 层贡献了几乎全部更新差异。
2. **解析修正**：Δ_i 可用一次 fp16 前向估计（`dO·(O_bf16 − O_fp16)`），dQ 修正需一次 V:=K 的前向，dK 修正需一次 dO:=Δ⊙Q 的反向，总成本抵消吞吐收益，不推荐。
3. 不要再试：减 V_sink 的 shift（2.3 节）、改 mask、改 Adam 参数。

## 6. 相关脚本

- `scripts/attention_dterm_probe.py`：抓真实激活 + 单层五变体误差 + H1/H2 预测；
- `scripts/attention_dterm_cpu_analysis.py`：peaked 行的 float64 解剖；
- `scripts/cudnn_fp16_vjp_equivalence.py`：v2（直接调规则）与 v1（反向重算）fp16 VJP 在真实激活上的等价性检查；
- `scripts/attention_precision_update_probe.py [config] [exp_name] [step]`：真实 Adam 状态上的更新空间对比，锚点用 fp16 验收 run 的 step-1000 checkpoint。
