# 启用 flash attention 提速：全层训练发散与分层稳定方案

**日期**：2026-08-30

**目标**：用 cuDNN fused attention（`use_cudnn_attention=true`）替代显式 attention，缩短训练时间。

**机器**：8×H100 80GB（`cs-76d36-a14d7-server`）

**状态**：**根因机制已定位，并得到一个通过 3000 步验收的保守修复。** 全 18 层 cuDNN 的约 1.5% 原始梯度差异具有不同于人造乘性噪声的坐标结构，经过 Adam 预条件后变成约 3.35% 的参数更新差异，并使约 0.79% 的更新坐标翻转符号；误差在早层产生时还会穿过后续网络放大。仅让最后 3/18 层使用 cuDNN，把更新差异降到 0.847%、符号翻转降到 0.193%，并从 step 0 稳定跑满 3000 步。该方案解决了发散，但实测步时约 `1.5 s/step`，只保留了小部分吞吐收益，尚未达到全层 cuDNN 的速度目标。

## 为什么值得继续追

同机同任务的实测步时：

| 实现 | 步时 | 来源 |
|---|---|---|
| 显式 attention | `1.645 s/step` | 0828 run，98714 s / 60000 步，含日志与 checkpoint 摊销 |
| cuDNN + stop-gradient mask | 约 `1.05 s/step` | 0829 run 的 `[TIMING]` 普通步 |
| cuDNN + dummy-key mask | 约 `1.38 s/step` | 2026-08-30 诊断 run |

按 60000 步计算，`1.645 -> 1.05` 约省 **9.9 小时**。

> [!IMPORTANT]
> `1.05 s/step` 取自一个**已经在发散**的 run。数值失效不改变 kernel 的执行成本，所以该数字作为吞吐参考是可用的；但正式验收时必须在一个不发散的 cuDNN run 上重测。

## 失效现象

`use_cudnn_attention=true` 时，训练在前约 1000 步与已验证基线逐点吻合，随后偏离并单调发散，最终变为 NaN。

```text
step    0828 显式（基线）      0829 cuDNN 9.14        cuDNN 9.19
 200    0.9244 /  2.89        0.9333 /  3.58         0.9348 /  3.53
1000    0.4220 /  2.77        0.4306 /  2.50         0.4319 /  3.16
1100    0.4077 /  1.79        0.4784 /  3.16         0.4723 /  3.37
1200    0.3742 /  2.12        0.6484 / 31.40         0.6600 / 35.49
2500    0.2457 /  1.33        5.7692 / 675.08        4.7055 / 759.18
5900    0.1763 /  1.36        nan                    （已停止）
```

**因果起点是 step 1000–1200，不是 NaN 出现的 step 5900。** 后者只是累积发散溢出的位置。这个区别把复现成本从 27 小时降到约 35 分钟，是本次调查最有操作价值的一条。

### 与 2026-08-29 事故是两种不同的失效

| | 2026-08-29（H200） | 2026-08-30（H100） |
|---|---|---|
| 现象 | **瞬时**：step 16300 正常 → 16400 全 NaN | **缓慢**：贴合 1000 步后发散 4700 步 |
| 根因 | 全空 mask 行的 Q 梯度 NaN | 数值扰动 + 配方裕度不足 |
| 修复 | `_stop_gradient_for_fully_masked_queries` | 未解决 |

全空 mask 修复是有效的，它消除了第一种模式。当前问题带着该修复仍然发生。

## 责任归属：单变量 A/B

同机、同数据集、同超参、同 seed，唯一变量是 `use_cudnn_attention`：

| | `cve5hiyt`（0828） | `orbnh1b7`（0829） |
|---|---|---|
| commit | `d0aec27` | `51d9af3` |
| `use_cudnn_attention` | 字段不存在（显式路径） | `true` |
| 结果 | step 59900 **loss 0.0589** | step 5900 **NaN** |

`git diff d0aec27 HEAD -- src/ scripts/` 共 72 行增删，全部属于该开关本身。排除 LR、数据、配置。

## 结论：cuDNN 内核是正确的

### 完整模型梯度 A/B

同一份 pi05_base 权重、同一个真实 batch、同一个 rng，唯一变量是开关（`scripts/grad_ab_probe.py`，cuDNN 9.19，batch 32，FSDP 8 卡）：

```text
batch 0   loss 显式=5.859448  cuDNN=5.851340   |grad| 104.31 vs 103.65   ratio=0.9937
batch 1                                        |grad|  67.19 vs  67.45   ratio=1.0038
batch 2                                        |grad|  70.71 vs  70.61   ratio=0.9986

相对梯度差：1.75e-02 / 1.66e-02 / 2.12e-02
按模块：PaliGemma 1.87e-02  action_in_proj 1.66e-02
        time_mlp_in 1.46e-02  time_mlp_out 1.20e-02  action_out_proj 5.87e-03
```

**相对差约 2%，梯度范数比值 0.994–1.004，没有任何模块偏离整体水平。** 这是 bf16 误差经 18 层 Gemma 反向累积的正常量级——单层测量中 cuDNN 对 fp32 参考解的相对误差是 `5.4e-03`、显式 bf16 路径是 `3.2e-03`，两者同阶。cuDNN 给出的不是更错的答案，只是另一个同样合法的 bf16 近似。

### 扰动的噪声/确定性分解

在同一 batch 上重复调用 cuDNN 反向，把差异拆开：

```text
batch    总差异(cuDNN vs 显式)   随机(cuDNN vs cuDNN)   确定性分量
  0          1.6993e-02             9.7433e-03          1.3922e-02
  1          1.7036e-02             8.7706e-03          1.4605e-02
  2          2.2268e-02             9.1202e-03          2.0315e-02
```

随机部分约 `0.9%`，来自 cuDNN 反向的 atomic 累加（同一输入两次反向 bitwise 不同）。确定性部分约 `1.5%`，是主要成分。

注意「确定性」在这里的含义：**给定 batch 可复现**，不等于**跨 batch 方向一致**。换一个 batch，舍入差异的方向随之改变；梯度范数比值为 1.00，也看不出系统性缩放。后续多 batch 实验已直接否定固定方向假设，见[跨 batch 固定方向假设不成立](#跨-batch-固定方向假设不成立)。

## 已排除的候选

全部在本机、真实训练形状下实测：`T=1018`（768 图像 + 200 prompt + 50 action）、`num_heads=8`、`num_kv_heads=1`、`head_dim=256`、bf16、含全空行的真实 block mask。参考解为同一显式路径在 fp32 下的结果。

- **cuDNN 版本**：9.14 与 9.19 的发散曲线逐点重合（step 1200 分别为 31.40 与 35.49）。
- **「静态序列长度非 128 倍数导致 SDPA backward 错误」**（cuDNN 9.15.1 release note）：
  ```text
  T=1018 (%128=122)   dQ rel-L2: 显式 3.23e-03   cuDNN 3.21e-03   NaN=0
  T=1024 (%128=0)     dQ rel-L2: 显式 3.23e-03   cuDNN 3.21e-03   NaN=0
  ```
  两者无差异，均无 NaN。**该假设在本机本形状上不成立。**
- **库打包错配**：环境统一为单源 pip 栈后（见下），照样发散。
- **logit 幅度**：std 从 1 扫到 80，cuDNN 最差 `7.2e-03`、显式 `3.5e-03`，同量级。
- **head_dim=256**（Hopper 上 cuDNN ≥9.5 才放开的新路径）：与 head_dim=128 表现一致。
- **cotangent 稠密度**：只在 action token 上非零 与 在全部有效 token 上稠密，结果相同。
- **dBias 反向路径**：`should_export_dbias` 要求 `bias.shape[0]==1`，实际 batch 为 256，不进入该路径。
- **分片**：`activation_sharding_constraint` 使用 `PartitionSpec(DATA_AXIS)`，只按 batch 分片，attention 拿到完整序列。
- **两版 mask 修复**：c1b70b2 给全空行开 dummy key，51d9af3 保留原 mask 只对 q 做 stop_gradient。在长/短两种 padding 密度下，两者在有效行上给出**逐位相同**的 dQ/dK/dV。且事故记录显示 c1b70b2 版本同样发散：从 step 15000 的 loss `0.145` 涨到 step 18600 的 `0.7085`，逐 step 保护在那里报告最近 100 步有 8 次更新非有限。**两版修复都不是原因，也都不是解药。**
- **降低 `peak_lr`**：`2.5e-5 -> 1.5e-5` 只把发散起点从 step 约 1150 推迟到约 1450：
  ```text
  step   cuDNN @1.5e-5
  1200   0.3954 /  2.95
  1400   0.4089 /  2.84
  1500   0.5155 /  7.17
  1700   0.9654 / 26.85
  1900   1.1773 / 88.68
  ```
  **LR 是速率参数，不是开关。**
- **提高 `optimizer.b2`**：`0.95 -> 0.99` 未延后发散起点。2026-08-30 在统一后的 cuDNN 9.19 单源环境中，从 step 0 跑满 3000 个更新；8×H100、batch 256，W&B、周期保存和最终保存均关闭。启动前 `scripts/check_cuda_stack.py` 判定 PASS。所有已输出的指标均为有限值，但曲线仍在 step 1100–1200 分叉：
  ```text
  step    loss       grad_norm    param_norm
   200    0.9082        3.1334     1802.3866
  1000    0.4321        3.7520     1802.5017
  1100    0.5189        4.3141     1802.5402
  1200    0.9354       44.6635     1802.5638
  1300    1.6713       52.6050     1802.5797
  1500    2.2757       30.3204     1802.6188
  2000    3.7292      193.8313     1802.7924
  2500    4.7271      422.8436     1803.0002
  2700    6.4255     1521.7623     1803.0616
  2900    4.5535      819.7691     1803.1382
  ```
  训练完整执行了 3000 个更新，最后一个正式指标窗口是 step 2900。step 1200 的显式基线为 `loss=0.3742 / grad_norm=2.12`，因此新 run 已超过 3× 梯度验收上限约 7 倍。step 2500 的 loss `4.7271` 又几乎复现了 `b2=0.95` cuDNN run 的 `4.7055`；较高 `b2` 只降低了该点的梯度峰值（`422.84` vs `759.18`），没有改变发散轨迹。普通非日志 step 的 dispatch 约 `1.29–1.39 s`，无数据断供或系统错误。实验结束后 8 张 GPU 均归零，实验目录为空，没有污染 checkpoint。**提高二阶矩窗口不是解法。**
- **显式 attention + 1.5% 逐 step 独立随机梯度扰动**：没有发散。2026-08-31 在训练配置中加入默认关闭的 `gradient_noise_scale` 诊断开关；显式梯度为 `g`，先生成 `r = g * N(0, I)`，再令 `delta = 0.015 * ||g|| / ||r|| * r`，优化器实际接收 `g + delta`。模型 RNG 保持原样，扰动 RNG 独立并按 step 可复现，因此每一步都精确满足 `||delta|| / ||g|| = 0.015`，同时扰动能量按原梯度能量分布，而不是按参数量集中到最大张量。

  8×H100、batch 256、显式 attention、原 `b2=0.95`，从 step 0 跑满 3000 个更新；W&B、周期保存和最终保存均关闭。正式运行前，CPU helper/配置回归 `58 passed`，生产 batch 的 2-step GPU 校准也连续报告 `gradient_noise_rel_norm=0.0150`。正式曲线：
  ```text
  step    loss       applied grad_norm    clean grad_norm    noise rel-L2
   100    2.8529          3.9906              3.9903           0.0150
   200    0.9262          4.0429              4.0433           0.0150
   500    0.5704          2.0811              2.0806           0.0150
   900    0.4335          2.8625              2.8631           0.0150
  2200    0.2698          1.7088              1.7083           0.0150
  2500    0.2476          1.8600              1.8613           0.0150
  2700    0.2387          1.5861              1.5854           0.0150
  2900    0.2411          1.6389              1.6389           0.0150
  ```
  训练循环耗时 `1:22:45`，另有首次数据冷启动约 6.5 分钟；普通 step 的 dispatch 约 `1.57–1.64 s`。最后一个正式指标窗口是 step 2900。step 2500 的 loss `0.2476` 与历史显式基线 `0.2457` 基本重合，而同期 cuDNN 9.19 已经是 `4.7055 / 759.18`。实验结束后 8 张 GPU 均归零，目录为空，没有污染 checkpoint。

  **结论**：配方并非对任意 1.5% 梯度扰动都没有裕度。该结果排除的是「逐 step 独立、零均值、按梯度能量加权」这一类随机扰动；它尚不能排除真实 cuDNN 差分具有跨 batch 的固定投影、时间相关性或不同的参数内结构。

### 两个 mask 变体：等价，但一个明显更快

`_stop_gradient_for_fully_masked_queries`（51d9af3）与 `_make_cudnn_attention_mask_safe`（c1b70b2）数值等价，但 dummy-key 版本要在**每一层**内重建整个 `(B, 1, T, S)` 掩码，batch 256 下实测 `1.38 s/step`，stop-gradient 版本约 `1.05 s/step`。H200 文档记录的 `1.91 -> 2.31 s/step` 回退是同一件事。

因此代码保留 stop-gradient 版本。51d9af3 的标题「Preserve the **fast** cuDNN mask」描述的正是这个性能取舍，它不是缺陷来源。

## 2026-08-31 后续调查：根因机制与稳定折中

### 跨 batch 固定方向假设不成立

`scripts/grad_ab_direction_probe.py` 对 8 个真实 batch 分别计算显式梯度和 3 次 cuDNN 反向的均值，再比较 `delta_b = g_cudnn,b - g_explicit,b`：

```text
跨 batch delta 两两 cosine：-0.004043 ± 0.126301，范围 [-0.247516, +0.226915]，n=28
||delta|| / ||g_explicit||：  1.556024e-02 ± 2.76e-03
cuDNN 同 batch 随机分量：    8.752676e-03
cos(delta, gradient)：       -0.00113 ± 0.225
cos(delta, parameter)：      +0.000016
```

差分没有稳定的跨 batch 方向，也不沿当前梯度或参数方向累积。因此没有继续做「显式 attention + 固定方向人造噪声」；该实验不会复现真实差分结构。

### batch 到达顺序不是原因

`strict_batch_order=true` 的全 18 层 cuDNN run 仍在原位置分叉：

```text
step 1000   loss=0.4304   grad_norm=2.3201
step 1100   loss=0.4593   grad_norm=2.1643
step 1200   loss=0.6034   grad_norm=15.4116
```

显式基线 step 1200 为约 `0.3742 / 2.12`。固定 sampler 输出顺序没有改变失效，因此 DataLoader worker 完成时序不是根因。后续诊断均保留 `strict_batch_order=true`，确保曲线可逐点比较。

### 关键机制：Adam 放大了坐标级差异

`scripts/optimizer_update_ab_probe.py` 从上述严格顺序 run 的 step 1000 checkpoint 恢复真实 Adam 状态，对同一真实 batch、同一 rng 计算实际优化器更新。原始梯度 L2 差异仍只有约 1.5%，但进入 Adam 后：

```text
                         相对更新差异     更新 cosine     符号翻转坐标
真实全层 cuDNN 差分          3.3521%         0.999440        0.7885%
旧的 1.5% 人造乘性噪声       0.7313%         --              0.1459%
```

真实差分造成的 Adam 更新差异是稳定人造噪声的 **4.6 倍**，符号翻转比例是 **5.4 倍**。主导部分在 PaliGemma（更新差异 3.353%、符号翻转 0.789%）。这解释了为何「相同梯度相对 L2」的人造噪声稳定，而 cuDNN 会发散：旧噪声按 `|g|` 分配能量，几乎不扰动小梯度/小二阶矩坐标；真实舍入差分具有不同的参数内结构，会被 Adam 的逐坐标归一化放大。

把 Adam `eps` 从 `1e-8` 提高到 `1e-6`/`1e-5`，真实更新相对差异只从 3.35% 降到 3.22%/3.08%，符号翻转仍约 0.79%，不是有效修复。

### 优化器时间平滑只能延迟，不能修复

将 `b1` 从 `0.9` 提高到 `0.99`（`b2` 保持 `0.95`）后，全层 cuDNN 在 step 1200 暂时健康，但随后同样发散：

```text
step 1200   loss=0.4027   grad_norm=2.1161
step 1400   loss=0.4950   grad_norm=6.2470
step 1500   loss=0.7486   grad_norm=17.0
step 2000   loss=2.749    grad_norm=67
step 2900   loss=8.844    grad_norm=562
```

它只把分叉推迟约 300 步，和降低 LR、提高 `b2` 的结果一致：改变累积速率，不改变误差来源。

### 层位置比层数更重要

为 `Pi0Config` 增加 `cudnn_attention_layer_start` 和 `cudnn_attention_num_layers`，可只在连续的 Gemma 层段使用 cuDNN。其余层仍走显式 attention；默认值保持历史上的全层行为。CPU 模型回归 `8 passed`。

在 step 1000 的真实 Adam 状态上做更新空间 A/B：

```text
cuDNN 层段       相对更新差异     更新 cosine     符号翻转坐标
全 18 层            3.3118%         0.999451        0.7769%
最前 6 层           3.2061%         0.999486        0.7548%
中间 6 层           2.2169%         0.999754        0.5141%
最后 6 层           1.1633%         0.999932        0.2720%
最前 3 层           3.3016%         0.999455        0.7720%
中间 3 层           1.9894%         0.999802        0.4634%
最后 3 层           0.8470%         0.999964        0.1926%
稳定人造噪声参考     0.7313%         --              0.1459%
```

只融合最前 3 层几乎等同全 18 层，说明早层的微小扰动会经后续 15 层反向/前向传播放大；不能按融合层数线性估计风险。最后 3 层最接近已经跑稳的人造噪声参考，因此选作首个训练候选。

### 最后 3 层方案跑满 3000 步

配置：`configs/diag_cudnn_last3.yaml`，8×H100、batch 256、`strict_batch_order=true`、原始 Adam（`b1=0.9, b2=0.95`）、从 pi05_base step 0 开始、W&B 和 checkpoint 保存关闭。层索引 15–17 使用 cuDNN，其余 15 层显式。正式曲线：

```text
step    loss      grad_norm    param_norm
   0    6.4029      65.5204     1802.3865
 100    2.8439       5.1838     1802.3851
 500    0.5719       2.4526     1802.4014
 900    0.4349       2.5621     1802.4742
1000    0.4240       2.3789     1802.5061
1100    0.4050       2.1847     1802.5416
1200    0.3757       2.3869     1802.5775
1400    0.3390       1.8067     1802.6504
1600    0.3238       2.6505     1802.7225
2000    0.2793       1.8254     1802.8666
2500    0.2463       2.0116     1803.0460
2900    0.2424       1.4769     1803.1906
```

训练完整执行 3000 个更新；最后一个正式指标窗口是 step 2900。原全层 cuDNN 在 step 1200 已到 `0.6034 / 15.41`，b1=0.99 延迟方案在 step 2000 已到 `2.749 / 67`；本方案的曲线与显式稳定基线一致，满足 3000 步验收。

训练循环耗时 `1:18:02`，普通步约 `1.5 s/step`；另有 404 秒的 64-worker 首批冷启动。相对本任务显式路径约 `1.57–1.64 s/step` 只有约 4–9% 加速，明显低于全层 cuDNN 的收益。**因此这是数值稳定的保守折中，不是最终吞吐目标。** 按用户指示，本轮到此停止，不继续运行最后 6 层或更大范围的训练。

## 2026-08-31 调查计划原稿（结果见上）

以下保留当时的计划作为调查轨迹。第 1 项已执行并否定固定方向；因此第 2 项不再能复现真实差分，未执行。实际随后转向 Adam 更新空间和分层位置分析，并得到上面的最后 3 层方案。

### 1. 测量真实 cuDNN 差分的跨 batch 方向一致性

扩展 `scripts/grad_ab_probe.py`，对多个真实 batch 计算 `delta_b = g_cudnn,b - g_explicit,b`，并报告：

- 不同 batch 的 `delta_b` 两两 cosine；
- `delta_b` 在显式梯度、参数向量和前一 batch 差分上的投影；
- 上述量按 PaliGemma、action projection 和 time MLP 分模块的结果；
- 同一 batch 多次 cuDNN 反向取平均后再计算，尽量剥离约 0.9% 的 atomic 随机分量。

若跨 batch cosine 或固定投影显著非零，说明真实差分存在可累积方向；若接近零，则需要检查人造噪声与 cuDNN 差分在参数内分布、尾部和时间相关性上的差别。

### 2. 显式 attention + 时间相关的固定方向扰动

复用 `gradient_noise_scale=0.015`，但让随机方向不再按 step 变化，每一步只根据当前 `||g||` 重新缩放同一参数坐标方向。若这一版本发散而逐 step 独立版本不发散，时间相关性就是关键杠杆；若仍稳定，则必须更直接地复现真实 cuDNN 差分结构。

### 3. `optimizer.weight_decay` `1e-10 -> 1e-4` 量级

当前几乎为零。加大给慢漂移一个回复力。改动比前两项大，收敛行为也会变，排在后面。

### 4. 若以上都不成立：绕开 attention 数值的提速路径

- `max_token_len` `200 -> 168`（全数据扫描得到的 tokenized state 最大长度是 165，留 3 token 余量）：减少约 3% 的序列长度，不触碰数值；
- 重新采集 XProf trace，确认剩余最大的 SigLIP/Gemma GEMM 与 fusion；
- 提高 `save_interval`，减少 checkpoint 对下一步的数据等待尖峰。

这三项合计的收益远小于 flash attention 的 36%，但零风险。

### 不要重复的弯路

- **不要再查 cuDNN 版本或库打包。** 9.14 与 9.19 表现一致，单源栈已验证。
- **不要再改 mask 处理。** 两个变体逐位等价，且都发散。
- **不要只调 `peak_lr`。** 按累积机制，任何取值大概率都只是推迟。
- **不要再单独提高 `optimizer.b2`。** `0.99` 从 step 0 跑满 3000 个更新，仍在 step 1100–1200 分叉，并在 step 2700 出现 `grad_norm=1521.76`。
- **不要重复逐 step 独立的 1.5% 随机梯度扰动。** 显式 attention 已跑满 3000 步，step 2500 loss `0.2476`，没有复现发散；下一步应测试真实差分的方向结构或时间相关扰动。
- **不要跨 run 比较 `grad_norm`。** 本仓库默认 `strict_batch_order: false`，各 run 的 batch 序列不同，早期 `grad_norm` 差异来自数据顺序而非实现差异。曾据此误判 cuDNN 存在系统性梯度放大，被受控 A/B（范数比 1.00）否定。

## 验收要求

> [!WARNING]
> 本次曾在 step 1300 仍健康时判定「降 LR 已解决」，随后 step 1450 开始发散。**判据必须是跑满预定步数并看完整条曲线，不能是「越过某一步还正常」。**

启用 `use_cudnn_attention` 的验收：

1. 从 **step 0** 起跑满 **3000 步**（最后 3 层方案实测训练循环约 78 分钟，另有约 6.7 分钟首次数据冷启动）；
2. 与同配置显式 attention 的基线曲线逐点对比 step 200/500/1000/1500/2000/3000 的 `loss`；
3. 全程 `grad_norm` 不得超过基线同点的 3 倍；
4. 通过后才能进入长跑，且长跑期间仍需按日志窗口盯 `loss` 与 `grad_norm`。

40 步吞吐短跑、从已收敛 checkpoint 出发的短跑、以及只验证「梯度有限」的 smoke test，都**不是**有效的正确性验收。2026-08-29 的修复通过了全部这三类检查，仍然发散。

建议增加发散护栏：`grad_norm` 已在日志 step 计算，当它连续多个日志窗口超过滚动中位数的固定倍数时中止训练。本次若有该护栏，会在 step 1300 停下而不是跑满 60000 步。

## 环境

### dispatcher 与 engine 版本错配（已修复）

事故期间 `cuda_versions.cudnn_get_version()` 返回 `91400`，但进程实际加载的是 **9.10.2 的 dispatcher stub 驱动 9.14.0 的引擎库**：

```text
site-packages/nvidia/cudnn/lib/libcudnn.so.9          <- 125 KB，dispatcher stub，9.10.2 wheel
$CONDA_PREFIX/lib/libcudnn_graph.so.9.14.0            <- 真正执行的引擎库
$CONDA_PREFIX/lib/libcudnn_engines_precompiled.so.9.14.0
$CONDA_PREFIX/lib/libcudnn_ops.so.9.14.0
```

`cudnnGetVersion()` 的返回值来自引擎库，因此它**无法**发现这种错配。上一份文档要求「必须确认输出 91400」的验证方法是不充分的。正确做法是枚举进程实际映射的库：

```bash
python scripts/check_cuda_stack.py
```

它遍历 `/proc/self/maps`，在 `libcudnn*` 来自多个目录、或文件名版本与运行时报告不一致时判 FAIL，然后在真实训练形状上跑一次前向+反向并与 fp32 参考解对比。**在混装环境下该脚本判 FAIL，而三项数值检查全部通过**——这正是只看数值 smoke test 会漏掉的部分。

### 统一到 pip cu128 栈（已完成）

目标：CUDA Toolkit 12.8 + pip cu128 PyTorch + pip cuDNN 9.19，移除 conda cuDNN。

依赖解析上的关键事实：

- `torch 2.10.0+cu128` 把 cuDNN 钉死在 `nvidia-cudnn-cu12==9.10.2.21`（`==`，非 `>=`）；
- `torch 2.11.0+cu128` 钉 `nvidia-cudnn-cu12==9.19.0.56` 和 `cuda-toolkit==12.8.1`；
- PyPI 默认的 `torch 2.11.0` 是 **CUDA 13 构建**，钉 `nvidia-cudnn-cu13`，必须走 cu128 索引；
- `jax-cuda12-plugin` 只要求 `nvidia-cudnn-cu12>=9.1,<10.0`，不构成约束。

**torch 版本决定了训练进程加载哪个 cuDNN**，尽管 torch 在本仓库只用于 DataLoader。这是 `pyproject.toml` 把 torch 下界提到 `>=2.11` 的原因。

另外两处容易漏掉的：

- `torchcodec` 的 `+cu128` 构建只存在于 pytorch 索引，默认 PyPI 上的同版本号没有 `+cu128` 后缀，需在 `[tool.uv.sources]` 中路由；
- 本地 fork `lerobot-xense`（editable 安装）在自己的 `pyproject.toml` 中声明 `torch<2.11.0`、`torchcodec<0.11.0`、`torchvision<0.26.0`，**必须同步放宽**，否则解析失败。不能改用 PyPI 的 lerobot 0.6——那会丢掉 fork 中的 xense 相机与 flexiv 机器人代码。

启动命令必须**去掉** `LD_LIBRARY_PATH="$CONDA_PREFIX/lib"`。移除 conda cuDNN 后该目录仍保留 conda-forge 的 `libcublas`/`libcudart`/`libnccl` 等，继续导出会让它们遮蔽 pip cu128 的同名库，等于换一个库重演同一类错配。

## 处置

- 0829 run 在 step 5900 之后的所有 checkpoint（10000–50000）均为 NaN，无保留价值；
- `cve5hiyt`（0828，显式 attention）仍是已跑满 60000 步的生产结果，loss `0.0589`；
- 最后 3 层 cuDNN 已通过 3000 步短程验收，但尚未跑 60000 步，不能把它等同于完整生产验收；
- 需要绝对稳妥时继续使用 `use_cudnn_attention: false`；需要先获取小幅吞吐收益时，可用 `use_cudnn_attention: true, cudnn_attention_layer_start: 15, cudnn_attention_num_layers: 3` 进入受监控长跑；
- 不要把 `use_cudnn_attention: true` 且 `cudnn_attention_num_layers: null`（全 18 层）用于生产训练。

## 相关脚本

- `scripts/check_cuda_stack.py` —— 验证进程实际加载的 cuDNN 栈是否单源，并在真实训练形状上做前向/反向数值检查；
- `scripts/grad_ab_probe.py` —— 完整模型梯度 A/B，比较显式与 cuDNN 的逐参数梯度，并把差异拆成随机与确定性分量；
- `scripts/grad_ab_direction_probe.py` —— 多真实 batch 的 cuDNN 差分方向、投影和随机分量；
- `scripts/optimizer_update_ab_probe.py` —— 从真实 checkpoint 恢复 Adam 状态，比较梯度差分进入优化器后的更新误差与符号翻转；
- `scripts/hybrid_update_ab_probe.py` —— 比较不同 cuDNN 层段的 Adam 更新空间误差。
