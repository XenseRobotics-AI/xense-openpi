# 未来触觉预测的触觉预训练方案

日期：2026-09-16（第 2 版）。状态：实现建议，未实施。以当前工作区为起点，不依赖任何已有 checkpoint 的结果。
依据：RATG（`/home/li/papers/Representation-Aligned Tactile Grounding for.pdf`）、STAR、N0、
`vtla_future_tactile_posttraining_plan.md`、xense-openpi 代码（HEAD cd7684c，含未提交变更）。

第 2 版改动：删掉原"步骤 1：独立未来触觉预测器"。输入编码器和目标编码器都直接用原始 ImageNet
FastViT-T12，只做 BN 统计校准，不单独预训练。原步骤 2/3/4 顺次改为步骤 1/2/3，主实验（策略 + LTP）
展开写在 §2.3。

## 1. 目标与原则

目标：让 π0.5 + FastViT 触觉的策略真正依赖触觉。手段：用"当前观测 + 动作 → 未来触觉"这个自监督
目标给 action expert 的中间表示施压（RATG 的 Latent Tactile Predictor，LTP）。未来触觉只在训练时
用，推理路径不变。

三条原则：

1. 编码器不单独预训练。原始 FastViT 校准 BN 后直接进策略；触觉表示由策略训练本身塑造。
2. 每一步的验收是因果干预（换触觉看输出变不变），不是 loss 曲线。
3. 一次只改一个变量。

## 2. 模型架构

### 2.1 基础策略（步骤 1 使用，也是部署结构）

```text
head / left_wrist / right_wrist RGB (3×224×224×3)
        │ SigLIP (So400m)                       prompt + 离散 state
        ▼                                              │ embed
   3×256 image tokens ────────────────┬────────────────┘
                                      ▼
                     PaliGemma 2B  ── prefix hidden / KV ──────────────┐
                                                                       │
4 路触觉 (4×224×224×3)                                                  │
        │ FastViT-T12（共享，ImageNet 权重，BN 统计校准后冻结）            │
        │ → 4×1024                                                     │
        │ tactile_proj (1024→1024)                                      │
        ▼                                                              ▼
   4 tactile tokens ──┐                                    action expert（Gemma 300M，
                      ├─ suffix (54 tokens) ─────────────▶  18 层，宽 1024，adaRMS 时间条件）
   50 action tokens ──┘                                              │
   (action_in_proj: 32→1024，输入 A_τ = τ·ε + (1-τ)·A)               ▼
                                                    suffix_out[:, -50:] → action_out_proj → v_t
```

注意力可见性（`make_attn_mask` 按 `key_block <= query_block`，prefix=0、tactile=1、action=2）：

| query \ key | prefix | tactile | action |
|---|---|---|---|
| prefix | ✓ | ✗ | ✗ |
| tactile | ✓ | ✓ | ✗ |
| action | ✓ | ✓ | ✓ |

时间条件 `adarms_cond` 是 `[B,1024]`，广播到全部 54 个 suffix token（非 RTC 路径）。

编码器归一化：FastViT 的 BN 在 Flax 实现里固定 `use_running_average=True`，ImageNet 统计与凝胶图像
不匹配。做法：`resize_with_pad` 改 center crop，在训练集触觉帧上重估全部 BN 统计写回初始化权重，
训练期间冻结统计（`scripts/audit_tactile_bn.py` 已有重估逻辑）。审计结论：BN 重估前 ImageNet 统计把
触觉特征幅度压了约 290×，但没丢信息，接触标签仍可从特征线性读出（约 0.79）。这是"不预训练也能直接用
原始 FastViT"的依据。

### 2.2 目标编码器 E_tac（步骤 2 的监督目标，离线算）

```text
未来触觉帧 T_{t+k,s} (224×224×3, center crop)
        │ E_tac = 原始 FastViT-T12 的冻结拷贝（与 2.1 同一份初始化权重、同一份校准 BN；训练中永不更新）
        ▼
   f ∈ R^1024（全局平均池化输出）
        │ 训练集统计：均值 μ，协方差 → PCA 取前 d=256 维，白化
        ▼
   z_{k,s} ∈ R^256，各维零均值单位方差
```

为什么这样定目标：

- **为什么是一份冻结拷贝而不是策略里那份可训编码器。** 目标若随输入编码器一起训，编码器可以把输出
  塌成常数让 `L_tac=0`（RATG 用同一个编码器，论文未讨论这一点）。冻结拷贝彻底排除塌缩，也让目标在
  整个训练过程中固定，可以离线算成标签库。
- **为什么原始 ImageNet 权重够用。** 目标编码器只需要"未来帧之间的差异在特征里可分辨"，不需要特征
  语义正确。2.1 的审计说明校准 BN 后特征保留了接触信息。若步骤 2 的目标对照（换 16×16 像素场）反而
  更好，说明这个前提不成立，再考虑别的目标编码器。
- **为什么 PCA 白化到 256 维而不是 1024 维逐维标准化。** ImageNet 特征在凝胶图像上有效秩低（幅度压缩
  290× 就是这个现象），逐维标准化会把大量近常数维放大成噪声维，头会花容量去拟合噪声。PCA 白化把方差
  集中到少数方向，零预测基线 loss 恰好 = 1，好解释。1024 维逐维标准化作为变体保留。
- **为什么是未来帧而不是当前帧。** 当前触觉已经在输入里，预测它是复制任务，头可以从触觉 token 直接
  抄，不给动作流施加任何压力。预测未来触觉必须联合当前触觉、当前视觉与即将执行的动作，是一个前向模型。

### 2.3 步骤 2：策略 + LTP（主实验）

#### 2.3.1 整体数据流

```text
                     prefix（同 2.1，VLM 流，18 层）
                        │  ↑ 每层 suffix 都能 attend 到同层 prefix 的 K/V
   4 tactile + 50 action tokens
        │
        ▼
   action expert Block 1 … Block m ──▶ h^(m) ∈ R^{B×54×1024} ──▶ Block m+1 … Block 18 ──▶ final_norm
                                          │                                                   │
                     训练时才有 ──────────┤                                     action_out_proj(suffix_out[:, -50:])
                                          │                                                   │
                                          │                                                v_t → L_flow = ‖v_t − (ε − A)‖²
                                          ▼
                       h_act^(m) = h^(m)[:, 4:54, :]   （只取 50 个动作位置，见 2.3.3）
                                          │
                                          ▼
              LTP：20 个可学习 query ── 单层 cross-attention（K/V = h_act^(m)）── MLP ── Linear → ẑ ∈ R^{B×5×4×256}
                                          │
                                          ▼
                       L_tac = mean_{valid (k,s)} ‖ẑ_{k,s} − z_{k,s}‖²      z 来自 2.2 的标签库，无梯度
                                          │
                       L = L_flow + λ·L_tac，λ=1（前 1k 步线性 warm-up）
```

推理时 LTP 不存在，前向与 2.1 逐位相同。

#### 2.3.2 h^(m) 是什么，从哪里取

action expert 是 `gemma.Module` 里的第 2 个 expert，18 个 `Block` 用 `nn.scan` 堆叠。每个 Block 对
suffix 流做 `x ← x + attn(adaRMS(x))`，再 `x ← x + ffn(adaRMS(x))`。`h^(m)` 定义为第 m 个 Block 的
输出残差流，即第 m+1 个 Block 的 `pre_attention_norm` 之前的值，形状 `[B, 54, 1024]`，前 4 个位置是
触觉 token，后 50 个是动作 token。

取法：给 scan 加一个 per-layer 输出（`nn.scan` 的 `out_axes`），带出 `[18, B, 54, 1024]` 的逐层
suffix 残差流，取第 m−1 个切片。显存 18×B×54×1024 bf16，batch 256 约 0.5 GB。不拆 scan，参数树
布局与现有 checkpoint 保持一致。只在 `compute_loss` 里开这个开关，`sample_actions` 不开。

m 的选择：由步骤 1 的逐层线性探针决定（对每层 `h^(l)` 动作位置拟合 ridge 预测未来触觉场，误差最小
的层）。RATG 在 π0 的 18 层 expert 上最小值落在第 5–9 层。只挂一层；多层同时挂 RATG 报告更差
（38–60 vs 74）。

#### 2.3.3 LTP 头的结构

```text
输入：h_act^(m) ∈ R^{B×50×1024}，key mask = 动作位置的 input_mask（全 True）

query 表：Q_{k,s} = e_k + p_s，e ∈ R^{5×1024}（horizon k∈{10,20,30,40,50}），p ∈ R^{4×1024}（pad s）
          → 20 个 query，形状 [B, 20, 1024]（batch 内广播）

块 1（唯一的一层）：
   kv_norm = RMSNorm(h_act^(m))
   Q = W_q·query, K = W_k·kv_norm, V = W_v·kv_norm       8 头 × 128 维
   u = query + W_o·softmax(QKᵀ/√128)·V                    [B, 20, 1024]
   u = u + MLP(RMSNorm(u))                                 1024 → 4096 (GELU) → 1024
   ẑ = W_out·RMSNorm(u)                                    1024 → 256，reshape 到 [B, 5, 4, 256]

参数量 ≈ 4·1024² + 8·1024² + 1024·256 ≈ 13M（expert 300M 的 4%）
```

每个设计点的理由：

- **为什么 K/V 只取 50 个动作位置，不取 4 个触觉位置。** 残差流的性质是：触觉位置 s 在第 m 层的值
  = 输入触觉 token + 逐层累加的更新，输入触觉 token 的拷贝始终留在里面。如果头能读触觉位置，它会直接
  从这份拷贝预测未来触觉（这是最短路径），梯度只流进 `tactile_proj` 和编码器，动作位置不受任何压力。
  那样得到的是"更好的触觉 token"，等价于 RATG 的编码器预训练（48），不是中间层挂头（74）。
  只读动作位置，预测未来触觉的唯一途径是：动作 token 在第 1..m 层通过 attention 把触觉信息拉进自己
  的残差流并保留下来。这恰恰是我们要的因果链：触觉 → 动作 token 表示 → 动作。
  对照：K/V 取全部 54 个位置，预期 `L_tac` 更低但反事实探针效应更弱。
- **为什么用可学习 query 的 cross-attention，不用平均池化 + MLP。** 50 个动作位置里触觉信息可能集中
  在少数位置（例如接触发生的时刻附近），平均池化会把它冲掉。20 个 query 各自去找对自己的 (k,s) 有用的
  位置；horizon k 的 query 天然应该关注 chunk 中第 k 步附近的动作 token。
- **为什么只有一层。** 头的容量必须小于"从零算出未来触觉"所需的容量，否则头自己把活干了，主干不用变。
  头只做"读出 + 线性对齐"，前向模型本身必须由 expert 第 1..m 层完成。这是"用辅助损失塑造主干表示"和
  "多任务学习"的分界。若发现 `L_tac` 降不下去且逐层探针说 `h^(m)` 里确实有这份信息，再加一层。
- **为什么 query 拆成 e_k + p_s。** 5 个 horizon 和 4 个 pad 共享结构：同一 pad 在不同 horizon 的
  预测应该相关，同一 horizon 的 4 个 pad 也相关。加法分解让参数从 20×1024 降到 9×1024，也让 horizon
  之间能共享"往前看多远"的模式。
- **为什么四路分别监督，不平均。** 左右夹爪、每爪两片 pad 的不对称是抓取状态的核心信息（哪边先碰到、
  哪边滑）。平均掉就把最有用的信号平均掉了。
- **为什么损失是 MSE 而不是 cosine。** 目标已白化，MSE 的零预测基线 = 1，可直接读出"解释了多少方差"；
  cosine 会丢掉幅度，而幅度（接触强弱）正是要的。

#### 2.3.4 梯度走向：哪些参数被 L_tac 更新

```text
L_tac ─▶ LTP 参数
      ─▶ h_act^(m) 动作位置
            ─▶ Block 1..m 的 expert 侧参数（attention、FFN、adaRMS 调制）
            ─▶ 动作位置对触觉位置的 attention ─▶ 触觉 token ─▶ tactile_proj ─▶ FastViT（若可训）
            ─▶ 动作位置对 prefix 的 attention ─▶ Block 1..m 的 VLM 侧 K/V 投影（若 VLM 未冻结）
            ─▶ action_in_proj（噪声动作的嵌入）、time_mlp（adaRMS 条件）
Block m+1..18、action_out_proj：不在 L_tac 的计算图上，只受 L_flow。不需要 stop-gradient。
```

要点：

- **m 之后的层只受 L_flow 是结构性的**，不是靠 stop-gradient。`L_tac` 只依赖 `h^(m)`，而 `h^(m)`
  不依赖第 m+1..18 层。这些层的职责保持纯粹：把已经带触觉的中间表示解码成动作。
- **动作条件是免费的。** flow matching 的训练输入是 `A_τ = τ·ε + (1−τ)·A`。τ 小的样本，动作 token
  几乎是干净的未来动作，`h_act^(m)` 里自然带"即将执行的动作"，预测未来触觉是真正的动作条件前向模型。
  τ 大的样本，动作 token 是噪声，头只能靠动作位置从 prefix + 触觉拉来的信息预测，退化成无动作条件的
  预测。同一条前向、同一个头，两种模式按 τ 自动切换，不需要第二次前向。
  监控：`L_tac` 按 τ 分箱。若使用了动作条件，低 τ 箱的 `L_tac` 应明显低于高 τ 箱；若两箱持平，说明
  动作位置没把动作内容用起来，先查 `action_in_proj` 的梯度。
- **共享参数的梯度比 r_g = ‖∇_θ L_tac‖ / ‖∇_θ L_flow‖**，θ 取 Block 1..m 的 expert 参数。目标区间
  0.1–0.5。低于 0.1 说明辅助损失没在塑造主干（λ 调大或 m 调深），高于 0.5 说明主干被辅助任务带跑
  （λ 调小）。

#### 2.3.5 为什么是这个位置、这个目标：RATG 的证据链

RATG 在 SmolVLA 上的单任务实机成功率（每任务 20 次）：

| 方案 | 成功率 |
|---|---|
| 无触觉 | 18 |
| 触觉当输入 | 41 |
| 编码器预训练后接入 | 48 |
| 头挂 VLM 输出 | 58 |
| 头挂 expert 最后一层 | 62 |
| 头挂 expert 中间层 | 74 |
| 多层同时挂 | 38–60 |
| latent 目标 vs 像素目标 | 80 vs 55 |

读法：

- **触觉当输入（41）→ 中间层挂头（74）**：说明问题不在触觉信息进没进模型，而在 action expert 有没有
  被逼着用它。模仿损失单独不会逼：单任务录制下 RGB + 本体感觉已经能预测动作，触觉是冗余输入，梯度
  最省事的解是忽略它。辅助损失把"必须用触觉"变成硬约束。
- **编码器预训练（48）明显低于挂头（74）**：更好的触觉 token 不等于被使用的触觉 token。这是删掉原
  步骤 1 的直接依据，也是 2.3.3 里头不读触觉位置的依据。
- **VLM 输出（58）< 最后一层（62）< 中间层（74）**：VLM 输出在 expert 上游，expert 仍可忽略；最后一层
  已经特化成输出速度场，再塞一个目标是两个任务抢同一个表示；中间层是"世界状态被组装、还没被解码成
  动作"的地方，在这里施压，下游各层消费的输入就带了触觉。
- **多层同时挂更差**：每层都被拉向同一个目标，层间失去分工。
- **latent（80）> 像素（55）**：像素场里光照、凝胶纹理、标记点的方差远大于接触引起的方差，头和主干会
  花容量拟合无关方差。固定编码器的 latent 把接触相关变化集中到少数方向。

RATG 用的是 PaXini 触觉阵列（taxel），不是光学凝胶；数字只能定性参考，方向排序是可信的。

#### 2.3.6 与现有代码的接口

实现状态（2026-09-17，JAX / pi05）：步骤 2 的 8–11 项与步骤 1 的第 2、3、4 项已实现，入口如下。

| 项 | 位置 |
|---|---|
| 逐层 suffix hidden 开关 | `gemma.Module.__call__(..., return_suffix_hidden=True)` 多返回 `[depth, B, S, D]`；`Block` 多一个静态参数 `collect_hidden`，关闭时编译图不变 |
| `Observation.aux_targets` | `models/model.py`，`from_dict` / `to_dict` / 两个 preprocess 都透传；serving 不设即为 None |
| LTP 头 | `models/tactile_future_head.py`（`TactileFuturePredictor`），挂在 `Pi0TactileFastVit.tactile_future_head` |
| `compute_loss` 覆写 | `models/pi0_tactile_fastvit.py`：返回 `{"flow", "tac", "tac_mask", "tac_by_time"}`；`Pi0._flow_forward` 抽出了共用前向 |
| 模型配置 | `Pi0TactileFastVitConfig.tactile_future_layer`（m，None 关闭）、`_horizons`、`_dim`、`_num_heads`、`_head_dim`、`_mlp_dim`、`_kv`（action / all） |
| 训练循环 | `scripts/train.py`：`flow + λ·tac`，λ 由 `TrainConfig.aux_loss_weight` / `aux_loss_warmup_steps` 线性 warm-up；记录 `loss/flow`、`loss/tac`、`tac/time_bin{0..3}`（按 τ 四分位）、`aux/grad_ratio`（r_g，`log_aux_grad_ratio`，只在 log 步多算一次反向） |
| 参数组 LR | `TrainConfig.param_lr_scales`（路径正则 → 倍率，`optimizer.create_optimizer`），LTP 头用 4.0，编码器 0.1× 用同一机制 |
| 标签库 | `scripts/compute_tactile_future_labels.py` → `episode_offsets.npy`、`feat_tac.npy`、`z_tac.npy`、`pixel_field.npy`、PCA 基、`meta.json`（含 `y_delta_rms`） |
| 查表 transform | `transforms.InjectTactileFutureLabels`（`target` = latent / pixel_delta），由 `LeRobotBiFlexivTactileDataConfig.tactile_future_labels_path` 接入 repack；越界置 invalid，用 LeRobot `index` 校验库与数据集一致 |
| 示例配置 | `configs/_examples/pi05_base_bi_flexiv_bottle_sorting_0917_fastvit_ltp_h100.yaml` |

未实现：RTC 与 LTP 同开（配置直接拒绝）、步骤 1 的 BN 重估写回与三个探针脚本、步骤 3 的多数据集混合。

- `Observation.aux_targets`：可选字段，装 `future_tactile_z [B,5,4,256]` 与 `future_tactile_mask [B,5]`；
  `from_dict`/preprocess 透传；部署时为 None。
- `Pi0TactileFastVit.compute_loss` 覆写：调 `PaliGemma.llm` 时开逐层输出开关，取 `h^(m)` 动作位置送
  LTP；返回 `{"flow": ..., "tac": ...}`，`train.py` 加权求和并逐项记录。
- LTP 参数放在 `tactile_future_head` 命名空间，`missing_regex` 已覆盖 `.*tactile.*`，从基座 checkpoint
  载入时不报缺失。
- 关闭 LTP（λ=0 且不取逐层输出）时，前向逐位等于 2.1。

## 3. 三步计划

### 步骤 1：接入策略、探针、选层（8×H100，10k–20k 步）

- 做：§2.1 的策略，编码器用校准 BN 后的原始 FastViT。1a 编码器冻结；1b 编码器 0.1× LR。对照：四路
  触觉 `image_mask=False` 的无触觉基线，同 seed 同数据同步数。
- 测：
  1. 反事实探针：同 RGB/state/prompt/噪声，换触觉（heavy↔light 配对、全零、时间错位、episode 内
     shuffle、pad 互换），量最终动作 chunk 的变化。
  2. **逐层线性探针**（RATG §3.2）：冻结 1a checkpoint，对 VLM 输出和 expert 18 层每层的动作位置
     hidden 各拟合闭式 ridge 预测 §4.2 的未来场；τ∈{0.25,0.5,0.75,1.0}，有/无触觉输入；附触觉
     shuffle 对照。产出误差随层曲线，最小值所在层 = 步骤 2 的 m 的候选。
  3. **逐层敏感度探针**（§6.1）：同一 checkpoint，batch 内 roll 触觉 / roll VL，量 18 层动作位置
     hidden 与 `v_t` 的 `S_tac`、`S_vl`、`S_x`。产出 `S_tac` 随层曲线。与无触觉基线同图对比。
     选层规则：取线性探针误差最小、且 `S_tac` 已明显上升的最浅层。两条曲线不一致时以线性探针为准，
     `S_tac` 曲线记录下来作为步骤 2 前后的对照基线。
- 意义：给步骤 2 一个同管线基线，并选层。预期这一步的反事实探针效应很弱（RATG 41 vs 18 的差距主要
  来自任务本身），不是终点。
- 可选 1c：前 N 千步以高比例把三路 RGB `image_mask=False` 逼 expert 学触觉接口（N0 Stage 2），
  之后降到 0.2。零模型改动。

### 步骤 2：策略 + LTP（主实验）

- 做：§2.3。变体一次只换一个：
  - 接口对照：同一头挂最后一层、挂 VLM 输出。
  - 读出对照：K/V 取全部 54 位置（含触觉位置）。
  - 目标对照：换 16×16 像素场（§4.2 的 `y_delta`），头的输出维改 768。
  - 多层同时挂（RATG 报告更差，用来确认）。
- 测：与步骤 1 相同的反事实探针；分块留出的分箱准确率；`L_tac` 按 τ 分箱；r_g；闭环实机。
- 过线：换触觉引起的动作系统性变化与真实行为差同量级，且出现在需要触觉的接触相位；分箱准确率高于
  无触觉基线。头预测得好但动作不变，判定未达成。
- 备选叠加：adaRMS 触觉调制 `adarms_cond = time_emb + W_tac(mean_s z_s)`，`W_tac` 零初始化。只在
  步骤 2 显示 hidden 已带触觉分量但动作仍不变时上。

### 步骤 3：多数据集

步骤 2 过线后，扩到本地 9 个同构 bi_flexiv 触觉数据集（约 98.6 小时，需核实），任务均衡采样，
按 session 切分。目的：单任务录制下 RGB 和本体感觉就能预测任务标签，模仿损失不需要触觉；多任务
混合打破这一点。前置工程：多数据集混合、episode 过滤。

## 4. 数据与标签

### 4.1 标签库

`scripts/compute_tactile_future_labels.py`：对每个数据集顺序解码一遍 4 路触觉视频，每帧 center crop
有效成像区域后：

- resize 到 16×16×3 uint8，写 `(num_frames, 4, 16, 16, 3)` memmap（步骤 1 探针目标、步骤 2 像素对照）；
- 过冻结 `E_tac`（§2.2）得 1024 维特征，训练集上算 μ 与 PCA 基，写 `z_tac (num_frames, 4, 256)`
  float16 与 PCA 基/均值（步骤 2 主目标）；

另存 episode 帧偏移表、训练集每 horizon 每 pad 的 `y_delta` RMS。bottle-sorting：像素场 0.42 GB，
latent 0.28 GB。

`transforms.InjectTactileFutureLabels`（`repack_transforms`，与 `InjectTactileReference` 同模式）：
按 `(episode_index, frame_index + k)` 查表，写出 `future_tactile_delta/state [K,4,16,16,3]`、
`future_tactile_z [K,4,256]`、`future_tactile_mask [K]`；越界置 invalid。serving 不跑 repack。

### 4.2 目标

`D(T)` = 16×16×3 下采样场，`T_ref` = `compute_tactile_refs.py` 产出的该 episode 夹爪张开参考帧：

- 变化场 `y_delta[k,s] = D(T[t+k,s]) − D(T[t,s])`（步骤 1 探针、步骤 2 像素对照）
- 状态场 `y_state[k,s] = D(T[t+k,s]) − D(T_ref[s])`（探针辅助）
- latent `z[k,s] = PCA_white(E_tac(T[t+k,s]))`（步骤 2 主目标）

像素场按训练集固定尺度归一化，零预测基线 loss = 1。不逐图标准化，不给当前/未来帧独立光度增广，
几何变换共用。四路分别监督。

### 4.3 时间与切分

`a_t` 对应 `T_t → T_{t+1}`，先核对时间戳。30 fps 下 K 直接用帧偏移。按 episode/session 切分，验证集
保持自然分布，另报接触子集（`‖y_delta‖` 分位数作代理）。

## 5. 训练细节

| | 步骤 1 | 步骤 2 | 步骤 3 |
|---|---|---|---|
| 可训参数 | 1a：策略（编码器冻结）；1b：+编码器 0.1× LR | 策略 + LTP，编码器同步骤 1 选定的方式 | 同 2 |
| VLM / SigLIP | 冻结 SigLIP，VLM 按配置 | 同 1 | 同 1 |
| LR | 现有配置 2.5e-5 | 同 1，LTP 1e-4 | 同 2 |
| 损失 | flow | flow + λ·L_tac，λ=1，warm-up 1k | 同 2 |
| 监控 | 反事实探针、逐层线性探针、逐层敏感度 S_tac/share | + r_g（0.1–0.5）、按 τ 分箱的 L_tac、share 相对步骤 1 的增量 | 同 2 + 按任务分箱 |
| 数据加载 | 现有 tactile 配置 | + 标签库查表 | + 多数据集混合 |

## 6. 验收与量化

### 6.1 触觉依赖度的量化：逐层敏感度探针

问题：触觉支路的信息对 action expert 的输出有多大影响。回答方式沿用 N0-VTLA 的
`scripts/probe_z_tactile_dependence.py`（文档 `docs/TACTILE_CAUSAL_PROBE.md`）的因果扰动框架：
对输入做受控扰动，量表示的变化，用触觉扰动的敏感度除以 VL 扰动的敏感度得归因比 R。

**测量点必须比 N0 更靠下游。** N0 的 z 由 latent query 同时读 VL 上下文和触觉 token 算出，才可能被
VL 架空。我们的 4 个触觉 token 只由触觉图经 FastViT 与 `tactile_proj` 算出，与 VL 无连接，对它算 R
恒为无穷大，没有信息量。我们的失效模式是"触觉 token 没问题，expert 不看它"，所以测量点在 expert 内部
与输出：

| 测量点 | 形状 | 回答的问题 |
|---|---|---|
| 触觉 token | [B, 4, 1024] | 只做塌缩检查：跨样本去均值余弦接近 1 说明 token 不随样本变，后面的数字全部作废 |
| `h^(l)` 动作位置，l = 1…18 | [18, B, 50, 1024] | 触觉信息在哪一层进入动作流、进了多少；同时服务步骤 1 的选层 |
| 单步 `v_t` | [B, 50, 32] | 触觉最终改变了多少动作 |
| 最终动作 chunk（可选，跑完整去噪） | [B, 50, 32] | 同上，行为级 |

**五个变体**，同一 batch、同一配对噪声 ε、同一 τ，每个变体一次前向：

| 变体 | 扰动 | 隔离的量 |
|---|---|---|
| real | 无 | 参考 |
| null | 四路触觉 `image_mask=False` | 去掉触觉后的响应（N0 是差分置零，这里没有差分） |
| tac-shuffle | 四路触觉图与 mask 在 batch 内 roll 一位，其余不动 | 触觉敏感度 |
| vl-swap | 三路 RGB 与 prompt 在 batch 内 roll 一位，state 与触觉不动 | VL 敏感度 |
| pad-pert | prompt mask 多遮 N 个尾部 token | 红鲱鱼检测，响应应接近 0 |

**每个测量点报三个敏感度、两个比值**，全部用去均值余弦（先减 batch 内 real 的均值，否则共享常量把
原始余弦推到 1，分辨率全无）：

```text
S_tac = 1 − cos_cent(real, tac-shuffle)         触觉敏感度
S_vl  = 1 − cos_cent(real, vl-swap)             VL 敏感度
S_x   = 1 − cos_cent(real[i], real[i+1])        跨样本自然变化，敏感度的天花板

R     = S_tac / S_vl                             N0 的归因比
share = S_tac / S_x                              触觉解释了多大比例的输出变化
```

按 τ ∈ {0.25, 0.5, 0.75, 1.0} 各报一组；按接触相位子集（`‖y_delta‖` 分位数代理）与非接触子集分开报。

**判定，不照搬 N0 的阈值。** N0 的 R ≥ 3 针对的是理应只由触觉决定的 z。对动作输出，视觉决定动作的
大部分是正确的，健康的触觉策略 R 也可能小于 1。用以下三条判：

1. 触觉 token 未塌缩，pad-pert 响应接近 0，否则先修再测。
2. `share` 在接触子集显著高于非接触子集，且在 `v_t` 处非零；无触觉基线在同一测量点为 0。
3. `S_tac` 随层曲线有一个明确的上升层，之后不回落到 0；步骤 2 之后该曲线相对步骤 1 整体上移，
   `v_t` 处的 `share` 增量与 heavy↔light 真实行为差 `delta_F`（`compute_pair_metrics`）同量级。

绝对数值的参照物是真实行为差与无触觉基线，不是固定阈值。

**与现有重型探针的分工。** `scripts/tactile_counterfactual_probe.py` 是 batch_size=1、heavy↔light
配对、跑完整去噪轨迹的工具，一对样本四次完整采样。轻量探针单前向、按 batch 算，便宜两个量级，负责
18 层曲线与大样本统计；重型探针在选出的 m 层与配对样本上做行为级验证。

**实施要点。**

- batch 必须跨 episode 抽，并混合接触帧与非接触帧。同 episode 相邻帧 roll 之后触觉几乎不变，`S_tac`
  会被系统性低估。可直接用 heavy↔light 配对作为 roll 的伙伴。
- 复用 `test/tactile_counterfactual/runner.py` 的模型加载与数据集构建；逐层 hidden 依赖 `gemma.py`
  的逐层输出开关（§7 第 3 项）；变体生成可按 N0 的 `variant_obs` 逐行翻译成 JAX。
- 自检：real 跑两次逐位相等（确定性前向）；tac-shuffle 与 real 的触觉图逐位不等。
- 输出 JSON 与 Markdown 表，每层一行，每 τ 一列组。

### 6.2 其他验收项

- 策略：反事实探针（`scripts/tactile_counterfactual_probe.py`，已有 heavy↔light 配对与 hidden/action
  测量，加全零、时间错位、shuffle、pad 互换条件）；训练级无触觉基线与推理时去触觉联合解释；分块
  留出分箱准确率；闭环实机。hidden cosine、梯度非零只是诊断。
- LTP：`L_tac` 相对零预测基线（=1）；按 horizon、pad、接触子集、τ 分箱；触觉 shuffle 后应回到基线附近。
  头预测得好只是必要条件。
- 消融一次一个变量，同数据同预算。

## 7. 工程清单

步骤 1：
1. `tactile_encoders/fastvit.py`：center crop、BN 统计写回
2. 配置：`freeze_filter`、`rgb_mask_prob`；参数组 LR（1b）
3. `gemma.py`：`nn.scan` 逐层 suffix hidden 输出开关（`out_axes`，训练时才开）
4. `scripts/compute_tactile_future_labels.py`（像素场 + `z_tac` 列）
5. `scripts/probe_future_tactile_layers.py`（逐层线性探针）
6. `scripts/probe_tactile_sensitivity_layers.py`（§6.1 逐层敏感度探针：五变体、三测量点、
   S_tac/S_vl/S_x/R/share，按 τ 与接触子集分箱，JSON + Markdown 输出）
7. 反事实探针加条件

步骤 2：
8. `transforms.InjectTactileFutureLabels`
9. `Observation.aux_targets`（可选字段，`from_dict`/preprocess 透传，部署为 None）
10. `compute_loss` 允许返回 dict；`train.py` 加权求和逐项记录、r_g、按 τ 分箱
11. `Pi0TactileFastVit`：LTP 模块（`tactile_future_head`）、`compute_loss` 覆写取 `h^(m)` 动作位置

步骤 3：
12. `DataConfig` 多 `repo_id` + 任务均衡采样；loader episode 过滤

开工前单测：标签库查表与在线解码同帧一致；越界 mask 正确；关闭 LTP 时前向逐位等于基线；`E_tac`
与策略里 `tactile_encoder` 初始化逐位相同；推理不需要任何未来字段。

## 8. 来源

- RATG：`/home/li/papers/Representation-Aligned Tactile Grounding for.pdf`（arXiv 2607.14609）。
  探针协议 §3.2；LTP §3.3；SmolVLA 结果见 §2.3.5；λ=1。
  注意：PaXini taxel 阵列而非光学凝胶；每任务 20 次实机；目标编码器与输入编码器同一个且未讨论塌缩，
  本文用冻结的原始 FastViT 拷贝规避。
- STAR：`/home/li/papers/STAR.pdf`，稀疏多时刻 {10..50}。
- N0：`/home/li/papers/N0VTLA.pdf`，预测接触变化、Stage 2 遮 VL 通路学接口。
- N0-VTLA 探针：`/home/li/hubo/N0-VTLA/scripts/probe_z_tactile_dependence.py` 与
  `docs/TACTILE_CAUSAL_PROBE.md`，五变体 / 去均值余弦 / 归因比 R 的框架来源；其测量点 z 不适用于本架构，
  见 §6.1。
- `docs/vtla_future_tactile_posttraining_plan.md`：方法草案。
- 现有工具：`scripts/compute_tactile_refs.py`、`scripts/audit_tactile_bn.py`、
  `scripts/tactile_counterfactual_probe.py`、`transforms.InjectTactileReference`。
