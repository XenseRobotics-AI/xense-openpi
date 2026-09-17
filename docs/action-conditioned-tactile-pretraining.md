# 未来触觉预测的触觉预训练方案

日期：2026-09-17（第 5 版）。状态：步骤 2 的代码已实现并通过单测；步骤 1 的两个逐层探针已在本机对
步骤 1a 的 10k 步 checkpoint（`checkpoints/pi05_base_bi_flexiv_bottle_sorting_0915_fastvit_h100/10000`）
正式跑完，结论见 §6.1 末"正式跑"：触觉 token 有信息、expert 不消费，选层候选 `l10`（§3 步骤 1）。
方案本身不依赖任何已有 checkpoint 的结果。

第 5 版改动：写入步骤 1 两个探针在 10k checkpoint 上的正式结果（线性探针 2048 帧、敏感度探针 512 帧）。
依据：RATG（`/home/li/papers/Representation-Aligned Tactile Grounding for.pdf`）、STAR、N0、
`vtla_future_tactile_posttraining_plan.md`、xense-openpi 代码（HEAD 8d563f2 + 工作区的探针改动）。

第 4 版改动：写入步骤 1 两个探针的实际实现（`scripts/probe_future_tactile_layers.py`、
`scripts/probe_tactile_sensitivity_layers.py`、公共件 `test/tactile_counterfactual/layer_probe.py`），
以及实现时定下的几个细节：`vl-swap` 在 pi05 下连离散 state 一起换；`pad-pert` 默认只遮 2 个尾 token；
接触代理有 `delta` / `state` 两种；线性探针默认对 50 个动作位置做平均池化；§7 第 7 项的四个新条件先进
轻量探针。附冒烟首跑观察（§6.1 末）。

第 3 版改动：去掉 BN 统计重估。输入编码器和目标编码器都用未经任何改动的原始 ImageNet FastViT-T12
权重，预处理只保留 center crop。理由见 §2.1。

第 2 版改动：删掉原"步骤 1：独立未来触觉预测器"。输入编码器和目标编码器都直接用原始 ImageNet
FastViT-T12，不单独预训练。原步骤 2/3/4 顺次改为步骤 1/2/3，主实验（策略 + LTP）展开写在 §2.3。

## 1. 目标与原则

目标：让 π0.5 + FastViT 触觉的策略真正依赖触觉。手段：用"当前观测 + 动作 → 未来触觉"这个自监督
目标给 action expert 的中间表示施压（RATG 的 Latent Tactile Predictor，LTP）。未来触觉只在训练时
用，推理路径不变。

三条原则：

1. 编码器不单独预训练，也不改它的统计。原始 FastViT 直接进策略；触觉表示由策略训练本身塑造。
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
        │ FastViT-T12（共享，原始 ImageNet 权重，BN 统计冻结）             │
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

编码器归一化：FastViT 的 BN 在 Flax 实现里固定 `use_running_average=True`，走的是 ImageNet 统计，
与凝胶图像不匹配。本方案不动这些统计，预处理只把 `resize_with_pad` 换成 center crop
（`DataConfig.tactile_resize_mode` 默认已是 `center_crop`）。

为什么不重估 BN。`scripts/audit_tactile_bn.py` 的审计结论是：ImageNet 统计把触觉特征幅度压了约
290×，但没丢信息，接触标签仍可从特征线性读出（约 0.79）。幅度压缩对两侧都无害——策略侧
`tactile_proj` 是可训线性层，尺度它自己会放回来；目标侧 §2.2 的 PCA 白化本来就要重新定尺度。
重估换来的收益是这个已经被吸收掉的尺度，代价却是一条新的失效路径：E_tac 与策略初始化必须逐位同权重，
多一步写回就多一处两边错配的机会，而错配是静默的——标签会变成"另一个编码器眼里的未来"，loss 曲线
照样下降。省掉这一步，这条不变量由"两边都不做任何事"保证。这也是"不预训练也能直接用原始 FastViT"
的依据。
若步骤 2 的 `L_tac` 降不到零预测基线以下，且 16×16 像素场对照明显更好，再回来考虑重估。

### 2.2 目标编码器 E_tac（步骤 2 的监督目标，离线算）

```text
未来触觉帧 T_{t+k,s} (224×224×3, center crop)
        │ E_tac = 原始 FastViT-T12 的冻结拷贝（与 2.1 逐位同一份初始化权重与 BN 统计；训练中永不更新）
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
  语义正确。2.1 的审计说明即便沿用 ImageNet BN 统计，特征也保留了接触信息。若步骤 2 的目标对照
  （换 16×16 像素场）反而更好，说明这个前提不成立，再考虑别的目标编码器。
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
| 逐层线性探针（步骤 1 第 2 项） | `scripts/probe_future_tactile_layers.py`：对 `vlm`、`l0`（suffix 输入）、`l1..l18` 各拟合闭式 ridge 预测 §4.2 的未来场（`--target pixel_delta`，或 `latent`）；τ×{real, null, tac-shuffle} 全组合，按 episode 切验证集，输出每层 nMSE、`null−real` 增益、best layer、按 horizon/pad/接触子集分解 |
| 逐层敏感度探针（§6.1） | `scripts/probe_tactile_sensitivity_layers.py`：五变体 + `--extra`（tac-zero / pad-swap / tac-timeshift），测量点 tactile token、`l0..l18`、`v_t`、最终 chunk；每 τ、每子集报 `S_tac/S_vl/S_null/S_pad/S_x/R/share`，自检 real 两次逐位相等、触觉变体像素确实变了 |
| 探针公共件 | `test/tactile_counterfactual/layer_probe.py`：`load_setup`（复用 runner 的模型/归一化加载，探针默认关 cuDNN attention 与 LTP 头）、`FutureTactileStore`（标签库查表 + 像素场接触代理）、`sample_frames`（batch 内 episode 互不相同、接触/非接触各半）、`LayerForward`（固定 τ、ε 的训练式前向，逐层动作位置残差流）、`make_variant`、去均值余弦、闭式 ridge（d>n 走对偶形式） |

未实现：RTC 与 LTP 同开（配置直接拒绝）、重型反事实探针 `tactile_counterfactual_probe.py` 的新增条件
（全零 / 时间错位 / shuffle / pad 互换目前只在轻量敏感度探针里有，见 §7 第 7 项）、步骤 3 的多数据集混合。

- `Observation.aux_targets`：可选字段，装 `future_tactile_z [B,5,4,256]` 与 `future_tactile_mask [B,5]`；
  `from_dict`/preprocess 透传；部署时为 None。
- `Pi0TactileFastVit.compute_loss` 覆写：调 `PaliGemma.llm` 时开逐层输出开关，取 `h^(m)` 动作位置送
  LTP；返回 `{"flow": ..., "tac": ...}`，`train.py` 加权求和并逐项记录。
- LTP 参数放在 `tactile_future_head` 命名空间，`missing_regex` 已覆盖 `.*tactile.*`，从基座 checkpoint
  载入时不报缺失。
- 关闭 LTP（λ=0 且不取逐层输出）时，前向逐位等于 2.1。

## 3. 三步计划

### 步骤 1：接入策略、探针、选层（8×H100，10k–20k 步）

- 做：§2.1 的策略，编码器用原始 FastViT（ImageNet 权重与 BN 统计，center crop）。1a 编码器冻结；
  1b 编码器 0.1× LR。对照：四路
  触觉 `image_mask=False` 的无触觉基线，同 seed 同数据同步数。
- 测：
  1. 反事实探针：同 RGB/state/prompt/噪声，换触觉（heavy↔light 配对、全零、时间错位、episode 内
     shuffle、pad 互换），量最终动作 chunk 的变化。
  2. **逐层线性探针**（RATG §3.2）：冻结 1a checkpoint，对 VLM 输出和 expert 18 层每层的动作位置
     hidden 各拟合闭式 ridge 预测 §4.2 的未来场；τ∈{0.25,0.5,0.75,1.0}，有/无触觉输入；附触觉
     shuffle 对照。产出误差随层曲线，最小值所在层 = 步骤 2 的 m 的候选。

     实现（`scripts/probe_future_tactile_layers.py`）：
     - 层：`vlm`（PaliGemma 输出对有效 prefix token 做 mask 平均，2048 维，与 τ 无关）、`l0`
       （suffix 输入，即 `action_in_proj(x_t)`，触觉尚未混入）、`l1..l18`（每个 Block 输出的残差流，
       `return_suffix_hidden`）。取 50 个动作位置。
     - 特征：默认 `--features mean`，50 个位置平均成 1024 维；`meanpos` 再拼上第 k−1 个动作 token
       （k 为 5 个 horizon），6144 维，此时 ridge 走对偶形式（n<d）。
     - 目标：默认 `--target pixel_delta`，`[5,4,768]` 展平，逐 pad 用标签库的 `y_delta_rms` 归一化；
       `latent` 可选。每帧要求 5 个 horizon 都在 episode 内（抽帧时 `frame + 50 < 长度`）。
     - 条件：`real`、`null`（触觉 `image_mask=False`）、`tac-shuffle`（batch 内 roll）× 4 个 τ，
       同一帧同一噪声。`vlm` 与触觉条件无关，只算一次。
     - 拟合：特征按训练集标准化，闭式 ridge，`λ = α·tr(XᵀX)/d`，α 在 1e-3…1e3 网格上按验证集 nMSE 选；
       验证集按 episode 切（`--val-fraction 0.25`）。报 nMSE = 验证 MSE / 训练均值预测的 MSE，即 1−R²，
       零预测≈1。另按 horizon、pad、接触/非接触子集分解。
     - 选层：每个 τ 下 `real` 条件 nMSE 最小的 expert 层；同时报 `nMSE(null) − nMSE(real)` 最大的层。
       前者是"哪层最能读出未来"，后者是"哪层的可读性真正来自触觉 token"。两者一致最好；若增益处处≈0，
       说明可读性全来自视觉/state/干净动作，选层只能靠敏感度曲线，且本身就是"expert 不看触觉"的证据。
     - 特征存 `features.npy`（float16 memmap `[cond, tau, layer, frame, dim]`），可离线重拟合。
  3. **逐层敏感度探针**（§6.1）：同一 checkpoint，batch 内 roll 触觉 / roll VL，量 18 层动作位置
     hidden 与 `v_t` 的 `S_tac`、`S_vl`、`S_x`。产出 `S_tac` 随层曲线。与无触觉基线同图对比。
     选层规则：取线性探针误差最小、且 `S_tac` 已明显上升的最浅层。两条曲线不一致时以线性探针为准，
     `S_tac` 曲线记录下来作为步骤 2 前后的对照基线。

  两个探针的命令（本机单卡，输出在 `outputs/tactile_layer_probes/<linear|sensitivity>/<时间戳>/`，
  各含 `results.json`、`report.md`、`frames.json`）：

  ```bash
  CKPT=checkpoints/pi05_base_bi_flexiv_bottle_sorting_0915_fastvit_h100/10000
  CFG=pi05_base_bi_flexiv_bottle_sorting_0915_fastvit_h100
  LABELS=assets/tactile_future_labels/bottle-sorting-0810
  python scripts/probe_future_tactile_layers.py --config-name $CFG --checkpoint-dir $CKPT \
      --labels-dir $LABELS --num-frames 2048 --batch-size 16 --num-workers 8
  python scripts/probe_tactile_sensitivity_layers.py --config-name $CFG --checkpoint-dir $CKPT \
      --labels-dir $LABELS --num-frames 512 --batch-size 16 --extra
  ```

  10k checkpoint 的正式结果见 §6.1 末"正式跑"：线性探针谷底 `l10`/`l11`（nMSE 0.964，触觉增益 ≈0），
  敏感度探针 `v_t` 处 share ≈1e-6，选 **m = 10**。

  读法：线性探针 `report.md` 的"layer choice"表给出每个 τ 下 `real` 条件 nMSE 最小的 expert 层，
  以及 `null−real` 增益最大的层（增益为 0 说明该层的可预测性全来自视觉/state/动作，不来自触觉 token）；
  敏感度探针的 verdict 段给出触觉 token 是否塌缩、pad-pert 是否≈0、`v_t` 处 share 在接触子集是否更高，
  以及每个 τ 下 `S_tac` 曲线的起升层与峰值层。两个脚本对无触觉基线 checkpoint 同样可跑。
  注意：pi05 的离散 state 在 prompt 里，`vl-swap` 会连 state 一起换，`S_vl` 是"视觉 + 本体感觉"敏感度。
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

另存 episode 帧偏移表、训练集每 horizon 每 pad 的 `y_delta` RMS。

bottle-sorting-0810 实测（2026-09-17 建库，160 episodes / 136,959 帧 / 547,836 个拟合向量，
单卡 61 分钟，37 帧/s）：`pixel_field.npy` 402 MB + `z_tac.npy` 268 MB + PCA 基 1.1 MB，
另有 `feat_tac.npy` 1.1 GB，合计 1.8 GB。**`feat_tac.npy` 建完即可删**：它只用于拟合 PCA，
`InjectTactileFutureLabels` 只读 `z_tac` 和 `pixel_field`。删掉后训练实际需要 0.67 GB。

PCA 谱与 d 的选择。256 维解释 1024 维总方差的 99.56%；在保留的 256 维内部，第 1 维占 51.8%、
前 32 维 95.4%、前 64 维 97.7%、前 128 维 99.2%。白化按 `1/sqrt(特征值)` 缩放，末维相对首维放大
127×，所以"尾部维会不会只是被放大的编码器噪声"是个真问题——若成立，后 128 维将占掉一半的 MSE
预算却只携带 0.8% 的方差，§2.2 反对 1024 维逐维标准化的理由就会反过来适用于白化本身。

实测否掉了这个担心：逐维算 episode 内 lag-1 自相关（40 个 episode），全部 256 维都在 0.91 以上
（前 8 维 0.977，第 224–256 维 0.916），256 维内不存在噪声底——尾部维是低幅度但时间连贯的信号。
故 `--pca-dim` 保持 256。保留一条余地：30 fps 下的时间平滑性也能被缓慢漂移的伪迹满足，真正的判据
是尾部维能否由"动作 + 当前触觉"预测，即步骤 2 的 `L_tac` 按维度分块看。z_tac 的维是有序的、白化是
逐维的，若要降 d 只需切片，不必重建库。

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

实测的 `y_delta_rms`（bottle-sorting-0810 全量）在左右夹爪之间差约 3 倍，且在全部 5 个 horizon 上
一致：左爪两片 pad 在 k=10 时是 5.0e-4，右爪两片是 1.4–1.7e-3；同爪的两片彼此几乎相等。这是稳定的
左右不对称，不是噪声。它正是 §2.3.3"四路分别监督，不平均"要保住的信号，同时也意味着像素场对照的
归一化必须逐 pad 做——用一个全局 RMS 会让右爪主导损失。

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

**实现（`scripts/probe_tactile_sensitivity_layers.py`，2026-09-17）。**

- 抽帧（`layer_probe.sample_frames`）：每个 batch 的 episode 互不相同（roll 的伙伴必然来自别的
  episode），奇偶槽位交替要接触帧 / 非接触帧。接触代理来自标签库的 16×16 像素场：`delta` =
  `‖(D(T_{t+10}) − D(T_t)) / rms_10‖`（文档原定义，衡量接下来 1/3 秒的触觉变化），`state` =
  `‖(D(T_t) − D(T_0)) / rms_50‖`（离 episode 首帧的距离，衡量当前是否处于接触态）。阈值取随机池
  （4096 帧）的 `--contact-quantile`（默认中位数）。不给 `--labels-dir` 则不分子集。
- 前向（`LayerForward`）：训练式全序列前向（prefix + suffix 一次算，无 KV cache），固定 τ 与噪声，
  `x_t = τ·ε + (1−τ)·A`，A 是该帧的干净动作块（抽帧时要求 `frame + 50 < 长度`）。cuDNN attention 默认
  关掉、LTP 头关掉，走显式 attention，保证逐位可复现。一次前向返回触觉 token、`l0..l18` 动作位置、
  `v_t`。最终 chunk 用生产采样器（RTC 配置则用 RTC 采样器）跑 `--chunk-steps` 步。
- 变体（`make_variant`）：五个核心变体如上表；`--extra` 追加 `tac-zero`（触觉图置 0，即 [−1,1] 里的
  中灰，mask 不动）、`pad-swap`（左右爪互换：key 0↔2、1↔3）、`tac-timeshift`（同 episode 后
  `--time-shift` 帧的触觉，默认 30 帧）。这三个就是 §7 第 7 项的条件，先在这里落地。
  两处与原表的出入：(a) pi05 的离散 state 在 prompt 里，`vl-swap` roll prompt 时 state 一起被换，
  `S_vl` 实际是"视觉 + 本体感觉"敏感度，无法在不重新 tokenize 的前提下只换视觉；(b) prompt 尾部是
  `;\nAction: `，`pad-pert` 默认只遮 2 个尾 token（`--pad-extra`），多遮会吃掉 state 数字，就不再是
  红鲱鱼。
- 指标：每个测量点对每个变体算逐样本 `1 − cos_cent`，中心是该 batch 内 real 的均值；`S_x` 用
  `real[i]` 对 `real[i+1]`。汇总报 mean/std/n，分 all / contact / noncontact；R 与 share 用均值之比，
  分母小于 1e-7（float32 舍入级，例如 `l0` 处的 `S_vl`）时记 n/a。
- 自检：real 两次前向 `action_stream` 逐位相等（冒烟实测 0.0）；每个触觉变体与 real 的触觉图逐行不等。
- 输出：`results.json`（全部统计）、`report.md`（每 τ 一张表，每测量点一行）、`frames.json`
  （抽到的帧、接触标记、阈值）。verdict 段直接给出三条判定所需的数：触觉 token 的 `S_x`（塌缩检查）、
  `v_t` 处 `S_pad/S_vl`（红鲱鱼）、`v_t` 处 share 的接触 / 非接触对比、每个 τ 下 `S_tac` 曲线的起升层
  （首次达到峰值一半）与峰值层。

**冒烟首跑（2026-09-17，10k checkpoint，16 帧 / 15 个 episode，τ∈{1.0, 0.5}，仅作管线验证，
数字不能当结论）。**

- 触觉 token 未塌缩：`S_x = 1.16`；`tac-shuffle` 处 1.16、`tac-zero` 1.00、`tac-timeshift` 0.51，
  即时间错位 1 秒的触觉图与当前差异约为跨样本差异的一半。
- 动作流几乎不看触觉：`S_tac` 在 `l1..l18` 从 1e-7 单调升到 3e-5，`v_t` 处 1e-6，share≈1e-6；
  同一位置 `S_vl` 从 3e-3 升到 0.9，`v_t` 处 0.1–0.3。R 在 1e-5 量级。
- `null`（mask 掉触觉 token）的响应 1e-4…1e-2，比 shuffle 大两个量级：去掉 4 个 key 改变了 attention
  的归一化，比换 token 内容影响大。所以 `S_tac` 用 shuffle 而不用 null 是对的。
- `pad-pert` 在 `v_t` 处 `S_pad/S_vl ≈ 0.01`，红鲱鱼检查通过；但在 `l17/l18` 达到 0.03–0.04，比 `S_tac`
  大三个量级，提示深层对 prompt 尾部格式 token 的敏感度已高于对触觉的敏感度。
- 这与 §3 步骤 1 的预期一致（"这一步的反事实探针效应很弱"）：10k 步、模仿损失单独训练下，触觉 token
  有信息但 expert 不消费。正式跑（512 帧、4 个 τ、`--extra`）与无触觉基线的同图对比，是步骤 2 前后
  对照的基线。
- 线性探针冒烟（64 帧、44 个训练行、1024 维）nMSE 全部 >1，属训练行不足，无结论；正式跑 2048 帧。

**正式跑（2026-09-17，同一 10k checkpoint，本机 5090 单卡；输出
`outputs/tactile_layer_probes/linear/10k_2048/` 与 `outputs/tactile_layer_probes/sensitivity/10k_512/`）。**

线性探针：2048 帧 / 160 episodes，按 episode 切 1517 训练行 / 531 验证行，`--features mean`，
`--target pixel_delta`，τ∈{0.25,0.5,0.75,1.0} × {real, null, tac-shuffle}。前向 767 s，拟合 516 s。

| 层 | τ=0.25 | τ=0.5 | τ=0.75 | τ=1.0 |
|---|---|---|---|---|
| vlm | 0.9729 | 0.9729 | 0.9729 | 0.9729 |
| l0 | 0.9905 | 0.9926 | 0.9964 | 1.0000 |
| l5 | 0.9691 | 0.9677 | 0.9706 | 0.9700 |
| l8 | 0.9663 | 0.9641 | 0.9670 | 0.9678 |
| **l10** | **0.9657** | **0.9639** | 0.9646 | 0.9637 |
| l11 | 0.9662 | 0.9640 | **0.9646** | **0.9637** |
| l14 | 0.9699 | 0.9694 | 0.9698 | 0.9692 |
| l18 | 0.9750 | 0.9750 | 0.9746 | 0.9745 |

- 曲线形状与 RATG 一致：`l0` ≈ 1（干净动作本身几乎不含未来触觉），沿层单调下降到 `l10`/`l11`
  的谷底，再回升到 `l18`。四个 τ 的最优层都是 `l10` 或 `l11`（两者差 <1e-3），接触子集最优层 `l8`–`l11`。
  谷底比 `vlm`（0.973）好 0.009，比 `l18` 好 0.011。ridge 的 α 选在网格中段（10），不是边界。
- 可读出的方差极小：最好也只有 3.6%（nMSE 0.964）。按 horizon 单调（k=10 时 0.987，k=50 时 0.953，
  越远越可读，即读出的是慢变的接触趋势），按 pad 左右分明（左爪两片 ≈1.01，即读不出；右爪两片
  0.90–0.94），接触子集 0.96 略好于非接触 0.97。
- **可读性完全不来自触觉 token**：`null − real` 增益在所有层、所有 τ 都在 ±8e-4 以内，中间层反而略负
  （去掉触觉 token 后动作位置对未来触觉略更可读）；`tac-shuffle − real` 在 1e-6 量级，即换掉触觉内容
  对动作位置的特征没有任何影响。3.6% 全来自视觉 / state / 干净动作。
- 副产品：同一批帧的 flow loss，`real` 与 `tac-shuffle` 逐位相同（τ=0.25 都是 5.46e-3），`null`
  高 35–70%（7.30e-3）。去掉 4 个 key 改变了 attention 归一化，模型对"触觉 token 在不在"敏感、对
  "触觉 token 是什么"不敏感。这条数据本身就否定了"推理时把触觉 mask 掉来验证无触觉基线"的做法。

敏感度探针：512 帧 / 153 episodes，接触 / 非接触各 256，四个 τ，`--extra`，最终 chunk 10 步去噪。
前向 771 s。自检：real 重复前向逐位相等（0.0），四个触觉变体像素均变了。

| 测量点（τ=1.0） | S_tac | S_vl | S_null | S_pad | S[tac-timeshift] | S_x | share |
|---|---|---|---|---|---|---|---|
| tactile_tokens | 1.076 | 0 | 0 | 0 | 0.492 | 1.076 | 1 |
| l1 | 2.4e-7 | 1.9e-3 | 1.3e-4 | 2.8e-5 | 2.2e-7 | 1.066 | 2.2e-7 |
| l10 | 3.5e-6 | 0.045 | 2.2e-3 | 7.7e-4 | 3.5e-6 | 1.068 | 3.3e-6 |
| l14 | 1.1e-5 | 0.224 | 6.9e-3 | 6.7e-3 | 1.1e-5 | 1.074 | 1.0e-5 |
| l18 | 3.1e-5 | 0.627 | 8.6e-3 | 0.032 | 3.1e-5 | 1.100 | 2.8e-5 |
| v_t | 1.2e-6 | 0.091 | 3.5e-4 | 1.0e-3 | 1.2e-6 | 1.071 | 1.1e-6 |
| chunk（10 步） | 2.2e-6 | 1.134 | 6.5e-3 | 0.012 | 2.2e-6 | 1.134 | 1.9e-6 |

- 判定 1：触觉 token 未塌缩（`S_x=1.08`，tac-zero 1.00、pad-swap 1.00、timeshift 30 帧 0.49）；
  pad-pert 在 `v_t` 处 `S_pad/S_vl` = 0.011–0.028，红鲱鱼通过。前提成立，后面的数字有效。
- 判定 2 不成立：`v_t` 处 share 在 1e-6 量级，四个 τ 下接触子集与非接触子集完全相同（1.10e-6 vs
  1.10e-6）；最终 chunk 处 share 1.9e-6。逐样本 std 与均值同量级（`v_t` τ=1.0：1.18e-6 ± 0.25e-6），
  是稳定的极小值，不是噪声。
- 判定 3 不成立：`S_tac` 沿层从 2e-7 单调升到 `l18` 的 3e-5，无起升层（"半峰"落在 `l13`–`l16` 只是单调
  曲线的算术结果），到 `v_t` 又跌回 1e-6。`S_vl` 同位置 2e-3 → 0.63 → 0.09，R 全程 1e-5–1e-3。
  τ 越小（动作越干净）`S_tac` 略大（τ=0.25 时 `v_t` 处 3.0e-6），但仍差四个量级。
- 三个 `--extra` 条件（tac-zero、pad-swap、tac-timeshift）在动作流上与 shuffle 给出相同数字
  （例如 `l18` 三者都是 3.1e-5），即 expert 对触觉内容的任何改动都一视同仁地不响应；tac-zero 略大
  （1.3e-4）是因为它把 token 幅度也改了。
- 深层对 prompt 尾部 2 个格式 token 的敏感度（`l17`/`l18` 的 `S_pad` 0.03–0.05）比对触觉高三个量级，
  冒烟观察在大样本上成立。

**结论与选层。** 步骤 1a（10k 步、仅模仿损失、编码器冻结）的 expert 完全不消费触觉 token：线性探针的
触觉增益 ≈0，敏感度探针 share ≈1e-6，与 §3 步骤 1 的预期一致，也是 §2.3.5 "触觉当输入 41 vs 中间层
挂头 74" 那条差距在本架构上的直接证据。选层规则"线性探针误差最小、且 `S_tac` 已明显上升的最浅层"
里第二个条件在这个 checkpoint 上不可用（`S_tac` 无起升层），按规则以线性探针为准：**m = 10**
（四个 τ 下的谷底，`l11` 等价备选），落在 RATG 报告的 5–9 层区间的深端。这条曲线（`l10` 谷底
0.964、`S_tac` 单调 2e-7→3e-5、`v_t` share 1e-6）就是步骤 2 前后对照的基线。
无触觉基线 checkpoint 的同图对比仍待跑（需要先训练 1a 的 `image_mask=False` 对照）。

### 6.2 其他验收项

- 策略：反事实探针（`scripts/tactile_counterfactual_probe.py`，已有 heavy↔light 配对与 hidden/action
  测量；全零、时间错位、shuffle、pad 互换四个条件目前在轻量探针里，重型探针尚未加）；训练级无触觉基线与推理时去触觉联合解释；分块
  留出分箱准确率；闭环实机。hidden cosine、梯度非零只是诊断。
- LTP：`L_tac` 相对零预测基线（=1）；按 horizon、pad、接触子集、τ 分箱；触觉 shuffle 后应回到基线附近。
  头预测得好只是必要条件。
- 消融一次一个变量，同数据同预算。

## 7. 工程清单

步骤 1：
1. ~~center crop~~ 已有：`DataConfig.tactile_resize_mode` 默认 `center_crop`。BN 统计重估已从方案中
   去掉（§2.1），此项无剩余工作。
2. 配置：`freeze_filter`、`rgb_mask_prob`；参数组 LR（1b）
3. `gemma.py`：`nn.scan` 逐层 suffix hidden 输出开关（`out_axes`，训练时才开）
4. `scripts/compute_tactile_future_labels.py`（像素场 + `z_tac` 列）
5. ~~`scripts/probe_future_tactile_layers.py`（逐层线性探针）~~ 已实现（2026-09-17）
6. ~~`scripts/probe_tactile_sensitivity_layers.py`（§6.1 逐层敏感度探针：五变体、三测量点、
   S_tac/S_vl/S_x/R/share，按 τ 与接触子集分箱，JSON + Markdown 输出）~~ 已实现（2026-09-17）
7. 反事实探针加条件：全零、时间错位、shuffle、pad 互换已作为 `--extra` 变体进了第 6 项的轻量探针
   （单前向、hidden 与 `v_t`、最终 chunk 级）；重型 `tactile_counterfactual_probe.py`（完整去噪轨迹、
   heavy↔light 配对）尚未加这些条件

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
