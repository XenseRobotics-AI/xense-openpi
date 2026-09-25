# 触觉专家级联流匹配（Cascaded Flow Matching）接入 π₀.₅

把去噪轨迹 τ∈[0,1] 切成两段：**上段由 action expert 负责（触觉不可见），下段由一个新增的第三专家 "tactile expert" 负责（触觉可见）**。两个专家预测的是同一个速度场 `ε − A`，只是 τ 区间不相交，所以触觉专家积分出来的就是最终动作，不是残差修正量。

参考实现：`/home/li/hubo/T-Rex`（Qwen3-VL + MoT 三专家）。本文档是把那套逻辑移植到本仓库 JAX 侧的设计方案。

---

## 0. 范围与已定决策

| 项 | 决策 |
|---|---|
| 目标框架 | **只做 JAX（Flax NNX）**。PyTorch 侧（`models_pytorch/`）与 `convert_jax_model_to_pytorch.py` 本次不动 |
| 模型变体 | **只支持 `pi05=True`**。`pi05=False` 在 config `__post_init__` 里直接 raise |
| 异步 slow/fast | **不做**。两段在 `sample_actions` 内部背靠背同步跑完，对外签名不变，serving 零改动 |
| RTC | **不做**。`enable_training_time_rtc=True` 在 config `__post_init__` 里直接 raise |
| LoRA | **不做**。全量训练，`get_freeze_filter` 不动 |
| action expert 是否看得到触觉 | **看不到**。触觉 token 从 suffix 移出，只进第三专家的流 |
| 降级路径（`forward_flow_action_full`） | **不做**。没有"触觉掉线时 action expert 独立跑满 10 步"的推理分支 |
| 训练时构造 KV 的噪声 | **用独立噪声，对齐 T-Rex**。与构造 `x_τ` 的噪声不同源；同噪声变体留到服务器上做 A/B |
| 触觉专家权重初始化 | **随机（xavier），对齐 T-Rex**。从 action expert 拷贝降为 A/B 的另一臂，理由见 §5 |
| 触觉 dropout | **0**。T-Rex 的 10% 是为降级路径服务的，降级路径砍掉后它变成反向正则，理由见 §4.3 |
| 触觉编码器 | 复用现有 FastViT-T12 分支（`tactile_encoders/` 注册表 + `tactile_proj`），不改 |

与现有 `Pi0TactileFastVit`（`src/openpi/models/pi0_tactile_fastvit.py`）的关系：**并列的另一条路线**，现有实现保留不动，作为 A/B 的对照组。

---

## 1. T-Rex 参考逻辑（摘要）

骨干 `T-Rex/qwen_vla/modeling_qwen3vl_mot.py`：一个 Qwen3-VL，三套并行专家权重（latent / action / tactile），每层各自 QKVO + norm + MLP，token 拼在序列维上做**联合注意力**（`:200-270`），靠 `latent_indexes / action_indexes / tactile_indexes` 路由。

流被切成两段（`modeling_vla.py:17-25`）：

| 阶段 | 方法 | τ 区间 | 步数 |
|---|---|---|---|
| 上段 | `forward_flow_action_partial`（`:643`） | [τ_split, 1] | `split_step=6` |
| KV 刷新 | 同上，`refresh_clean_kv=True`（`:733-760`） | 在 τ_split 处 | 1 次前向 |
| 下段 | `tactile_flow_continue`（`:766`） | [0, τ_split] | `10 − 6 = 4` |

关键点：

- `dt = −1/num_steps_total` 两段共用，所以级联轨迹的积分步长与单体 10 步流一致。
- KV 刷新：上段跑完后，用 τ_split 时刻的 `x_split` **重写一遍 action 段的 KV**，让触觉专家看到的是自洽的 τ_split 动作上下文。
- 触觉专家的输入序列是 `[触觉观测 token | τ emb | x_τ 动作 token]`，读出头是独立的 `final_layer_tactile`。
- 训练（`T-Rex/scripts/train.py:1015-1113`）两个 loss：`L_flow`（action expert，τ∈[0,1] 全域，触觉盲）+ `w · L_flow_tactile`（触觉专家，τ∈(0,τ_split]，目标同为 `ε − A`）。
- **三次独立采噪**：`noisy_actions/timesteps` 用于 `L_flow`；`ahat_noise`（`:1048`）用于 no_grad rollout 拿 KV；`eps_r/time_r`（`:1067-1072`）用于构造触觉专家的训练点。三者互不相同。
- 触觉 dropout 10%：把触觉**信号置零**（不是置 None），`:1083-1096`。
- 触觉专家权重 **xavier 随机初始化**，不从 action expert 拷贝（`:846-854`）。

---

## 2. 本仓库的落点

### 2.1 骨干：`gemma.Module` 已经是 N 专家泛化的

`src/openpi/models/gemma.py`：

- `configs` 是 `Sequence[Config]`，`xs` 里给 `None` 表示该专家本轮不出场（`Attention.__call__:344-449`、`Module.__call__:603-625`）。
- 权重命名 `_name(name, i)`（`:657`）：第 0 个专家无后缀，之后 `_1`、`_2`。注释里写"实际只用两个"，但代码没有硬编码 2。
- 约束（`:346-348`、`:563`）：所有专家必须同 `num_heads / num_kv_heads / head_dim / depth`。`gemma_300m`（width 1024, depth 18, heads 8/1, head_dim 256）完全满足。

**结论：骨干一行不用改。**

### 2.2 KV cache 比 T-Rex 干净

`Attention` 里 `k = concat([cache_k, k])` 之后**整条返回**（`gemma.py:392-394`、`:447`）。所以"prefix ⊕ action suffix"的缓存直接是某次前向的第二个返回值，不需要 T-Rex 那套 `crop` / `_clone_dynamic_cache`。KV 刷新在本仓库是天然免费的——就是"多跑一次前向并接住它的 kv"。

### 2.3 token 布局（pi05）

```
expert 0 (paligemma, 2048)   prefix:  3×256 图像 token + 200 文本 token（state 已离散化进文本）
expert 1 (action,   1024)   suffix:  50 个动作 token（pi05 无 state token，时间走 adaRMS）
expert 2 (tactile,  1024)   tstream: 4 个触觉 token + 50 个动作 token（时间走自己的 adaRMS）
```

`ar_mask`：

- prefix：全 `False`（双向）
- suffix：`[True] + [False]*49`（一个块，块内双向，看得到 prefix）
- tstream：`[True, False, False, False] + [True] + [False]*49`（两个块：触觉块 + 动作块，动作块看得到触觉块）

这个写法和现有 `pi0_tactile_fastvit.py:104` 的 `tactile_ar` 一致，也保持了 pi0 "动作 token 互相双向" 的既有语义。注意与 T-Rex 的差别：T-Rex 骨干是全因果的，它的 x_τ token 之间是因果的；本仓库保持块内双向。

---

## 3. 推理流程（同步，`sample_actions`）

记 `N = num_steps`（默认 10），`S = cascade_split_step`（默认 6），`τ_split = 1 − S/N`，`dt = −1/N`。

```
Phase 0  prefix
  prefix_tokens, prefix_mask, prefix_ar = embed_prefix(obs)          # 触觉已被过滤掉
  _, kv_prefix = llm([prefix_tokens, None, None],
                     mask=make_attn_mask(prefix_mask, prefix_ar),
                     positions=cumsum(prefix_mask)-1)

Phase A  t: 1.0 → τ_split，S 步，action expert
  （与今天的 sample_actions 逐行相同，只是 llm 入参补成三元）
  llm([None, suffix_tokens, None], kv_cache=kv_prefix,
      adarms_cond=[None, cond_a, None])
  v_t = action_out_proj(suffix_out[:, -50:]);  x ← x + dt·v_t
  while_loop cond: time >= tau_split - dt/2

Phase A' KV 刷新（1 次前向，结果丢弃，只要 kv）
  suffix_tokens_split = embed_suffix(obs, x_split, τ_split)
  _, kv_split = llm([None, suffix_tokens_split, None], kv_cache=kv_prefix, ...)
  # kv_split 长度 = prefix_len + 50

Phase B  t: τ_split → 0，N−S 步，tactile expert
  tac_tokens = [4 触觉 token（FastViT→tactile_proj）| action_in_proj_tac(x_t) ×50]
  llm([None, None, tac_tokens], kv_cache=kv_split,
      adarms_cond=[None, None, cond_t])
  v_t = action_out_proj_tac(tac_out[:, -50:]);  x ← x + dt·v_t
  while_loop cond: time >= -dt/2
```

**Phase A 的终止条件**：`time >= tau_split - dt/2`。在 `time == tau_split` 时为假（因为 `-dt/2 > 0`），在前一步为真，正好跑满 S 步。这与现有 `cond` 的 `time >= -dt/2` 是同一个式子在 `tau_split = 0` 时的特例，保持风格一致，且对 `num_steps` 被 trace 成动态值仍然安全。

**Phase B 的 mask**（query 54，key `prefix_len + 50 + 54`）：

```
to_prefix = einops.repeat(prefix_mask, "b p -> b s p", s=54)
to_suffix = ones((b, 54, 50))                       # τ_split 的动作上下文对触觉全可见
within   = make_attn_mask(tac_mask, tac_ar)          # (b, 54, 54)
full     = concat([to_prefix, to_suffix, within], axis=-1)
```

`tac_mask = concat([4 路触觉图像的 image_masks, ones(b,50)])`。`make_attn_mask` 的 `valid_mask`（`pi0.py:59`）会把缺失的触觉视角在 key 侧一并屏蔽，与现有实现行为一致。

**Phase B 的 positions**：`pos = (sum(prefix_mask) + 50) + cumsum(tac_mask) - 1`，即接在 action suffix 之后。等价于 T-Rex 的 `_extend_position_ids(latent_pos, n_action_in_cache, n_tac_seq)`。注意这意味着同一个动作 index 在两条流里拿到不同的 RoPE 位置——这是 T-Rex 的行为，本方案对齐。

---

## 4. 训练流程（`compute_loss`）

```
# --- 项 1：action expert，触觉盲，τ∈[0,1] 全域（与现有 pi05 loss 完全相同）---
ε_a ~ N(0,I);  t_a ~ Beta(1.5,1)*0.999+0.001
x_t = t_a·ε_a + (1−t_a)·A;   u_a = ε_a − A
(prefix_out, suffix_out, _), kv_full = llm([prefix, suffix, None],
                                           adarms_cond=[None, cond_a, None])
loss_a = mean((action_out_proj(suffix_out[:,-50:]) − u_a)², axis=-1)

# --- 项 2：tactile expert，τ∈(0,τ_split] ---
kv_prefix = stop_gradient(kv_full[:, :, :prefix_len])      # 见 4.1
ε_kv ~ N(0,I)                                              # 独立噪声（对齐 T-Rex）
x_split = stop_gradient(rollout S 步(action expert, 起点 ε_kv, kv_prefix))
kv_split = stop_gradient(刷新前向(x_split, τ_split, kv_prefix))

ε_t ~ N(0,I);  t_t ~ Beta(1.5,1)*0.999+0.001               # 与 ε_a/t_a 独立
τ = t_t · τ_split
x_τ = τ·ε_t + (1−τ)·A;   u_t = ε_t − A
(_, _, tac_out), _ = llm([None, None, tac_tokens(x_τ, τ)],
                         kv_cache=kv_split, adarms_cond=[None, None, cond_t])
loss_t = mean((action_out_proj_tac(tac_out[:,-50:]) − u_t)², axis=-1)

return loss_a + w · loss_t          # 两项都是 (b, ah)，签名不变
```

### 4.1 prefix KV 复用（省掉一次昂贵的前缀前向）

项 2 需要一份 prefix KV。**不要**为它单独再跑一次 prefix 前向（那是 SigLIP×3 + 2B 骨干过 968 token，是整个 step 里最贵的一块）。

项 1 的前向返回值第二项现在被丢弃（`pi0.py:301` 的 `, _`），它是 `(k, v)`，序列布局是 `[prefix | suffix]`，切 `[:, :, :prefix_len]` 即得 prefix KV。

**这个切片是精确等价的**，理由：`k/v` 是每个 token 自身隐状态的线性投影；而 prefix token 的隐状态与 suffix 是否存在无关——`make_attn_mask` 下 suffix 首 token 的 `ar=True` 使 prefix 无法注意到 suffix（`pi0.py:56`），prefix 的 RoPE 位置也与单独前向时相同。所以 prefix 的每层 K/V 逐位相同。

### 4.2 梯度隔离

JAX 没有 `torch.no_grad`，靠 `jax.lax.stop_gradient` 截断**离开 rollout 的所有数组**（`x_split` 与 `kv_split` 的 k、v）。截断后反向不会经过 rollout 回流到 action expert 参数，XLA 也会把这些前向的反传 DCE 掉。

### 4.3 触觉 dropout：默认 0，不要照抄 T-Rex 的 10%

T-Rex 留这 10% 是为 `forward_flow_action_full` 那条降级路径服务的——要让触觉专家学会"触觉掉线时也能跑"。本方案砍掉了降级路径，这个理由就不存在了。

更关键的是它的方向是反的：dropout 教给模型的是"触觉为零时也要给出好动作"，而触觉专家**本来就能只靠 KV 给出好动作**（见 §8.1）。这 10% 的步数等于在直接训练那个我们想避免的无触觉解，不是中性正则。

次要代价：训练里见过全零触觉会让"置零"这个反事实探针变成 in-distribution 从而钝化。无论 dropout 取多少，**探针一律用 shuffle（打乱 batch 内触觉），不要用置零**——这是之前 LTP 那轮吃过的亏。

字段保留（`tactile_dropout`），默认 0.0。若后续确实要做降级推理再打开；实现上置零**投影后的触觉 token** 而不是输入图像，比经 FastViT 编码一张全零图更省也更确定。

### 4.4 开销

每个 training step 额外多出 `S + 1 + 1 = 8` 次 suffix 规模的前向（54 token × 18 层 × attend ~1018 key），其中 7 次在 `stop_gradient` 下、1 次带梯度。相对主前向（SigLIP + 2B 过 968 token + 反传）是小头，**粗估 +15~30% step 时间，需实测**。

显存：新专家 311M 参数，fp32 参数 1.24 GB + AdamW 一二阶矩 2.49 GB ≈ **+3.7 GB**。80 GB H100 上按 memory 里现有 batch 配置需要留意余量。

⚠️ `use_cudnn_attention` 只在 `kv_cache is None` 时生效（`gemma.py:428`）。所有级联前向都带 kv_cache，因此走 `explicit_attention`。主 loss 前向仍走 cuDNN（float16 内核），两者不冲突。

### 4.5 分项日志（可选但建议做）

`compute_loss` 折叠成一个标量后，`loss_a` 与 `loss_t` 在日志里就分不开了，而这两条曲线是判断触觉分支是否在学的主要依据。

改法：`scripts/train.py:281` 的 `loss_fn` 返回 `(loss, aux)`，`nnx.value_and_grad(..., has_aux=True)`，`info` 字典里多两个 key。约 5 行。需要 `compute_loss` 额外吐出分项——可以加一个 `compute_loss_with_aux` 方法，避免动 `BaseModel` 的签名。

---

## 5. 权重初始化与加载

新增参数：

| 位置 | 参数 | 来源 |
|---|---|---|
| `.*llm.*` 下所有 `_2` 后缀段 | `q_einsum_2` `kv_einsum_2` `attn_vec_einsum_2` `mlp_2` `pre_attention_norm_2` `pre_ffw_norm_2` `final_norm_2`（含 adaRMS 的 `Dense_0`） | **从 `_1`（action expert）拷贝** |
| 顶层投影 | `action_in_proj_tac` `action_out_proj_tac` `time_mlp_in_tac` `time_mlp_out_tac` | **从无后缀版本拷贝** |
| 触觉分支 | `tactile_encoder` `tactile_proj` | FastViT ImageNet 权重 / 随机（`__init__` 内已处理） |

**默认用随机（xavier），拷贝作为 A/B 的另一臂。** 表里"从 `_1` 拷贝"是 `tactile_expert_init: copy_action` 那一臂的行为。

拷贝看起来更优（pi05_base 的 action expert 已经是训练好的速度预测器，拷完 step 0 的级联链就约等于基线单体 10 步流，收敛快、§7.3 的等价性测试也好做），但它有一个具体的坏处：

> 拷贝之后 `loss_t` 从一开始就接近最优，**整条触觉支路拿到的梯度幅度从 step 0 起就很小**。而 `tactile_proj` 是随机初始化的、需要实质梯度才能被塑造成有用的投影。小梯度 + 随机投影 ⇒ 这条支路大概率停在噪声状态，成为死分支。

随机初始化下 `loss_t` 初期很大，包括 `tactile_proj` 在内的所有通路都拿到实质梯度，触觉支路至少会被塑形。

⚠️ 但**初始化不是决定性的杠杆**。随机初始化下模型同样会先学 KV 这条强信号、学完进平台期，触觉照样可能被忽略。初始化只影响"多快走到捷径"，不影响"会不会走"。真正的杠杆是 §8.2 的 `cascade_split_step`。

§7.3 的等价性测试是结构测试不是训练，跑它的时候临时切到 `copy_action` 即可。

实现：新增 `CascadeInitWeightLoader(CheckpointWeightLoader)`，在 `_merge_params` 之前对扁平化的 loaded dict 做一次段级重命名复制（路径按 `/` 切段，段名结尾 `_1` → `_2`；顶层 `x` → `x_tac`）。YAML 里 `missing_regex` 仍需覆盖 `(.*tactile.*)`，否则 FastViT 那 450 个叶子会被丢掉、在 step 0 之前挂在 pytree 结构校验上（现有配置的注释已记录过这个坑）。

---

## 6. 改动清单

| 文件 | 改动 | 规模 |
|---|---|---|
| `src/openpi/models/gemma.py` | **无**（确认 N 专家已支持） | 0 |
| `src/openpi/models/pi0_config.py` | 抽出"专家 config 列表 + `use_adarms` 列表"的构造方法，供子类覆盖 | 小 |
| `src/openpi/models/pi0.py` | `__init__` 改用上述列表；3 处 `llm([...])` 调用补成按专家数长度（加个 `_llm_inputs` 小 helper）。RTC 两处一并补齐或留空（新 config 会禁用 RTC） | 小 |
| `src/openpi/models/pi0_tactile_expert_config.py` | 新增。字段见 §6.1，`__post_init__` 里 raise 掉 `pi05=False` 和 `enable_training_time_rtc=True` | 新文件 |
| `src/openpi/models/pi0_tactile_expert.py` | 新增。`embed_tactile` / 重写 `sample_actions` / 重写 `compute_loss`。`embed_suffix` 直接用基类（不含触觉） | ~250 行 |
| `src/openpi/training/weight_loaders.py` | 新增 `CascadeInitWeightLoader` | ~40 行 |
| `src/openpi/training/registry.py` | 注册新 config 类型 + 新 loader | 2 行 |
| `src/openpi/models/model.py` | **无**。复用 `ModelType.PI05_TACTILE`（数据变换需求完全相同，`training/config.py:139` 的 match 分支不用动） | 0 |
| `configs/_examples/<新>.yaml` | 新训练配置 | 新文件 |
| `scripts/train.py` | 可选：分项 loss 日志（§4.5） | ~5 行 |
| `src/openpi/models/pi0_tactile_expert_test.py` | 新增，用 `dummy` gemma variant 跑结构 smoke + §7 的等价性断言 | 新文件 |

### 6.1 新增配置字段

```yaml
model:
  type: Pi0TactileExpertConfig
  pi05: true                      # 强制
  enable_training_time_rtc: false # 强制
  tactile_expert_variant: gemma_300m
  tactile_expert_init: random             # random | copy_action，见 §5
  cascade_total_steps: 10
  cascade_split_step: 4                   # τ_split = 0.6，见 §8.2；不要默认用 6
  tactile_loss_weight: 1.0
  tactile_dropout: 0.0                    # 见 §4.3，不要照抄 T-Rex 的 0.1
  # 以下继承自现有触觉分支
  tactile_encoder_name: fastvit_t12
  tactile_pretrained_path: ~/.cache/fastvit_t12_apple_dist_in1k_flax/params.safetensors
  tactile_compute_dtype: bfloat16

weight_loader:
  type: CascadeInitWeightLoader
  params_path: gs://openpi-assets/checkpoints/pi05_base/params
  missing_regex: (.*tactile.*)
```

---

## 7. 验证计划

上服务器之前，本地按顺序过：

1. **结构 smoke**：`dummy` variant（width 64 / depth 4）建模型，跑通 `compute_loss` + `sample_actions`，断言形状。
2. **退化等价性 A**：`cascade_split_step == cascade_total_steps` ⇒ Phase B 零步，`sample_actions` 输出应与基线 pi05 **逐位一致**。
3. **退化等价性 B**：临时切 `tactile_expert_init: copy_action` + 临时把触觉 token 数设为 0 ⇒ 级联 S+(N−S) 步的输出应与基线 10 步**几乎重合**（残差只来自 τ_split 处的 KV 刷新与位置偏移）。这条能一次性抓出 mask / positions / KV 拼接的错误。注意这只是结构测试，训练默认仍用 `random`。
4. **训练 smoke**：100 步。`loss_a` 曲线应与基线（去掉 suffix 触觉的 pi05）重合——action expert 的计算路径确实没变；`loss_t` 应从 ≈`loss_a` 的量级开始下降。
5. **服务器 A/B**，按优先级：
   - `cascade_split_step` ∈ {4, 2}（§8.2，最重要）
   - KV rollout 噪声 **独立**（本方案默认）vs **与 `x_τ` 同源**
   - `tactile_expert_init` `random`（默认）vs `copy_action`（§5）
6. **触觉敏感性探针**（关键，每个 arm 都要做）：对最终动作做触觉 **shuffle** 反事实，测 `‖Δaction‖`。只看 `loss_t` 下降完全不够——`loss_t` 降到很低但对触觉零响应，正是 LTP 0917 那轮的实际结果。

---

## 8. 风险与未决项

### 8.1 级联到底买到了什么（先把预期校准好）

诚实的结论：**级联让"触觉专家"成为必需品，但没有让"触觉输入"成为必需品。**

后 4 步的输出只能由触觉专家产生、action expert 在结构上无法完成这段积分——这一点是真的，与现在"触觉 token 拼进 suffix"有本质区别。但触觉专家完全可以只靠 cached KV（VL 前缀 + τ_split 的动作上下文）把这几步积完，对它自己的触觉 token 零响应。那样付出 311M 参数和 +3.7 GB 显存，换来的是同一个结果。这与之前 LTP 那轮的失败模式同构。

相对现有单专家方案，确定能拿到的增量只有三条：

1. **参数隔离**：学用触觉不会破坏 action expert 已有能力。单专家时预训练解是很强的吸引子，"保持原解、忽略触觉"是局部最优——这多半就是探针测出触觉增益≈0 的成因。
2. **独占计算通道**：触觉不必和 968 个 VL token 争注意力带宽。
3. **可归因性**：后几步里触觉专家相对 action expert 唯一多出来的信息就是那几个触觉 token，所以反事实探针的结果变得干净可解释。这是**测量**上的好处，不是行为上的保证。

### 8.2 τ_split 决定触觉能影响的决策层级（最重要的旋钮）

τ_split = 0.4 时 `x_split = 0.4·ε + 0.6·A`，动作已大体确定；后 4 步是精修，一个 4 步 Euler 积分无法把样本搬到分布的另一个模态上。**也就是说 S=6 把触觉放在了只能做微调的位置。**

这对"接触力反馈式的局部伺服、抓握力调整"是合适的；对**离散决策**就不合适——而离散决策恰恰是当前触觉数据集的形态（`Xense/bottle-sorting-0810` 的 prompt 是"重的放远箱、轻的放近箱"，这个二元决策在高 τ 段就做完了，而那段是触觉盲的）。

| S | τ_split | 触觉负责 | 触觉能影响的层级 |
|---|---|---|---|
| 6 | 0.4 | 4 步 | 只有精修 |
| 4 | 0.6 | 6 步 | 部分模态选择 |
| 2 | 0.8 | 8 步 | 大部分模态选择 |

**建议第一批从 S ∈ {4, 2} 扫起。** 若目标任务是称重分拣这类决策型的，S=6 可能从结构上就注定测不出触觉效果，跑了也说明不了问题。

若 S 扫到 2 仍然测不到触觉敏感度，那问题多半不在这套结构，而在"动作是否本来就能从视觉完全预测出来"——那是数据/任务属性，换任何架构都救不回来，该回头检查数据而不是继续调这里。

### 8.3 其他

1. **独立噪声导致的训练/推理分布不一致**。推理时 KV 里的 `x_split` 与正在去噪的 `x_τ` 是同一条轨迹；训练时（对齐 T-Rex）不是。这是 §7.5 A/B 的对象。
2. **RoPE 位置重复**：同一动作 index 在 action 流与 tactile 流里拿到不同位置。对齐 T-Rex，但如果 §7.3 的残差偏大，这是第一个该怀疑的地方。
3. **显存 +3.7 GB**，batch size 可能需要下调；下调后与基线的对比要保持 global batch 一致（memory 里记过 `batch_size` 是全局值这个坑）。
4. **PyTorch 导出会断**：三专家 checkpoint 无法被 `convert_jax_model_to_pytorch.py` 消化。若部署走 JAX serving 则无影响。

### 8.4 若捷径确实发生了，下一步的候选手段

按"先便宜后昂贵"排：

1. 调小 S（§8.2），最便宜，先做。
2. 触觉敏感度正则：对 shuffle 触觉下的输出差异给一个显式奖励项。直接针对失效模式，但有让模型学出"与任务无关的触觉抖动"的风险。
3. 检查数据里触觉是否真的携带视觉不可得的信息（§8.2 末段）。
4. ~~辅助预测 loss（让触觉 token 去预测某个触觉派生量）~~ ——LTP 那轮已经试过，结果是走 VL 捷径，不要重复。
