# 实验记录：触觉分支加不加，实机动作一样——到底谁在忽略触觉

**日期**：2026-08-21 起，2026-08-22 续写（第 7 节、8.6 节、第 9 节）
**状态**：A、B、C、D、F 已完成；E（实机复测）待做
**结论**：**信息在 token 里，丢在 action expert 的读出上。**
触觉 token 的重/轻标签用分块留出线性探针可读到 **0.788**（该传感器像素级上限约 0.81），
但 action expert 对它的传输是零——同一次触觉替换在两个不同 base 观测下产生的隐层扰动
`cos = −0.001 ± 0.0017`（chance 0.0044），到最终动作只剩 **0.03%** 的系统性成分。
编码器确实把整个触觉流形压了约 290 倍，**机制已确认是 BatchNorm 的 running stats
仍冻结在 ImageNet 统计量上**——但压掉的是幅度不是信息：事后重估 BN 把幅度撑开 290 倍，
可分性 0.788 → 0.742，一点没变。
另有一个独立的、确定的实机通路 bug（见第 3 节），会让触觉在推理时被完全屏蔽。

> **08-21 的阶段性结论已被 8.6 节收窄**：当时记的是"编码器塌缩，action expert
> 没有东西可忽略，L3 无法检验"。那是对**幅度**的正确描述、对**内容**的错误推论。
> 结论的完整演变见 9.0。
**硬件**：本机 RTX 5090（32 GB）；训练在远程 8×H100
**模型**：`Pi0TactileFastVitConfig`（Pi05 主干 + FastViT-T12 触觉 token 进 action_expert）
**Checkpoint**：`checkpoints/pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100/59999`（60000 步，8月19日）
**数据集**：`Xense/bottle-sorting-0810`（160 episodes / 136,959 帧 / 7 相机 / 30 fps）

---

## 1. 现象与初始猜想

实机验证时，带触觉编码器的模型和不带的模型，动作表现没有区别。
瓶子分拣任务里"重→远箱 / 轻→近箱"的判据本应来自触觉，所以这等于触觉分支没起作用。

**初始猜想**：action expert 直接忽略了 FastViT 出来的特征向量。

---

## 2. 先分层：三种"忽略"症状相同，修法完全不同

| 层级 | 含义 | 候选根因 |
|---|---|---|
| **L1** | 触觉图根本没到模型 | 客户端 / plumbing，train-inference skew |
| **L2** | 图到了，但 FastViT 输出没有判别力 | `use_running_average=True` 用 ImageNet BN running stats 套凝胶图，特征塌成常数 |
| **L3** | token 有信息，但 action expert 不看它 | **初始猜想** |

三者在实机上表现一致，必须逐层排除，不能直接跳到 L3。

---

## 3. 代码审查先行：一个不需要实验就能确认的事实（L1）

`examples/bi_flexiv_rizon4_rt/env.py:93-95`：

```python
for cam_name, img in raw_obs["images"].items():
    if "_depth" in cam_name or "tactile" in cam_name:
        continue
```

`real_env.py` 在 commit 8d5b88f 之后已经能正确读出并改名 4 路触觉，但 `env.py` 这一层在把
observation 交给 policy 之前又把所有含 `tactile` 的 key 全部丢掉。
`git log -- examples/bi_flexiv_rizon4_rt/env.py` 确认 8d5b88f 没有碰过这个文件；
类 docstring 第 58 行仍写着 "tactile cameras are silently ignored by the policy"。

**为什么这会精确产生观察到的现象**：服务端 `BiFlexivTactileInputs` 对缺失的触觉相机零填充并置
`image_mask=False`（`bi_flexiv_policy.py:173-175`），然后

- `make_attn_mask` 的 `valid_mask = input_mask[:,None,:] * input_mask[:,:,None]`（`pi0.py:48`）
  → action token 对 tactile 位置的注意力被完全屏蔽；
- `positions = sum(prefix_mask) + cumsum(suffix_mask) - 1`（`pi0.py:544`）
  → 被屏蔽的 tactile token 不推进 RoPE 位置，action token 的位置编码与"没有触觉分支"时逐位一致。

即 **`image_mask=False` 时触觉分支在数学上等价于不存在**，模型退化为普通 pi05。
和对照组行为一致是必然的，与 action expert 学没学会用触觉无关。

**待确认的时间线矛盾**：`policy.py:177-233` 的 `_audit_tactile_inputs_once`（commit ccd26e6）
在遇到 `image_mask=False` 时是直接 `raise` 的。checkpoint 写于 8月19日 19:04，
实机测试很可能跑在该 audit 落地之前。需要核对实机所用 commit，
以及服务端日志有无 `[TACTILE SERVER] ... mask=True std=...`。

---

## 4. 实验设计

| 编号 | 目的 | 方法 | 判据 |
|---|---|---|---|
| **A** | 梯度是否到过触觉分支 | 读 checkpoint `train_state` 里 tactile 的 Adam 二阶矩 `nu`；`params` 里的 `tactile_encoder/*` 与 ImageNet 预训练权重逐张量对比 | `nu ≈ 0` 或权重逐位相同 → 训练时就是死路 |
| **B** | 编码器判别力（分离 L2） | 只跑 `tactile_encoder + tactile_proj`，比较接触 / 未接触图的 1024 维 token 距离，以同状态两帧为噪声地板 | 接触/未接触距离 ≈ 噪声地板 → L2 编码器塌了 |
| **C** | 端到端因果干预（分离 L3） | 固定 noise，只换右臂触觉图，比较 action chunk；对照：mask 全 False / 纯噪声触觉 / 换右腕视觉图 | `Δ_tac ≈ Δ_mask ≈ Δ_noise ≪ Δ_vis` → L3 坐实 |
| **D** | 敏感度数值证据 | `jax.grad` 求 `d‖actions‖/d(tactile_token)` 的 Frobenius 范数，与 action / prefix token 比 | 小 2–3 个数量级 → L3 直接证据 |
| **E** | 修完后实机复测 | 去掉 `env.py` 的 tactile 过滤，确认服务端 audit 打出 mask=True，同瓶空/满各跑一次 | 分箱行为出现差异 |

**注意事项**：离线实验必须关掉 RTC（不传 `prev_chunk_left_over`）。
config 是 `enable_training_time_rtc: true`，实机走 `training_time_rtc_sample_actions`，
返回 chunk 的前 `inference_delay` 段是被冻结的上一段拷贝，会系统性压掉任何差异——
这本身也是"实机看起来一样"的一个独立嫌疑。

**服务/加载配置**：`--policy.config=pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100_h100`
`--policy.dir=checkpoints/pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100/59999`。
该 config 构造出的模型是 501 个参数叶子（其中 450 个 tactile），与 checkpoint 的
`params/_METADATA` 完全一致；norm stats 在 checkpoint 的 `assets/` 内。

**素材**（标注在 B0 中被推翻，正确状态见该节）：
- `有水_episode_000001_frame_000479_right_tactile_{0,1}.jpg`
- `无水_episode_000021_frame_000302_right_tactile_{0,1}.jpg`
- `/home/li/下载/selected_tactile_frames/右臂未抓取_episode_000021_frame_000100_right_tactile_{0,1}.jpg`

B 最终没有依赖这几张 jpg 下结论：改为直接从 `Xense/bottle-sorting-0810` 按
`observation.state[19]`（right_gripper.pos）在**同一 episode 内**分出张开/闭合两类，
每类 12 帧。跨 episode 比较会被凝胶漂移和光照淹没，单帧对比也没有类内散度做参照。

---

## 5. 实验 A：梯度是否到过触觉分支

脚本：`scripts/audit_tactile_weights.py`（只偏序恢复 tactile 子树，约 60 MB，不动 12 GB 主干）

```
python scripts/audit_tactile_weights.py \
    --checkpoint-dir checkpoints/pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100/59999
```

### A1 Adam 矩

| 组 | leaves | `mu` rms | `nu` rms |
|---|---|---|---|
| `tactile_encoder` | 288 | 4.77e-07 | 8.30e-10 |
| `tactile_proj` | 2 | 8.88e-07 | 2.34e-10 |
| 主干参考（`action_in/out_proj`, `time_mlp_*`） | 8 | 4.69e-05 | 1.28e-06 |

**全零 `nu` 叶子：0 / 290。**

结论：梯度确实流过触觉分支，不存在"完全断路"。
但量级明显偏小——按 `sqrt(nu)` 估计的典型梯度幅度，`tactile_encoder` 约为主干的 1/40，
`tactile_proj` 约为 1/74。

### A2 权重相对 ImageNet 初值的位移

```
checkpoint tactile_encoder leaves=448  pretrained=448  matched=448
bit-identical to pretrained: 160 / 448
changed:                     288 / 448
relative movement: median=6.08e-03  mean=1.64e-02
largest: stages_2/blocks_4/token_mixer/layer_scale/gamma  rel=3.04e-01
tactile_proj/kernel  shape=(1024,1024)  std=3.12e-02
```

160 个逐位未变的张量**全部**是 BatchNorm 的 `mean` / `var`——这与
`fastvit.py` 里硬编码的 `use_running_average=True` 一致，属预期行为，不是 bug 本身，
但意味着 **BN 的 running stats 至今仍是 ImageNet 的统计量**，从未适配凝胶图。

288 个可训练张量全部发生了位移，中位相对变化 0.6%。

**A 的结论：训练侧没有断路，编码器确实被训练了，但训练得很轻。L1 的"梯度从没到过"被排除。**

---

## 6. 实验 B：编码器判别力

脚本：`scripts/audit_tactile_encoder.py`（只加载 `tactile_encoder` + `tactile_proj`，约 30 MB）

### B0 先修正实验前提：给的图标注是错的

用 `observation.state[19]`（right_gripper.pos）核对那几帧：

| 帧 | 标注 | right_gripper | right_tcp_z | 实际状态 |
|---|---|---|---|---|
| ep1 f479 | 有水 / 有接触 | **0.4716** | 0.2255 | 夹爪闭合，**在抓** |
| ep21 f302 | 无水 / 没有接触 | **0.4719** | 0.2261 | 夹爪闭合，**也在抓** |
| ep21 f100 | 右臂未抓取 | **1.0000** | 0.1276 | 夹爪张开，真的没抓 |

前两帧的夹爪开度差 0.0003、TCP z 差 0.6 mm，**都是抓取状态**。
所以这一对不是"接触 / 未接触"，而是"抓满瓶 / 抓空瓶"（重量差），而且跨 episode。
直接拿它们做对比，差异会被 episode 间的凝胶/光照漂移淹没——
实测 `mean|contact − no_contact|` = 3.75，`mean|no_contact_A − no_contact_B|` = 3.70，两者无法区分。

### B1 数据里到底有没有触觉信号——有，而且很强

沿 episode 21 均匀取 25 帧，测每帧与首帧的 `mean|d|`：

| frame | right_gripper | mean\|d vs f0\| |
|---|---|---|
| 0–241 | 1.0000（张开） | 0.000 → 0.017 |
| 282–402 | 0.4720（闭合） | **3.44 → 3.50** |
| 443–483 | 0.72 → 1.00（松开） | 1.75 → 1.64 |
| 644–765 | 0.4631（再次闭合） | **2.23 → 2.42** |

抓取瞬间信号跳变约 200 倍。**触觉数据本身是好的，抓/不抓一目了然。**

### B2 同 episode 内的判别力（正确的对比）

每个 episode 取 12 帧张开（未接触）+ 12 帧闭合（接触），比较类间距离与类内散度：

| episode | 输入 类内(闭) | 输入 类间 | 输入分离度 | token 类内(闭) | token 类间 | **token 分离度** | 类均值 1−cos |
|---|---|---|---|---|---|---|---|
| 21 | 0.004 | 0.013 | 3.25× | 0.0631 | 0.0637 | **1.011×** | 1.32e-06 |
| 1 | 0.004 | 0.010 | 2.50× | 0.0711 | 0.0841 | **1.182×** | 3.22e-06 |

对照（编码器的动态范围）：真实触觉图 vs 纯随机噪声图 → `1−cos ≈ 0.072`，token L2 ≈ 8.2。

也就是说编码器对**粗暴**的输入变化有反应（L2 8.2），
但整个真实触觉流形被压缩进 L2 ≈ 0.06 的一个点上（小 130 倍），
其中"抓 / 没抓"这个本应最容易的区分只占 1.01×，低于帧间抖动。

### B3 padding 是帮凶但不是主因

触觉原图是 400×700，`resize_with_pad` 到 224×224 后只有 224×128 是内容，
**43% 的像素是纯黑 padding**，纵向分辨率被压掉 3.1 倍。换掉 padding 后：

| 预处理 | pad 占比 | token 分离度 | 类均值 1−cos |
|---|---|---|---|
| `resize_with_pad`（当前） | 0.43 | 1.011× | 1.32e-06 |
| 直接拉伸到 224×224 | 0.00 | 1.428× | 8.00e-06 |
| 中心裁 400×400 再缩放 | 0.00 | 1.524× | 4.02e-05 |

去掉 padding 让判别力提升约 30 倍（按 `1−cos`），但绝对水平仍然是"分不开"。
**padding 值得修，但不是根因。**

### B 的结论

**L2 成立：编码器在真实触觉输入上塌缩了。**
输入端 3.3 倍的类间可分性，经过 FastViT + `tactile_proj` 后只剩 1.01 倍。
action expert 拿到的 4 个 token 在整个 episode 里近乎恒定（cos > 0.999997），
**所以"action expert 忽略了触觉特征"这个原始猜想目前无法证伪也无法证实——
它确实没有可用的东西可看。**

塌缩的候选机制（尚未逐一验证）：
1. BN running stats 冻结在 ImageNet 统计量上（A2 已确认 160 个 BN buffer 逐位未变），
   凝胶图的分布与 ImageNet 差异大，激活被推离 BN 的工作点；
2. 梯度太小（A1：约为主干的 1/40 ～ 1/74），60k 步不足以把编码器推成判别性表征；
3. 模仿学习损失对触觉几乎没有压力——轨迹大部分阶段靠视觉就能拟合，
   只有"重/轻分箱"那一下需要触觉，监督信号极稀疏；
4. `resize_with_pad` 丢掉 43% 有效面积 + 3.1 倍纵向下采样（B3，次要因素）。

---

## 7. 实验 C / D：端到端因果干预与逐级衰减（2026-08-22）

工具：`scripts/tactile_counterfactual_probe.py` + `test/tactile_counterfactual/`
（离线反事实探针，本节所有数字都出自它）。
配置：`configs/probes/water_weight_counterfactual_inline_ep0based.yaml`。

```bash
PYTHONPATH=.:src:packages/xense-client/src \
    python scripts/tactile_counterfactual_probe.py \
    --config configs/probes/water_weight_counterfactual_inline_ep0based.yaml
```

### 7.0 设计：2×2，四个条件共用同一份固定噪声

每一对 (满瓶帧, 空瓶帧) 跑四个条件：

| 条件 | base 观测（RGB + state + 动作历史） | 触觉 |
|---|---|---|
| `F_F` | 满 | 满（恒等，不换） |
| `F_E` | 满 | **空** |
| `E_E` | 空 | 空（恒等，不换） |
| `E_F` | 空 | **满** |

这个 2×2 是后面 7.5 节那个决定性判据的前提：**同一对触觉图像**被替换两次，
一次在 base=满 下、一次在 base=空 下，于是 nuisance 成分完全相同。

**第 4 节担心的 RTC 冻结前缀不适用**：`inference_mode: rtc`，但
`trace_sampler.py` 传的是 `prev_chunk_left_over=None` / `inference_delay=0`
（首次推理的 dummy 前缀），不存在"返回 chunk 前段是上一段拷贝"的压制。

### 7.1 前提核对（全部通过，否则后面的数都不作数）

- **帧索引偏移**：`sampled_frames.yaml` 是 1-based。对着 parquet 逐帧核对过，
  0-based 偏移版（`_ep0based`）才是对的——未偏移版有 2~6% 的帧根本不存在。
  偏移后 water/no_water 帧 **100%** 满足 `right_gripper.pos < 0.48`，
  not_grasped 帧 **0%** 满足。
- **重/轻标签有效**：松开夹爪时 `right_tcp.x` 分别是 0.918 / 0.457，100% 分离。
- **确定性**：`F_F` 重跑与首次 max abs diff = **0.0**（逐位相同）。
  所以所有 delta 都是真实因果效应，不是数值抖动。
- **相位混杂被否定**：担心过"探测帧太晚、决策已经做完"。数据不支持——
  早期帧（手臂尚未提交，n=299）效应 0.171%，后期帧 0.183%，基本一致。

### 7.2 实验 C：换掉触觉，动作几乎不动（500 对 × 4 条件 = 2000 次推理）

| 指标 | 数值 |
|---|---|
| 触觉换掉后 `final_action` 变化 RMS | **5.9e-4**（median 5.8e-4, max 1.2e-3） |
| 相对于动作本身（RMS 0.342） | **0.17%**（p95 0.20%，最坏 0.37%） |
| 单个元素最大变化 | 0.026（动作范围 ±1.97） |
| 相对于「重/轻」行为差异（RMS 0.103） | 单对 0.57%，**系统性成分 0.03%** |

作为标尺：`R_tcp.x`（决定远箱/近箱的那一维）的重/轻行为差异是 **0.385**，
触觉造成的偏移是 **0.000029** —— **0.01%**。

**三条独立证据说明这是「被忽略」，不是「效应小但真实」：**

1. **方向完全随机**。`cos(Δ_F, −Δ_E)` = −0.0000（t = −0.02）。
   如果触觉编码了一致的"重 vs 轻"方向，两者应当正相关。
   500 对平均后 delta 缩小到 0.045 倍 —— 精确等于 1/√500，独立噪声的理论值。
2. **扰动打在死维度上**。动作是 20D（18 TCP + 2 gripper）零填充到 32D。
   填充维 20–31 模型输出 ~0.001（等于零），却收到 4.7e-4 的触觉扰动 ——
   是真实维度所受扰动（5.7e-4 ~ 9.6e-4）的 **80%**。触觉进来后是近似各向同性的扰动。
3. **AdaRMS 通路完全没走**。150 对里 tactile 行的 `adarms_cond` 全部逐 bit 相同
   （`embed_suffix` 强制置零，见 7.4）。触觉只能通过 54 个 suffix token 里的
   4 个 tactile token 走 attention。

### 7.3 信号确实存在（像素级复核，独立于编码器）

用 32×32 原始右手触觉像素 + 逻辑回归，**按 episode 分组交叉验证**（防止记忆 episode）：

| 探测窗口 | 触觉像素 | 对照：手臂位置 `right_tcp.x` |
|---|---|---|
| 抓取早期 (phase<0.3) | **88.1% / AUC 0.941** | 44.2% / AUC 0.446（随机） |
| 抓取后期 (phase>0.7) | 93.8% / AUC 0.967 | 100% / AUC 1.000 |

关键在早期窗口：那时手臂还没朝任何一个箱子走（Cohen d = −0.10，本体感觉完全无法区分），
**只有触觉知道答案，而且线性可读**。但在这批帧上触觉换掉造成的动作变化是 0.169%，
和全体平均（0.174%）没有区别。

### 7.4 实验 D：逐级衰减（50 对，打开 `save_tactile_tokens` / `save_fastvit_features`）

运行 `outputs/tactile_counterfactual/20260822_070615`，全部 sanity gate 与 500 对运行一致。

| stage | signal | delta | relative |
|---|---|---|---|
| 1. FastViT features (4×1024) | 0.4720 | 2.225e-03 | 0.47% |
| 2. `tactile_proj` tokens (4×1024) | 0.6582 | 2.268e-03 | **0.34%** |
| 3. action hidden, 去噪 step 0 | 0.4151 | 1.461e-03 | **0.35%** |
| 4. action hidden, 去噪 step 9 | 0.5432 | 4.473e-03 | 0.82% |
| 5. `v_t`, step 9 | 1.0300 | 7.504e-03 | 0.73% |
| 6. final action | 0.3368 | 5.904e-04 | 0.18% |

**这一步推翻了"attention 把 tactile token 压到零权重"的假设。**
token 变 0.34%，hidden 就变 0.35%（衰减因子 **1×**），到 step 9 反而放大到 0.82%。
幅度是足额传进去的。

同时排除的其它候选：

- **`W_K` / `W_V` collapse**：checkpoint 里 tactile 位置的 `1+scale` ≈ 1.0，没有塌缩。
- 但注意 tactile token 自身的 residual `gate` rms ≈ **0.03** —— 它们的残差流几乎冻结。
- `adarms_cond` 在 4 个 tactile 行上 rms = **0.000000**（`embed_suffix` 强制置零），
  在 50 个 action 行上是 0.106570。tactile token 拿到的调制只有学到的 Dense bias。
- 四路 token 变化幅度接近：cam0=0.3% cam1=0.3% cam2=0.3% cam3=0.4%
  （cam0/1 = 左手，cam2/3 = 右手，即拿瓶子的那只手）。

### 7.5 决定性判据：一致方向存不存在，以及能不能跨 base 复用

幅度传过去了，那丢的是什么？丢的是**结构**。三个测试，一个比一个严。

**(a) 跨 pair 一致方向**（`_direction.py` / `_transmit.py` part B）。
每一对的替换都产生一个差异向量；若该级存在一致的"重 vs 轻"编码，这些向量应当互相对齐。
用白化（kernel-ridge，λ=0.1）估计方向、留一法打分，n=100：

| stage | dim | 白化 LOO cos | chance | 倍数 |
|---|---|---|---|---|
| FastViT features | 4096 | **+0.1203** (t=+104.7) | 0.0156 | **7.7×** |
| `tactile_proj` tokens | 4096 | **+0.1171** (t=+95.5) | 0.0156 | **7.5×** |
| action hidden step 0 | 51200 | +0.0010 (t=+0.8) | 0.0044 | 0.2× |
| action hidden step 5 | 51200 | −0.0025 (t=−2.1) | 0.0044 | −0.6× |
| action hidden step 9 | 51200 | +0.0008 (t=+0.7) | 0.0044 | 0.2× |
| final action | 1600 | −0.0001 | 0.0250 | 0.0× |

**encoder 输出里有一个稳健、可复用的重/轻方向，过 `tactile_proj` 保留 97%
（0.1203 → 0.1171），进 action expert 就掉到 chance。**

**(b) 跨 base 迁移**（`_transmit.py` part C）。在 base=满 的样本上拟合方向，
到 base=空 的留出样本上打分：

| stage | 迁移 cos | chance | 倍数 |
|---|---|---|---|
| FastViT features | +0.0317 (t=+2.7) | 0.0156 | **2.03×** |
| `tactile_proj` tokens | +0.0309 (t=+2.5) | 0.0156 | **1.98×** |
| action hidden step 0 | −0.0002 | 0.0044 | −0.06× |
| action hidden step 9 | +0.0011 | 0.0044 | 0.24× |
| final action | +0.0023 | 0.0250 | 0.09× |

**(c) base 不变性——最干净的一个**（`_transmit.py` part A）。
利用 7.0 的 2×2：同一对触觉图像替换两次，

```
dF = h(base=满, tac=满) − h(base=满, tac=空)
dE = h(base=空, tac=满) − h(base=空, tac=空)
```

触觉输入完全相同，只有 base 不同，所以 nuisance 成分被抵消掉了：

| stage | dim | cos(dF, dE) | t | chance | ‖dE‖/‖dF‖ |
|---|---|---|---|---|---|
| FastViT features | 4096 | **+1.0000** | — | 0.0156 | 1.000 |
| `tactile_proj` tokens | 4096 | **+1.0000** | — | 0.0156 | 1.000 |
| action hidden step 0 | 51200 | **−0.0010** | −0.6 | 0.0044 | 1.004 |
| action hidden step 9 | 51200 | +0.0027 | +1.6 | 0.0044 | 1.002 |
| final action | 1600 | +0.0040 | +1.0 | 0.0250 | 1.025 |

前两行 cos 精确等于 1 是 gate（token 只由触觉图算出，与 base 无关），说明测量链路正确。
到 action expert 的隐层就是 0：**完全相同的触觉变化，在不同 base 下把隐层推向互不相关的方向**，
而幅度一点没丢（‖dE‖/‖dF‖ = 1.00）。95% 上界 cos < 0.003，即触觉扰动里
**不到 0.3% 的幅度是触觉输入的可复用函数**。

### 7.6 测量地板与正对照（`_control.py`）

隐层以 float16 存盘，先确认这不是精度问题：

```
float16 ulp of stored hidden values : 1.713e-04
quantisation noise rms (两数组相减) : 6.992e-05
tactile-swap delta rms              : 1.401e-03
noise / delta (幅度)                : 5.0%
=> 对 cos 的最大衰减                : 0.2%
```

真 cos 若是 0.5 会被测成 0.499。**不是测量地板。**

正对照用同样的 2×2 换 base（同样的隐层张量、同样的 float16）：

| 扰动 | 相对幅度 | 跨另一因子的 cos |
|---|---|---|
| tactile swap（step 0） | 0.35% | **−0.0010** |
| base swap（step 0） | 131.49% | **+1.0000** |

正对照通过，但它是**平凡通过**的——base 效应比触觉大 375 倍，cos≈1 是算术必然。
真正的论据是上面那个量化上界。

### 7.7 C / D 的结论

```
凝胶图                     差分线性探针 0.88（分块留出，见 8.4）
  ↓ FastViT + BN            幅度压缩 ~290×（机制见 8.6）
tactile token              只在 0.15%~0.34% 的球内变化，但标签仍可读到 0.788（见 8.6）
  ↓ tactile_proj            一致方向保留 97%，不是瓶颈
  ↓ action expert readout   幅度全传（1.0×），base-invariant 成分 = 0    ★ 真正的断点
final action               系统性成分 0.03%
```

**L3 成立，而且是主要故障。** attention 没有被堵住，`W_K`/`W_V` 没塌，
tactile token 被足额读入——只是读进去的东西对触觉内容没有可复用的依赖。
这正是一条**从没被训练用起来**的输入通路的样子。

**一个诚实的保留**：没有等幅度对照来排除"这个 stack 对任何 ~0.35% 的扰动
都会产生 base 相关的随机旋转"（数值混沌）。这只影响 7.5 的定位表述，
不影响主结论——主结论站在 7.2 的行为层结果上，与隐层分析无关。
要做这个对照需要给 runner 加一个条件：扰动某个非触觉输入到匹配的 hidden delta 量级。

分析脚本随运行目录留存：`outputs/tactile_counterfactual/20260822_060037/_*.py`
（`_analyze.py` `_analyze_hidden.py` `_analyze_phase.py` `_probe_tactile_signal.py`
`_final_scale.py` `_extract_*.py`）与 `20260822_070615/_*.py`
（`_attenuation.py` `_direction.py` `_transmit.py` `_control.py`），
中间产物 `_*.npz` 可直接复用。

---

## 8. 重训前的数据审计（2026-08-21 追加）

目标实验：两个外观完全相同的瓶子，有水→远箱、无水→近箱，
两类之间**唯一**的差别是夹取时的触觉图。用它来验证触觉 encoder 是否真的影响 action。
前提是标签在触觉之外无处可查。逐通道审计如下。

脚本：`scripts/audit_bottle_sorting_leakage.py`

### 8.1 标签与录制结构

按"松开夹爪时右臂 TCP x"二分（远箱 x≈0.93，近箱 x≈0.45）：
**71 个有水（远） / 89 个无水（近）**，多数类基线 **0.556**。

**录制是强块状的**：25 个连续块，最大连续 30 个同类 episode。

```
ep  0- 39: 1101111111110111111100000000000000000000
ep 40- 79: 0000000000111111111100000000001111111111
ep 80-119: 1101101100110011111100000000000000000000
ep120-159: 0100000000111011111000000000001111111111
```

这决定了**留出集必须按块切，不能按 episode 随机切**——相邻 episode 共享光照、
凝胶状态和装置漂移，随机切会让任何通道都"看起来有效"。

### 8.2 prompt 没有泄漏

`meta/tasks.parquet` 只有一条 task 字符串，全部 160 个 episode 共用：
`"Pick up the bottle, and place it in the far bin if it is heavy, or in the near bin if it is light."`

### 8.3 逐通道线性探针（grasp+15 帧，绝对图像）

| 通道 | 随机划分 | **按块划分** |
|---|---|---|
| state（20 维本体感觉） | 0.731 | 0.481 |
| head 相机 | 0.694 | 0.544 |
| right_wrist | 0.681 | 0.487 |
| right_tactile_0 | 0.744 | 0.613 |
| tactile_0 + tactile_1 | 0.750 | 0.581 |
| 基线 | 0.556 | 0.556 |

**随机划分下每个通道都"有效"，这全是块状录制的假象。**
按块划分后视觉和本体感觉都回落到基线——
**这是好消息：视觉和本体感觉不会跨 session 泄漏标签，实验设计成立。**

补充：本体感觉的可分性随时间上升（抓取前 Cohen d≈0.7 且最大维是**没参与动作的左臂** z，
属装置漂移；grasp+20 帧起 `rtcp_z` 升到 d≈1.2；grasp+90 帧 d≈3.3 —— 那时轨迹已经分叉，属预期）。

### 8.4 差分触觉探针：信号确实存在

绝对图像的探针被 session 外观支配。改用
`|I(grasp+k) − I(夹爪张开时的参考帧)|`，抵消凝胶/光照漂移、只留形变：

| 采样时刻 | tactile_0 | tactile_1 | 两路合并（按块划分） |
|---|---|---|---|
| grasp+10f (0.33s) | 0.656 | 0.494 | 0.594 |
| grasp+25f (0.83s) | 0.781 | 0.556 | 0.750 |
| **grasp+45f (1.5s)** | 0.806 | 0.581 | **0.881** |

**结论：一个线性探针在严格分块留出下能达到 88%。**
重量信号真实存在、可解码，且：

1. **必须是差分表征**。绝对单帧只有 0.58（≈基线），差分能到 0.88。
   当前架构每步只喂一张触觉图、无历史无参考——
   **这很可能是编码器塌缩最深层的原因**：它拿不到"未形变基准"，
   而每个 session 的凝胶外观又在漂移。
2. **信号随抓取后时间增强**：0.33s 时才 0.64，1.5s 时 0.88。负载是在提起过程中建立的。
3. **tactile_0（顶）扛了绝大部分信号**，tactile_1（底）接近基线。

---

---

## 8.5 差分输入的实现与端到端标定（2026-08-21）

实现落在：`transforms.TactileDifference` / `transforms.InjectTactileReference` /
`transforms.fit_square`、`bi_flexiv_policy.BiFlexivTactileDiffInputs`、
`config.LeRobotBiFlexivTactileDiffDataConfig`、`scripts/compute_tactile_refs.py`、
config `pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_diff_h100`。

**参考帧放在 repack、差分放在 data_transforms**：推理时 `policy_config` 只跑
`data_transforms + model_transforms`、**不跑 repack**。所以按 episode 查表的参考帧注入
只能在 repack（仅训练），差分本身在 data_transforms（训练/推理共用）。
推理侧参考帧由客户端在 `env.reset()` 后抓取并随每帧发送。

**首帧合格性**：160/160 个 episode 的 frame 0 双夹爪均为 1.0000（张开），
`compute_tactile_refs.py` 会在写盘前强制校验，不合格直接拒绝。

**参考帧存 CHW**：注入的参考帧和真实相机一起进 `data["images"]`，
而 `_decode_bi_flexiv` 会对其中每一项做 CHW→HWC 转置。存 HWC 会被转坏。

**增益标定**（过完整管线，center_crop，分块留出 48×48 线性探针）：

| gain | std@grasp+45 | 饱和像素 | 探针 tac2 | 探针 tac2+3 |
|---|---|---|---|---|
| 1 | 0.066 | 0% | 0.738 | 0.887 |
| 4 | 0.201 | 2.2% | 0.738 | 0.881 |
| **8（默认）** | **0.270** | **4.4%** | 0.775 | **0.881** |
| 16 | 0.362 | 7.6% | 0.819 | 0.863 |
| 32 | 0.492 | 16.6% | 0.819 | 0.838 |

**0.88 的可分性完整穿过了整条管线。** 增益过高反而伤：clip 会压掉形变**幅度**，
而幅度正是"满瓶 vs 空瓶"的判据。gain=8 是拐点。

**顺带修掉的第二个客户端 bug**：`main.py` 向 `BiFlexivRizon4RTEnvironment` 传
`tactile_camera_mapping=`，而 `env.py` 的 `__init__` 不接受该参数——客户端启动即 TypeError。
同样是 8d5b88f 改了 main.py / real_env.py 却漏改 env.py。

**已验证**：参考帧与当前帧相同时差分严格为 0；客户端漏发参考帧时报错而非静默零填充；
`fit_square` 三种模式对已经是 224×224 的输入幂等（客户端和服务端各调一次也不会二次缩放）。

---

## 8.6 实验 F：BN 统计重估 —— 塌缩的机制确认，以及一个反转（2026-08-22）

第 6 节把编码器塌缩列为最靠前的技术根因，候选机制之一是
「BN running stats 冻结在 ImageNet 统计量上」（A2 已确认 160 个 buffer 逐位未变，
`tactile_encoders/fastvit.py` 有 4 处硬编码 `use_running_average=True`）。
这条不需要重训就能判定：在真实触觉帧上重估 BN 的 running stats，再量一次判别力。

脚本：`scripts/audit_tactile_bn.py`（自包含）。做法是把 `fastvit.py` 的模块全局 `nn`
换成一个只改写 `BatchNorm` 的代理，让 80 个 BN 走 batch-统计模式——**不动源码、
不做任何梯度步，只有那 160 个 buffer 变**。linen 路径与 nnx 路径的前向做了
逐位对齐 gate（`max|diff| = 0.000e+00`）。

```bash
PYTHONPATH=.:src python scripts/audit_tactile_bn.py \
    --probe-config configs/probes/water_weight_counterfactual_inline_ep0based.yaml \
    --calib-frames 384 --pairs 300
```

标定用 384 帧 × 4 路 = 1536 张图，取自 158 个 episode，**评测 episode（1、21）被留出**。
完整输出存在 `outputs/tactile_counterfactual/_bn_restat_20260822.log`。

### 8.6.1 ImageNet 统计确实离触觉数据很远

```
mean: 中位相对偏移 0.465,  p90 1.610,  max 44.802
var : 中位相对偏移 0.360,  p90 0.544,  max  1.987
```

重估之后编码器的动态范围整个撑开：

| 指标（右手上传感器 cam2） | BEFORE | AFTER | 倍数 |
|---|---|---|---|
| 重/轻 token 扰动 `‖Δt‖/‖t‖` | **0.154%** | **44.6%** | **×290** |
| 抓/没抓 类均值 1−cos（ep1 cam2） | 6.6e-07 | 3.6e-02 | ×5×10⁴ |
| 真实触觉 vs 随机噪声图 1−cos | 0.072 | 0.85 | ×12 |

注意 BEFORE 那一列的 1−cos 是 1e-7 量级，**已经在 float32 的分辨极限上**
（有几个直接读到 0.0）。第 6 节记的 1.32e-6 还是高估——真实塌缩比当时测到的更极端。
而"真实触觉 vs 噪声"这个对照值 0.072，与 6.2 节独立测到的 0.072 完全一致，
说明两次测量是同一套东西。

**结论 1：第 6 节猜的机制是对的。ImageNet BN 统计就是那 ~130× 压缩的原因。**

### 8.6.2 反转：压掉的是幅度，不是信息

类均值距离太粗——下游 action expert 对 token 的读出是线性的（`W_V`），
所以决定性的问题是**线性可分性**，而不是类均值离得远不远。
用对偶岭回归探针，按 episode 分块留出（训练 ep<66 / 测试 ep≥66，测试集 n≈300）：

| cam | 位置 | BEFORE | AFTER |
|---|---|---|---|
| tactile_0 | 左手上 | 0.570 | 0.543 |
| tactile_1 | 左手下 | 0.530 | 0.493 |
| **tactile_2** | **右手上** | **0.788** | 0.742 |
| tactile_3 | 右手下 | 0.709 | 0.603 |

（chance = 0.500；λ=1 / 0.1 / 0.01 三档，上表取 λ=1，另两档见 log）

类间/类内分离比：BEFORE 0.62–2.52，AFTER 0.63–1.39——**没变**。
白化 LOO cos（cam2）：2.06× chance → 1.99× chance——**也没变**。

**幅度撑开了 290 倍，可读性一点没提高。**（AFTER 略降是因为事后换统计量，
让训练好的权重处在它没见过的分布上。）撑开是各向同性的：信号和 nuisance 一起放大。

**结论 2：BN 不是信息瓶颈。**

> 附带说明：本节还跑了一个"抓/没抓跨 episode 线性探针"（训 ep1 测 ep21 及反向），
> 但每类只有 12 帧、n=24，结果在 0.21~0.96 之间乱跳，**功效不足，不作为证据**。
> 上面那张分块留出表（n≈300）才是可依赖的。

### 8.6.3 真正重要的那个数：0.788

**触觉 token 里，重/轻标签本来就以 0.788 的分块留出线性可分度存在着。**

对照 8.4 节同一个传感器的像素级数字：绝对帧 0.613，差分帧 0.806。也就是说——
**训练好的编码器是在原始绝对像素之上*增加*了判别力（0.613 → 0.788），不是在破坏它。**

（这两个数来自不同特征空间和不同帧集合，不是严格受控对比，方向可信但幅度别当精确值。
0.788 这个数本身是干净的：n≈300，λ=1 与 λ=0.1 分别是 0.788 / 0.712。）

### 8.6.4 第 6 节的 L2 判决需要收窄

> ~~「编码器塌缩了，action expert 没有可用的东西可看，所以 L3 无法检验」~~
>
> **「编码器把幅度压了约 290 倍，但内容保住了 —— 0.788 已接近该传感器 ~0.81 的像素上限。」**

L3 因此是可检验的，而第 7 节已经检验完了：action expert 对这 0.788 的传输是零。

### 8.6.5 BN 仍然要修，但理由变了

从「恢复信息」变成「**让梯度和 attention 能动**」：

token 的重量变化只占范数的 0.15%，意味着 action expert 收到的实际上是
「常数 + 微扰」—— attention logits 几乎不动（贡献退化成一个学到的常数偏置），
而对判别方向的梯度相对于那个常数是可忽略的。
**这很可能就是 A1 测到触觉分支梯度只有主干 1/40 ~ 1/74 的直接原因。**
撑到 ~45% 之后这条路才通。

**具体怎么修：换 GroupNorm，不要在训练好的 checkpoint 上重估 running stats。**

1. 事后换统计量是 off-distribution 的，8.6.2 的 AFTER 那一列就是证据；必须在训练时就正确。
2. 每张触觉图有 43% 是 `resize_with_pad` 的纯黑边（6.3 节），
   **这些常数像素会主导 batch 统计量**。也就是说第 9 节的第 3 项（去 padding）
   和第 4 项（BN）是耦合的，要一起改。
3. GroupNorm 还顺带消掉训练/推理的统计量不一致，以及对 batch 组成的依赖。

---

## 9. 结论与后续（2026-08-22 更新）

### 9.0 结论是怎么变的

| 时间 | 当时的判断 | 被什么推翻/收窄 |
|---|---|---|
| 初始 | L3：action expert 忽略了触觉特征 | —— |
| 08-21（第 6 节） | L2：编码器塌缩，L3 无法检验（"没东西可看"） | —— |
| 08-22（第 7 节） | L3 确认：token 有一致方向，action expert 传输为零 | 反事实探针 |
| 08-22（8.6 节） | L2 收窄为「幅度塌缩」；**信息一直在 token 里（0.788）**，L3 是主要故障 | BN 重估 + 线性探针 |

一句话：**初始猜想是对的，只是当时没有工具证明它。**

### 9.1 已确定

1. **训练侧通路是活的**（A）。梯度流过触觉分支，288 个可训练张量全部位移。
   但梯度幅度只有主干的 1/40 ~ 1/74。
2. **编码器把幅度压了约 290 倍**（B、8.6.1）。机制确认是
   **BN running stats 冻结在 ImageNet 统计量上**（mean 中位偏移 0.47，p90 1.61，max 44.8）。
3. **但信息没丢**（8.6.2、8.6.3）。触觉 token 的重/轻标签分块留出线性可分度 **0.788**，
   高于原始绝对像素的 0.613，接近差分像素的 0.806。
4. **action expert 不读它**（第 7 节）。幅度足额传入（token 0.34% → hidden 0.35%，
   增益 1.0×），但 base-invariant 成分为零（`cos(dF,dE) = −0.001 ± 0.0017`，chance 0.0044），
   到最终动作只剩 **0.03%** 的系统性成分，`R_tcp.x` 上是 **0.01%**。
5. **实机通路曾被切断**（第 3 节，代码级确定，已修）。这意味着最初那次
   「带触觉/不带触觉实机一样」的观察**本身不含信息**——但结论已不依赖它，
   第 7 节是在触觉确实进了模型的前提下独立复现的。
6. **数据本身没问题**（B1、8.4）。抓取瞬间触觉信号跳变约 200 倍；
   分块留出下差分像素探针 0.88。

### 9.2 已排除的假设

| 假设 | 证据 |
|---|---|
| 梯度从没到过触觉分支 | A1：290 个叶子无一 `nu ≈ 0` |
| attention 权重 ≈ 0，tactile token 被压没 | 7.4：token 0.34% → hidden 0.35%，增益 1× |
| `W_K` / `W_V` collapse | 7.4：tactile 位置 `1+scale` ≈ 1.0 |
| 探测帧太晚、决策已做完（相位混杂） | 7.1：早期帧 0.171% vs 后期 0.183% |
| float16 存储精度导致隐层 cos 测不出来 | 7.6：对 cos 的最大衰减 0.2% |
| 视觉/本体感觉跨 session 泄漏标签 | 8.3：分块划分后全部回落到基线 |
| BN 是信息瓶颈 | 8.6.2：重估后幅度 ×290，可分性 0.788 → 0.742 |

### 9.3 后续动作（按杠杆重排）

原表的排序建立在「编码器没东西可看」之上。8.6 节把那个前提改了，所以重排：

| # | 动作 | 理由 | 状态 |
|---|---|---|---|
| 1 | **辅助损失，两个头**：头 A 挂在 `tactile_proj` 之后的 4 个 token 上预测重/轻；**头 B 挂在 action expert 的 action-token hidden 上**，同样预测重/轻 | 瓶颈已定位到传输。只加头 A 不够——token 可以很有判别力而 readout 依然不传，这正是第 7 节测到的现象。头 B 直接对坏掉的那一级施压。权重取 0.1×flow loss 量级 | 待做 |
| 2 | **BN → GroupNorm**（连带去掉 `resize_with_pad`，改中心裁或拉伸） | 8.6.5：不是为了恢复信息，是为了让 token 不再是「常数 + 0.15% 微扰」，梯度和 attention logits 才动得起来。两者耦合，必须一起改，且训练/推理两侧同步 | 待做 |
| 3 | **打破 session 捷径**：(a) 按录制块切 train/val；(b) head/wrist RGB 强 per-episode 光度增广；(c) **RGB modality dropout**——以 p≈0.3 把 head+wrist 的 `image_mask` 置 False | 8.1/8.3：25 个连续块、最大 30 个同类连续，随机划分下 head 相机 0.694、state 0.731。模型训练时看到的正是随机划分那一侧，**模仿损失完全可以靠 session 外观压到底，永远不需要触觉**——这才是 A1 那个 1/40 梯度的根因。(c) 通路现成（`bi_flexiv_policy.py:71-90`），不用改架构；但别全程 mask，机械臂需要视觉去够瓶子，建议只 mask wrist 或只在 grasp 之后的窗口 mask | 待做 |
| 4 | **触觉改差分输入**（`..._fastvit_diff_h100`，8.5 节已实现并标定） | 仍然值得做（8.4：该传感器 0.613 → 0.806，两路合并 0.88），但边际收益比 8.4 的像素级数字暗示的小——编码器已经自己把绝对帧做到了 0.788。从「必做前提」降为「值得做」 | 已实现，未训练 |
| 5 | 触觉分支单独调高学习率（10–40×） | A1：梯度只有主干的 1/40 ~ 1/74。需要新增 param-group（optax `multi_transform` + nnx 路径过滤，目前 `TrainConfig` 只有 `freeze_filter`）。低风险，但单独做没用 | 待做 |
| 6 | 架构层：让 tactile token 参与 adaRMS、或允许 attend prefix | 7.4：`ar_mask` 首 token 为 True（块边界，看不到 prefix）、`adarms_cond` 强制为 0、residual `gate` rms ≈ 0.03 —— 这 4 个 token 是上下文无关的静态向量。但在 1–3 之前动架构会毁掉归因 | 待做 |
| 7 | A/B 对照：同 seed / 同数据 / 同步数，训「有触觉」与「触觉 mask 全 False」 | 唯一能干净归因到触觉的对照。mask=False 时触觉在数学上等价于不存在（第 3 节） | 待做 |
| 8 | 实机复测（原实验 E） | 第 3 节的通路已修；但在 1–3 落地前实机看不出东西 | 待做 |

### 9.4 验收指标：重训完直接用探针量，不用等实机

反事实探针现在可以当**任何一次重训的验收测试**用，50 对约 3 分钟：

| 指标 | 当前（59999） | 目标 |
|---|---|---|
| `‖Δtoken‖ / ‖token‖` | 0.34% | ≥ 5% |
| token 层白化 LOO cos | 0.1171（7.5× chance） | ≥ 0.5 |
| **`cos(dF, dE)` @ action hidden** | **−0.001**（chance 0.0044） | **≥ 0.3** ← 「action expert 真的在读触觉」的直接判据 |
| final action 系统性成分 | 0.03% | 与重/轻行为差同量级 |
| token 层分块留出线性探针 | 0.788 | ≥ 0.85 |

配上离线分箱准确率（**分块留出**，抓取后预测的 action chunk 指向远箱还是近箱），
线性探针 0.88 是参考上界。

### 9.5 待确认 / 未做的对照

- **等幅度对照**（7.7）：扰动某个非触觉输入到匹配的 hidden delta 量级（~0.35%），
  重跑 7.5(c)，以排除「这个 stack 对任何微扰都会 base 相关地随机旋转」。
  需要给 runner 加一个条件。只影响定位表述，不影响主结论。
- 实机测试跑的是哪个 commit？服务端日志里有没有 `[TACTILE SERVER] ... mask=True std=...`？
  这决定第 3 节的 L1 是不是当次实机的实际原因。（现在已不影响结论，仅为归档完整性。）

---

## 附录：本文涉及的工具

| 路径 | 用途 | 引入 |
|---|---|---|
| `scripts/audit_tactile_weights.py` | 实验 A：Adam 矩 / 权重相对 ImageNet 的位移 | 08-21 |
| `scripts/audit_tactile_encoder.py` | 实验 B：编码器判别力 | 08-21 |
| `scripts/audit_bottle_sorting_leakage.py` | 第 8 节：逐通道泄漏审计 | 08-21 |
| `scripts/compute_tactile_refs.py` | 8.5 节：差分参考帧预计算 | 08-21 |
| `scripts/tactile_counterfactual_probe.py` + `test/tactile_counterfactual/` | 实验 C / D：离线反事实探针 | 08-22 |
| `scripts/build_counterfactual_config.py` | 由采样帧生成探针 YAML | 08-22 |
| `configs/probes/*.yaml` | 探针配置（`_ep0based` 是正确的那个） | 08-22 |
| `scripts/audit_tactile_bn.py` | 实验 F：BN 统计重估 | 08-22 |
