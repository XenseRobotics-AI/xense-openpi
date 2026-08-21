# 实验记录：触觉分支加不加，实机动作一样——到底谁在忽略触觉

**日期**：2026-08-21
**状态**：进行中（A、B 已完成，C、D、E 待做）
**结论（阶段性）**：**触觉编码器在本数据上已经塌缩**——输入能把"抓住/没抓住"分开约 3.3 倍，
经过 FastViT + `tactile_proj` 之后只剩 1.01 倍（类均值 cos = 0.999999）。
action expert 收到的是一个几乎恒定的向量，所以"action expert 忽略触觉"这个猜想
在修好编码器之前无法被检验——它没有东西可忽略。
另有一个独立的、确定的实机通路 bug（见第 3 节），会让触觉在推理时被完全屏蔽。
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

## 7. 实验 C / D / E

（待做——须在第 9 节的数据/表征问题修好后再做，否则无意义）

---

## 8. 重训前的数据审计（2026-08-21 追加）

目标实验：两个外观完全相同的瓶子，有水→远箱、无水→近箱，
两类之间**唯一**的差别是夹取时的触觉图。用它来验证触觉 encoder 是否真的影响 action。
前提是标签在触觉之外无处可查。逐通道审计如下。

脚本：`scripts/audit_bottle_sorting_leakage.py`

### 9.1 标签与录制结构

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

### 9.2 prompt 没有泄漏

`meta/tasks.parquet` 只有一条 task 字符串，全部 160 个 episode 共用：
`"Pick up the bottle, and place it in the far bin if it is heavy, or in the near bin if it is light."`

### 9.3 逐通道线性探针（grasp+15 帧，绝对图像）

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

### 9.4 差分触觉探针：信号确实存在

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

## 9. 阶段性结论与后续

### 已确定

1. **训练侧通路是活的**（A）。梯度流过触觉分支，288 个可训练张量全部位移。
   但梯度幅度只有主干的 1/40 ～ 1/74，属"训了但训得很轻"。
2. **编码器塌缩**（B）。输入端 3.3 倍的"抓/没抓"可分性，出编码器只剩 1.01 倍，
   类均值 cos = 0.999999。**这是目前最靠前的技术根因。**
3. **实机通路被切断**（第 3 节，代码级确定）。`env.py:95` 丢掉所有 tactile key，
   服务端零填充 + `image_mask=False`，触觉在数学上等价于不存在。
4. **数据本身没问题**（B1）。抓取瞬间触觉信号跳变约 200 倍。

### 对原始猜想的回答

"action expert 忽略了 tactile 特征"——**目前无法检验**。
编码器给出的 4 个 token 在整段轨迹里近乎恒定，action expert 即使完全正常工作也学不到东西。
必须先修 L1（通路）和 L2（表征），才谈得上验证 L3。

### 后续动作（按优先级）

| # | 动作 | 理由 | 状态 |
|---|---|---|---|
| 1 | 删掉 `env.py` 的 tactile 过滤 | 不修的话任何模型改动在实机上都看不出效果 | **已完成** |
| 2 | **触觉改差分输入**：喂 `当前帧 − 本 episode 夹爪张开时的参考帧`，或把两帧堆成 6 通道 | 8.4 节：绝对单帧线性可分 0.58（≈基线），差分 0.88。这是最大的杠杆，也是编码器塌缩最可能的深层原因 | 待做 |
| 3 | 触觉图改用拉伸或中心裁，不要 `resize_with_pad` | 当前 43% 输入是黑边，纵向下采样 3.1×；实测判别力提升约 30 倍。**改了必须训练/推理两侧同步改** | 待做 |
| 4 | 让 FastViT 的 BN 用数据自身统计量（换 GroupNorm，或在触觉数据上重估 running stats） | A2 确认 BN buffer 仍是 ImageNet 的 | 待做 |
| 5 | 按**录制块**切留出集（不是按 episode 随机切） | 8.1/8.3：随机切会让所有通道虚假有效（0.75 vs 0.58） | 待做 |
| 6 | 建离线评测指标：留出块上，抓取后预测的 action chunk 指向远箱还是近箱 → 分箱准确率 | 不需要实机就能量化，比肉眼看机械臂强得多；线性探针 0.88 是参考上界 | 待做 |
| 7 | A/B 对照：同 seed/同数据/同步数，训「有触觉」与「触觉 mask 全 False」两个模型 | 唯一能干净归因到触觉的对照。mask=False 时触觉在数学上等价于不存在（第 3 节） | 待做 |
| 8 | 给触觉分支单独调高学习率，或先用辅助任务（"抓/没抓"、"重/轻"分类）预训练编码器 | A1 显示梯度只有主干的 1/40～1/74；模仿损失对触觉的监督过于稀疏 | 待做 |
| 9 | 修完 2–4 后重跑实验 B 确认判别力，再跑 C / D 验证 L3 | 顺序不能颠倒 | 待做 |

### 待用户确认

- 实机测试跑的是哪个 commit？服务端日志里有没有 `[TACTILE SERVER] ... mask=True std=...`？
  这决定第 3 节的 L1 是不是当次实机的实际原因。
