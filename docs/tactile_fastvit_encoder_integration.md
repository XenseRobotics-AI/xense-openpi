# 触觉 FastViT-T12 编码器接入 π₀.₅

把 4 路触觉图像（左/右臂 × 顶/底）通过 FastViT-T12 编码后注入 π₀.₅ 的 suffix，**不**走 PaliGemma/SigLIP 主视觉通路。训练和推理都基于 training-time RTC。

实现入口：

- 模型：`src/openpi/models/pi0_tactile_fastvit.py` + `pi0_tactile_fastvit_config.py`
- 编码器：`src/openpi/models/tactile_encoders/{base,fastvit}.py`（注册表 + Flax 版 FastViT）
- 数据：`src/openpi/policies/bi_flexiv_policy.py::BiFlexivTactileInputs`、`training/config.py::LeRobotBiFlexivTactileDataConfig`
- Train config：`pi05_base_bi_flexiv_earbuds_case_assembly_with_lid_operation_rtc_tactile_fastvit_a100`
- 推理：`examples/bi_flexiv_rizon4_rt/{real_env,env,main}.py`

---

## 1. 关键决策

| 项 | 决策 |
|---|---|
| 触觉相机布局 | 左臂顶/底 + 右臂顶/底 = **4 路** |
| 触觉视觉路径 | **只走 FastViT → suffix token**，不进 SigLIP prefix |
| FastViT 训练状态 | 始终参与训练，不在 freeze filter 中特殊处理 |
| 跨框架 | **Flax NNX 重写**（与 `pi0.py` 一致），权重一次性从 PyTorch 转过来 |
| 触觉 token 与 adaRMS | 不进 `adarms_cond`，只走 token 通道 |
| 触觉图分辨率 | 224×224（与视觉相机一致） |
| 编码器耦合度 | 走 `tactile_encoders/` 注册表，未来可替换 |

---

## 2. 架构与数据流

### 2.1 训练数据流

```
LeRobot dataset
  │ LeRobotBiFlexivTactileDataConfig (repack)
  │ BiFlexivTactileInputs            (重命名为 base_/wrist_/tactile_*_rgb)
  │ ModelTransformFactory            (ResizeImages 224×224, Tokenize, Pad)
  ▼
Observation { images: 3 cam + 4 tactile, ... }
  │ Pi0TactileFastVit._preprocess_observation
  │   → preprocess_observation_tactile(image_keys=IMAGE_KEYS_TACTILE_4)
  ▼
                        ┌──────────────────────────────────────┐
embed_prefix (子类 override) │ 过滤掉 tactile keys              │
                        │ 仅 base/wrist 3 张图 → SigLIP        │ → prefix tokens
                        └──────────────────────────────────────┘
                        ┌──────────────────────────────────────┐
embed_suffix             │ 4 张 tactile 一次性 batched 进 FastViT│
                        │ + tactile_proj → 4 个 suffix token   │
                        │ 拼到原 state + action+time 之前       │ → suffix tokens
                        └──────────────────────────────────────┘
  │ PaliGemma LLM（prefix + suffix 一起，attn_mask 由 ar block 决定）
  ▼
suffix_out[:, -action_horizon:] → action_out_proj → flow 速度
```

### 2.2 suffix token 排布

```
[tactile_0] [tactile_1] [tactile_2] [tactile_3] | [action_0 ... action_49]
   ar=T        ar=F        ar=F        ar=F        ar=T   F   F  ...   F
 ←──── 触觉块（块内互看；不可读 action；action 可读触觉） ────→
```

- 触觉块首位 `ar=True` 切断与 prefix 的反向读取，后 3 位 `False` 让 4 张触觉互相可见。
- action 块通过 attention 在每层 transformer 重新拉取 tactile / prefix 的 KV，所以 tactile 信号会被多层混合后进入 action 表征。
- pi05 时 adaRMS 仍由 timestep MLP 产生，tactile 位置补零（见 `pi0_tactile_fastvit.py:116-121`）；这只影响 tactile 自身被 timestep 调制与否，不切断 attention 通路。

### 2.3 为什么 tactile 不进 SigLIP

历史实现里 tactile keys 留在 `obs.images` 中，被基类 `Pi0.embed_prefix` 的循环 (`pi0.py:138`) 同时送进 SigLIP，导致：

- prefix image tokens 从 3×256=768 涨到 7×256=1792（多 1024），LLM 整段 sequence 翻倍以上；
- SigLIP 本身多编了 4 张图。

实测在 H100×8、batch=256 上 step 时间从 1.4s 涨到 3.6s（约 2.6×），瓶颈在 GEMM / fusion / NCCL。

修复方法：`Pi0TactileFastVit` override `embed_prefix`，用 `dataclasses.replace` 浅拷贝出一个不含 tactile keys 的 `Observation` 给基类。原 `obs` 仍交给 `embed_suffix` 的 FastViT 路径。

### 2.4 RTC 兼容性

`_compute_loss_training_time_rtc` 和 `training_time_rtc_sample_actions` 只依赖 `embed_prefix` / `embed_suffix` 的返回签名，对 suffix 内 token 数无感知；`v_t = action_out_proj(suffix_out[:, -action_horizon:])` 仍取尾部 action 段。RTC 损失无需改动。

---

## 3. 模型层改动

### 3.1 `pi0.py`（基类）

唯一改动：把 `compute_loss` / `sample_actions` / `training_time_rtc_*` 中的 `_model.preprocess_observation(...)` 改为 `self._preprocess_observation(...)`，基类默认实现仍调原函数。

不引入任何触觉字段，保持基类与触觉解耦。

### 3.2 `Pi0TactileFastVit`（子类）

继承 `Pi0`，新增三处：

1. `__init__`：构造 `tactile_encoder`（注册表）+ `tactile_proj`（Linear → action expert width），保存 `_tactile_keys`。
2. `_preprocess_observation`：改用 `preprocess_observation_tactile(image_keys=IMAGE_KEYS_TACTILE_4)`，让 4 路 tactile 走专用增广（小裁剪 + 颜色抖动 + 高斯噪声）。
3. `embed_prefix`：用 `dataclasses.replace` 过滤掉 tactile keys 后调 `super().embed_prefix()`。
4. `embed_suffix`：把 4 张 tactile stack 成 `(B*N, H, W, 3)` 一次性过 FastViT（`use_running_average=True`，每个 op 都是 batch-invariant），proj 后 reshape 回 `(B, N, w)` 拼到 `super().embed_suffix(...)` 返回的 base tokens 前面。adaRMS per-token cond 时补 0 行对齐长度。

### 3.3 `Pi0TactileFastVitConfig`

继承 `Pi0Config`，加：

- `tactile_encoder_name: str = "fastvit_t12"`
- `tactile_pretrained_path: str | None`（Flax safetensors）
- `tactile_image_keys: tuple[str, ...]`（默认 `tactile_0_rgb ... tactile_3_rgb`）
- `tactile_compute_dtype: str = "bfloat16"`（FastViT 前向 / matmul 精度；参数仍 fp32）
- `model_type` 返回 `PI05_TACTILE` / `PI0_TACTILE`
- `inputs_spec` 在 3 cam 之外加 4 tactile 占位

不重写 `get_freeze_filter`，触觉模块（encoder + proj）默认全部参与训练。

### 3.4 触觉编码器抽象

`tactile_encoders/`：

- `base.py::TactileEncoder`：`feature_dim: int`、`__call__(images: (B, H, W, 3)) -> (B, feature_dim)`。
- `__init__.py::build_tactile_encoder(name, ...)`：注册表。
- `fastvit.py`：FastViT-T12 的 Flax/NNX 实现，输入 `(B, H, W, 3) ∈ [-1, 1]`，内部还原到 `[0, 1]` 再做 ImageNet 标准化。**仅训练分支**，不实现 reparameterize。

权重转换由 `scripts/convert_fastvit_torch_to_flax.py` 完成，支持 `.safetensors / .pth / .pt / .bin`，`--verify-numerics` 用固定输入校验 `max|diff| < 1e-3`。

---

## 4. 数据 / Policy / 推理

- `model.py`：新增 `IMAGE_KEYS_TACTILE_4 = (base, left_wrist, right_wrist, tactile_0..3_rgb)`。`preprocess_observation_tactile` 已按 `"tactile" in key` 走触觉分支，无需改动。
- `bi_flexiv_policy.py::BiFlexivTactileInputs`：在 `BiFlexivInputs` 基础上多映射 4 路触觉到 `tactile_{0..3}_rgb`。约定：0=左顶，1=左底，2=右顶，3=右底。
- `training/config.py::LeRobotBiFlexivTactileDataConfig`：repack 加 4 路触觉键；`ModelTransformFactory` 加 `PI05_TACTILE / PI0_TACTILE` 分支（与 PI05/PI0 等价，因为 ResizeImages 对所有 image key 透明）。
- 推理客户端 `examples/bi_flexiv_rizon4_rt/`：
  - `real_env._POLICY_CAMERAS` 加 4 路触觉；
  - `env.get_observation` 不再跳过 tactile，统一 `resize_with_pad(224,224) + HWC→CHW`；
  - `main.py` 默认 `enable_tactile_sensors=True`，加 4 个 `--*_tactile_*_cam` 映射开关。
- 服务端 `serve_policy.py` 无需改：根据 train config 自动构造 `BiFlexivTactileInputs`。

---

## 5. 端到端用法

### 5.1 准备 FastViT 权重

```bash
mkdir -p model/fastvit_t12_apple_dist_in1k
# 任选：huggingface-cli download timm/fastvit_t12.apple_dist_in1k --local-dir <上面路径>
#       或自己 finetune 的 .pth/.pt/.bin 直接放进去

uv run scripts/convert_fastvit_torch_to_flax.py \
    --torch-checkpoint-dir model/fastvit_t12_apple_dist_in1k \
    --out-path model/fastvit_t12_apple_dist_in1k_flax/params.safetensors \
    --verify-numerics
```

### 5.2 训练

```bash
# 1) 归一化统计
uv run scripts/compute_norm_stats.py \
    --config-name pi05_base_bi_flexiv_earbuds_case_assembly_with_lid_operation_rtc_tactile_fastvit_a100

# 2) 8 卡训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    pi05_base_bi_flexiv_earbuds_case_assembly_with_lid_operation_rtc_tactile_fastvit_a100 \
    --exp-name=tactile_fastvit_$(date +%Y%m%d_%H%M) \
    --fsdp-devices 8

# 冒烟（5 步定位 shape / 权重对齐）
uv run scripts/train.py \
    pi05_base_bi_flexiv_earbuds_case_assembly_with_lid_operation_rtc_tactile_fastvit_a100 \
    --exp-name=smoke --batch-size 8 --num-train-steps 5 --overwrite
```

`weight_loader` 指 `pi05_base`，只补 SigLIP/PaliGemma/ActionExpert；FastViT 权重在 `__init__` 内由 `TactileFastVitEncoder` 单独加载本地 safetensors。`missing_regex` 允许 `tactile_encoder|tactile_proj` 缺失。

### 5.3 推理

服务端：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_base_bi_flexiv_earbuds_case_assembly_with_lid_operation_rtc_tactile_fastvit_a100 \
    --policy.dir=<checkpoint dir>
```

客户端（BiFlexiv 实机 + RTC）：

```bash
python -m examples.bi_flexiv_rizon4_rt.main \
    --host <server> --port 8000 \
    --rtc_enabled --enable_tactile_sensors \
    --left_tactile_top_cam left_tactile_top \
    --left_tactile_bottom_cam left_tactile_bottom \
    --right_tactile_top_cam right_tactile_top \
    --right_tactile_bottom_cam right_tactile_bottom
```

如果 lerobot 相机名与默认值相同，`--*_cam` 可省。

---

## 6. 风险与注意点

| 风险 | 处理 |
|---|---|
| FastViT BN 在小 batch 下不稳定 | 默认 `use_running_average=True` 用冻结 stats；当前 batch=256/FSDP=8（每卡 32）足够稳，必要时再换 GroupNorm |
| 权重转换名字漏映射 | 转换脚本必须输出 `missing/unexpected keys`，`--verify-numerics` 严格 < 1e-3 |
| 推理时缺触觉相机 | `BiFlexivTactileInputs` 缺失时填零图 + `image_mask=False`，触觉 token 被 attention mask 抑制 |
| LeRobot 数据集未含 4 路触觉 | 先录到数据集，否则 `RepackTransform` 会 key 缺失 |
| 与旧 `pi0_tactile.py`（2 路 + SigLIP prefix）路径冲突 | 不复用旧代码；新链路独立命名（`pi0_tactile_fastvit*`、`IMAGE_KEYS_TACTILE_4`） |

---

## 7. 更换视觉编码器

想把 FastViT 换成例如 `dinov2_vits14`：

1. 新建 `tactile_encoders/dinov2.py`，实现 `TactileEncoder` 接口。
2. 在 `tactile_encoders/__init__.py::build_tactile_encoder` 注册 `name`。
3. train config 改 `tactile_encoder_name` + `tactile_pretrained_path`。

不动 `Pi0TactileFastVit`、不动 `pi0.py`、不动数据通路。
