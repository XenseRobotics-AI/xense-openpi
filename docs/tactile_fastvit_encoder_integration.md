# 触觉 FastViT-T12 编码器接入 π₀.₅ 的实施方案

> 目标 train config：`pi05_base_bi_flexiv_earbuds_case_assembly_with_lid_operation_rtc_0429_h100`
> 编码器源码：`src/openpi/models/fastvit_t12_apple_dist_in1k.py`（PyTorch / timm）
> 修改点：在 `Pi0` 的 `embed_suffix` 中注入 4 张触觉图的 embedding，**作为 `Pi0TactileFastVit` 子类**（不直接改 `pi0.py` 主类）。
> 训练与推理均使用 training-time RTC。

---

## 0. 关键决策（用户确认）

| 项 | 决策 |
|---|---|
| 触觉相机布局 | **左臂 ×2（顶+底）+ 右臂 ×2（顶+底）= 共 4 路** |
| FastViT 参数 | **始终参与训练**，不在 freeze filter 内做特殊处理 |
| 跨框架方案 | **方案 A：把 FastViT 移植成 Flax/NNX**，一次到位 |
| 触觉 token 与 adaRMS | **不进 `adarms_cond`**，只走 token 通道 |
| 触觉图分辨率 | **224×224**（与视觉相机一致） |
| 与 π 模型耦合度 | **松耦合**：编码器走 `tactile_encoders/` 抽象接口，未来可替换 |
| 框架 | **JAX/Flax NNX**（与 `pi0.py` 主链一致；不动 PyTorch 训练分支） |
| 推理 | **training-time RTC** + WebSocket 客户端 `examples/bi_flexiv_rizon4_rt/main.py`，需新增触觉相机映射 |

---

## 1. 总体架构

```
src/openpi/models/
├── pi0.py                              # 基础 Pi0（新增 1 个 _preprocess_observation hook）
├── pi0_tactile_fastvit.py              # 新建：继承 Pi0，覆写 embed_suffix
├── pi0_tactile_fastvit_config.py       # 新建：继承 Pi0Config，加触觉相关字段
├── fastvit_t12_apple_dist_in1k.py      # 保留：PyTorch 实现（权重转换时用一次）
└── tactile_encoders/                   # 新建：触觉编码器抽象（可扩展其他视觉编码器）
    ├── __init__.py                     # 注册表 build_tactile_encoder(name, ...)
    ├── base.py                         # TactileEncoder Protocol：(b,h,w,3) -> (b, feat_dim)
    └── fastvit.py                      # Flax 版 FastViT-T12 + 加载本地权重

scripts/
└── convert_fastvit_torch_to_flax.py    # 新建：PyTorch → Flax 权重转换

src/openpi/policies/
└── bi_flexiv_policy.py                 # 扩展：新增 BiFlexivTactileInputs（含 4 路触觉）

src/openpi/training/
└── config.py                           # 扩展：新增 LeRobotBiFlexivTactileDataConfig + 新 train config + ModelTransformFactory 分支

src/openpi/models/model.py              # 扩展：IMAGE_KEYS_TACTILE_4（左右各2）；preprocess 接受新 keys

examples/bi_flexiv_rizon4_rt/
├── real_env.py                         # 扩展：_POLICY_CAMERAS 加 4 路触觉
├── env.py                              # 扩展：触觉相机也走 resize+CHW 通路
└── main.py                             # 扩展：默认开启触觉，新增 --tactile_*_cam 映射开关
```

### 1.1 数据流（训练）

```
LeRobot dataset
   │
   │  repack (LeRobotBiFlexivTactileDataConfig)
   ▼
{ "images": { head, left_wrist, right_wrist,
              left_tactile_top, left_tactile_bottom,
              right_tactile_top, right_tactile_bottom },
  "state": ..., "actions": ..., "prompt": ... }
   │
   │  BiFlexivTactileInputs
   ▼
{ "image":  { base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb,
              tactile_0_rgb, tactile_1_rgb, tactile_2_rgb, tactile_3_rgb },
  "image_mask": {...}, "state": ..., "actions": ..., "prompt": ... }
   │
   │  ModelTransformFactory (ResizeImages 224×224 + Tokenize + PadStatesAndActions)
   ▼
Observation
   │
   │  Pi0TactileFastVit._preprocess_observation
   │    (调 preprocess_observation_tactile, image_keys=IMAGE_KEYS_TACTILE_4)
   ▼
Observation (with augmentations)
   │
   │  embed_prefix : SigLIP(3 visual) + lang  → prefix tokens
   │  embed_suffix : FastViT(4 tactile) + state + action+time  → suffix tokens
   ▼
PaliGemma + ActionExpert → action chunk
```

### 1.2 suffix token 排布（重点）

```
[tactile_0]  [tactile_1]  [tactile_2]  [tactile_3] | [state] | [action_0 ... action_49]
   ar=T         ar=F         ar=F         ar=F        ar=T       ar=T  F  F ...  F
  ←─── 触觉块（块内互相可见，块外看不见前缀，状态/动作可看见触觉） ───→
```

- 触觉块**整体**作为一个新 ar block：第一位 `True` 切断与 image/lang 前缀的交互，后 3 位 `False` 让 4 张触觉互相可见。
- `state` token 仍然是一个独立 block（保留原有语义），可看见触觉。
- `action` tokens 共 `action_horizon` 个，仍然是一个 block，可看见触觉与 state。
- `adarms_cond` 仅由 timestep MLP 产生，**触觉特征不进 adaRMS**。

### 1.3 RTC 兼容性

`_compute_loss_training_time_rtc` 与 `training_time_rtc_sample_actions` 都只依赖 `embed_prefix` / `embed_suffix` 的返回签名，对 suffix 内部 token 数量无感知；`v_t = action_out_proj(suffix_out[:, -self.action_horizon:])` 仍取尾部 action token。**RTC 损失公式无需改动**。

---

## 2. 跨框架问题：FastViT PyTorch → Flax NNX 移植

### 2.1 为什么必须移植

- `pi0.py` 用 `flax.nnx` + `jax.numpy`，被 `jit/grad/FSDP` 包裹；
- 若用 `jax.pure_callback` 调用 PyTorch FastViT：① 反向传播无法回传到 FastViT；② host-callback 性能差；③ FSDP 多卡放置麻烦。
- 用户要求 **FastViT 参数始终参与训练**，所以必须把 FastViT 也变成 Flax 可微分模块。

### 2.2 Flax 实现要点

`src/openpi/models/tactile_encoders/fastvit.py` 中实现的 Flax 版本严格对应 PyTorch 结构：

| PyTorch (timm) | Flax (linen) | 说明 |
|---|---|---|
| `nn.Conv2d` | `nn.Conv(..., feature_group_count=groups)` | NCHW → NHWC，stride/padding 一致 |
| `nn.BatchNorm2d` | `nn.BatchNorm` | 注意 `use_running_average` 训练 False / 推理 True |
| `ConvNormAct` | 同名子类 | conv + BN（无激活，激活在外） |
| `SqueezeExcite` | 自定义模块 | GAP → 1×1 → ReLU → 1×1 → Sigmoid |
| `MobileOneBlock` | 同名子类 | 多分支训练：`conv_kxk` 列表 + `conv_scale` + `identity` BN，前向求和后激活 |
| `ReparamLargeKernelConv` | 同名子类 | `large_conv` + `small_conv`（小核 zero-pad 到大核） |
| `RepMixer` | 同名子类 | `norm` MobileOne + `mixer` MobileOne + `layer_scale`，残差 |
| `ConvMlp` | 同名子类 | 7×7 depthwise conv（NHWC） + 1×1 fc + GELU + 1×1 fc |
| `RepMixerBlock` | 同名子类 | token_mixer + mlp + drop_path + layer_scale |
| `FastVitStage` | 同名子类 | downsample(PatchEmbed) + blocks(scan-free Sequential) |
| `FastVit` | 同名子类 | stem + stages + final_conv + GAP → (B, 1024) |
| `forward_features` | `__call__` | 不带 head，输出 1024-d 池化向量 |

**仅训练分支**：不实现 `reparameterize()`（融合分支等价但破坏可训练性）；如未来需要导出推理用 frozen 模型再加。

**输入约定**：`(B, H, W, 3) ∈ [-1, 1]`（与 SigLIP / pi0 一致）。**FastViT 自带的 ImageNet 归一化（mean/std）** 在编码器内部转成 ImageNet 标准化：先把 [-1, 1] 还原回 [0, 1]，再做 `(x - mean) / std`。

### 2.3 权重转换脚本

`scripts/convert_fastvit_torch_to_flax.py`：

1. 加载 PyTorch FastViT（`load_local_pretrained` 从 `checkpoint/fastvit_t12_apple_dist_in1k/`）。
2. 调用 `_torch_to_flax_state_dict` 把 PyTorch 参数树名映射到 Flax 参数树名：
   - `weight (out, in, k, k)` → `kernel (k, k, in, out)`（depthwise: groups=in_ch, Flax 用 `feature_group_count`，kernel 形状 `(k, k, in/groups, out)`）；
   - BatchNorm `running_mean/running_var/weight/bias` → linen BN 的 `mean/var/scale/bias`（params + batch_stats 双 collection）；
   - `head.fc.*` 跳过（num_classes=0）。
3. 用 `safetensors` 或 `flax.serialization` 保存到 `checkpoint/fastvit_t12_apple_dist_in1k_flax/{params.safetensors}`。
4. **等价性测试**：固定输入 `x ∈ R^{2,224,224,3}`，分别跑 PyTorch / Flax，比较 1024-d 输出 `max |diff| < 1e-3`。

---

## 3. 模型层改动

### 3.1 `pi0.py` 唯一改动：加 preprocess hook

```python
class Pi0(_model.BaseModel):
    def _preprocess_observation(self, rng, observation, *, train):
        """Hook so subclasses can swap in tactile-aware preprocess."""
        return _model.preprocess_observation(rng, observation, train=train)

    # compute_loss / sample_actions / training_time_rtc_sample_actions
    #   原本：observation = _model.preprocess_observation(...)
    #   改为：observation = self._preprocess_observation(...)
```

这是 `pi0.py` 中**仅有的 4 处替换**，**不引入触觉任何字段**——保持 Pi0 与触觉解耦。

### 3.2 触觉编码器抽象

`src/openpi/models/tactile_encoders/base.py`：

```python
class TactileEncoder(nnx.Module):
    """Protocol for tactile-image encoders.

    Implementations must expose:
      - feature_dim: int  (output channels per image)
      - __call__(images: (B, H, W, 3)) -> (B, feature_dim)
    """
    feature_dim: int

    def __call__(self, images): ...
```

`src/openpi/models/tactile_encoders/__init__.py`：

```python
def build_tactile_encoder(name: str, *, rngs, **kwargs) -> TactileEncoder:
    if name == "fastvit_t12":
        from .fastvit import TactileFastVitEncoder
        return TactileFastVitEncoder(rngs=rngs, **kwargs)
    raise ValueError(f"Unknown tactile encoder: {name}")
```

未来加新编码器只需写一个 `tactile_encoders/<new>.py`，注册一个 `name`，不动 `Pi0TactileFastVit`。

### 3.3 `Pi0TactileFastVit`

```python
class Pi0TactileFastVit(Pi0):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        from openpi.models.tactile_encoders import build_tactile_encoder
        self.tactile_encoder = build_tactile_encoder(
            config.tactile_encoder_name,
            rngs=rngs,
            pretrained_path=config.tactile_pretrained_path,  # 可为 None
        )
        action_expert_width = _gemma.get_config(config.action_expert_variant).width
        self.tactile_proj = nnx.Linear(
            self.tactile_encoder.feature_dim, action_expert_width, rngs=rngs
        )
        self._tactile_keys = config.tactile_image_keys

    def _preprocess_observation(self, rng, observation, *, train):
        return _model.preprocess_observation_tactile(
            rng, observation, train=train, image_keys=_model.IMAGE_KEYS_TACTILE_4,
        )

    def embed_suffix(self, obs, noisy_actions, timestep):
        # 1) 4 张触觉 → encoder → (b, feat_dim) → proj → (b, w)
        tactile_tokens, tactile_mask = [], []
        for key in self._tactile_keys:
            feat = self.tactile_encoder(obs.images[key])
            feat = self.tactile_proj(feat)
            tactile_tokens.append(feat[:, None, :])
            tactile_mask.append(einops.repeat(obs.image_masks[key], "b -> b 1"))
        tactile_tokens = jnp.concatenate(tactile_tokens, axis=1)
        tactile_mask = jnp.concatenate(tactile_mask, axis=1)
        tactile_ar = jnp.asarray([True] + [False] * (len(self._tactile_keys) - 1))

        # 2) 复用父类得到 state + action+time tokens
        base_tokens, base_mask, base_ar, adarms_cond = super().embed_suffix(
            obs, noisy_actions, timestep
        )

        # 3) 拼接：tactile 在最前
        tokens = jnp.concatenate([tactile_tokens, base_tokens], axis=1)
        input_mask = jnp.concatenate([tactile_mask, base_mask], axis=1)
        ar_mask = jnp.concatenate([tactile_ar, base_ar], axis=0)
        return tokens, input_mask, ar_mask, adarms_cond
```

### 3.4 配置

`Pi0TactileFastVitConfig(Pi0Config)`：

```python
@dataclasses.dataclass(frozen=True)
class Pi0TactileFastVitConfig(Pi0Config):
    tactile_encoder_name: str = "fastvit_t12"
    # Flax 权重 .safetensors 路径；None → 从零初始化
    tactile_pretrained_path: str | None = None
    tactile_image_keys: tuple[str, ...] = (
        "tactile_0_rgb", "tactile_1_rgb", "tactile_2_rgb", "tactile_3_rgb",
    )

    @property
    def model_type(self) -> _model.ModelType:
        # 复用现有 PI05_TACTILE 枚举；ModelTransformFactory 已有兜底
        return _model.ModelType.PI05_TACTILE if self.pi05 else _model.ModelType.PI0_TACTILE

    def create(self, rng):
        from openpi.models.pi0_tactile_fastvit import Pi0TactileFastVit
        return Pi0TactileFastVit(self, rngs=nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size=1):
        # 3 visual + 4 tactile
        ...
```

不重写 `get_freeze_filter`——继承父类，所有触觉模块（encoder + proj）保持可训练。

---

## 4. 数据 / Policy / Preprocess 通路

### 4.1 `model.py`

```python
IMAGE_KEYS_TACTILE_4 = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
    "tactile_0_rgb",
    "tactile_1_rgb",
    "tactile_2_rgb",
    "tactile_3_rgb",
)
```

`preprocess_observation_tactile` 的现有逻辑已通过 `"tactile" in key` 判断走触觉增广分支（小裁剪 + 颜色 + 高斯噪声），4 路 token 命中同一分支即可。

### 4.2 `bi_flexiv_policy.py`

新增 `BiFlexivTactileInputs`（继承自 `BiFlexivInputs` 的实现思路）：

```python
@dataclasses.dataclass(frozen=True)
class BiFlexivTactileInputs(transforms.DataTransformFn):
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "head", "left_wrist", "right_wrist",
        "left_tactile_top", "left_tactile_bottom",
        "right_tactile_top", "right_tactile_bottom",
    )

    def __call__(self, data):
        # 与 BiFlexivInputs 相同，再多映射 4 路触觉到 tactile_{0..3}_rgb
        ...
```

约定 `tactile_0..3_rgb` 对应：

| 模型键 | 物理位置 |
|---|---|
| `tactile_0_rgb` | left arm 顶部触觉 |
| `tactile_1_rgb` | left arm 底部触觉 |
| `tactile_2_rgb` | right arm 顶部触觉 |
| `tactile_3_rgb` | right arm 底部触觉 |

### 4.3 `training/config.py`

新增 `LeRobotBiFlexivTactileDataConfig`：

```python
@dataclasses.dataclass(frozen=True)
class LeRobotBiFlexivTactileDataConfig(LeRobotBiFlexivDataConfig):
    repack_transforms: ... = dataclasses.field(
        default=_transforms.Group(
            inputs=[_transforms.RepackTransform({
                "images": {
                    "head":                  "observation.images.head",
                    "left_wrist":            "observation.images.left_wrist",
                    "right_wrist":           "observation.images.right_wrist",
                    "left_tactile_top":      "observation.images.left_tactile_top",
                    "left_tactile_bottom":   "observation.images.left_tactile_bottom",
                    "right_tactile_top":     "observation.images.right_tactile_top",
                    "right_tactile_bottom":  "observation.images.right_tactile_bottom",
                },
                "state": "observation.state",
                "actions": "action",
                "prompt": "task",
            })]
        )
    )

    def create(self, assets_dirs, model_config):
        # 与父类一致，data_transforms 使用 BiFlexivTactileInputs
        ...
```

`ModelTransformFactory` 增加 `PI05_TACTILE` / `PI0_TACTILE` 分支（与 PI05 / PI0 一致，因为 ResizeImages 对所有 image key 透明），如未来需要也可以在这里加 ResizeTactileImages。

### 4.4 新增 train config

```python
TrainConfig(
    name="pi05_base_bi_flexiv_earbuds_case_assembly_with_lid_operation_rtc_tactile_fastvit_h100",
    model=pi0_tactile_fastvit_config.Pi0TactileFastVitConfig(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=True,
        enable_training_time_rtc=True,
        max_delay=10,
        tactile_encoder_name="fastvit_t12",
        tactile_pretrained_path="checkpoint/fastvit_t12_apple_dist_in1k_flax/params.safetensors",
    ),
    data=LeRobotBiFlexivTactileDataConfig(
        repo_id="Xense/earbuds_case_assembly_with_lid_operation",
        use_delta_cartesian_actions=True,
        default_prompt="Pick up each earbud case from the left stands, insert the matching earbuds, close the lid, and place the case on the middle stand",
        base_config=DataConfig(prompt_from_task=True),
    ),
    save_interval=2000,
    keep_period=10000,
    ema_decay=None,
    batch_size=256,
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=20000,
    num_workers=64,
    fsdp_devices=8,
),
```

`weight_loader` 仍指 `pi05_base`，只补充 SigLIP/PaliGemma/ActionExpert 的参数；FastViT 权重在 `__init__` 内由 `TactileFastVitEncoder` 自行加载本地 `safetensors`（不与 orbax 主权重共享路径）。

---

## 5. 推理客户端适配

### 5.1 `real_env.py`

```python
_POLICY_CAMERAS = (
    "head", "left_wrist", "right_wrist",
    "left_tactile_top", "left_tactile_bottom",
    "right_tactile_top", "right_tactile_bottom",
)
```

`get_images` 不变：缺失的相机不会进 dict，由 client 端补 mask=False。

### 5.2 `env.py`

`get_observation` 中**不再跳过 tactile 相机**，把它们也 `resize_with_pad(224, 224)` + `HWC→CHW`，加入 `processed_images`。`raw_images`（给录制器）继续排除 tactile（避免无谓占用磁盘，可选）。

### 5.3 `main.py`

- 默认 `enable_tactile_sensors=True`。
- 新增 4 个映射开关，允许把 lerobot 真实相机名映射为 policy 期望的 `left_tactile_top` 等：

  ```python
  left_tactile_top_cam: str = "left_tactile_top"
  left_tactile_bottom_cam: str = "left_tactile_bottom"
  right_tactile_top_cam: str = "right_tactile_top"
  right_tactile_bottom_cam: str = "right_tactile_bottom"
  ```

  传入 `BiFlexivRizon4RTEnvironment` 后由 `real_env.get_images()` 做改名。

### 5.4 服务端不需要改

服务端的 `serve_policy.py` 根据 train config 的 `data` 字段自动构造同样的 `BiFlexivTactileInputs`，客户端 image dict 中含 4 路触觉即可直接吃。

---

## 6. 端到端脚本用法

### 6.1 准备 FastViT 权重

`src/openpi/models/fastvit_t12_apple_dist_in1k.py` 已经是 **self-contained** 实现（不依赖 timm），自带 `load_state_dict_from_path` 既支持 `.safetensors`，也支持 `.pth/.bin/.pt`（包括带 `{"state_dict": ...}` 包装的 checkpoint）。`_find_checkpoint_in_dir` 自动识别以下文件名：`model.safetensors`、`pytorch_model.bin`、`pytorch_model.pth`、`model.pth`、`model.pt`。

```bash
# 1) 下载 PyTorch 权重到约定目录（任选一种格式）
mkdir -p checkpoint/fastvit_t12_apple_dist_in1k
# 例如：huggingface-cli download timm/fastvit_t12.apple_dist_in1k --local-dir checkpoint/fastvit_t12_apple_dist_in1k
# 或：直接把自己训练好的 .pth/.pt/.bin 放进该目录

# 2) 转成 Flax 版
uv run scripts/convert_fastvit_torch_to_flax.py \
    --torch-checkpoint-dir checkpoint/fastvit_t12_apple_dist_in1k \
    --out-path checkpoint/fastvit_t12_apple_dist_in1k_flax/params.safetensors \
    --verify-numerics
```

`--verify-numerics` 会跑一次 PyTorch / Flax 等价性测试，要求 `max|diff| < 1e-3`。如果你后续用自己 fine-tune 过的 FastViT（同一架构），同一脚本也可以转换——只需把对应文件放到 `--torch-checkpoint-dir` 即可。

### 6.2 训练流程

```bash
# 1) 计算归一化统计量（必须，新加触觉相机也需要 stats）
uv run scripts/compute_norm_stats.py \
    --config-name pi05_base_bi_flexiv_earbuds_case_assembly_with_lid_operation_rtc_tactile_fastvit_h100

# 2) 训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    pi05_base_bi_flexiv_earbuds_case_assembly_with_lid_operation_rtc_tactile_fastvit_h100 \
    --exp-name=tactile_fastvit_run_$(date +%Y%m%d_%H%M) \
    --fsdp-devices 8
```

冒烟测试（小 batch 跑通 5 步，定位 shape / 权重对齐问题）：

```bash
uv run scripts/train.py \
    pi05_base_bi_flexiv_earbuds_case_assembly_with_lid_operation_rtc_tactile_fastvit_h100 \
    --exp-name=smoke \
    --batch-size 8 --num-train-steps 5 --overwrite
```

### 6.3 推理：服务端

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_base_bi_flexiv_earbuds_case_assembly_with_lid_operation_rtc_tactile_fastvit_h100 \
    --policy.dir=checkpoints/pi05_base_bi_flexiv_earbuds_case_assembly_with_lid_operation_rtc_tactile_fastvit_h100/<exp_name>/<step>
```

### 6.4 推理：客户端（BiFlexiv 实机，含 RTC）

```bash
python -m examples.bi_flexiv_rizon4_rt.main \
    --host <server> --port 8000 \
    --rtc_enabled \
    --enable_tactile_sensors \
    --left_tactile_top_cam left_tactile_top \
    --left_tactile_bottom_cam left_tactile_bottom \
    --right_tactile_top_cam right_tactile_top \
    --right_tactile_bottom_cam right_tactile_bottom
```

> 4 个 `--*_cam` 默认值已与训练时一致，**如果你的 lerobot robot 相机名相同，可以省略**。

### 6.5 验证脚本（强烈推荐先跑）

```bash
# 1) Flax FastViT 单元测试（前向输出维度、确定性）
uv run python -m pytest src/openpi/models/tactile_encoders/test_fastvit.py -xvs

# 2) 模型 forward smoke：构造 fake_obs(2)，跑一次 embed_prefix + embed_suffix，检查 token 数
uv run python -m openpi.models.pi0_tactile_fastvit  # 模块自带 __main__ smoke
```

---

## 7. 风险 & 注意事项

| 风险 | 处理 |
|---|---|
| FastViT BN 在小 batch 下不稳定 | π₀.₅ 默认 batch_size=256 / FSDP=8，每卡 32 — 足够；如需要可考虑替换 BN→GroupNorm（届时再评估） |
| 权重转换名字漏映射 | 转换脚本必须输出 `missing/unexpected keys`，并在 `--verify-numerics` 中严格 < 1e-3 |
| 触觉相机分辨率 ≠ 模型输入 | `ModelTransformFactory` 的 `ResizeImages(224, 224)` 对所有 image key 透明，会自动 resize |
| 推理时缺触觉相机 | `BiFlexivTactileInputs` 缺失时填零图 + `image_mask=False`，模型仍能跑（触觉 token 被 attention mask 抑制） |
| LeRobot 数据集尚未含 4 路触觉 | 先把触觉录制进数据集（更新 repo_id 或本地 root），否则训练时 `RepackTransform` 会报 key 缺失 |
| 与 `pi0_tactile.py`（2 路触觉 + SigLIP + prefix）路径冲突 | 不复用旧代码；新增独立命名（`pi0_tactile_fastvit*`、`IMAGE_KEYS_TACTILE_4`） |

---

## 8. 替换为其他视觉编码器（未来）

想把 FastViT-T12 换成例如 `dinov2_vits14`：

1. 新建 `src/openpi/models/tactile_encoders/dinov2.py`，实现 `TactileEncoder` 接口，提供 `feature_dim`、`__call__`。
2. 在 `tactile_encoders/__init__.py` 的注册表中加一行 `if name == "dinov2_vits14": return TactileDinoV2Encoder(...)`。
3. 修改 train config 中 `tactile_encoder_name="dinov2_vits14"` + `tactile_pretrained_path=<对应权重>`。
4. 不动 `Pi0TactileFastVit`、不动 `pi0.py`、不动数据通路。

这是用户"未来可能更换视觉编码器，不要把编码器和 pi 模型耦合得很深"的实现保证。



会的。我把整条路径走一遍说清楚为什么。

  一、注意力路径：tactile 信息怎么进 action 的

  把整段 [prefix + tactile + actions] 的 ar_mask 摊开看：

  ┌──────────────────────┬──────┬─────────────────────────────┬─────────────┐
  │          段          │ 长度 │           ar_mask           │ cumsum 块号 │
  ├──────────────────────┼──────┼─────────────────────────────┼─────────────┤
  │ prefix (images+text) │ P    │ [False]*P                   │ 0           │
  ├──────────────────────┼──────┼─────────────────────────────┼─────────────┤
  │ tactile              │ 4    │ [True, False, False, False] │ 1           │
  ├──────────────────────┼──────┼─────────────────────────────┼─────────────┤
  │ actions              │ 50   │ [True, False, ..., False]   │ 2           │
  └──────────────────────┴──────┴─────────────────────────────┴─────────────┘

  make_attn_mask 的规则（pi0.py:47）：token i 能注意到 token j ⟺ cumsum[j] ≤ cumsum[i]。

  - action token（块 2）→ tactile token（块 1）：1 ≤ 2，允许注意。
  - action token（块 2）→ prefix（块 0）：允许（这是原本就有的）。
  - tactile token（块 1）→ prefix（块 0）：允许（tactile 可以读图像和语言上下文）。
  - tactile token（块 1）→ action token（块 2）：2 > 1，不允许——tactile 不被 action 反向污染，这就是 ar_mask 的"块单向"语义。

  所以每一层 transformer 里，action token 都能通过 attention 把 tactile 的 key/value 拉过来。经过若干层 stack 后，suffix_out[:, 
  -50:]（即最后 50 个 action token 的隐藏态）已经累积了 tactile 信息。最后 v_t = self.action_out_proj(suffix_out[:, 
  -50:])（pi0.py:320）就把这份带有触觉特征的隐藏态映射成 flow 速度，进入 loss / 推理。

  二、adarms_cond 补零会不会切断 tactile→action 的信息？

  不会。把它们彻底拆开看：

  - attention 决定"哪个 token 看得到哪个 token 的内容"——由 ar_mask / attn_mask 控制；tactile→action 通路在这里成立。
  - adarms_cond 只进 RMSNorm，作用是"在 attention/FFN 之前对每个 token 的隐藏态做 scale/shift 调制"（gemma.py:135-137），是一种
  per-token 归一化条件，不是门控也不是 attention mask。
  - 给 tactile 位置补零，等价于"tactile token 自身不被 diffusion timestep 调制"，但它仍然是一个有内容、有 KV、可被 query 的 token。

  换句话说：补零影响的是"tactile token 出去时长什么样"，不影响"它能不能被 action 看到"。design 里 "tactile 不进 adaRMS"
  这句话指的就是前者，不是切断信息流。

  三、信息流动的端到端简图

  tactile_image (b, 224, 224, 3)
      │ FastViT-T12 + tactile_proj
      ▼
  tactile_feat (b, 4, action_expert_width)        ← 4 个 token
      │ 拼到 suffix 前面
      ▼
  suffix = [tactile(4) | action(50)]              shape (b, 54, emb)
      │ + prefix (image+text) 一起进 PaliGemma.llm
      │ 每层 transformer:
      │   - RMSNorm(cond=0 for tactile, cond=time_emb for action)
      │   - self-attention(全 token，但受 ar_mask 限制为单向块结构)
      │       └─ action_token.Q · tactile_token.K  →  attention weight  ★ 这里把 tactile 拉进来
      │       └─ ... · prefix_token.K              →  原有的 image+text 通路
      │   - FFN
      ▼
  suffix_out (b, 54, emb)
      │ suffix_out[:, -50:]  取后 50（action 位置）
      ▼
  v_t = action_out_proj(...)                       ← 已经隐含 tactile 信息

  四、几个值得知道的副作用 / 注意点

  1. 每一层都注入一次：tactile→action 不是只在最后一层一次性接进来，而是每个 transformer block 都会重新做一次 attention，所以 tactile
  信号实际上是被多层逐步抽取/精炼的。
  2. action token 之间也共享 tactile：50 个 action token 都能各自从 tactile 那儿拿信息，不是只有第一个。
  3. time conditioning 仍然独立于 tactile：action token 的 RMSNorm 用 time_emb 调制，告诉网络"现在是 flow 的哪个时刻"；tactile
  是内容性条件，两个职责不冲突。
  4. 若想让 tactile 也享受 time-cond（即让 tactile 表征也随 diffusion step 变化），把那段补零改成 time_emb 的 broadcast 即可。但目前 doc
   决策是不要，所以保持零。

  结论：当前实现，输出 action 一定带 tactile 信息，且经过每一层都重新混合，不只是最后一层。如果你想验证，可以拿一个固定 batch、把 4 张
  tactile 图替换成全零（或随机扰动）跑两遍 compute_loss，看 v_t 是否改变——改了就证明通路确实活的。

