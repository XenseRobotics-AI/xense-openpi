
工作分支：

`feature/tac-encoder`

目标不是修改训练算法，在test文件夹里，实现一个**离线 tactile counterfactual inference diagnostic tool**，用于检查已经训练好的 `Pi0TactileFastVit` JAX checkpoint 中，FastViT 触觉信息是否真正影响 action expert 和最终 action chunk。

## 0. 开始编码前必须先阅读现有代码

不要根据猜测重新实现 OpenPI pipeline。

首先阅读并理解至少以下代码：

* `src/openpi/models/pi0_tactile_fastvit.py`
* `src/openpi/models/pi0.py`
* `src/openpi/models/model.py`
* `src/openpi/training/data_loader.py`
* `src/openpi/training/config.py`
* BiFlexiv tactile 对应的 data config / transforms / policy
* 当前 YAML config loader
* 当前 checkpoint/policy 加载代码
* 当前 LeRobot dataset 加载逻辑

重点确认：

1. `Pi0TactileFastVit.embed_suffix()` 如何生成：

   * FastViT features
   * `tactile_proj`
   * 4 tactile tokens
   * suffix tokens
   * suffix mask
   * suffix AR mask
   * `adarms_cond`

2. `Pi0.sample_actions()` 如何：

   * preprocess observation
   * 生成/接收 noise
   * encode prefix
   * 构建 KV cache
   * 每个 flow denoising step 调用 `embed_suffix`
   * 得到 `suffix_out`
   * 取最后 `action_horizon` 个 action hidden tokens
   * 调用 `action_out_proj`
   * 得到 `v_t`
   * Euler update `x_t`
   * 得到最终 action chunk

3. LeRobot v3 数据如何通过当前项目已有的 DataConfig/transforms 转换成模型真正使用的 `Observation`。

必须尽可能复用现有 dataset / transforms / normalization / checkpoint loading，不允许为该 probe 手工重新实现一套不一致的 preprocessing。

---

# 1. 实验目标

输入 YAML 中定义的 500 对样本。

YAML来自sampled_frames.yaml

每一对包含：

* 一个有水瓶子的 `(episode_index, frame_index)`
* 一个无水瓶子的 `(episode_index, frame_index)`

记为：

`Full_i`
`Empty_i`

对每一对执行四次 inference：

### F_F

基础 observation 来自 Full_i；
4 路 tactile 也来自 Full_i。

### F_E

基础 observation 仍来自 Full_i；
只把 4 路 tactile 替换成 Empty_i 的 tactile。

必须保证以下字段与 F_F 完全相同：

* non-tactile RGB
* robot state
* prompt/task
* 其他 observation 字段

只有 tactile 不同。

### E_E

基础 observation 来自 Empty_i；
tactile 来自 Empty_i。

### E_F

基础 observation 来自 Empty_i；
只把 tactile 替换为 Full_i 的 tactile。

500 pairs × 4 runs = 2000 inference runs。

---

# 2. YAML 配置

设计一个清晰的 YAML schema，例如：

```yaml
dataset:
  repo_id: Xense/xxx
  revision: null
  root: null

model:
  config_name: xxx
  checkpoint_dir: checkpoints/xxx/39999

experiment:
  inference_mode: standard
  num_steps: 10
  base_seed: 12345
  batch_size: 1

output:
  dir: outputs/tactile_counterfactual
  shard_size: 50

  save_fastvit_features: true
  save_tactile_tokens: true
  save_suffix_tokens: false
  save_adarms_cond: true
  save_action_hidden: true
  save_v_t: true
  save_x_t: true

pairs:
  - pair_id: 0
    full:
      episode_index: 10
      frame_index: 245
    empty:
      episode_index: 510
      frame_index: 231
```

支持任意 pair 数量，不要把代码写死为 500。

启动时验证：

* `pair_id` 唯一；
* full/empty 条目均存在；
* episode/frame 在 dataset 中存在；
* tactile keys 全部存在；
* 同一个 `(episode, frame)` 不应意外重复，若重复给 warning；
* config model type 确实支持 tactile branch。

---

# 3. Pair noise 设计

counterfactual 实验必须固定 flow initial noise。

对于 pair `i`：

```text
seed_full_base  = base_seed + 2 * pair_id
seed_empty_base = base_seed + 2 * pair_id + 1
```

要求：

```text
F_F 与 F_E 使用完全相同的 initial noise；
E_E 与 E_F 使用完全相同的 initial noise。
```

即：

```text
noise(F_F) == noise(F_E)
noise(E_E) == noise(E_F)
```

必须在 metadata 中保存 noise seed。

不要让 `sample_actions()` 自己为每个 counterfactual run 重新随机产生 noise。

---

# 4. Tactile swap

不要交换 FastViT feature。

第一阶段实验必须交换 **4 路 tactile image**，让数据重新经过完整链路：

```text
tactile RGB
→ tactile preprocessing
→ FastViT
→ tactile_proj
→ tactile tokens
→ action expert
→ action
```

创建独立 helper，例如：

```python
make_counterfactual_observation(
    base_observation,
    tactile_donor_observation,
)
```

严格保证只有配置中定义的 4 个 tactile image keys 及对应 tactile masks 来自 donor。

禁止修改：

* robot state
* normal RGB cameras
* prompt
* task
* timestamps/metadata that affect model input

提供 debug assertions 验证 swap 是否干净。

---

# 5. 不修改 production `sample_actions()` API

不要为了 probe 修改正常部署代码中 `sample_actions()` 的返回值。

实现独立 diagnostic function，例如：

```python
trace_sample_actions(...)
```

数学计算必须与当前 `Pi0.sample_actions()` 等价。

可以抽取公共 helper，但不要改变现有模型的 inference semantics。

首先复制/重构当前 `sample_actions()` 的逻辑，并使其额外返回 trace。

---

# 6. 每次 inference 保存的中间变量

需要记录以下层级。

## Metadata

每个 run：

```text
run_id
pair_id
condition                 # F_F / F_E / E_E / E_F

base_label                # full / empty
tactile_label             # full / empty

base_episode_index
base_frame_index

tactile_episode_index
tactile_frame_index

noise_seed

config_name
checkpoint_dir
num_steps
action_horizon
action_dim
```

---

## Tactile representation

保存：

```text
fastvit_features
```

预期：

```text
(4, fastvit_dim)
```

以及：

```text
tactile_tokens
```

即 `tactile_proj` 后、进入 suffix 的 4 个 token：

```text
(4, action_expert_width)
```

如实现上不宜侵入 `embed_suffix()`，可以增加 diagnostic helper，但必须保证数值与真实 `embed_suffix()` 中使用的 tensor 完全一致。

提供 sanity check：

```text
manual tactile tokens
vs
embed_suffix() 前四个 tokens
```

应该 `allclose`。

---

## Suffix information

根据 YAML 开关保存：

```text
suffix_tokens
suffix_input_mask
suffix_ar_mask
adarms_cond
```

记录实际 shape。

不要假设 `adarms_cond` 一定是 3D。

标准 inference 和 RTC 路径可能不同。RTC目前始终开启

---

## Transformer action hidden state

每一个 denoising step 保存：

```text
action_hidden
```

定义为当前代码中的：

```python
suffix_out[:, -action_horizon:, :]
```

预期单次：

```text
(action_horizon, action_expert_width)
```

---

## Velocity prediction

每一步保存：

```text
v_t = action_out_proj(action_hidden)
```

shape：

```text
(action_horizon, action_dim)
```

---

## Flow trajectory

每一个 denoising step 保存：

```text
timestep
x_t_before
v_t
x_t_after
```

其中必须满足当前 sampler：

```python
x_t_after = x_t_before + dt * v_t
```

最后保存：

```text
final_action
```

并确认其与正常 `sample_actions(..., noise=same_noise)` 的输出数值一致。

---

# 7. 最重要的 equivalence test

在正式实验之前，对一个 observation 执行：

```text
normal_action =
    model.sample_actions(
        ...,
        noise=fixed_noise,
        num_steps=N,
    )

trace_action =
    trace_sample_actions(
        ...,
        noise=fixed_noise,
        num_steps=N,
    ).final_action
```

要求：

```python
np.testing.assert_allclose(...)
```

误差只允许来自合理的浮点误差。

如果 trace implementation 与 production sampler 不一致，脚本必须失败，不能继续跑 2000 次 experiment。

---

# 8. Counterfactual sanity checks

正式运行前，对至少一个 pair 自动检查：

### Check 1

`F_F` 与 `F_E`：

non-tactile camera input 必须完全相同。

### Check 2

state 必须完全相同。

### Check 3

prompt/task 必须完全相同。

### Check 4

4 路 tactile 应存在差异。

报告每一路：

```text
mean_absolute_pixel_difference
```

### Check 5

使用相同 observation + 相同 fixed noise 重复 inference 两次：

final action 应 allclose。

### Check 6

F_F 与 F_E 使用完全相同 initial noise。

E_E 与 E_F 同理。

### Check 7

确认 4 路 tactile masks 有效，并保存 mask。

若任何关键检查失败，默认 fail-fast。

提供可选 `--skip-strict-validation`，但默认不允许忽略。

---

# 9. 数据加载

数据为 Hugging Face 上的 LeRobot v3 dataset。

优先使用仓库目前训练使用的 LeRobot dataset API。

不要直接自己手工扫描 parquet/video，除非现有 LeRobot API 确实不能完成指定 `(episode_index, frame_index)` 随机读取。

需要实现高效的：

```python
get_sample(episode_index, frame_index)
```

若 LeRobotDataset 使用全局 index，需要在初始化阶段建立：

```text
(episode_index, frame_index) → dataset/global index
```

映射。

只建立一次，不要每个 inference 全 dataset 搜索。

---

# 10. 使用训练时相同 transforms

必须通过 config 得到正确的：

* DataConfig
* repack transforms
* data transforms
* normalization stats
* model transforms

目标是使从 dataset 得到的 observation 与真实训练/推理时模型看到的 observation 一致。

不要自己复制归一化常数。

如果 checkpoint/config 中缺少 norm stats 或 tactile key，明确报错。

---

# 11. TrainConfig

原始

TrainConfig(
        name="pi05_base_bi_flexiv_bottle_sorting_0817_fastvit_h100",
        model=pi0_tactile_fastvit_config.Pi0TactileFastVitConfig(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pi05=True,
            enable_training_time_rtc=True,
            max_delay=10,
            tactile_encoder_name="fastvit_t12",
            # Path to a Flax-format FastViT checkpoint produced by
            # scripts/convert_fastvit_torch_to_flax.py. Set to None to train the
            # encoder from scratch.
            tactile_pretrained_path="/root/localstorage/hf_cache/fastvit_t12_apple_dist_in1k_flax/params.safetensors",
        ),
        data=LeRobotBiFlexivTactileDataConfig(
            repo_id="Xense/bottle-sorting-0810",
            use_delta_cartesian_actions=True,
            default_prompt="Pick up the bottle, and place it in the far bin if it is heavy, or in the near bin if it is light.",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        save_interval=10000,
        keep_period=10000,
        ema_decay=None,
        batch_size=256,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            # pi05_base has no tactile branches; allow them to be missing so the
            # freshly-initialized tactile_encoder (FastViT pretrained) and the
            # random-init tactile_proj survive the merge.
            missing_regex=r".*(lora|tactile_encoder|tactile_proj).*",
        ),
        num_train_steps=60000,
        num_workers=64,
        fsdp_devices=8,
    ),

---

# 12. 输出格式

创建一个有日期时间的文件夹，每个run输出一个yaml文件，保证人类可阅读

---

# 13. dtype / 存储

模型计算保持原始 dtype。

写盘时：

* `final_action`, `v_t`, `x_t` 建议 float32；
* `action_hidden` 可以配置保存为 float16/bfloat16 equivalent，以减少空间；
* mask 保存 bool；
* metadata 不存 ndarray。

YAML 增加：

```yaml
output:
  hidden_storage_dtype: float16
```

---

# 14. 自动计算基础 sensitivity metrics

虽然主要目的是保存 raw trace，但运行结束后顺便生成基本 summary。

对于每一 pair：

```text
delta_F =
    final_action(F_F)
    -
    final_action(F_E)

delta_E =
    final_action(E_E)
    -
    final_action(E_F)
```

计算：

```text
final_action_rms
final_action_l2
per_step_l2
per_action_dim_rms
```

同样对每个 denoising step 计算：

```text
action_hidden_rms_difference
v_t_rms_difference
x_t_rms_difference
```

保存到 summary/result metadata。

不要只计算一个全局 mean。

---

# 15. 建议 CLI

例如：

```bash
python scripts/tactile_counterfactual_probe.py \
    --config configs/probes/water_weight_counterfactual.yaml
```

额外支持：

```text
--pair-start
--pair-end
--max-pairs
--dry-run
```

例如测试：

```bash
python scripts/tactile_counterfactual_probe.py \
    --config ... \
    --max-pairs 2 \
    --dry-run
```

`dry-run` 至少验证：

* YAML
* dataset
* checkpoint
* sample indexing
* tactile keys
* observation transform
* model loading

可选择不执行完整 2000 inference。

---

# 16. Batch

首先保证 `batch_size=1` 正确。

结构上允许未来 `batch_size > 1`。

不要为了第一版性能优化牺牲 paired experiment 的正确性。

在 batch 模式中仍必须保证每个 counterfactual pair 使用对应的固定 noise。

---

# 17. 日志

运行时打印类似：

```text
Loaded dataset: ...
Pairs: 500
Expected runs: 2000

Model: Pi0TactileFastVit
Checkpoint: ...
Action horizon: ...
Action dim: ...
FastViT dim: ...
Action expert width: ...

[1/500]
F_F done
F_E done
E_E done
E_F done

delta_F action RMS: ...
delta_E action RMS: ...
```

每写完一个 shard 输出 checkpoint/progress 信息。

程序中断后最好支持 resume，跳过 metadata 中已经完整保存的 run/pair。

---

# 18. 不要做的事情

不要：

1. 修改模型训练 loss；
2. 改变 checkpoint；
3. 重新训练 FastViT；
4. 手工实现一套与训练不同的 normalization；
5. 使用不同 random noise 比较 F_F/F_E；
6. 把 RGB/state 和 tactile 一起 swap；
7. 默认使用 RTC clean prefix；
8. 用 frame-level label 猜测代替 YAML 显式 pair；
9. 为了方便而绕过原模型 preprocessing；
10. 在没有验证 trace sampler 与 production sampler 等价前运行完整实验。

---

# 19. 测试

至少增加以下 unit/integration tests：

### YAML parsing

能正确读取 pair config。

### Dataset indexing

指定 episode/frame 得到正确 sample。

### Tactile-only swap

swap 后只有 tactile fields 发生变化。

### Noise pairing

counterfactual 两次 run noise 完全一致。

### Trace equivalence

`trace_sample_actions().final_action`
与
`sample_actions(... fixed noise)`
allclose。

### Shape validation

检查：

```text
fastvit_features
tactile_tokens
action_hidden
v_t
x_t
final_action
```

shape 与 model config 一致，不写死 50/20/1024。

---

# 20. 完成后请给我

完成编码后不要只说“已实现”。

请输出：

1. 新增/修改了哪些文件；
2. 每个文件职责；
3. 实际读取到的 dataset key 名；
4. 实际读取到的 tactile key 名；
5. 实际 model class；
6. FastViT feature shape；
7. tactile token shape；
8. suffix shape；
9. adarms_cond shape；
10. action_hidden shape；
11. v_t shape；
12. final_action shape；
13. trace vs production sampler equivalence test 结果；
14. 一条只跑 2 pairs 的测试命令；
15. 完整 500 pairs 的正式运行命令。

如发现当前仓库 API 与本文假设不一致，以仓库当前 `feature/tac-encoder` 代码为准，不要强行套用这里的函数名；在最终说明中明确指出差异。
