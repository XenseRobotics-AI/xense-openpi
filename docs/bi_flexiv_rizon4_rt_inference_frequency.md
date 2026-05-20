# BiFlexiv Rizon4 RT RTC 推理流程与频率分析

本文按以下客户端启动命令梳理推理流程和频率关系：

```bash
python -m examples.bi_flexiv_rizon4_rt.main \
    --args.host 192.168.142.221 \
    --args.port 8000 \
    --args.bi-mount-type diagonal \
    --args.inner-control-hz 1000 \
    --args.interpolate-cmds \
    --args.runtime-hz 30 \
    --args.rtc-enabled
```

这条命令启用了 RTC，因此客户端执行路径是：

```text
Runtime(30Hz)
  -> PolicyAgent
  -> RTCActionChunkBroker
  -> WebsocketClientPolicy
  -> policy server
  -> Policy.infer()
  -> model.sample_actions / training_time_rtc_sample_actions
  -> action chunk
  -> RTC ActionQueue
  -> 每个 runtime step 取一个 20D action
  -> robot.send_action(action_dict)
```

## 1. 本次运行参数

显式传入的参数：

| 参数 | 本次值 | 作用 |
|---|---:|---|
| `host` | `192.168.142.221` | policy server 地址 |
| `port` | `8000` | policy server 端口 |
| `bi_mount_type` | `diagonal` | 传给 LeRobot `BiFlexivRizon4RTConfig` 的安装方式 |
| `inner_control_hz` | `1000` | 机器人底层 RT 控制频率参数 |
| `interpolate_cmds` | `True` | 传给 LeRobot，用于底层命令插值 |
| `runtime_hz` | `30` | 客户端 runtime 目标循环频率 |
| `rtc_enabled` | `True` | 使用 `RTCActionChunkBroker` |

未显式传入但本次仍生效的默认参数：

| 参数 | 默认值 | 本次影响 |
|---|---:|---|
| `action_queue_size_to_get_new_actions` | `30` | RTC 队列剩余动作数小于等于 30 时触发新推理请求 |
| `execution_horizon` | `50` | 转发给 policy/model 的 RTC 参数；客户端队列触发逻辑不直接用它 |
| `blend_steps` | `0` | RTC merge 时不做线性 blend |
| `default_delay` | `4` | RTC warmup/初始延迟估计，单位是 runtime step |
| `dry_run` | `False` | action 会实际发给机器人 |
| `record` | `False` | 不启用 recorder subscriber |
| `pico4_intervention` | `False` | 不启用人工干预 |

注意：`Args.bi_mount_type` 注释中写的是 `"forward" or "side"`，但 `main.py` 本身没有校验字符串，会把 `diagonal` 原样传给 LeRobot 配置；是否支持该值取决于外部 `lerobot` 机器人实现。

## 2. 三个频率概念

这套系统里有三层频率，不能混成一个概念。

| 概念 | 本文含义 | 本次运行值 |
|---|---|---:|
| 客户端单步动作消费频率 | `RTCActionChunkBroker.infer()` 每次从 `ActionQueue` 取出一个 20D action 的频率 | 目标 `30Hz` |
| Python 动作下发频率 | `environment.apply_action()` 调 `robot.send_action(action_dict)` 的频率 | 目标 `30Hz` |
| 机器人底层伺服频率 | LeRobot/Flexiv RT 内部控制线程频率，由 `inner_control_hz` 配置 | `1000Hz` |

`runtime_hz=30` 让 `Runtime` 目标周期为：

```text
1 / 30 = 33.3ms
```

每个 runtime step 的顺序是：

```text
observation = environment.get_observation()
action = agent.get_action(observation)
environment.apply_action(action)
subscriber.on_step(observation, action)
```

在本次运行中没有 dry-run，所以 `apply_action()` 会进入 `BiFlexivRizon4RTRealEnv.step()`，把 20D action 转成 `action_dict` 并调用：

```python
self.robot.send_action(action_dict)
```

因此：

```text
客户端单步动作消费频率 ≈ Python 动作下发频率 ≈ 30Hz
机器人底层伺服频率 = 1000Hz
```

30Hz 是 Python 侧新目标动作下发节奏；1000Hz 是底层控制层如何跟踪、保持或插值这些目标动作的节奏，具体实现在外部 `lerobot` 包中。

## 3. RTC 客户端流程

启用 `--args.rtc-enabled` 后，`main.py` 实例化：

```python
RTCActionChunkBroker(
    policy=ws_client_policy,
    frequency_hz=args.runtime_hz,  # 本次为 30
    action_queue_size_to_get_new_actions=args.action_queue_size_to_get_new_actions,  # 默认 30
    rtc_enabled=True,
    execution_horizon=args.execution_horizon,  # 默认 50
    blend_steps=args.blend_steps,  # 默认 0
    default_delay=args.default_delay,  # 默认 4
    delta_state_dim=18,
)
```

RTC broker 有两个并行角色：

1. **主线程角色**：每个 runtime step 从 `ActionQueue` 取一个单步 action 返回给 `Runtime`。
2. **后台线程角色**：当队列剩余动作数小于等于阈值时，异步向 policy server 请求新的 action chunk。

主线程逻辑：

```text
每 33.3ms 左右：
  更新 latest observation
  从 ActionQueue.get() 取一个 action
  返回 {"actions": action}
  environment.apply_action()
```

后台线程触发条件：

```python
if self._action_queue.qsize() <= self._action_queue_size_to_get_new_actions:
    request new action chunk
```

本次阈值为：

```text
Q = action_queue_size_to_get_new_actions = 30 steps
```

也就是当队列里剩余不超过 30 个单步动作时，后台线程开始请求下一块动作。

## 4. RTC 下推理请求频率和动作块生成频率

### 4.1 二者是否一样

在 RTC 下：

```text
server 推理请求频率 = 客户端后台线程向 server 发起 WebSocket infer 请求的频率
动作块生成频率 = server 完成 policy.infer 并返回一个 action chunk 的频率
```

正常稳定运行、没有请求失败时，二者基本一致，因为：

```text
一次 WebSocket infer 请求 -> server 执行一次 Policy.infer() -> 返回一个 action chunk
```

它们不是两个独立时钟。区别只在时间相位：请求先发出，动作块会在一次网络往返和模型推理延迟后返回。

### 4.2 本次运行的近似频率

设：

```text
f = runtime_hz = 30Hz
Q = action_queue_size_to_get_new_actions = 30
H = server 每次返回的 action chunk 长度
```

对当前 BiFlexiv Pi05 配置，模型默认 `action_horizon=50`，并且 `BiFlexivOutputs` 输出 `[50, 20]` 动作块。因此通常取：

```text
H ≈ 50
```

在稳定状态下，队列从约 50 个动作被客户端以 30Hz 消费到 30 个动作时触发下一次推理：

```text
触发间隔 ≈ (H - Q) / f
         ≈ (50 - 30) / 30
         ≈ 0.667s
```

所以：

```text
server 推理请求频率 ≈ 1 / 0.667s ≈ 1.5Hz
server action chunk 生成频率 ≈ 1.5Hz
```

这只是近似值。实际频率会受到以下因素影响：

- server 模型推理耗时；
- WebSocket round-trip latency；
- RTC merge 时按 `real_delay` 截掉新 chunk 前几步；
- runtime 是否稳定跑到 30Hz；
- 后台线程 1ms 轮询粒度；
- server 实际模型配置的 `action_horizon` 是否为 50。

## 5. RTC 队列 buffer 与延迟

本次队列触发阈值 `Q=30`，runtime 为 `30Hz`，因此触发新推理时旧队列还能支撑：

```text
buffer 时间 = Q / runtime_hz = 30 / 30 = 1.0s
```

含义是：当后台线程开始请求新 chunk 时，如果 server round-trip + 模型推理 + merge 能在约 1 秒内完成，主线程通常不会耗空队列。

RTC broker 会把推理耗时转换成 step 数：

```python
time_per_step = 1.0 / frequency_hz
inference_delay_steps = ceil(latency / time_per_step)
```

本次：

```text
1 step = 33.3ms
default_delay = 4 steps ≈ 133ms
```

正常运行时，下一次传给模型的 `estimated_delay_steps` 会基于最近真实 delay 的最大值再加 `delay_margin=2`，并 clamp 到当前队列可用长度内。

RTC 的目标是让模型生成新 chunk 时冻结前 `estimated_delay_steps` 个动作，使新 chunk 前缀与旧队列即将执行的动作一致；merge 时按实际消耗步数 `real_delay` 截断新 chunk，从而减少切换抖动。

## 6. RTC warmup

`Runtime._run_episode()` 在正式控制循环前调用：

```python
warmup_obs = environment.get_observation()
agent.warmup(warmup_obs)
```

RTC broker 覆写了 `warmup()`，会在正式控制循环前等待后台线程完成两个阶段：

1. Phase 1：第一次推理，用于 JIT/编译，结果不执行。
2. Phase 2：带 `prev_chunk_left_over` 形状的第二次推理，结果 merge 到队列，作为正式控制循环的初始动作来源。

因此，本次 RTC 模式下，第一步控制通常不会再承担 JIT 编译延迟；正式 30Hz runtime 开始前，队列已经有可执行动作。

## 7. Policy server 侧流程

server 侧入口通常是：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_base_bi_flexiv_pack_6_cosmetic_bottles_lora \
    --policy.dir=<checkpoint_dir>
```

`scripts/serve_policy.py` 本身没有固定 Hz。它做的是：

1. 根据 `--policy.config` 找到 `TrainConfig`。
2. 通过 `policy_config.create_trained_policy()` 加载 checkpoint 和 norm stats。
3. 组装 `openpi.policies.policy.Policy`。
4. 启动 `WebsocketPolicyServer` 等待客户端请求。

每收到一次客户端 WebSocket 消息，server 执行：

```text
unpack observation
extract __rtc_kwargs__
policy.infer(obs, **rtc_kwargs)
pack and send action chunk
```

因此 server 侧频率完全由客户端 RTC broker 的请求节奏驱动。

## 8. 视觉、语言与动作块的关系

本次客户端每个 runtime step 都会更新 latest observation，但不是每个 observation 都会发送到 server 推理。只有 RTC 后台线程触发请求时，才会把当时最新的 observation 发给 server。

对当前 BiFlexiv 输入：

```python
{
    "state": obs["qpos"],
    "images": {
        "head": ...,
        "left_wrist": ...,
        "right_wrist": ...,
    },
    "images_raw": ...,  # 录制用，policy transform 不使用
}
```

server 侧 `Policy` 输入 transform 会做：

1. `BiFlexivInputs()`：把三路相机映射为模型标准 key：
   - `head` -> `base_0_rgb`
   - `left_wrist` -> `left_wrist_0_rgb`
   - `right_wrist` -> `right_wrist_0_rgb`
2. `Normalize(...)`：归一化 state/action 相关数据。
3. `ResizeImages(224, 224)`：确保图像分辨率是模型输入尺寸。
4. `TokenizePrompt(...)`：将语言指令 token 化。Pi05 下 state 也会作为离散输入并入 prompt token 流。
5. `PadStatesAndActions(action_dim=32)`：模型内部动作维度 pad 到 32。

当前客户端通常不发 `prompt`，语言指令多半来自 server 侧：

- `serve_policy.py --default_prompt`，或
- 训练配置中的 `LeRobotBiFlexivDataConfig.default_prompt`。

例如 `pi05_base_bi_flexiv_pack_6_cosmetic_bottles_lora` 配置自带打包六个化妆品瓶子的默认英文指令。

所以本次 RTC 下的频率关系是：

```text
视觉观测进入模型的频率 ≈ server 推理请求频率 ≈ 1.5Hz
语言 prompt tokenize/embed 频率 ≈ server 推理请求频率 ≈ 1.5Hz
语言指令内容更新频率通常 = 0Hz  # 内容是静态默认 prompt
server action chunk 生成频率 ≈ 1.5Hz
客户端单步动作消费/下发频率 ≈ 30Hz
机器人底层伺服频率 = 1000Hz
```

也就是说，server 侧不是每个 30Hz 单步动作都重新看一次图像和语言；它大约每 0.67 秒拿一次最新视觉观测和同一条语言指令，生成一块约 50 步的动作序列，然后客户端在 RTC 队列中 merge 并以 30Hz 逐步下发。

## 9. 本次运行的关键结论

```text
客户端模式 = RTC
runtime_hz = 30Hz
inner_control_hz = 1000Hz
action_queue_size_to_get_new_actions = 30
execution_horizon = 50
blend_steps = 0
default_delay = 4 steps ≈ 133ms
模型 action_horizon ≈ 50
```

频率汇总：

| 项目 | 本次估计频率 | 说明 |
|---|---:|---|
| 客户端单步动作消费频率 | `30Hz` | 每个 runtime step 从 RTC 队列取一个 20D action |
| Python 动作下发频率 | `30Hz` | 每个 runtime step 调一次 `robot.send_action` |
| 机器人底层伺服频率 | `1000Hz` | 由 `inner_control_hz=1000` 传给 LeRobot |
| server 推理请求频率 | 约 `1.5Hz` | 队列从 50 消费到 30 时触发下一次请求 |
| server action chunk 生成频率 | 约 `1.5Hz` | 一次请求返回一个 chunk，时间上晚于请求发起 |
| 视觉观测进入模型频率 | 约 `1.5Hz` | 每次 server 推理使用触发时最新 observation |
| 语言 prompt 参与计算频率 | 约 `1.5Hz` | 内容通常是静态默认 prompt |
| 语言内容更新频率 | 通常 `0Hz` | 除非客户端或 server 改变 prompt |

一次完整关系可以概括为：

```text
约每 0.667s：
  RTC 后台线程向 server 发送一次最新 observation
  server 用三路图像 + 当前 state + 默认 prompt 生成一个 action chunk
  RTC broker 将新 chunk merge 到队列

约每 33.3ms：
  runtime 从 RTC 队列取一个 20D action
  Python 调 robot.send_action(action_dict)

约每 1ms：
  底层 LeRobot/Flexiv RT 控制层按 1000Hz 执行/插值/跟踪目标
```

## 10. 调参提示

1. 如果希望模型更频繁地利用新视觉观测，需要提高 server 推理请求频率。本次 RTC 下主要由 `action_queue_size_to_get_new_actions`、模型 chunk 长度 `H` 和 `runtime_hz` 共同决定。
2. 增大 `action_queue_size_to_get_new_actions` 会更早请求新 chunk，视觉更新更频繁，队列 buffer 更大，但 server/GPU/network 压力也更大。
3. 降低 `runtime_hz` 会降低动作下发频率，并延长同样 step 数对应的真实时间；`default_delay=4` 在 30Hz 是 133ms，在 20Hz 会变成 200ms。
4. 如果出现 `Action queue exhausted`，说明推理/网络延迟相对队列 buffer 太大，优先检查 server latency、网络和 `action_queue_size_to_get_new_actions`。
5. `execution_horizon=50` 会传给模型作为 RTC 参数，但客户端后台线程是否触发新请求主要看 `ActionQueue.qsize() <= action_queue_size_to_get_new_actions`。
