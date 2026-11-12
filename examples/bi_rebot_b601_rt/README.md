# BiRebot B601 RT 推理（含 RTC）

对应 `Xense/bi_rebot_09_pick_and_place_1023` 和
`LeRobotBiRebotB601DataConfig`。沿用 Flexiv 示例的 WebSocket policy、
`RTCActionChunkBroker`、Runtime / DecoupledRuntime。

## 启动

在 GPU 机器启动训练所得 checkpoint：

```bash
python scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_base_bi_rebot_09_pick_and_place_rtc_1023_h100 \
  --policy.dir=/path/to/checkpoint
```

机器人端在 xense-openpi 根目录运行，使用安装了当前 `lerobot-xense`、
Rebot 原生 SDK 和本仓库 `xense-client` 的环境（与采集数据的环境一致）：

```bash
python -m examples.bi_rebot_b601_rt.main \
  --args.run pick-and-place-rtc --args.host <server-ip>
```

检查相机、状态和模型输出时可使用 dry-run。它仍会连接机器人，执行 recipe
中的启动复位、夹爪初始化和退出回零，**只禁止下发模型动作**：

```bash
python -m examples.bi_rebot_b601_rt.main \
  --args.run dry-run --args.host <server-ip>
```

所有参数支持 `--help`；优先级为默认值 < run YAML < CLI。普通 chunk 推理：

```bash
python -m examples.bi_rebot_b601_rt.main \
  --args.robot-recipe dm-taccap00-09 --args.host <server-ip> \
  --args.action-horizon 50
```

也可直接加载 lerobot 的采集/遥操作 recipe，读取其中的 `robot:` 块。
相对 URDF 路径按 recipe 所在目录及其祖先目录解析，支持从 openpi 目录启动：

```bash
python -m examples.bi_rebot_b601_rt.main \
  --args.robot-recipe /path/to/lerobot-xense/recipes/record/bi_rebot_b601_rt/dm-taccap00-09.yaml \
  --args.rtc-enabled --args.host <server-ip>
```

## 数据和硬件约定

- 20 维顺序：左 TCP `[0:9]`、左夹爪 `[9]`、右 TCP `[10:19]`、右夹爪 `[19]`。
  两臂 TCP 各自在自己的基座坐标系中，旋转为矩阵前两列。
- 驱动的 `head`、`left_left_wrist`、`right_right_wrist` 分别映射到 policy 的
  `head`、`left_wrist`、`right_wrist`。腕部双前缀来自双臂驱动，匹配实际数据。
  相机名称变化时用 `--args.head-camera`、`--args.left-wrist-camera`、
  `--args.right-wrist-camera` 指定；缺少任何一路会报错。
- 模型输出是绝对 TCP 目标。客户端不再做 delta 转换；外置夹爪限制在 `[0, 1]`。
  两臂动作的有限性和旋转有效性均在发送左臂之前检查。
- 默认不采集触觉。`--args.task` 可覆盖 checkpoint 的默认任务文本。
- `recipes/dm-taccap00-09.yaml` 使用平台 09 采集 recipe 的硬件、控制参数和复位姿态，
  通过 `motor_family: dm` + `taccap_follower` 选择 SDK 内置的
  `DM_TacCap00/urdf/rebot_b601_dm_taccap_v1_gripper.urdf` 及对应 offset 文件。
  双臂 USB 桥按右臂先插、左臂后插对应 `auto:first` / `auto:second`。
  迁移到其他平台时应替换 recipe，保持模型、TCP、夹爪和相机与训练数据相符。

## RTC 和执行节奏

`--args.rtc-enabled` 启用异步取动作、延迟估计、剩余动作前缀引导及队列合并。
客户端向服务端发送 **绝对 20 维动作前缀**；服务端使用当前状态及 Rebot
DataConfig 的 `[9 TCP, 1 gripper, 9 TCP, 1 gripper]` mask 做 delta 编码和归一化。
服务端需要本仓库支持 RTC 的代码及上面的 RTC 训练配置。

默认 `runtime_hz: 30` 与采集频率一致，底层 Rebot 控制器自行做 100 Hz 插值。
队列剩余 20 步时请求下个 chunk，初始延迟 4 步，延迟余量 2 步，默认不开线性混合。
dry-run 会打印 RTC 延迟、队列长度等指标。

可用 `--args.action-hz 30` 解耦观测与动作线程，保持动作消费频率独立于图像读取。
此时 RTC 延迟估计使用 `action_hz`；额外的 paced 队列默认仅 1 步，减少前缀之外
的缓存延迟。提高 `action_hz` 会加快执行训练轨迹，不会自动对动作做时间插值。

启动只执行一次初始复位，后续 episode 使用驱动的 `rt_moving` 等待真实复位完成；
超时/故障直接传播。`--args.no-go-to-start` 关闭启动和 episode 复位；
`--args.reset-on-exit False` 关闭断开时回零。Ctrl+C 停止执行并清理连接，
第二次 Ctrl+C 强制退出。dry-run 和测试均不能代替真机验证。
