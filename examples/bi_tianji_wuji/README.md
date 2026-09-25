# 天机双臂 + Wuji 双手部署（无 RTC）

入口为 `python -m examples.bi_tianji_wuji.main`，从 xense-openpi 仓库根目录运行。
使用训练配置 `pi05_base_bi_tianji_wuji_block_sort_0925_h100`
（`configs/_examples/pi05_base_bi_tianji_wuji_block_sort_0925_h100.yaml`）。

客户端移植自 TacXense 已上真机验证的 `examples/bi_tianji_wuji`
（配置 `qwen3_5_2b_bi_tianji_wuji_block_sort_0920_a15_backbone_lr_split_h200`），
上位机侧的状态／动作顺序、图像预处理、执行时序与生命周期均保持一致，只替换了期望的训练配置名。

## 上位机环境

使用 `lerobot-xensehand` conda 环境及该仓库的硬件 SDK。
不要用 openpi 的完整依赖覆盖上位机的 LeRobot 分支。
客户端仅需本仓库的 `xense_client`、NumPy、Pillow、msgpack、websockets、tyro、PyYAML；
recipe 解码还使用硬件环境中的 draccus。无需在上位机加载模型或安装训练依赖。

```bash
conda activate lerobot-xensehand
cd /home/li/hubo/xense-openpi
python -m pip install -e ./packages/xense-client --no-deps
python -m pip install 'tyro>=0.9.5' PyYAML 'websockets>=14' msgpack pillow
python -c 'import lerobot; print(lerobot.__file__)'
python -m examples.bi_tianji_wuji.main --help
```

`lerobot.__file__` 应位于 `/home/li/hubo/lerobot-xensehand`。
配置文件解码和 `--help` 不连接硬件。

## 启动推理服务器

在 GPU 服务器上运行，替换 checkpoint 路径：

```bash
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi05_base_bi_tianji_wuji_block_sort_0925_h100 \
  --policy.dir=/path/to/checkpoints/pi05_base_bi_tianji_wuji_block_sort_0925_h100/<exp>/<step>
```

checkpoint 需包含 `params/` 及 `assets/Xense/TW-block-sort-0918/norm_stats.json`。
`serve_policy.py` 会把 `config` 和 `checkpoint_dir` 写进 server metadata；
上位机在连接机器人之前检查其中的训练配置名，不符则拒绝运行，并把 metadata 写入日志。

## 工位与启动

`recipes/block-sort.yaml` 的 IP、相机序列号及 TCP 来自现有采集 recipe。
首次使用先核对实际工位。双臂世界坐标变换和 Home 关节角默认继承
`TianjiArmWujiConfig`；若采集时覆盖过这些值，部署 recipe 必须写入相同值。
部署模板使用驱动默认速度上限 0.5 m/s、1.0 rad/s，低于采集 recipe 的
1.5 m/s、3.14 rad/s；真机确认跟随效果后按需要调整。
也可用 `--args.robot-recipe /path/to/record.yaml` 读取原采集文件的 `robot:` 部分。
加载器关闭深度、连接时回 Home、断开时手指回零；嵌套 `wuji:` 不支持，使用扁平 `wuji_*` 字段。

```bash
# 真实状态/图像输入，推理并记日志，不下发 policy 动作或复位命令
python -m examples.bi_tianji_wuji.main --args.run dry-run --args.host SERVER_IP

# 短程执行，开始前双臂及双手复位；每次只执行预测前 10 步
python -m examples.bi_tianji_wuji.main --args.run block-sort --args.host SERVER_IP \
  --args.action-horizon 10 --args.max-episode-steps 30

# 正常执行
python -m examples.bi_tianji_wuji.main --args.run block-sort --args.host SERVER_IP
```

dry-run 仍连接并使能驱动：Wuji 底层会发送当前位置保持命令，因此不等同于断电或只读连接。
它不会发送 policy 动作、启动复位、退出回 Home 或手指回零。
正常执行默认在每个 episode 开始前复位，等待双臂和双手完成；
使用 `--args.no-reset-on-start` 可从当前位置开始。
正常完成、Ctrl+C 和异常退出均不自动回 Home，而是取消机械臂流式目标、断开设备。
物理急停仍使用设备急停。

## 数据与时序

- 状态和动作严格使用 58 维：左 TCP 9 + 右 TCP 9 + 左手 20 + 右手 20，
  与 `Xense/TW-block-sort-0918` 的 `action` 字段名逐一对应。
  采集时 86 维 state 末尾的 28 维机械臂关节位置／速度不传给模型
  （训练侧由 `TruncateState(58)` 丢弃）。
- 每只手顺序为 index、middle、pinky、ring、thumb，每指四关节，
  使用驱动具名字段；底层 SDK 的设备关节顺序由 Wuji 驱动处理。
- 图像为三路（head、left_wrist、right_wrist）uint8 RGB，等比例补边至 224×224，发送 CHW。
- 服务端完成归一化、离散状态 token 化、padding 裁剪及 TCP delta 到 absolute 的恢复
  （`use_delta_cartesian_actions: true`，前 18 维 delta、手指 40 维 absolute）；
  上位机直接发送返回的绝对目标，不做额外 delta 累加或手指 0–1 裁剪。
- 每次换块先调用天机 `cancel()`，让机械臂保持最后命令、结束流式控制，
  双手由自身 worker 保持最后目标，然后读取新观测并等待推理。
  请求默认超时 10 秒；超时关闭连接并结束本次运行，不尝试使用迟到响应。
  不放宽驱动 0.5 秒命令 watchdog，不在推理等待期间重发旧轨迹。
- 执行频率默认 30 Hz（与数据集 FPS 一致），每次执行 1–50 步，可用 `--args.action-horizon` 调整。
  推理期间存在停顿，没有 RTC、预取或动作块融合；观测慢时也不会追赶式突发下发。
- `logs/bi_tianji_wuji.log` 追加记录配置、维度顺序、server metadata、
  推理耗时及每一步完整 58 维状态／动作；可用 `--args.log-file` 分开保存每次实验。

首版不包含 LeRobot 数据录制、Pico/Manus 接管或视频订阅。
