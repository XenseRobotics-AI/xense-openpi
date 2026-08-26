# 数据管线优化：跳过触觉解码 + 消除加载器停顿

**日期**：2026-08-19 实现与验证，2026-08-25 生产运行复核
**机器**：ebcloud H100（8×H100 80GB / 192 逻辑核 / 2 TB 内存）
**起因**：`parquet-vs-mp4` 实验记录 §8 里未执行的前两项建议

---

## 0. 结论速览


| 改动                                    | 目标                                | 实测结果                                                   |
| --------------------------------------- | ----------------------------------- | ---------------------------------------------------------- |
| **A. 跳过 4 路未用的触觉解码**          | §8 第一项，预估「约 1.8×」        | 数据侧**1.71×**，端到端 **0%**——预估的前提不成立        |
| **B1. 批次乱序交付** (`in_order=False`) | 每`num_workers` 步一次的约 6 s 停顿 | 受控 A/B：停顿**20 次 → 0 次**，最大等待 7.83 s → 0.43 s |
| **B2. 无限 sampler**                    | epoch 边界重建迭代器                | 边界停顿**63.0 s → 0 s**                                  |
| **B 合计（受控 400 步 A/B）**           | —                                  | 770.8 s →**715.7 s = 1.08×**                             |
| **B 合计（65,000 步生产运行）**         | —                                  | 数据停顿损失**6.84 h → 131 s**，约 **1.14×**             |
| 剩余瓶颈                                | —                                  | 加载器吞吐天花板约 1.84 s/batch，**不是排序问题**          |

一句话：**A 是对的优化但在H100上兑现不了；B 才是真正拿到的收益。**

---

## 1. 改动 A：`tactile` 开关

### 1.1 问题

`LeRobotDataset.__getitem__` 无条件遍历 `meta.video_keys` 解码**全部** 7 路视频流，
而 `LeRobotBiFlexivDataConfig` 的 repack 只取 `head` / `left_wrist` / `right_wrist`：

```
observation.images.head / left_wrist / right_wrist    640×480   ← 用
observation.images.{left,right}_tactile_{0,1}         400×700   ← 解完直接丢
```

H.264 随机寻址解码只有顺序解码的 1/200，所以这 4 路是纯浪费。
（Xense 的 bi_flexiv 数据集全都是这个结构：optical-module-insertion、pick_up_cube、
unscrew-the-bottle-cap、bottle-sorting、stack-cubes 均为 3 用 + 4 触觉。）

### 1.2 实现

- `DataConfig.tactile: bool = False`
- `SelectiveVideoLeRobotDataset` 覆盖 `_get_query_timestamps`，把非白名单的流从查询里滤掉，
  `_query_videos` 就永远不会被要求解它们
- 启动时校验：若 repack 引用了被禁用的流，直接 `ValueError`，不会静默拿到黑图

**没有改 lerobot，没有转换任何数据。**

```yaml
data:
  type: LeRobotBiFlexivDataConfig
  repo_id: Xense/<dataset>
  base_config:
    prompt_from_task: true
    tactile: false        # 省略即为 false；用触觉的模型（Pi05Tactile 等）才设 true
```

### 1.3 结果


| 指标                            | tactile=True | tactile=False       | 倍数       |
| ------------------------------- | ------------ | ------------------- | ---------- |
| 单样本`__getitem__`（H100）     | 262.4 ms     | **153.3 ms**        | **1.71×** |
| 单样本`__getitem__`（RTX 5090） | 192.1 ms     | 103.2 ms            | 1.86×     |
| 全 transform 链输出             | —           | 20 样本**逐位相同** | —         |

省下的 109.1 ms/样本，与 §8 预估的「288 ms 解码里的 110 ms」几乎完全吻合。

**但 400 步受控 A/B 显示端到端毫无变化：**


| 指标               | tactile=True | tactile=False |
| ------------------ | ------------ | ------------- |
| 400 步总时长       | 768.4 s      | 770.8 s       |
| GPU dispatch 合计  | 615.0 s      | 617.6 s       |
| 等数据合计         | 139.6 s      | 139.5 s       |
| 停顿步数（>0.5 s） | 20           | 20            |
| 启动到首个 batch   | 413.7 s      | **376.4 s**   |

唯一实测到的收益是启动快 37.3 s（worker spawn 301 s 两边一样，差的是「单个 worker
独自造完一个 256 batch」的时间）。

### 1.4 为什么 1.71× 兑现成 0

§8 那句「约 1.8×」是 h200 箱 **batch 512** 的推算，当时 mp4 加载相对 GPU 的余量只有 11%。
这台机 batch 256 + 192 核，加载器本来就跟得上 GPU，多出来的余量 GPU 用不上。

更关键的是：第 2 节的诊断显示，剩下的等待是加载器的**吞吐天花板**，
而这个天花板与解码量无关——这正是 tactile 开关一秒都省不下来的直接原因。

### 1.5 保留它的理由

逐位无损、零风险、每样本少烧约 40% 的 CPU、每个 epoch 边界省约 37 s。
换更大 batch 或核更少的机器时它才会真正兑现。**但别指望它让当前配置变快。**

---

## 2. 改动 B：消除Dataloader停顿

### 2.1 诊断装置

为了不必每次花 20 分钟跑训练，做了一个复现装置：真数据 + 真 transform 链 + 真 torch
DataLoader，用 `sleep(1.55)` 替代 GPU 步，同时抓 torch 内部迭代器的两个私有状态：

```
outstanding = _send_idx - _rcvd_idx     已派给 worker 但还没回来的任务数
buffered    = _task_info 里已完成的结果  算完了、但还轮不到它交付的 batch 数
```

**这两个数把「算不出来」和「算出来了但过不去」区分开——是整件事里最值钱的一步。**

### 2.2 发现一：队头阻塞（head-of-line blocking）


| batch | wait   | outstanding | buffered |
| ----- | ------ | ----------- | -------- |
| 64    | 3.02 s | 1024        | 10       |
| 66    | 9.84 s | 1024        | **41**   |
| 128   | 3.16 s | 1024        | 10       |
| 130   | 9.66 s | 1024        | **41**   |

`outstanding` **恒为 1024**（= `prefetch_factor 16 × 64 workers`，任务管线一直是满的），
而停顿发生时 `buffered` 高达 41——**41 个 batch 已经算完躺在那儿，训练循环却在等**。

原因：torch 严格按 sampler 顺序交付。任务 *j* 派给 worker `j mod 64`，主进程必须先拿到
*j* 才能拿 *j+1*。这一轮里最慢的那个 worker，堵住排在它后面的所有 batch。每 64 步绕一圈，
就必然撞上一次最慢的那个。`prefetch_factor` 调多深都没用——问题不是没算，是不让过。

**修复**：`DataLoader(in_order=False)`（PyTorch ≥2.6 为这个场景加的官方参数）。
谁先算完谁先交付。这里排序毫无意义——sampler 本来就是随机打散的。

正确性已验证：`in_order=False` 下一个 pass 仍然**每个 index 恰好出现一次**，不重不漏，
只是交付顺序变了。

### 2.3 发现二：epoch 边界重建迭代器

`TorchDataLoader.__iter__` 在 sampler 耗尽时 `break` 出去重新 `iter()`。预取管线被整个
拆掉重建，训练循环要**等一个 worker 独自造完一整个 batch**。把 epoch 缩短到 80 个 batch
强制触发，实测：

```
EPOCH BOUNDARY at batch 80: rebuilding iterator
STALL batch 80: wait=62.96s outstanding=79 buffered=60
```

**63.0 秒** = 256 样本 × 246 ms/样本。

**修复**：`InfiniteSampler`——永不停止、每轮重新打散，`StopIteration` 再也不会触发。
代价为零：openpi 本来就不 checkpoint 数据加载器状态（`restore_state` 直接
`del data_loader`），batch 序列从来就不可恢复。

### 2.4 诊断装置上的结果

150 个 batch，模拟 1.55 s GPU 步：


| 变体                 | 总时长      | 冷启动 | 等数据 | 停顿数 | 最大等待   |
| -------------------- | ----------- | ------ | ------ | ------ | ---------- |
| 基线                 | 349.5 s     | 66.0 s | 49.9 s | 14     | 9.8 s      |
| `in_order=False`     | 336.3 s     | 57.3 s | 45.7 s | **0**  | **0.42 s** |
| 基线（epoch=80）     | 376.8 s     | 67.1 s | 76.2 s | 6      | **63.0 s** |
| 两项都修（epoch=80） | **330.0 s** | 53.6 s | 43.1 s | **0**  | **0.39 s** |

### 2.5 受控真实训练 A/B（400 步，同机同配置）


| 指标                   | 修复前  | 修复后      | 倍数       |
| ---------------------- | ------- | ----------- | ---------- |
| **400 步总时长**       | 770.8 s | **715.7 s** | **1.08×** |
| GPU dispatch 合计      | 617.6 s | 578.4 s     | 1.07×     |
| 等数据合计             | 139.5 s | 122.7 s     | 1.14×     |
| **停顿步数（>0.5 s）** | **20**  | **0**       | —         |
| 最大`next_batch`       | 7.83 s  | **0.43 s**  | 18×       |
| 中位`next_batch`       | 0.030 s | 0.290 s     | —         |

修复前的 20 次停顿：步 63/64/69、127/128/133、191/192/197、255/256/261、319/320/325、
383/384/389——每 64 步一簇。**修复后一次都没有。**

> 中位 `next_batch` 从 0.030 s 涨到 0.290 s **不是退步**：同样的产能缺口，以前攒成每 64 步
> 一次的 6 秒尖峰，现在均摊到每一步。总等待反而少了 16.8 s。

---

## 3. 生产运行复核（2026-08-25）

**配置**：`pi05_base_bi_flexiv_unscrew_bottle_cap_0824_h100`
（`Xense/unscrew-the-bottle-cap-0821`，100 ep / 84,419 帧，batch 256，workers 64，fsdp 8）
**运行**：64,951 步 / 30.60 h

日志确认补丁生效：`skipping video decode for 4 of 7 stream(s)`，且调用栈行号是
`data_loader.py:352`（旧代码 `:292`）。

### 3.1 与另一次运行的对照（注意：不是受控 A/B）

对照的 `优化前.log` 是**另一个任务、另一个数据集**
（`earbuds_case_insertion_teleop_rtc_0715`），绝对时长不可比，只有归一化后的停顿指标有意义。


|                                      | 优化前（earbuds，99,951 步 / 51.46 h） | 优化后（unscrew，64,951 步 / 30.60 h） |
| ------------------------------------ | -------------------------------------- | -------------------------------------- |
| **平均每步**                         | 1.853 s                                | **1.696 s**                            |
| **数据停顿次数**（`next_batch`>3 s） | **3,345**                              | **36**                                 |
| **数据停顿总损失**                   | **6.84 h**（占 13.3%）                 | **131 s**（占 0.12%）                  |
| 最大`next_batch`                     | 10.32 s                                | 5.38 s                                 |

归一化到每万步：停顿次数 **334.7 → 5.5**（少 61 倍），损失 **2464 s → 20 s**（少 123 倍）。

### 3.2 这次运行省了多少

数据集只有 84,419 帧，**一个 epoch 才 329 步**——65,000 步里有 **197 次 epoch 边界**。
按修复前每次约 63 s 算，光边界就该吃掉 **3.45 h**。

日志验证：落在 epoch 边界附近（±2 步）的数据停顿 **0 次**（无限 sampler 生效）；
落在 `num_workers` 倍数上的也没有（乱序交付生效）。

按优化前的停顿比率外推，这次本该损失约 **4.45 h**，实际只损失 **0.04 h**：


|                  |               |
| ---------------- | ------------- |
| 实际用时         | 30.60 h       |
| 不优化的估计用时 | 约 35.0 h     |
| **提速**         | **约 1.14×** |

### 3.3 残留的 36 次停顿：全是 checkpoint，不是加载器

那 36 次 `next_batch > 3 s` 的步号是 1001、2001、3001…——**36/36 全部落在 checkpoint
保存的下一步**。`save_interval=1000`，65,000 步存了 64 次，每次阻塞 7–11 s（合计 389 s），
异步落盘线程又和 64 个 worker 抢 CPU，导致下一步取数慢 3–5 s。

合计约 **520 s（0.14 h）**，很好摘：

```yaml
save_interval: 5000      # 从 1000 提到 5000，checkpoint 开销降到 1/5
keep_period: 10000
```

---

## 4. 剩下的瓶颈：吞吐天花板，不是排序

修完之后 `buffered` **恒为 0**——结果一到就被取走，中位等待 0.21–0.29 s，峰值 0.36 s。
这说明加载器已经在满负荷跑，稳态产能约 **1.84 s/batch**，而 GPU 每 1.55 s 就要一个。
残留的约 14% 等待就是这个差额。

折算：64 workers 每秒产出 139 样本 = **每 worker 460 ms/样本**，而单进程实测只要 153 ms
——并发到 64 个 worker 时每样本成本涨了 3 倍，在 192 核的机器上并没有线性扩展。

**这个天花板不在解码上**：直接证据是 tactile 开/关对等待时间毫无影响。
候选嫌疑（未验证，按可疑度排序）：

1. 每个 worker 里跑 `jax.image.resize`（`resize_with_pad` 是 jax 实现，64 个 worker 进程各自初始化 JAX）
2. `_collate_fn` 逐样本 `np.stack`
3. numpy 数组经 multiprocessing queue 的 pickle 传输（每 batch 约 115 MB uint8）

**下一步该测什么**：在 64 路并发下给 worker 内的 `__getitem__` 分段计时
（解码 / transform / collate 各占多少），再决定优化哪一段。在此之前别再猜。

---

## 5. 代码位置与开关

改动全部在远程 `~/localstorage/xense-openpi`（**本机仓库保持未改**）：


| 文件                                                          | 改动                                                                                             |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `src/openpi/training/config.py`                               | `DataConfig.tactile`、`TrainConfig.strict_batch_order`                                           |
| `src/openpi/training/data_loader.py`                          | `SelectiveVideoLeRobotDataset`、`_resolve_decode_video_keys`、`InfiniteSampler`、`in_order` 接线 |
| `src/openpi/training/data_loader_test.py`                     | 10 个新测试（`pytest src/openpi/training/` 共 **68 passed**）                                    |
| `configs/README.md`、`configs/_examples/_FULL_REFERENCE.yaml` | `tactile` 文档                                                                                   |

```yaml
data:
  base_config:
    tactile: false          # 默认；true = 解码触觉流

strict_batch_order: false   # 默认；true = 恢复严格顺序交付
```

**两个开关的默认值就是优化后的行为**，已有 yaml 不改也在用；写出来只是让配置自解释。

**`strict_batch_order` 的取舍**：默认 `false` 意味着 batch 的**交付顺序**不再逐次可复现。
每个 index 每轮仍恰好出现一次，样本内容不受影响，但两次同 seed 的训练看到的 batch 顺序会不同。
需要逐 batch 复现时设 `true`，代价是队头阻塞回来。
（注意：因为 openpi 不保存数据加载器状态，**断点续训本来就已经打乱了顺序**。）

---

## 6. 复现

诊断装置与基准脚本在远程 `/root/tac_bench/`：

```bash
conda activate lerobot-xense && cd ~/localstorage/xense-openpi
export PYTHONPATH=$PWD/src

# 触觉解码 A/B + 逐位一致性
JAX_PLATFORMS=cpu python /root/tac_bench/bench_decode.py --n 60

# 停顿诊断（--in-order / --infinite / --epoch-batches 组合出 4 个变体）
JAX_PLATFORMS=cpu python /root/tac_bench/diag_rotation.py \
    --batches 150 --epoch-batches 80 --in-order false --infinite --label fixed

# in_order=False 的覆盖性验证
python /root/tac_bench/check_in_order.py
```

---

## 7. 给后来者的四条

1. **先看停顿的周期再动手。** 周期 = `num_workers` 是队头阻塞，周期 = 一个 epoch 的步数是
   边界重建，周期 = `save_interval` 是 checkpoint——都不是「解码太慢」。看错了会花一天优化
   解码然后发现端到端纹丝不动。
2. **`buffered` 这个数值千金难买。** 停顿时它是 0 说明真的算不出来，是 41 说明算出来了
   过不去——两种情况的修法完全相反。
3. **别把别的机器上的预估当结论。** §8 的「1.8×」在 h200 箱 batch 512 下是对的，
   搬到 batch 256 + 192 核就一点不剩。
4. **别拿两个不同任务的日志比绝对时长。** 要比就比归一化后的指标（每万步的停顿次数与损失），
   或者干脆跑受控 A/B。
