# 数据管线优化：跳过触觉解码 + 消除加载器停顿

**日期**：2026-08-19
**机器**：ebcloud H100（8×H100 80GB / 192 逻辑核 / 2 TB 内存）
**数据集**：`Xense/optical-module-insertion-0731`（151 episodes / 592,853 帧 / 7 相机 / 30 fps）
**配置**：`pi05_base_bi_flexiv_optical_module_insertion_0819_h100`（pi05 + RTC，batch 256，num_workers 64，fsdp 8）
**起因**：`docs/parquet-vs-mp4-experiment.md` §8 里未执行的前两项建议

---

## 0. 结论速览

| 改动 | 目标 | 实测结果 |
|---|---|---|
| **A. 跳过 4 路未用的触觉解码** | §8 第一项，预估「约 1.8×」 | 数据侧 **1.71×**，端到端 **0%** ——预估的前提不成立 |
| **B1. 批次乱序交付** (`in_order=False`) | 每 `num_workers` 步一次的约 6 s 停顿 | 停顿 **14 次 → 0 次**，最大等待 9.8 s → 0.42 s |
| **B2. 无限 sampler** | epoch 边界重建迭代器 | 边界停顿 **63.0 s → 0 s**，短 epoch 场景整体 **1.14×** |
| **B 合计，真实训练 400 步** | — | 770.8 s → **715.7 s（1.08×）**，停顿 **20 次 → 0 次** |
| 剩余瓶颈 | — | 加载器吞吐天花板，约 1.84 s/batch，**不是排序问题** |

一句话：**A 是对的优化但在这台机器上兑现不了；B 才是真正拿到的收益。**

---

## 1. 改动 A：`tactile` 开关

### 1.1 问题

`LeRobotDataset.__getitem__` 无条件遍历 `meta.video_keys` 解码**全部** 7 路视频流，
而 `LeRobotBiFlexivDataConfig` 的 repack 只取 `head` / `left_wrist` / `right_wrist`：

```
observation.images.head / left_wrist / right_wrist         640×480   ← 用
observation.images.{left,right}_tactile_{0,1}              400×700   ← 解完直接丢
```

H.264 随机寻址解码是顺序解码的 1/200，所以这 4 路是纯浪费。

### 1.2 实现

- `DataConfig.tactile: bool = False`（`src/openpi/training/config.py`）
- `SelectiveVideoLeRobotDataset` 覆盖 `_get_query_timestamps`，把非白名单的流从查询里滤掉，
  `_query_videos` 就永远不会被要求解它们（`src/openpi/training/data_loader.py`）
- 启动时校验：若 repack 引用了被禁用的流，直接 `ValueError`，不会静默拿到黑图

**没有改 lerobot，没有转换任何数据。**

YAML 用法（默认即关闭）：

```yaml
data:
  type: LeRobotBiFlexivDataConfig
  repo_id: Xense/optical-module-insertion-0731
  base_config:
    prompt_from_task: true
    tactile: false        # 省略即为 false；用触觉的模型才设 true
```

### 1.3 结果

| 指标 | tactile=True | tactile=False | 倍数 |
|---|---|---|---|
| 单样本 `__getitem__` 中位数（H100） | 262.4 ms | **153.3 ms** | **1.71×** |
| 单样本 `__getitem__` 中位数（RTX 5090） | 192.1 ms | 103.2 ms | 1.86× |
| 全 transform 链输出 | — | 20 样本**逐位相同** | — |

省下的 109.1 ms/样本，与文档 §8 预估的「288 ms 解码里的 110 ms」几乎完全吻合。

**但 400 步真实训练 A/B 显示端到端毫无变化：**

| 指标 | tactile=True | tactile=False |
|---|---|---|
| 400 步总时长 | 768.4 s | 770.8 s |
| GPU dispatch 合计 | 615.0 s | 617.6 s |
| 等数据合计 | 139.6 s | 139.5 s |
| 停顿步数（>0.5 s） | 20 | 20 |
| 启动到首个 batch | 413.7 s | **376.4 s** |

唯一的实测收益是启动快 37.3 s（worker spawn 301 s 两边一样，差的是"单个 worker 独自造完一个
256 batch"的时间）。

### 1.4 为什么 1.71× 兑现成 0

文档 §8 的「约 1.8×」是 h200 箱 **batch 512** 的推算，当时 mp4 加载相对 GPU 的余量只有 11%。
这台机 batch 256 + 192 核，加载器本来就跟得上 GPU，多出来的余量 GPU 用不上。

**更关键的是**：后面第 2 节的诊断显示，剩下那 18% 的等待是加载器的**吞吐天花板**，
而这个天花板与解码量无关——这正是 tactile 开关一秒都省不下来的直接原因。

### 1.5 保留它的理由

逐位无损、零风险、每样本少烧约 40% 的 CPU、每个 epoch 边界省约 37 s。
换更大 batch 或核更少的机器时它才会真正兑现。**但别指望它让当前配置变快。**

---

## 2. 改动 B：消除加载器停顿

### 2.1 诊断方法

400 步训练日志显示 20 次停顿，位置精确落在 **64 的倍数**附近（= `num_workers`），每次约 6 s。
为了不必每次都花 20 分钟跑训练，做了一个复现装置：真数据 + 真 transform 链 + 真 torch
DataLoader，用 `sleep(1.55)` 替代 GPU 步，同时抓 torch 内部迭代器的两个私有状态：

```
outstanding = _send_idx - _rcvd_idx      已派给 worker 但还没回来的任务数
buffered    = _task_info 里已完成的结果   算完了、但还轮不到它交付的 batch 数
```

这两个数把"算不出来"和"算出来了但过不去"区分开。

### 2.2 发现一：队头阻塞（head-of-line blocking）

| batch | wait | outstanding | buffered |
|---|---|---|---|
| 64 | 3.02 s | 1024 | 10 |
| 66 | 9.84 s | 1024 | **41** |
| 128 | 3.16 s | 1024 | 10 |
| 130 | 9.66 s | 1024 | **41** |

`outstanding` **恒为 1024**（= `prefetch_factor 16 × 64 workers`，任务管线一直是满的），
而停顿发生时 `buffered` 高达 41——**41 个 batch 已经算完躺在那儿，训练循环却在等**。

原因：torch 严格按 sampler 顺序交付。任务 j 派给 worker `j mod 64`，
主进程必须先拿到 j 才能拿 j+1。这一轮里最慢的那个 worker，堵住了排在它后面的所有 batch。
每 64 步绕一圈，就必然撞上一次最慢的那个。`prefetch_factor` 调多深都没用——
问题不是没算，是不让过。

**修复**：`DataLoader(in_order=False)`（PyTorch ≥2.6 的官方参数，正是为这个场景加的）。
谁先算完谁先交付。这里排序毫无意义——sampler 本来就是随机打散的。

正确性已验证：`in_order=False` 下一个 pass 仍然**每个 index 恰好出现一次**，不重不漏，
只是 batch 的交付顺序变了。

### 2.3 发现二：epoch 边界重建迭代器

`TorchDataLoader.__iter__` 在 sampler 耗尽时 `break` 出去重新 `iter(self._data_loader)`。
预取管线被整个拆掉重建，训练循环要**等一个 worker 独自造完一整个 batch**。

把 epoch 缩短到 80 个 batch 强制触发，实测：

```
[epoch_base]   EPOCH BOUNDARY at batch 80: rebuilding iterator
[epoch_base]   STALL batch 80: wait=62.96s outstanding=79 buffered=60
```

**63.0 秒**，= 256 样本 × 246 ms/样本。这个量和"启动到首个 batch"是同一回事。
按 60000 步 / epoch 2316 步算，一次训练要撞 25 次。

**修复**：`InfiniteSampler`——永不停止、每轮重新打散的 sampler，
`StopIteration` 再也不会触发，边界消失。

代价为零：openpi 本来就不 checkpoint 数据加载器状态
（`checkpoints.restore_state` 直接 `del data_loader`），batch 序列从来就不可恢复。

### 2.4 结果

150 个 batch，模拟 1.55 s GPU 步，tactile=False：

| 变体 | 总时长 | 冷启动 | 等数据(1..149) | 停顿数 | 最大等待 | epoch 重建 |
|---|---|---|---|---|---|---|
| 基线 | 349.5 s | 66.0 s | 49.9 s | 14 | 9.8 s | 0 |
| `in_order=False` | 336.3 s | 57.3 s | 45.7 s | **0** | **0.42 s** | 0 |
| 基线（epoch=80） | 376.8 s | 67.1 s | 76.2 s | 6 | **63.0 s** | 1 |
| 两项都修（epoch=80） | **330.0 s** | 53.6 s | 43.1 s | **0** | **0.39 s** | **0** |

同样有 epoch 边界的对比：**376.8 s → 330.0 s = 1.14×**，而这只包含 1 次边界。
停顿从"每 64 步一次 6 秒尖峰 + 边界一次 63 秒"变成**一次都没有**。

### 2.5 真实训练验证（400 步，同一配置同一机器）

| 指标 | 修复前 | 修复后 | 倍数 |
|---|---|---|---|
| **400 步总时长** | 770.8 s | **715.7 s** | **1.08×** |
| GPU dispatch 合计 | 617.6 s | 578.4 s | 1.07× |
| 等数据合计 | 139.5 s | 122.7 s | 1.14× |
| **停顿步数（>0.5 s）** | **20** | **0** | — |
| 最大 `next_batch` | 7.83 s | **0.43 s** | 18× |
| 中位 `next_batch` | 0.030 s | 0.290 s | — |

修复前的 20 次停顿：步 63/64/69、127/128/133、191/192/197、255/256/261、319/320/325、383/384/389
——每 64 步一簇，每次 5.4–7.8 s。**修复后一次都没有。**

中位 `next_batch` 从 0.030 s 涨到 0.290 s 不是退步：同样的产能缺口，
以前攒成每 64 步一次的 6 秒尖峰，现在均摊到每一步。总等待反而少了 16.8 s。

这 400 步里**没有** epoch 边界（epoch = 2316 步），所以无限 sampler 的收益不在这张表里。
外推到 60000 步的完整训练：

- 轮转停顿：0.138 s/步 × 60000 ≈ **2.3 小时**
- epoch 边界：25 次 × 63 s ≈ **0.4 小时**
- 合计约 **2.7 小时**，相对约 32 小时的训练即约 **8.5%**

---

## 3. 剩下的瓶颈：吞吐天花板，不是排序

修完之后 `buffered` **恒为 0**——结果一到就被取走，中位等待 0.28 s。
这说明加载器已经在满负荷跑，稳态产能约 **1.84 s/batch**，而 GPU 每 1.55 s 就要一个。
残留的 43 s 等待（约 16%）就是这个差额。

折算下来：64 workers 每秒产出 139 样本 = **每 worker 460 ms/样本**，
而单进程实测只要 153 ms——**并发到 64 个 worker 时每样本成本涨了 3 倍**，
在 192 核的机器上并没有线性扩展。

**这个天花板不在解码上**：直接证据是 tactile 开/关对 139.6 s 的等待毫无影响。
候选嫌疑（未验证，按可疑度排序）：

1. 每个 worker 里跑 `jax.image.resize`（`resize_with_pad` 是 jax 实现，64 个 worker 进程各自初始化 JAX）
2. `_collate_fn` 逐样本 `np.stack`
3. numpy 数组经 multiprocessing queue 的 pickle 传输（每 batch 约 115 MB uint8）

**下一步该测什么**：在 64 路并发下给 worker 内的 `__getitem__` 分段计时
（解码 / transform / collate 各占多少），再决定优化哪一段。在此之前别再猜。

---

## 4. 代码位置

| 文件 | 改动 |
|---|---|
| `src/openpi/training/config.py` | `DataConfig.tactile`、`TrainConfig.strict_batch_order` |
| `src/openpi/training/data_loader.py` | `SelectiveVideoLeRobotDataset`、`_resolve_decode_video_keys`、`InfiniteSampler`、`in_order` 接线 |
| `src/openpi/training/data_loader_test.py` | 9 个新测试（`pytest` 共 **15 passed**） |
| `configs/README.md`、`configs/_examples/_FULL_REFERENCE.yaml` | `tactile` 文档 |

### 开关

```yaml
data:
  base_config:
    tactile: false          # 默认；true = 解码触觉流

strict_batch_order: false   # 默认；true = 恢复严格顺序交付
```

**`strict_batch_order` 的取舍**：默认 `false` 意味着 batch 的交付顺序不再可复现。
每个 index 每轮仍恰好出现一次，样本内容不受影响，但两次同 seed 的训练看到的 batch 顺序会不同。
需要逐 batch 复现时设 `true`，代价是队头阻塞回来。
（注意：因为 openpi 不保存数据加载器状态，**断点续训本来就已经打乱了顺序**。）

---

## 5. 复现

诊断装置和基准脚本在远程机 `/root/tac_bench/`：

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

## 6. 给后来者的三条

1. **先看停顿的周期再动手。** 周期 = `num_workers` 是队头阻塞，周期 = 一个 epoch 的步数是边界重建，
   两者都不是"解码太慢"。看错了会花一天优化解码然后发现端到端纹丝不动。
2. **`buffered` 这个数值千金难买。** 停顿时它是 0 说明真的算不出来，是 41 说明算出来了过不去——
   两种情况的修法完全相反。
3. **别把别的机器上的预估当结论。** §8 的「1.8×」在 h200 箱 batch 512 下是对的，
   搬到 batch 256 + 192 核就一点不剩。
