# 实验记录：用 PNG-in-parquet 替代 mp4 能否加快训练

**日期**：2026-08-18 ～ 08-19
**结论**：**不采用。** 性价比不成立——加载快 3.7 倍，训练总时长只快 1.12 倍，代价是 307 倍存储。
**硬件**：h200-16（8×H200 / 192 核 / 2 TB 内存 / 14 TB RAID）
**数据集**：`Xense/optical-module-insertion-0731`（151 episodes / 592,853 帧 / 7 相机 / 30 fps）
**训练配置**：pi05 + RTC，batch 512，num_workers 64，fsdp 8

---

## 1. 动机

`lerobot-record --dataset.video=false` 会把相机帧作为 `image` 特征、以 PNG 字节内嵌进 parquet，
读取时不需要视频解码。假设是：训练时的 H.264 随机寻址解码开销很大，换成 parquet 能显著提速。

初步测量支持这个假设——单样本 `__getitem__` 耗时 290 ms，其中 **99% 是视频解码**：

```
full __getitem__          290.4 ms
  非视频部分（state/action/parquet 读取）   2.2 ms
  视频解码                                288.1 ms  (99%)
```

根因不是"解码慢"，是**随机寻址**：顺序解码 5000 fps/核，随机时间戳解码只有 25 fps/核，差 200 倍。
每次取样都要在 H.264 流里 seek 到最近关键帧再往前解。

---

## 2. 关键前提：先确认瓶颈在哪

分析已有的 40,000 步训练日志（38.4 h）后，假设的前提被推翻了：

| | 时长 | 占比 |
|---|---|---|
| GPU 计算（40000 × 2.76 s） | **30.7 h** | 80% |
| 等数据（2427 步停顿，占 6.1%） | **7.4 h** | 19% |

94% 的步里 `next_batch = 0.03 s`，loader 完全跟得上。
**任何数据侧优化的天花板是 1.24×。**

> 教训：先看 `[TIMING] next_batch=` 再决定要不要优化数据管线。这一步几乎零成本。

---

## 3. 实验方法

- 全量转换 151 episodes → PNG-in-parquet，**保持原分辨率、保留全部 7 路相机**，
  使 A/B 只有"存储格式"一个变量，模型输入逐像素相同
- 两个训练配置除 `repo_id` 外逐字段一致，各跑 260 步，机器无其他负载
- 另测单进程单样本延迟、64-worker 持续吞吐（消费 400 batch 以超过预取深度 128）

**转换验证**：60 个随机样本 × 7 相机，`max |src − dst| = 0.000 / 255`；
action / state / timestamp / index / task 零不匹配。转换无损。

**转换成本**：64 核 22.6 分钟；产出 737 GB，另加约 838 GB Arrow 缓存。

---

## 4. 结果

### 4.1 端到端（260 步）

| 指标 | mp4 | parquet | 倍数 |
|---|---|---|---|
| **总时长** | 1585 s | **1416 s** | **1.12×** |
| GPU dispatch 中位数 | 2.76 s | 2.76 s | 对照一致 ✓ |
| worker spawn | 325 s | 346 s | 0.94× |
| 启动到首个 batch | 523 s | 440 s | 1.19× |
| 停顿步数 | 27 / 260 | 20 / 260 | — |
| 停顿损失时长 | 131 s | 145 s | 0.90× |

### 4.2 加载吞吐

| 测量项 | mp4 | parquet | 倍数 |
|---|---|---|---|
| 单样本（单进程） | 312.5 ms | 61.5 ms | 5.1× |
| 单样本（64 workers） | 311.1 ms | 83.6 ms | 3.7× |
| 持续 batch 生产 | 2.489 s | 0.669 s | 3.7× |
| 相对 GPU（2.76 s/batch）余量 | **1.11×** | **4.13×** | — |

### 4.3 为什么 3.7× 只兑现成 1.12×

GPU 每步 2.76 s，260 步就是 718 s 不可压缩计算。
mp4 的 loader **本来就跟得上**（2.489 vs 2.76 s/batch），只是余量仅 11%。
parquet 把余量拉到 4.1 倍，但这部分 GPU 用不上。

两个数据集都在 page cache 里（2 TB 内存，实测磁盘读约 1 MB/s），
所以 parquet 这个数字还是**最好情况**，内存小的机器复现不出来。

---

## 5. 为什么 parquet 大这么多

平均每帧每相机从 **578 字节** 涨到 **178 KB**。不是 parquet 容器低效（内嵌的就是原始 PNG 字节），
而是视频编码和逐帧图片编码不是一个量级。同一批真实帧实测（head 相机 640×480，300 帧）：

| 编码方式 | 每帧 | 相对原始 RGB |
|---|---|---|
| 原始 RGB | 900.0 KB | 1.0× |
| PNG level 1（本次转换） | 160.7 KB | 5.6× |
| PNG level 6（原生录制） | 133.0 KB | 6.8× |
| WebP 无损 | 74.8 KB | 12.0× |
| JPEG q95 | 47.4 KB | 19.0× |
| JPEG q90 | 32.2 KB | 27.9× |
| H.264 全关键帧 (g=1, crf23) | 8.3 KB | 108.7× |
| H.264 正常 (g=30, crf23) | 1.2 KB | 775.2× |
| **H.264 实际录进数据集的** | **0.3 KB** | 2863.7× |

511 倍差距 = 三件事叠加：

| 成因 | 贡献 | 说明 |
|---|---|---|
| 无损 vs 有损 | 19.4× | PNG 保留每个比特，包括传感器噪点 |
| 帧内 vs 帧间 | 7.1× | 30 fps 相邻帧几乎一样，H.264 只存差分 |
| 编码器激进程度 | 3.7× | 录制码率比 crf23 还低不少 |

**反直觉的推论**：录进去的 mp4 只有 0.3 KB/帧，块效应和细节丢失早已发生。
转成 PNG 是在**无损保存一份已经被压坏的画面**——737 GB 保护的是压缩伪影，不是真实细节。

---

## 6. 转换出的 parquet 与原生录制的差异

功能等价（feature 定义、列结构、读取路径完全一致，`LeRobotDataset` 分不出区别），字节不同：

| 项目 | 原生 `--dataset.video=false` | 本次转换 |
|---|---|---|
| 像素来源 | 相机直出 | mp4 解码结果（已过一轮 H.264 有损） |
| PNG 压缩等级 | 6 | 1 |
| parquet 压缩 | snappy + dictionary | zstd |
| 文件切分 | 累积到 100 MB 换文件 | 每 episode 一个 |
| 该数据集文件数 | 约 7,400 个 | 151 个 |
| row group | 由写入批次决定 | 200 帧 |

两点值得记住：

- 原生用 level 6，比 level 1 小约 17%——同样数据原生录制约 610 GB 而非 737 GB
- `data_files_size_in_mb` 默认 100 MB，内嵌图像后每帧 1.24 MB，每文件只装得下约 80 帧，
  会产生七千多个碎文件。**每 episode 一个文件是更合适的布局**
- 转换**救不回**原始画质。要保住原始画质只能录制时就设 `--dataset.video=false`

---

## 7. 副产品：三个与本结论无关、但值得单独处理的发现

### 7.1 `_query_hf_dataset` 按整行取数（阻断性）

`LeRobotDataset._query_hf_dataset`（`src/lerobot/datasets/lerobot_dataset.py:1021`）
取 50 步 action chunk 时按**整行**索引。mp4 数据集不受影响，因为
`get_hf_features_from_features` 会跳过 `dtype: video` 的列，parquet 里没有像素。
但 image 数据集的图像内嵌在行里，于是每个样本要解 **50 × 7 = 350 张 PNG 再全部丢掉**。

不修的话 parquet **比 mp4 慢 9.4 倍**（3212.5 ms vs 350.5 ms），
会直接导出"parquet 更慢"的错误结论。

```diff
  # src/lerobot/datasets/lerobot_dataset.py :1021  (_query_hf_dataset)
- try:    result[key] = torch.stack(self.hf_dataset[key][relative_indices])
- except: result[key] = torch.stack(self.hf_dataset[relative_indices][key])
+ views = getattr(self, "_column_views", None)
+ if views is None:
+     views = self._column_views = {}
+ view = views.get(key)
+ if view is None:
+     view = views[key] = self.hf_dataset.select_columns([key])   # Arrow 零拷贝
+ result[key] = torch.stack(view[relative_indices][key])
```
（另需在 `_ensure_hf_dataset_loaded()` 里加 `self._column_views = {}` 以便重载时失效）

| | 修复前 | 修复后 | |
|---|---|---|---|
| mp4 | 350.5 ms | 312.5 ms | 1.12× |
| parquet | 3212.5 ms | 61.5 ms | 52× |

输出逐位相同，302 个 dataset 测试通过（残留 2 failed / 8 errors 在未打补丁时同样存在）。

**本次已按要求回滚。** 若日后使用任何 image 数据集，必须先合入此补丁。
对 mp4 路径本身也有 1.12× 的小幅收益。

### 7.2 源数据集的 episode 元数据存在重复行（会坑到任何元数据重写）

`Xense/optical-module-insertion-0731` 的 `meta/episodes/file-000.parquet` 含全部 151 个 episode，
而 `file-001..024` 是 ep 94–140 的重复行（断点续录留下的，值完全一致）。

`load_episodes()`（`src/lerobot/datasets/utils.py:376`）**不去重**，
而 `meta.episodes[i]` 是**按位置**索引的——源数据集能正常工作，
只是因为 `file-000` 恰好在 glob 排序里排第一。

第一版转换继承了重复行，`dataset_from_index` 整体偏移一个 episode，
episode 边界附近的 action chunk padding 就错了。
**像素和所有逐帧字段全程正确**，25 个抽样里只有 3 个暴露出问题——极易漏掉。

> 任何 merge / 筛选 / 转换 episode 元数据的操作，都必须先按 `episode_index` 去重。

### 7.3 停顿的两个真实来源（与存储格式无关）

**epoch 边界**：大停顿每 1157 步一次（592,853 ÷ 512 = 1157.9，正好一个 epoch）。
`TorchDataLoader.__iter__` 在边界重建迭代器，预取队列排空，
然后整个训练循环等**一个 worker 独自造完一整个 batch**（512 × 单样本延迟 = mp4 上 159 s）。
40000 步日志里这项 36 次 × 约 210 s ≈ **2.1 h**。

**worker 轮转**：260 步实验里大停顿精确落在第 63 / 127 / 191 / 255 步，
间隔正好等于 `num_workers=64`，**两个 arm 完全一样**。
torch 严格按 worker 轮转顺序交付 batch，每绕一圈必然等最慢的那个，预取再深也没用。

---

## 8. 建议（未执行，供后续参考）

| 优先级 | 动作 | 收益 | 代价 |
|---|---|---|---|
| 高 | 跳过 4 路未使用的触觉相机解码 | 占 288 ms 解码里的 110 ms（38%），mp4 余量 11% → 约 1.8× | 一个相机白名单 |
| 高 | `TorchDataLoader.__iter__` 换成无限 sampler | 40000 步里约 2.1 h | 约 10 行 |
| 中 | `num_workers` 64 → 128 | 192 核机器上余量翻倍 | 改配置 |
| — | 转 parquet 提速 | 1.12× | 307× 存储 + 每数据集 23 分钟 |

**第一项的依据**：`_get_query_timestamps`（`lerobot_dataset.py:1008`）遍历全部 `meta.video_keys`，
但 `BiFlexivInputs.EXPECTED_CAMERAS` 只用 head 和两个 wrist——
4 路触觉解码完直接丢弃。**不需要转换任何数据。**

**若日后仍要走 parquet**：存预缩放帧。openpi 无任何数据增强，图像直接进
`resize_with_pad(224, 224)`，所以存"保持长宽比缩放但**不加 padding**"的版本
（640×480 → 224×168）行为完全等价，体积降到 27–81 GB。

> ⚠️ 别存成 224×224 把黑边烤进去：`resize_with_pad` 对 float 输入填 **−1.0**，
> 而烤进 uint8 图里的黑边是 **0.0**，模型输入会被悄悄改掉。交给训练时去 pad 则语义不变。

再叠加 JPEG q90（32 KB/帧），全量只要几十 GB——反正原始画质早已被 H.264 压掉了。

---

## 9. 清理记录

实验产物已全部删除，代码已恢复原样（2026-08-19）：

| 项目 | 大小 |
|---|---|
| `Xense/optical-module-insertion-0731-img` 数据集 | 737 GB |
| HF Arrow 缓存 `default-690c2a6ecaa1ab9b` / `default-38ece6c6a8519bc3` | 838 GB |
| `checkpoints/ab_video` + `checkpoints/ab_image` | 62 GB |
| `assets/ab_video`、`assets/ab_image`（norm stats） | 小 |
| `configs/ab_video.yaml`、`configs/ab_image.yaml` | 小 |
| `~/v2i/`（转换器与基准脚本） | 小 |
| `lerobot-xense` 的 `_query_hf_dataset` 补丁 | 已 `git checkout` 回滚 |

**保留**：`default-57dc72e1e9a9d88b`（112 MB，2026-08-13 创建，属于原 mp4 数据集，早于本次实验）。

配套的可视化报告见 `.claude/parquet-vs-mp4.html`。
