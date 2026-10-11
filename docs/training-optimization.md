# 训练优化

JAX Gemma attention 在训练和推理中统一使用显式实现。cuDNN fused attention 的配置、
FP16 custom VJP 和专用数值检查已移除；模型参数结构没有变化，旧 checkpoint 权重仍可加载。
本次变更是实现清理，与 NaN 问题的修复分开处理。显式路径下也有训练发散记录，切换
attention 实现不能作为 NaN 已解决的证据。

## 配置迁移

升级前，从所有用于训练或推理的 YAML 的 `model` 块中删除
`use_cudnn_attention` 和 `cudnn_attention_dtype`，无论原值是 true、false 还是何种 dtype。
也要删除 Python 模型构造调用和 CLI 中对应的参数。

没有忽略旧字段的兼容层。YAML loader 直接构造配置对象，遗留字段会触发
`TypeError: unexpected keyword argument`。迁移时需要检查：

- 仓库中的共享 example YAML，以及各台训练机上被 gitignore 的 `configs/*.yaml`。
- 线上推理服务和 TRON2 上机服务使用的 YAML。`serve_policy` 按配置名重新读取配置，
  即使 checkpoint 权重兼容，旧 YAML 仍会导致服务启动失败。
- 尚未合入的 `feature/rlt-*`、`feat/tac-rex`、`feature/tac-encoder-ltp`、`TianjiWuji`
  和 `feat/insert-test-tube-*` 分支。rebase 或合并时清理配置字段及代码引用，避免重新引入。

`config_yaml_test.py` 会加载所有共享 example YAML，并检查配置枚举；它不能覆盖
其他机器上的私有配置。部署前应在实际训练和服务环境中确认所用配置可加载。
历史诊断快照记录的是当时的配置，重新用作训练配置时也需要迁移。

## 生产配置

Pi0.5 的优化示例见
[`pi05_base_bi_flexiv_earbuds_case_insertion_teleop_rtc_0904_h100.yaml`](../configs/_examples/pi05_base_bi_flexiv_earbuds_case_insertion_teleop_rtc_0904_h100.yaml)。

```yaml
model:
  type: Pi0Config
  pi05: true
  max_token_len: 128
  batch_image_views: true
data:
  base_config:
    tactile: false
strict_batch_order: false
```

`max_token_len: 128` 只针对上述耳机盒数据集做过验证：历史采样的 4037 个状态
（含各维极值样本）最长为 116 tokens。更换数据集、prompt 或状态维度后需重新检查
完整 transform 链的 token 长度；tokenizer 超长会警告并截断。不要直接套用 128。
训练前也需重新计算 norm stats，并检查 `lr_schedule.decay_steps` 是否符合训练计划。

## 保留的优化

- DataLoader worker 返回 CPU Torch Tensor，通过共享内存传输 batch；主进程以
  `tensor.numpy()` 的零拷贝视图构造 JAX array，减少大批量 NumPy 数据的进程间复制。
- 训练结束显式关闭 DataLoader，回收 worker。
- 梯度和参数范数仅在 `log_interval` 步计算，减少全模型归约；loss 每步计算。
- `batch_image_views` 将多路相机按样本交错合批，只调用一次 SigLIP，然后还原各视角。
- 跳过未使用的触觉视频、乱序 batch 交付和无限 sampler。

`strict_batch_order: false` 可避免队头阻塞，但交付顺序不保证跨运行一致；对比训练数值时设为 `true`。
Gemma 与 SigLIP 保持默认的全量 rematerialization；历史 8×H100、batch 256 实验中，
关闭 remat 或保存更多残差均超出显存容量。

## 历史性能与当前验证

历史实现和排障过程可通过 Git 提交查阅：`476babe` 引入 fused attention；
`f42992e`、`ad08973`、`895811c` 记录生产验证和数值排障；`df9755f`、`3ea2b1c`
引入并优化 FP16 custom VJP；`cf07bbe`、`cf03854`、`e13ac0f` 涉及其他优化及文档整理。
这些提交同时涉及不同功能，本次只定向删除 attention 路径，保留其他优化。

2026-09-04，旧 FP16 VJP 初版在 8×H100、全局 batch 256、固定 seed 和严格 batch 顺序下，
从 pi05_base step 0 起跑，通过了当时的 3000 步验收。后续直接调用私有 forward/backward rule
的版本另做过真实激活等价性检查和 400 步对照。这些结果仅描述已移除的历史实现。

同日 H100 端到端实验中，旧 FP16 基线为 1.328 s/step；token 长度 200→128 后为
1.246 s/step；单独加入下述 XLA 标志为 1.281 s/step；二者组合为 1.193 s/step，
相对基线减少约 10.2%。口径为 400 步运行中 step 100 到 step 300 的日志墙钟差除以 200。
这些数字不是当前显式路径的吞吐，也不能直接推导删除后的性能变化。

删除后的 GPU 验证重点如下，实测前不宣称提速比例或 batch 512 已通过：

- 吞吐：在同一机器固定模型、数据、seed、严格 batch 顺序、batch、token 长度、XLA 标志、
  优化器及诊断逻辑，分别运行删除前后的版本。至少运行 400 步，按 step 100 到 300 的
  墙钟差除以 200，比较显式路径与旧实现的步时。
- 显存：H100 的 8 卡、全局 batch 256 已有显式路径运行经验；H200 耳机盒配置的
  8 卡、全局 batch 512、token 200 需要单独实测峰值显存和训练情况，不能用 batch 256 推断。
- 数值：检查 loss、梯度、参数和 optimizer state 是否有限，继续观察训练趋势。
  短跑不替代收敛验证；非有限值防护和 NaN 排障属于独立工作。

## 环境与启动

保持 pip CUDA 12.8 库来源一致，避免将 `$CONDA_PREFIX/lib` 加到 `LD_LIBRARY_PATH` 前面。
cuDNN 由 torch wheel 精确 pin 带入（jax-cuda12-plugin 只要求 `>=9.1,<10`）。
显式 Gemma attention 不再使用 fused attention 的 head_dim=256 专属版本下限；
CUDA 栈中的其他算子仍可能使用 cuDNN，库来源和版本一致性检查继续保留。

运行 `check_cuda_stack.py` 检查实际加载的库来源、cuDNN 文件版本及运行时版本一致性，
避免混装 dispatcher 和 engine。该脚本不再执行 attention 梯度比对，也不验证训练收敛性。

```bash
env -u LD_LIBRARY_PATH python scripts/check_cuda_stack.py

env -u LD_LIBRARY_PATH \
  XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true \
    --xla_gpu_all_gather_combine_threshold_bytes=1073741824 \
    --xla_gpu_reduce_scatter_combine_threshold_bytes=1073741824 \
    --xla_gpu_all_reduce_combine_threshold_bytes=1073741824 \
    --xla_gpu_enable_pipelined_all_gather=true \
    --xla_gpu_enable_pipelined_reduce_scatter=true \
    --xla_gpu_enable_while_loop_double_buffering=true" \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  python scripts/train.py <config> --exp-name=<name>
```

这些 XLA 标志作为整体做过历史验证，未逐项测量贡献。继续已有实验使用 `--resume`；
仅在确实需要覆盖目标实验目录时使用 `--overwrite`。
