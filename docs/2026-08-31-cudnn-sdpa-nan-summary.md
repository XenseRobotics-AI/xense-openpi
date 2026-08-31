# cuDNN fused Attention NaN 问题概述

## 问题原因

训练启用 JAX cuDNN fused Attention 后，前向 loss 起初正常，但反向传播先产生 NaN 梯度，随后参数和 loss 被持续污染。当前运行时为 cuDNN 9.14。

NVIDIA 已确认，旧版 cuDNN SDPA backward 在静态序列长度不是 128 倍数时可能产生错误结果或 NaN，并在 cuDNN 9.15.1 中修复。当前模型的 Attention 序列长度约为：

```text
3 × 256 个图像 token + 200 个文本 token + 50 个 action token = 1018
1018 % 128 = 122
```

这与 NVIDIA 报告的触发条件高度一致。相关 issue 还表明：即使前向输出正确，cuDNN backward 的 `dQ` 仍可能出现整行 NaN。

参考：

- [NVIDIA/cudnn-frontend #160](https://github.com/NVIDIA/cudnn-frontend/issues/160)
- [cuDNN 9.15.1 Release Notes](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.15.1/release-notes.html)

## 尝试解决办法

1. **暂时关闭 cuDNN Attention**：恢复显式 Attention 路径，保证生产训练数值有效。
2. **启用原生 padding 路径**：向 JAX `dot_product_attention` 传入值为完整物理长度的 `query_seq_lengths` 和 `key_value_seq_lengths`，强制 cuDNN 使用 padding kernel；现有 block mask 仍负责实际 token 屏蔽。
3. **升级 cuDNN runtime**：将实际加载版本从 9.14 升到至少 9.15.1，并检查启动日志确认版本。
4. **长度对照实验**：把 `max_token_len` 临时改为 206，使总长度变为 1024，以验证“非 128 倍数”是否为直接触发条件。

以上方案需要从有效 checkpoint 使用固定真实 batch 做逐 step 有限性检查。至少验证 500 个连续 step 的 loss、梯度、参数和优化器状态均为有限值后，才能重新用于生产训练。
