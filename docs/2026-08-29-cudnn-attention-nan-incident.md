# cuDNN Attention 全空 Mask 导致训练 NaN

**日期**：2026-08-29

**机器**：8×H200

**配置**：`pi05_base_bi_flexiv_earbud_case_insertion_0826_h200`

**影响范围**：从 clean step 16000 切换到 `use_cudnn_attention=true` 后的生产训练

## 摘要

生产训练恢复后，第一个记录窗口已经出现 NaN 梯度，随后 loss 和参数持续为 NaN。根因不是 cuDNN 动态库版本、数据加载、显存或普通的梯度爆炸，而是 cuDNN fused Attention 在 attention mask 存在整行全 `False` 时的反向数值行为：前向输出有限，但 Q 梯度为 NaN。

Pi0.5 的 prompt 会补齐到 `max_token_len=200`。padding token 在 `make_attn_mask` 中既不是有效 query，也不是有效 key，因此必然生成全 `False` query 行。原显式 Attention 用有限的 `big_neg` 处理这些行，不会产生同样的 Q 梯度 NaN；直接切换到 cuDNN 实现暴露了这一差异。

训练已停止。当前最后一个确定干净且仍保留的 checkpoint 是 step 15000；step 16000 已被保留策略删除，step 20000–55000 均已污染。

## 现象与时间线

切换前的原 Attention 训练指标正常：

```text
Step 16000: grad_norm=0.6721, loss=0.1429, param_norm=1806.9792
Step 16100: grad_norm=1.1132, loss=0.1456, param_norm=1806.9926
Step 16200: grad_norm=0.8198, loss=0.1402, param_norm=1807.0051
Step 16300: grad_norm=0.9735, loss=0.1410, param_norm=1807.0171
```

从 step 16000 checkpoint 用 cuDNN Attention 恢复后：

```text
Step 16100: grad_norm=nan, loss=0.1185, param_norm=nan
Step 16200: grad_norm=nan, loss=nan,    param_norm=nan
Step 16300 及以后: grad_norm=nan, loss=nan, param_norm=nan
```

checkpoint 目录名记录的是循环 step，而保存的 `train_state.step` 已经加一。因此恢复后第一个实际更新的前向 loss 仍有限，但该步反向产生 NaN 并污染参数；后续前向全部变为 NaN。

日志又对整个窗口的所有字段使用 `nanmean`。第一个窗口中唯一有限的首步 loss 会让聚合结果仍显示为 `0.1185`，从而掩盖其余 NaN。grad norm 和 param norm 只在日志 step 真正计算，所以当时已经直接显示 NaN。

## 最小复现

在 JAX 实际加载 cuDNN runtime `91400` 的同一环境中，使用 BF16 GQA 和含 16 个全空 query 行的 mask：

```text
forward output finite: true
Q/K/V gradient finite: [false, true, true]
Q gradient NaN count: 16384
```

为每个全空 query 行临时开放第 0 个 key 后：

```text
Q/K/V gradient finite: [true, true, true]
Q/K/V gradient NaN count: [0, 0, 0]
valid query rows unchanged: true
```

这解释了完整训练的全部现象：首步前向正常、首步梯度 NaN、更新后参数 NaN、随后 loss NaN。

## 修复

cuDNN 分支保留原始 attention mask，不再给全空行增加 dummy key。调用 `jax.nn.dot_product_attention` 前用 `any(..., axis=-1)` 找出有效 query；Q 的前向值逐元素不变，但全空 query 行通过 `stop_gradient` 从反向链路中移除。这样保持原 mask 语义和有效 query 输出，同时把无效 Q 梯度固定为 0。

训练循环曾增加逐 step 有限性保护。根据 2026-08-29 的生产运行要求，该保护已移除，恢复原更新路径；训练仍按原逻辑在日志边界输出 `loss`、`grad_norm` 和 `param_norm`，不再自动拒绝更新或因非有限值停止。最初曾把 `1.91 -> 2.3 s/step` 归因于该保护，后续 A/B 推翻了这一判断：移除保护后连续有限训练仍约为 `2.31 s/step`。

这是一项明确的吞吐优先取舍：日志只能事后暴露数值异常，不能保证异常更新不会进入参数或下一个 checkpoint。

## 验证要求

修复验收按以下顺序执行：

1. 单元测试确认 Q 前向逐元素不变，且只停止全空 query 行的梯度；
2. H200/cuDNN 最小反向测试确认 Q/K/V 梯度全部有限；
3. 从 step 15000 复制到独立诊断实验目录，禁用 W&B 和 checkpoint 保存；
4. 至少运行 200–500 step，逐日志窗口确认 loss/grad norm/param norm 有限；
5. 通过后才能清理污染 checkpoint 或恢复生产训练。

## 2026-08-29 修复验证结果

代码与单元回归：

- `python -m compileall` 通过；
- `git diff --check` 通过；
- `src/openpi/models/pi0_test.py`：`8 passed`；
- `scripts/train_test.py`：`2 passed`，包括两步训练和 checkpoint 恢复；
- H200/cuDNN BF16 GQA 真实序列形状测试：原始反向产生 71680 个 Q 梯度 NaN；Q stop-gradient 后 Q/K/V 梯度全部有限，NaN 数量均为 0，前向值和有效 query 输出逐元素不变。

真实训练使用 8×H200、全局 batch 512、cuDNN runtime 91400，从 clean step 15000 通过 `/tmp` 独立目录只读恢复。W&B、周期保存和最终保存均关闭。为缩短冷启动只启用 16 个 DataLoader worker，因此本次不统计吞吐。

主动结束前完成了 step 15001–15060 共 60 个更新；每 10 step 的检查结果如下：

```text
Step 15010: grad_norm=0.7763, loss=0.1438, param_norm=1806.8403, update_is_finite=1.0000
Step 15020: grad_norm=1.0680, loss=0.1666, param_norm=1806.8416, update_is_finite=1.0000
Step 15030: grad_norm=0.8248, loss=0.1507, param_norm=1806.8430, update_is_finite=1.0000
Step 15040: grad_norm=0.8843, loss=0.1576, param_norm=1806.8450, update_is_finite=1.0000
Step 15050: grad_norm=0.7334, loss=0.1528, param_norm=1806.8467, update_is_finite=1.0000
Step 15060: grad_norm=1.4659, loss=0.1655, param_norm=1806.8483, update_is_finite=1.0000
```

原缺陷会在恢复后的第一个更新产生 NaN，因此 60 个连续有限更新已经验证修复覆盖了已知故障。短跑结束后 8 张 GPU 显存均为 0 MiB，原 checkpoint 未被修改。恢复生产前仍建议完成 200–500 step 的长验证，以覆盖更多数据样本和训练阶段。

### 速度复核

上次长跑报告的普通 step `1.910590 s` 来自数值已经失效的区间：第一个生产日志窗口已出现 NaN 梯度，随后参数和 loss 持续为 NaN，因此不能作为有效训练的吞吐基线。修复后连续有限训练的普通 step 稳定约为 `2.31 s`。

16-worker 短跑曾出现 `dispatch=1.82–1.83 s`，但它发生在 80 秒左右的数据断供之后，是 GPU 空闲后的瞬时样本；同一运行的连续 step 仍为 `2.30 s`，并且每约 16 step 出现 30–85 秒 `next_batch` stall。生产继续使用 96 workers 以避免数据断供，约 8 分钟冷启动不计入稳态吞吐。

## 恢复注意事项

污染的 step 20000–55000 已删除，生产训练已从 step 15000 恢复并保存 step 18000。逐 step 保护在 step 18600 报告最近 100 步中有 8 次更新被拒绝，当时窗口聚合 loss 仍为有限值 `0.7085`，因此只看 loss 不足以证明每次更新都有限。移除保护后的生产训练从最后保留的 step 18000 继续。

原 step 16000 已删除，无法无损续接；从 step 15000 恢复意味着重跑约 1000 个有效 step。
