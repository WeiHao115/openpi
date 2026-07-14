# OpenPI π0.5 MOE 修改说明

本说明记录 OpenPI PyTorch π0.5 的 MOE 版本。原始 OpenPI 文件不修改，所有训练入口和模型代码都放在 `_MOE` 副本中。

## 新增文件

- `scripts/train_pytorch_MOE.py`
  - 从 `scripts/train_pytorch.py` 复制而来。
  - 训练入口改为读取 `openpi.training.config_MOE`。
  - 模型实例化改为 `openpi.models_pytorch.pi0_pytorch_MOE.PI0Pytorch`。
  - 加载原 pi05 预训练权重时使用 `strict=False`，避免新增 MOE/force/tactile 参数导致旧权重加载失败。

- `src/openpi/training/config_MOE.py`
  - 从 `src/openpi/training/config.py` 复制而来。
  - `pi0_config` 引用改为 `openpi.models.pi0_config_MOE`。
  - 原配置名保持不变，例如 `debug_pi05`、`pi05_lerobot_C16_ethernet_cable`，但通过 MOE 训练脚本运行时会构造 MOE 模型。

- `src/openpi/models/pi0_config_MOE.py`
  - 从 `src/openpi/models/pi0_config.py` 的 V2 实现整理而来。
  - 新增六维力、触觉和 MOE 配置项。
  - `load_pytorch` 指向 `openpi.models_pytorch.pi0_pytorch_MOE.PI0Pytorch`。

- `src/openpi/models_pytorch/pi0_pytorch_MOE.py`
  - 从 `src/openpi/models_pytorch/pi0_pytorch.py` 的 V2 实现整理而来。
  - 实现 force/tactile/MOE/联合 force-action denoise。

- `src/openpi/models_pytorch/README_MOE.md`
  - 本说明文件。

## 启动方式

不启动 MOE：

```bash
python /home/k202/openpi/openpi/scripts/train_pytorch.py <config_name> --exp_name <run_name>
```

启动 MOE：

```bash
python /home/k202/openpi/openpi/scripts/train_pytorch_MOE.py <config_name> --exp_name <run_name>
```

例如：

```bash
python /home/k202/openpi/openpi/scripts/train_pytorch_MOE.py debug_pi05 --exp_name debug_pi05_moe
```

你的已有配置也可以直接用 MOE 入口启动：

```bash
python /home/k202/openpi/openpi/scripts/train_pytorch_MOE.py pi05_lerobot_C16_ethernet_cable --exp_name pi05_lerobot_moe
```

## MOE 模块

`pi0_pytorch_MOE.py` 新增 `TokenMoE`：

- 每个 token 经过 `router: Linear(dim, num_experts)` 得到专家 logits。
- 对 logits 做 `top_k` 选择。
- 选中的专家权重经过 `softmax` 归一化。
- 未选专家权重为 0。
- 所有专家分别处理 token，最后按 router 权重加权求和。
- token 输入输出维度保持不变。

默认配置：

- `moe_num_experts = 4`
- `moe_top_k = 2`
- `moe_hidden_mult = 2`

## 新增编码器

- `ForceEncoder`
  - 输入 UMI 六维力窗口 `[B, force_window_size, force_dim]`。
  - 展平后经过 MLP，输出 force context token。

- `TactileEncoder`
  - 输入触觉 0/1 数值信号 `[B, tactile_dim]`。
  - 经过 MLP 输出 `[B, 1, C]` 触觉 token。

## Token 流程

MOE 放在 token 进入 VLM/Gemma 主干之前。

Prefix 包含：

- 图像 token
- 触觉 token
- 文本 token

Prefix 处理：

- `observation.force` / `observation.forces` / `observation.force_torque` 经 `ForceEncoder` 得到 `force_context`。
- 图像 token 与 `force_context` summary 拼接，经 `image_force_fuse` 融合后进入 `image_moe`。
- 触觉 0/1 信号经 `TactileEncoder` 后进入 `tactile_moe`。
- 文本 token 经 `embed_language_tokens` 后进入 `text_moe`。
- 图像 token 和触觉 token 共同求 `image_summary`，传给 suffix 中的 force token 使用。

π0.5 模式下 suffix 包含：

- noisy force token
- noisy action token

Suffix 处理：

- noisy force 经 `force_in_proj`，与 `force_context` 相加。
- force token 融合 `image_summary`，再进入 `force_moe`。
- noisy action 经 `action_in_proj` 后进入 `action_moe`。
- 时间信息仍通过 π0.5 原有 `time_mlp_in/out` 生成 AdaRMS 条件。

## 联合去噪

训练时：

- action 按 flow matching 构造 `x_t` 和 `u_t`。
- force target 优先读取 `observation.force_target` / `observation.target_force` / `observation.force_targets`。
- 如果没有 force target，则退回到输入 force 的最后一帧作为监督。
- Gemma 输出后拆分为 `force_out` 和 `action_out`。
- `force_out` 经 `force_out_proj` 预测六维力流向量。
- `action_out` 经 `action_out_proj` 预测动作流向量。
- 最终 loss 为 action loss 加 force loss。

推理时：

- `sample_actions` 同时迭代更新 `force_x_t` 和 action `x_t`。
- 返回值仍然是 action。
- 预测的六维力保存在 `self.last_predicted_forces`。

## 输入约定

MOE 文件不修改 OpenPI 原始 `Observation` dataclass。为了兼容现有数据结构，MOE 通过 `getattr` 读取可选字段：

- force:
  - `observation.force`
  - `observation.forces`
  - `observation.force_torque`

- tactile:
  - `observation.tactile`
  - `observation.tactile_signal`
  - `observation.tactile_signals`

- force target:
  - `observation.force_target`
  - `observation.target_force`
  - `observation.force_targets`

如果 force 或 tactile 缺失，MOE 会自动创建零 token，保证训练入口可以直接跑通。
