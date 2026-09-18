# MiniOneRec 复现指南

适用环境：**Linux x86_64、4 张 L40（48 GB）、Python 3.11**。以下命令在 `MiniOneRec/` 根目录执行，统一入口为 `run.sh`。

## 1. 安装环境

```bash
conda create -n minionerec-l40 python=3.11 -y
conda activate minionerec-l40
bash run.sh setup
```

使用 PyTorch 2.6.0 + CUDA 12.4，建议 NVIDIA 驱动 550 或更新。精简依赖见 `requirements-l40.txt`，无需安装 FlashAttention、vLLM、torchrec，也无需登录 W&B。

## 2. 配置路径

按需修改 **`config/industrial.sh`**，默认配置可直接运行：

| 内容 | 默认值 |
|---|---|
| GPU | `0,1,2,3` |
| 原始数据 / 处理后数据 | `data/raw/Amazon23/` / `data/Amazon23/` |
| Backbone | `models/Qwen3-1.7B/` |
| Embedding | `models/Qwen3-Embedding-4B/` |
| SID / SFT / RL 产物 | `outputs/amazon23_industrial_*` |
| 终端日志 / 评估结果 | `logs/` / `results/amazon23_industrial/` |

脚本每次自动读取配置，无需手动 `export`。HF 直连不可用时，在配置中启用 `HF_ENDPOINT` 镜像行。

## 3. 按顺序运行

```bash
bash run.sh download    # 从 Hugging Face 下载模型和 Amazon23 Industrial
bash run.sh prepare     # 预处理 → embedding → RQ-Kmeans+ SID → SFT/RL CSV
bash run.sh sft         # 4 卡 SFT
bash run.sh eval sft    # 单独评估 SFT：4 卡分片并行，再合并计算指标
bash run.sh rl          # 从 SFT final_checkpoint 开始，4 卡 RL
bash run.sh eval rl     # 单独评估 RL：4 卡分片并行，再合并计算指标
```

评估沿用官方分片并行方式，使用配置中的 4 张 GPU；每卡 batch 8、50 beams、max_new_tokens 256、length_penalty 0。SFT/RL 结果分别保存为 `results/amazon23_industrial/sft.json`、`rl.json`，汇总日志分别为 `logs/eval_sft.log`、`eval_rl.log`；各卡日志在对应的 `*_shards.*` 目录。必须指定 `eval sft` 或 `eval rl`。

`prepare` 中 constrained K-means 使用 CPU，embedding 使用 4 卡，RQ-Kmeans+ 使用配置中的第一张卡。脚本自动保存终端日志，某阶段报错即停止。重新执行 `prepare` 会从预处理开始；需要重跑某一步时使用：

```bash
bash run.sh preprocess
bash run.sh embedding
bash run.sh sid         # 初始化 → 训练 → 生成 SID
bash run.sh convert
```

单独生成 SID 用 `bash run.sh sid-index`，默认选取最近更新的 `best_collision_model.pth`；可通过 `SID_CKPT=/path/to/best_collision_model.pth bash run.sh sid-index` 指定自己训练的权重。同一 SID 输出目录不要并发训练。

SFT 续训示例：

```bash
bash run.sh sft --resume_from_checkpoint ./outputs/amazon23_industrial_qwen3_1.7b_sft/checkpoint-N
```

## 4. 查看日志

```bash
# 另开终端，激活相同环境后执行
bash run.sh tensorboard
```

访问 `http://127.0.0.1:6006`。远程训练时，可在本机执行 `ssh -L 6006:127.0.0.1:6006 用户名@服务器` 后访问。

## 5. 保留的关键设置

- **数据**：HF `McAuley-Lab/Amazon-Reviews-2023`，Industrial 类别。下载器固定 revision 并校验 SHA-256，清单见 `data/amazon23_industrial_manifest.json`；原始数据约 3.48 GB，预处理建议预留 64 GB CPU 内存。
- **日期**：对齐官方 Amazon23 Shell，2018-10-01 00:00:00 至 2023-09-01 00:00:00，含边界，按运行主机本地时区。CSV 文件名保留旧日期后缀，内容为上述时间范围的数据。
- **Embedding / SID**：Qwen3-Embedding-4B，2560 维，保留 mean pooling、不做 L2 归一化；RQ-Kmeans+ 为 3 × 256 码本，碰撞去重可能追加第 4 层。
- **SFT / RL**：Qwen3-1.7B 全参数训练，均为 4 卡、ZeRO-2、BF16、每卡 batch 16、梯度累积 16；SFT 有效 batch 1024，RL 为每步 1024 条候选（约 64 个独立 prompt）。SFT 10 epochs、RL 2 epochs，其余训练参数沿用现有脚本。
- **保存与评估**：最终权重在各自输出目录的 `final_checkpoint/`，TensorBoard 日志在 `tensorboard/`；Eval prompt 已与 SFT 对齐。

已完成本地小规模链路检查，尚未实测完整 L40 多卡训练、峰值显存与正式指标。RL 验证 batch 保留官方的每卡 128。

## 6. Qwen3-0.6B 对照实验

复用相同环境、预处理数据、Qwen3-Embedding-4B 向量、SID 和 CSV，已完成的 `prepare` 无需重跑。使用独立入口，4 卡、ZeRO-2、每卡 batch 16、梯度累积 16 及其余超参数保持一致：

```bash
bash run_qwen3_0.6b.sh download
bash run_qwen3_0.6b.sh sft
bash run_qwen3_0.6b.sh eval sft
bash run_qwen3_0.6b.sh rl
bash run_qwen3_0.6b.sh eval rl
bash run_qwen3_0.6b.sh tensorboard
```

0.6B 从 `Qwen/Qwen3-0.6B` 重新做 SFT，RL 默认加载它自己的 SFT `final_checkpoint`，不能复用 1.7B 权重。模型保存在 `models/Qwen3-0.6B/`，训练输出为 `outputs/amazon23_industrial_qwen3_0.6b_{sft,rl}/`；终端日志与评估结果分别位于原日志/结果目录下的 `qwen3_0.6b/`。0.6B 专用路径见脚本顶部，公共数据路径与 GPU 仍在 `config/industrial.sh` 配置。评估继续使用 4 卡，每次单独选择 SFT 或 RL；训练效果需要分别评估。
