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

RL 已修复 DeepSpeed 0.18.0 在退出时触发的 `BF16_Optimizer.destroy` 越界，并在最终保存完成后释放进程组。该修复只影响清理流程，无需重装依赖；已经保存且能正常评估的模型无需重训。

RL/评估的约束解码已识别合法 SID 的 EOS 终止状态，避免已结束 beam 触发 `No valid tokens` 误报；保持原 EOS 约束和分数，真正非法的前缀仍会告警。

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

## 7. Qwen3-1.7B 学习率对照

原基线峰值学习率为 `3e-4`。先试 `5e-4`，检查提高学习率是否加快验证 loss 下降；这只是对照实验，不保证更快或最终指标更好。保留 linear 衰减、warmup 20 步、10 epochs、4 卡 ZeRO-2、每卡 batch 16、累积 16 及全部数据配置。

```bash
bash run_qwen3_1.7b_lr.sh sft 5e-4
bash run_qwen3_1.7b_lr.sh eval sft 5e-4
bash run_qwen3_1.7b_lr.sh tensorboard 5e-4
```

从原始 Qwen3-1.7B 预训练权重重新做 SFT，复用现有 SID/CSV，无需重跑 `prepare`。输出在 `outputs/amazon23_industrial_qwen3_1.7b_sft_lr/5e-4/`，终端日志和评估结果分别在原目录下的 `qwen3_1.7b_lr/5e-4/`。默认模型路径为 `models/Qwen3-1.7B/`，可用 `QWEN3_17B_LR_MODEL` 覆盖；输出根目录可用 `QWEN3_17B_LR_OUTPUT_ROOT` 覆盖。同一学习率重复执行会复用目录。

省略学习率时默认 `5e-4`；也可将以上三条命令中的值都换成 `1e-4` 做降低学习率的对照。按相同 epoch 比较验证 loss，并结合各自最佳 checkpoint 的 HR/NDCG 判断；若 `5e-4` 明显震荡或验证效果更差，不应继续盲目提高。

## 8. Qwen3-0.6B：25 epochs 实验

固定 **0.6B / 3e-4 / 当前 RQ-Kmeans+**，复用已有 SID/CSV，无需重新 `prepare`。从原始预训练权重重新训练，保留 4 卡 ZeRO-2、每卡 batch 16、累积 16、warmup 20 步；linear 衰减覆盖 25 epochs，因此不是接着原 10 epochs 的学习率曲线续训。

```bash
bash run_qwen3_0.6b_25ep.sh sft
bash run_qwen3_0.6b_25ep.sh eval sft
bash run_qwen3_0.6b_25ep.sh tensorboard
```

25 epochs 是上限；仍约每半个 epoch 验证/保存，连续 3 次验证 loss 无改善时早停，`final_checkpoint` 为验证 loss 最佳模型。输出为 `outputs/amazon23_industrial_qwen3_0.6b_sft_25ep/`；日志和评估结果分别在原目录下的 `qwen3_0.6b_25ep/`，TensorBoard 对比原 0.6B 基线与本实验。模型路径复用 `QWEN3_06B_MODEL`，输出可用 `QWEN3_06B_25EP_SFT_OUTPUT_DIR` 覆盖；重复运行会复用目录。

### 15 epochs 对照

固定 **0.6B / 3e-4 / 原 RQ-Kmeans+**，改用 15 epochs 的 linear 衰减计划，从预训练权重重新 SFT。其余参数、约半个 epoch 验证/保存和原早停保持一致；数据复用原 SID/CSV，无需 `prepare`。

```bash
bash run_qwen3_0.6b_15ep.sh sft
bash run_qwen3_0.6b_15ep.sh eval sft
bash run_qwen3_0.6b_15ep.sh tensorboard
```

输出为 `outputs/amazon23_industrial_qwen3_0.6b_sft_15ep/`，可用 `QWEN3_06B_15EP_SFT_OUTPUT_DIR` 覆盖；日志/评估结果在原目录下的 `qwen3_0.6b_15ep/`。TensorBoard 对比 10ep、15ep、25ep；15 是上限，可能提前停止。使用原 RQ-Kmeans+ 的 `DATA_ROOT`，不要指向平衡 SID 目录。

## 9. Qwen3-0.6B：平衡 SID 消融

固定 **0.6B / 3e-4 / 10 epochs 上限**，只把 SID 换为已有平衡 RQ-KMeans 初始化分配。复用当前 4B、2560 维 mean pooling 向量的 `codes_constrained.npy`，无需重新下载、embedding 或聚类。训练从预训练权重开始，4 卡 ZeRO-2、每卡 batch 16、累积 16、linear、warmup 20 步和早停保持原设置。

```bash
bash run_qwen3_0.6b_balanced.sh prepare
bash run_qwen3_0.6b_balanced.sh sft
bash run_qwen3_0.6b_balanced.sh eval sft
bash run_qwen3_0.6b_balanced.sh tensorboard
```

`prepare` 读取原数据目录里的 `Industrial_and_Scientific.codes_constrained.npy`、item/index 和 train/valid/test CSV，保留每行商品 ID、标题、顺序与划分，只重映射 SID；新 index/CSV/info 写入 `data/Amazon23/variants/rqkmeans_balanced/`。原实验文件不覆盖，训练/评估前自动校验新数据。已准备好无需重复 `prepare`。

若分配文件只保存在 `outputs/`，第一条命令改为：

```bash
BALANCED_CODES_PATH=./outputs/Industrial_and_Scientific.codes_constrained.npy bash run_qwen3_0.6b_balanced.sh prepare
```

模型输出为 `outputs/amazon23_industrial_qwen3_0.6b_sft_balanced/`，日志/评估结果在原目录下的 `qwen3_0.6b_balanced/`。源数据、新数据和模型输出可分别用 `BALANCED_SOURCE_DATA_ROOT`、`BALANCED_DATA_ROOT`、`QWEN3_06B_BALANCED_SFT_OUTPUT_DIR` 覆盖；训练和评估使用同一新数据路径。TensorBoard 对比原 10ep RQ-Kmeans+ 基线；SID 改变引起的碰撞、辅助任务样本变化沿用现有构造逻辑。

### balanced 15 epochs 对照

复用上述平衡 SID 数据，无需重新 `prepare`。固定 **0.6B / 3e-4 / balanced SID**，从预训练权重重新训练，linear 衰减覆盖 15 epochs；四卡 ZeRO-2、每卡 batch 16、累积 16、warmup 20 步、约半个 epoch 验证/保存和原早停保持一致。

```bash
export CUDA_VISIBLE_DEVICES=3,4,5,6
bash run_qwen3_0.6b_balanced_15ep.sh sft
bash run_qwen3_0.6b_balanced_15ep.sh eval sft
bash run_qwen3_0.6b_balanced_15ep.sh tensorboard
```

模型输出为 `outputs/amazon23_industrial_qwen3_0.6b_sft_balanced_15ep/`（可用 `QWEN3_06B_BALANCED_15EP_SFT_OUTPUT_DIR` 覆盖）；日志/结果位于原目录下的 `qwen3_0.6b_balanced_15ep/`。自定义平衡数据路径继续用 `BALANCED_DATA_ROOT`，训练和评估前自动校验。15 是上限，可能早停；`final_checkpoint` 为验证 loss 最佳模型，TensorBoard 对比 balanced 10ep/15ep。

### balanced 15ep SFT 接 RL

从上述 **Embedding-4B / 2560维 / balanced / 15ep SFT** 的 `final_checkpoint` 开始，复用同一套 balanced 数据，无需重做 `prepare` 或 SFT。

```bash
export CUDA_VISIBLE_DEVICES=3,4,5,6
bash run_qwen3_0.6b_balanced_15ep.sh rl
bash run_qwen3_0.6b_balanced_15ep.sh eval rl
bash run_qwen3_0.6b_balanced_15ep.sh tensorboard
```

RL沿用 `rl.sh`：四卡ZeRO-2、每卡batch16、累积16、2 epochs、学习率`1e-5`、ranking reward、16个候选、beta=`1e-3`，其余参数保持不变。模型输出到 `outputs/amazon23_industrial_qwen3_0.6b_rl_balanced_15ep/`；终端日志为 `logs/qwen3_0.6b_balanced_15ep/rl.log`，评估结果为 `results/amazon23_industrial/qwen3_0.6b_balanced_15ep/rl.json`。TensorBoard包含SFT与RL各自的日志。

自定义SFT目录继续用 `QWEN3_06B_BALANCED_15EP_SFT_OUTPUT_DIR`，RL目录用 `QWEN3_06B_BALANCED_15EP_RL_OUTPUT_DIR`；数据路径继续用 `BALANCED_DATA_ROOT`，须与该SFT训练时一致。SFT评估仍单独运行 `bash run_qwen3_0.6b_balanced_15ep.sh eval sft`，已有结果无需重评。

## 10. Embedding-0.6B / 1024维 / balanced / 10ep

对比第9节的 **Embedding-4B / 2560维 / balanced / 10ep**。仅换embedding模型与维度；backbone仍为Qwen3-0.6B，保留mean pooling、无L2、`3e-4`、四卡ZeRO-2、每卡batch16、累积16、linear、warmup20和原早停。

```bash
export CUDA_VISIBLE_DEVICES=3,4,5,6
bash run_qwen3_0.6b_emb06b_balanced.sh download
bash run_qwen3_0.6b_emb06b_balanced.sh prepare
bash run_qwen3_0.6b_emb06b_balanced.sh sft
bash run_qwen3_0.6b_emb06b_balanced.sh eval sft
bash run_qwen3_0.6b_emb06b_balanced.sh tensorboard
```

沿用现有环境和Qwen3-0.6B backbone；`download`只下载HF的`Qwen/Qwen3-Embedding-0.6B`。`prepare`复用原商品元数据和CSV划分，重新四卡编码1024维向量、CPU平衡聚类（3×256、seed42），再按商品ID替换CSV中的SID；不运行RQ-Kmeans+。无需重下Amazon数据或重做时间过滤。

新数据、向量和码本在`data/Amazon23/variants/embedding06b_1024_balanced/`；模型在`outputs/amazon23_industrial_qwen3_0.6b_sft_emb06b_balanced/`；日志/结果子目录为`qwen3_0.6b_emb06b_balanced/`。训练从预训练权重开始，最多10ep、约半epoch验证一次，最终保存验证loss最佳模型。训练/评估前校验数据和编码清单，原实验产物不覆盖；已有本实验数据时跳过`prepare`。

自定义路径用`EMB06B_SOURCE_DATA_ROOT`（原CSV数据根目录）、`EMB06B_DATA_ROOT`（新数据目录）、`EMB06B_MODEL`（embedding模型），backbone仍用`QWEN3_06B_MODEL`。模型输出可用`QWEN3_06B_EMB06B_BALANCED_SFT_OUTPUT_DIR`覆盖。TensorBoard对比两种embedding的balanced **10ep**。

# rq-kmeans+
## 10 epochs
qwen3 1.7b sft
[1, 3, 5, 10, 20, 50]
NDCG:   [0.03823548 0.04090538 0.04299591 0.04527875 0.04822333 0.0525814 ]
HR      [0.03823548 0.04281383 0.04782528 0.0549403  0.06663367 0.08853554]

qwen3 1.7b sft lr5e-4
[1, 3, 5, 10, 20, 50]
NDCG:   [0.037988   0.04080594 0.04268284 0.0451673  0.04815509 0.05225238]
HR      [0.037988   0.04281383 0.04739219 0.0551259  0.06700489 0.08766937]

qwen3 0.6b sft
[1, 3, 5, 10, 20, 50]
NDCG:   [0.03959661 0.04283954 0.04414379 0.04649201 0.04952945 0.05347935]
HR      [0.03959661 0.04528862 0.04850585 0.05580647 0.0679948  0.08791685]

qwen3 0.6b rl
[1, 3, 5, 10, 20, 50]
NDCG:   [0.04473179 0.04851166 0.05079366 0.05333908 0.05582298 0.05996816]
HR      [0.04473179 0.05135185 0.05685826 0.06477758 0.07467673 0.09558869]

## 25 epochs
qwen3 0.6b sft
[1, 3, 5, 10, 20, 50]
NDCG:   [0.03848295 0.04160362 0.04293497 0.04529756 0.04758798 0.05095159]
HR      [0.03848295 0.04386562 0.04708284 0.05438347 0.06347831 0.08043061]

# balanced kmeans

## 10 epochs
qwen3 0.6b sft
[1, 3, 5, 10, 20, 50]
NDCG:   [0.04052466 0.04409607 0.04607142 0.04893972 0.05199004 0.05615395]
HR      [0.04052466 0.04671163 0.05153746 0.06038483 0.07251129 0.09348512]

## 15 epochs
qwen3 0.6b sft
[1, 3, 5, 10, 20, 50]
NDCG:   [0.04058652 0.043909   0.04631021 0.04903452 0.05210262 0.05665202]
HR      [0.04058652 0.04634041 0.05221803 0.06069418 0.07288251 0.09583617]

qwen3 0.6b rl
[1, 3, 5, 10, 20, 50]
NDCG:   [0.04801089 0.05313122 0.05568735 0.05916239 0.06278401 0.06792715]
HR      [0.04801089 0.05685826 0.06304523 0.07381055 0.08816433 0.11433521]


# Qwen3-Embedding-0.6B balanced kmeans

## 10 epochs
qwen3 0.6b sft
[1, 3, 5, 10, 20, 50]
NDCG:   [0.04027718 0.04310692 0.0449251  0.04756991 0.05080558 0.05520089]
HR      [0.04027718 0.04528862 0.04974324 0.05797191 0.07071707 0.09299016]

CUDA_VISIBLE_DEVICES=3,4,5,6 bash run_qwen3_0.6b_balanced.sh sft
CUDA_VISIBLE_DEVICES=3,4,5,6 bash run_qwen3_0.6b_balanced.sh eval sft
