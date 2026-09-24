# 兴趣预测实验

## 方案

```text
已有 SFT → Mid-train 兴趣激活 → RL 优化
用户历史 → 2个兴趣token → 下一个商品SID
```

- **SFT**：复用 Qwen3-0.6B / Embedding-4B、2560维 / balanced SID / 15ep 模型。
- **Mid-train**：取历史中出现最多的2个一级 SID 作为兴趣标签，同频优先近期，不足用 `<interest_none>` 补齐。兴趣 token 从对应一级 SID 的 embedding 和输出 head 初始化，之后独立训练；标签不使用目标商品。
- **RL**：用商品命中奖励优化兴趣和商品预测，两种模式均从同一个 Mid 模型开始。

一级 SID 是聚类先验，不一定对应真实类目。当前历史最多10条；生成商品时仍能看到历史，也不强制商品属于所预测的兴趣。

保留原多任务：Mid 保留 SID↔标题、历史→下一商品标题；RL 保留标题/描述→SID。只有 SID历史/标题历史→商品SID 的任务增加兴趣预测。

| RL模式 | 每个历史的采样方式 | 奖励与更新 |
|---|---|---|
| `16x1` | 16组兴趣，每组1个商品 | 商品命中为1；在16条轨迹间计算相对优势，同时更新兴趣和商品 |
| `4x4` | 4组兴趣，每组4个商品 | 兴趣：组内任一商品命中即1，在4组间比较；商品：按单个命中，在组内4个商品间比较 |

每组均为 **2个兴趣 token**。`4x4` 的兴趣损失只计一次；全0或全1的比较组没有相对优势，仍保留KL。无历史的商品识别任务，两种模式都生成16个商品。

## 运行

复用 [MYREADME.md](MYREADME.md) 的环境与 balanced 数据，在 `MiniOneRec` 目录执行。已有 SFT 时直接从 Mid 开始。

```bash
conda activate minionerec
export CUDA_VISIBLE_DEVICES=3,4,5,6

# 可选：重新训练或评估原SFT
bash run_interest.sh sft
bash run_interest.sh eval sft

# 兴趣激活
bash run_interest.sh mid
bash run_interest.sh eval mid

# 两种RL分别训练、评估
bash run_interest.sh rl 16x1
bash run_interest.sh eval 16x1

bash run_interest.sh rl 4x4
bash run_interest.sh eval 4x4

bash run_interest.sh tensorboard
```

| 默认参数 | Mid-train | RL |
|---|---|---|
| 学习率 / epochs | 3e-5 / 3 | 1e-5 / 2 |
| 调度 / warmup | linear / 20步 | cosine / 3% |
| 每卡 batch / 梯度累积 | 16 / 16 | 16条轨迹 / 16 |
| 模型保存 | 最佳验证loss | 最后权重 |

训练均为 **4卡、ZeRO-2、TensorBoard**；RL 的 KL 系数为0.001，每次更新共1024条商品轨迹、64个prompt组。Mid 参数是待验证的实验起点。

默认 SFT 来自 `outputs/amazon23_industrial_qwen3_0.6b_sft_balanced_15ep/final_checkpoint`。模型和日志保存至 `outputs/interest_qwen3_0.6b_balanced/{mid,rl_16x1,rl_4x4}/`，评估结果在 `results/interest_qwen3_0.6b_balanced/`。

修改参数或新开实验：

```bash
export INTEREST_ROOT=./outputs/interest_trial2
bash run_interest.sh mid --epochs 3 --lr 3e-5
bash run_interest.sh rl 4x4 --epochs 2 --lr 1e-5
```

`INTEREST_SFT_DIR` 可指定 SFT 目录（包含 `final_checkpoint`），`BALANCED_DATA_ROOT` 可指定数据目录。训练目录非空会拒绝重跑，当前不支持断点续训。已通过CPU测试，L40/ZeRO-2尚未实测；可在新目录为上述训练命令追加 `--max_steps 2` 做短跑。

## 评估与记录

评估使用4卡独立分片。Mid、两种RL均生成50条完整兴趣→商品路径，按联合概率排序，商品去重后计算HR/NDCG；去重后可能不足50件，同时记录平均唯一候选数。默认使用test集，验证集可用 `INTEREST_EVAL_SPLIT=valid bash run_interest.sh eval 4x4`。

两种新RL均用独立约束采样、二值命中奖励和约束集合内的log-prob；与原RL的beam采样、ranking奖励及全词表log-prob不同，因此对比旧RL时不能把差异全部归因于兴趣机制。

| 实验 | HR@10 | HR@50 | NDCG@50 | 平均唯一候选数 |
|---|---|---|---|---|
| 原 balanced15ep SFT | 0.06069418 | 0.09583617 | 0.05665202 | — |
| 原 balanced15ep → RL | 0.07381055 | 0.11433521 | 0.06792715 | — |
| Mid-train | | | | |
| RL 16x1 | | | | |
| RL 4x4 | | | | |
