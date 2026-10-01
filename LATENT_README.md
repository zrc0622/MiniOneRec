# 历史向量 → 3个中间 token → 推荐

对照基线：**balanced kmeans / 15 epochs / Qwen3-0.6B**，即 `run_qwen3_0.6b_balanced_15ep.sh`。商品 SID 仍使用原来的 **Embedding-4B / 2560维 / balanced**，数据划分和历史长度不变。

## 方案

1. 按时间顺序输入历史商品标题，用冻结的 Qwen3-Embedding-4B 得到2560维向量；保留 mean pooling、不做L2归一化。最多2048 tokens，超长时删除最早的完整标题，不输入下一商品。
2. 用三并列 VQ-VAE 分支压缩向量：每个码本256项、每项32维，共输出3个离散码。Encoder为 `2560→2048→1024→512→256→128→96`，96维拆成3个32维向量分别量化，再拼接送入对称Decoder：`96→128→256→512→1024→2048→2560`。复用仓库RQ-VAE的MLP实现、ReLU、Xavier初始化、dropout=0、无BN；总参数 **16,116,064（约1612万）**。训练集拟合、验证集选模型，之后冻结；**不保证三个分支分别对应独立语义兴趣**。每个码向量的32维与原RQ-VAE一致，但并列拼接后为96维，原RQ-VAE残差相加后仍为32维，效果需实验比较。
3. 从预训练 Qwen3-0.6B 重新 SFT，**没有 Mid-train**。只将主任务改成 `历史SID → 3个中间token → 下一商品SID`；保留原 SID↔标题、历史SID→下一标题任务，不新增历史文本推荐。主任务 loss 为 `latent CE + item CE`，两部分分别平均；辅助任务沿用原数据及普通CE。
4. 从同一个新 SFT 分别跑两种 RL：

| 模式 | 主任务采样与奖励 | 概率与优势 |
|---|---|---|
| `original` | 原 beam sampling，16条完整轨迹；原命中奖励+ranking奖励 | 原全词表logprob、样本标准差；整条轨迹同一优势 |
| `4x4` | 4组中间token，每组采样4个item；组内任一命中则该组奖励1 | 约束词表logprob；中间token按4组奖励归一化，item按组内4个命中值归一化，共享前缀只计一次 |

两种RL都保留原标题/描述→SID、历史标题→下一SID辅助任务，辅助任务仍直接生成SID。没有中间token伪标签命中奖励。`original`保留原RL优化配方，但因增加中间token和统一主任务prompt，属于适配版本；原始无中间token的完整基线仍保留。

## 运行

复用 [MYREADME.md](MYREADME.md) 的环境、下载模型和balanced数据准备；已有环境无需新增依赖。新实验均放在独立 `LATENT_ROOT`，已有基线模型不能直接当作新版SFT。

```bash
cd MiniOneRec
conda activate minionerec
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LATENT_ROOT=./outputs/latent_qwen3_0.6b_balanced15

bash run_latent.sh encode
bash run_latent.sh vq
bash run_latent.sh sft
bash run_latent.sh eval sft

bash run_latent.sh rl original
bash run_latent.sh eval original

bash run_latent.sh rl 4x4
bash run_latent.sh eval 4x4

bash run_latent.sh tensorboard
```

- SFT：4卡ZeRO-2，每卡16、累积16，有效batch1024；3e-4、最多15ep、linear、warmup20、约半epoch验证、earlystop3。`final_checkpoint`为验证总loss最佳模型；同时记录独立 `eval_item_ce`，总loss不能与基线CE直接比较。
- RL：4卡ZeRO-2，每卡16条轨迹、累积16；16条轨迹为同一prompt组，因此每次更新约64个prompt/1024条轨迹。2ep、1e-5、cosine、warmup3%、beta0.001、原参考模型同步；保存最终模型。
- VQ：第一张可见卡，默认最多50ep、1e-3、batch512、commitment0.25、验证重建误差earlystop5。可用 `bash run_latent.sh vq --epochs 30` 修改。码本使用率、perplexity和死码数量在 `labels/labels.json`；严重塌缩时先检查VQ，不急于训练LLM。

已有基线无需重跑。需要重新调用时：

```bash
bash run_latent.sh baseline sft
bash run_latent.sh baseline eval sft
bash run_latent.sh baseline rl
bash run_latent.sh baseline eval rl
```

## 结果

- 新模型：`$LATENT_ROOT/sft/final_checkpoint`、`rl_original/final_checkpoint`、`rl_4x4/final_checkpoint`。
- TensorBoard：各训练目录内；VQ日志在 `labels/tensorboard_vq`。RL同时记录中间token与item的零优势比例、多样性和命中率。
- 评估：4卡各跑一份模型、分片独立推理。结果在 `$LATENT_ROOT/eval/<阶段>_<划分>_b<预算>.<随机后缀>/merged.json.metrics.json`；同目录有逐条预测和每卡日志。
- 所有新版阶段默认相同的50条**联合路径**beam预算，按全词表联合概率排序，同商品取最高路径分数去重，可能不足50件。务必一起看 `mean_unique_candidates`、`fraction_with_50_candidates`，不要把50条路径当作50件商品。

```bash
# 用验证集选模型/搜索预算；修改预算后各阶段需保持一致比较
LATENT_EVAL_SPLIT=valid LATENT_EVAL_BEAMS=200 bash run_latent.sh eval sft
```

新训练拒绝覆盖非空目录。重新实验可换 `LATENT_ROOT`；编码/VQ产物冻结后可复制或链接到新目录复用。旧 `run_interest.sh` / `IDEA_README.md` 及其结果保持原样。

若已跑过旧版VQ（包括3×64维版本），本次可复用历史 `embeddings`，需在新目录重跑VQ、标签及后续SFT/RL。命令仍为 `bash run_latent.sh vq`，训练epoch、学习率等设置沿用上文。

本地验证覆盖tiny-Qwen CPU训练与评估；完整L40/DeepSpeed训练和效果仍需实际实验验证。


# 1
## sft
"HR@1": 0.03761677906329271,
"NDCG@1": 0.03761677906329271,
"HR@3": 0.0454742312689476,
"NDCG@3": 0.04219355195729239,
"HR@5": 0.05023819835426591,
"NDCG@5": 0.04414224913230786,
"HR@10": 0.06007548103693621,
"NDCG@10": 0.04728886254274613,
"HR@20": 0.07096454866052095,
"NDCG@20": 0.05004254262525347,
"HR@50": 0.07851265235414218,
"NDCG@50": 0.05159687782703717,