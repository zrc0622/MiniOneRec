# MiniOneRec 交接给服务器 Codex

更新：2026-10-02。本文件用于恢复上下文；服务器上的实际代码、配置、checkpoint 和完整日志为准。

**最重要的状态：新版 latent SFT 已完成；latent original RL 第0步报错，尚未修复。用户已删除本地 Codex 上次提供的 RL 生成修复，明确留给服务器 Codex 处理，不要认为补丁仍在或直接恢复旧补丁。**交接时本地仓库 clean，`latent/beam_search.py` 不存在。

## 1. 工作方式与环境

- 用户希望中文、简洁说明，命令用 `.sh` 组织；直接读取服务器实验产物，不再要求来回粘贴结果。
- 不擅自调整实验范围、算法、超参数或依赖；其它需要改的先说明并征得同意。保留已有基线和实验结果，新实验用独立目录。
- 明确保留 **mean pooling、不做 L2 归一化**；不要因 Qwen embedding 官方推荐 last-token pooling 而自行替换。
- 用户要求 VQ 网络不简化，当前指定结构见下文。
- AOP/星云技能只有当前请求明确说“使用aop skill”/“使用星云 skill”并表达调用意图时才使用；普通训练话题不触发。
- 服务器项目：`/data/zuorongchang/project/MiniOneRec`；环境：`conda activate minionerec`（Python 3.11）；GPU 为 L40，训练4卡，可用 `CUDA_VISIBLE_DEVICES=3,4,5,6`。
- 版本：PyTorch 2.6.0/CUDA 12.4、Transformers 4.57.1、TRL 0.24.0、Accelerate 1.10.1、DeepSpeed 0.18.0；完整精简依赖见 `requirements-l40.txt`。先核对已装版本，无需重装。
- 训练用 TensorBoard；评估为4张卡各加载完整模型、独立数据分片，不是张量并行。SFT、RL 分开评估。

## 2. 仓库与实验地图

| 内容 | 入口/文档 |
|---|---|
| 原复现、数据下载与环境、历史实验结果 | `MYREADME.md`；官方说明 `README.md` |
| 当前对照基线 | `run_qwen3_0.6b_balanced_15ep.sh` |
| 旧兴趣方案：一级SID伪标签 → Mid → RL | `interest/`、`run_interest.sh`、`IDEA_README.md` |
| 当前方案：历史文本embedding → 并列VQ → 联合SFT → RL | `latent/`、`run_latent.sh`、`LATENT_README.md` |
| 回归测试 | `tests/test_latent.py`、`tests/latent_distributed_smoke.py` |

最初复现使用 Qwen3-1.7B / Qwen3-Embedding-4B / RQ-Kmeans+；后来增加0.6B、学习率、训练轮次、balanced初始化和Embedding-0.6B消融。现在主要基线为 **Qwen3-0.6B / Embedding-4B、2560维 / balanced kmeans / 15ep**。NanoOneRec 是他人复现，仅供对照，不应自动照搬其配置。

Amazon23 Industrial 通过 HF `McAuley-Lab/Amazon-Reviews-2023` 下载。预处理日期对齐官方：2018-10-01 至 2023-09-01（含边界，按运行主机本地时区）；CSV旧文件名 `2016-10-2018-11` 不代表内容日期。当前历史最多10条。

balanced 商品SID复用 RQ-KMeans 的平衡初始化分配，不经过后续 RQ-Kmeans+ 优化；数据目录 `data/Amazon23/variants/rqkmeans_balanced/`，类别 `Industrial_and_Scientific`。已有数据无需重做。

## 3. 当前 latent 方案

动机来自 OneRec-Think：用自动构造的离散中间表示代替人工自然语言推理标签。当前模型生成3个中间token，再生成下一商品SID；生成商品时仍可见原历史，并无硬性类目限制。

1. **标签输入**：历史商品标题按时间顺序串接，冻结 Qwen3-Embedding-4B，FP32 masked mean pooling，取2560维、无L2。最多2048 tokens，超长删除最早的完整标题。仅使用历史，不输入下一商品。
2. **并列 VQ-VAE**：encoder `2560→2048→1024→512→256→128→96`，拆为3×32，每分支独立256项码本；量化后拼接96维，decoder对称。复用 `rq.models.layers.MLPLayers`，ReLU/Xavier、dropout=0、无BN，参数16,116,064。不是残差RQ，三个码也不保证对应独立语义兴趣。
3. **VQ训练**：仅train拟合、valid重建误差选模型，然后冻结导出各split标签；默认最多50ep、lr1e-3、batch512、commitment0.25、patience5。目标是重建历史embedding，未直接优化下一商品预测。
4. **SFT**：从预训练Qwen3-0.6B重新训练，无Mid。主任务由 `历史SID→下一商品SID` 改为 `历史SID→3个latent token→下一商品SID+换行+EOS`。保留原SID↔标题、历史SID→下一标题辅助数据。
5. **SFT loss**：主任务为分别平均的 `latent CE + item CE`（latent权重1）；辅助仍普通CE，再按样本数组合。与原token平均CE的梯度权重不完全一致，不能直接比较总loss；prompt也有相应调整。
6. **SFT设置**：4卡ZeRO-2、BF16、每卡16、累积16，有效batch1024；3e-4、15ep上限、linear、warmup20、约半epoch验证、earlystop3、保存最佳总验证loss。此轮实际跑满15ep。
7. **RL**：两种模式从同一个新版SFT开始，均2ep、1e-5、cosine、warmup3%、beta0.001、每卡16条轨迹/累积16；每16轨迹属于同一prompt，每次更新约64个prompt。保留原标题/描述→SID、历史标题→下一SID辅助任务（仍SID-only），无latent标签命中奖励。

| RL模式 | 采样与奖励 |
|---|---|
| `original` | beam sampling产生16条完整轨迹；原命中+ranking奖励、全词表logprob、样本标准差归一化；整条轨迹共享优势。不是独立采样16次。 |
| `4x4` | 独立采样4组latent，每组4个item；组内任一命中，兴趣组奖励1；兴趣优势在4组间归一化，item优势在组内4个命中值间归一化；共享前缀loss只计一次，使用约束词表logprob。 |

新增latent token的LLM embedding可学习，与商品一级SID不共享；VQ标签生成器在SFT/RL期间冻结。实际已运行VQ结构需查 `labels/labels.json` 的 `vq_options`：曾有旧3×64版本，仅靠SFT日志无法确认是哪一版。

## 4. 已完成结果与结论

下表是各README中用户给出的离线结果（按原标注为test）；核对同split、checkpoint、预算后再作严格比较。

| 实验 | HR@10 | HR@50 | NDCG@50 |
|---|---:|---:|---:|
| 原balanced15ep SFT | 0.06069418 | 0.09583617 | 0.05665202 |
| 原balanced15ep → RL | 0.07381055 | 0.11433521 | 0.06792715 |
| 旧兴趣Mid（2个历史高频一级SID标签） | 0.05871435 | 0.09645487 | 0.05535728 |
| 旧兴趣RL 16×1 | 0.06446823 | 0.08123492 | 0.05718366 |
| 旧兴趣RL 4×4 | 0.05809565 | 0.06552001 | 0.05229945 |
| **当前latent SFT** | **0.06007548** | **0.07851265** | **0.05159688** |

当前latent SFT还包括 HR@1=0.03761678、HR@20=0.07096455。相对原SFT，HR@10−1.02%、HR@50−18.08%、NDCG@50−8.92%。目前没有收到成功完成的latent RL结果，勿把旧interest RL混作新版结果。

**TensorBoard已核验**：`outputs/tensorboard_latent/events.out.tfevents.1790674956.ubuntu-4U-GPU-Server.637140.0`，4140步/15ep/30次验证，正常结束、最后一步总验证loss最佳。最终 `eval/loss=4.098320`、`eval/latent_ce=1.778711`、`eval/item_ce=2.285039`；基线验证CE=2.163991。最后13→15ep item CE仍下降约1%，并无持续验证反弹，但不能据此保证延长训练有效。

item CE是在给定正确VQ标签和此前正确SID token的teacher forcing条件下计算的；仍高于基线是负面信号，但prompt、loss权重与统计口径有差异，不能做严格单变量归因。

**用户额外补充（不一定已写入README）**：

- `mean_unique_candidates=27.89`、`fraction_with_50_candidates=0.0348`、`mean_valid_paths=50`。
- 即约44.22%的路径指向重复商品，96.52%的样本不足50件。当前eval预算是50条联合路径，按全词表联合logprob排序，同商品取最大路径分数去重；不是50个不同商品，也不是对latent路径概率求和。
- 已确认候选覆盖不足，但不能证明全部效果下降都由此造成。HR@1也下降约7.3%。
- VQ `code_statistics` 在test三分支平均 `used≈120/256`。这不等于严重塌缩或健康，仍需各分支perplexity与train统计；test没出现的码不等于训练死码。

当前判断：该版本是负结果，应区分搜索覆盖、标签预测误差、标签与推荐目标不一致。不要直接宣判所有“中间token”方案无效，也不要优先盲目加epoch。

## 5. 未解决：latent original RL启动报错

**用户已删除之前的修复，服务器 Codex需重新诊断/处理。本次交接没有修复代码。**

```text
trainer.train() → training_step() → _prepare_inputs()
→ latent.generation.rollout(mode='original') → model.generate()
→ grammar.allowed(...)
ValueError: Illegal SID prefix: [15]
进度：0/4928
```

这是RL训练第一步的rollout，不是SFT或离线eval；随后eval提示缺少 `rl_original/final_checkpoint` 是训练未完成的后果。无需因此重训SFT。

此前本地CPU/tiny-Qwen曾复现的线索（供重新验证，不等同于服务器已修好）：

- 日志显示 generation_config被Qwen默认值覆盖为 `temperature=0.6, top_p=0.95`。HF4.57.1可能将显式等于全局默认的配置替换成模型默认值；可检查 `use_model_defaults=False` 或直接生成参数覆盖的作用。
- HF beam sampling用不放回multinomial抽取约2×beam宽度的候选。top-p裁剪或float32概率下溢导致支持不足时，可能取到零概率/被mask的候选，再触发严格SID前缀校验。
- 之前测试随机tiny模型未覆盖真实checkpoint默认参数和尖锐概率分布；补这两类回归。正常候选充分时应检查与原beam采样一致，不能只吞掉异常或把非法SID纳入奖励。
- 曾尝试的临时实例级beam selector/log空间抽样补丁**已被用户删除**。旧分析/测试报告如果写“修复完成”，只描述当时已删除版本，不能作为当前验证结论；不要自动还原或宣称当前测试已覆盖它。
- NCCL barrier、累积1/16（日志采用DS16）、untested optimizer警告不是该traceback的直接原因。旧 `deepspeed_cleanup.py` 解决的是训练结束时BF16 optimizer析构越界，与本次第0步错误不同。

修复后先短跑验证再全量；新latent训练拒绝覆盖非空目录，失败目录需备份/换路径，不能直接覆盖既有结果。现有脚本不支持直接断点续训。

## 6. 入口、产物和建议的下一步

```bash
cd /data/zuorongchang/project/MiniOneRec
conda activate minionerec
export CUDA_VISIBLE_DEVICES=3,4,5,6
export LATENT_ROOT=./outputs/latent_qwen3_0.6b_balanced15

# 已完成编码/VQ/SFT时无需重跑它们。以下是入口，按当前任务选择执行。
bash run_latent.sh eval sft
# 待修复RL、备份失败输出后，从现有SFT开始：
bash run_latent.sh rl original && bash run_latent.sh eval original
bash run_latent.sh rl 4x4 && bash run_latent.sh eval 4x4
```

- 新版产物：`$LATENT_ROOT/{embeddings,labels,sft,rl_original,rl_4x4}/`；模型在各训练目录的 `final_checkpoint/`，训练日志在 `tensorboard/`，VQ日志在 `labels/tensorboard_vq/`。
- 每轮训练的 `latent_run.json` 记录参数/来源哈希；`labels/labels.json` 记录实际VQ结构和码本统计。先直接读取服务器的这些文件。
- 评估：`$LATENT_ROOT/eval/<阶段>_<划分>_b<预算>.<随机后缀>/merged.json` 和 `merged.json.metrics.json`；含逐条预测、latent路径、metadata，旁边有每卡日志。
- 原基线SFT：`outputs/amazon23_industrial_qwen3_0.6b_sft_balanced_15ep/final_checkpoint`。它是无latent基线，不能替代新版SFT用于latent RL。

此前只提出、**尚未确认执行**的诊断：

1. 复用新版SFT，在valid集比较50/200路径预算，同时看唯一候选数/HR；增加预算需单独报告，不当作相同算力下的模型提升。

```bash
LATENT_EVAL_SPLIT=valid LATENT_EVAL_BEAMS=50 bash run_latent.sh eval sft
LATENT_EVAL_SPLIT=valid LATENT_EVAL_BEAMS=200 bash run_latent.sh eval sft
```

2. 给定该历史的正确VQ标签，再仅生成item，与模型自行生成latent比较；同时测latent各槽/三token整组准确率。**此评估入口尚未实现**，需要用户确认继续。标签仅来自历史，不泄露下一商品；但运行时调用标签生成器，属于诊断，不是现有纯LLM推理路径。
3. 读取完整VQ统计：各split各slot的used/dead/perplexity、实际结构和hash。根据诊断再决定改标签、loss权重或搜索；不默认扩充网络/训练预算。

本地历史详细证据在仓库外 `../analysis/latent_sft_comparison_20261001/`、`../analysis/latent_generation_fix_20261001/`、`../analysis/interest_comparison_20260928/`，可能未同步服务器。关键事实已收录本文件，后续直接从服务器真实产物核验即可。
