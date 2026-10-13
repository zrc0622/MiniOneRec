> **新服务器当前工作见 [NEW_SERVER_HANDOFF.md](NEW_SERVER_HANDOFF.md)。** 以下开头描述旧版历史兴趣对照；新版商品及兴趣 embedding 配方、教师字符兴趣任务与阶段边界以交接文档为准。

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
## 四种 SFT 搜索评估（不重新训练）

新增 `run_latent_eval_four.sh` 依次评估以下四种方案，每种都使用4卡独立数据分片：

| 方案 | 默认搜索预算 | 商品排序 |
|---|---|---|
| `joint_max` | 200条 latent+SID 联合路径 | 同商品最高联合路径分数 |
| `hierarchical_max` | 先取4组 latent，每组50条商品路径 | 同商品最高联合路径分数 |
| `hierarchical_sum` | 与上项相同的4×50候选生成方法 | 同商品已搜索到的唯一完整路径做 log-sum-exp |
| `fixed_latent` | latent beam宽4，取最佳完整组合，再搜50条商品路径 | 商品条件概率排序（同一 latent 的联合分数排序等价） |

所有路径分数都使用全词表 logprob，包含3个latent、完整SID、换行和EOS，length_penalty=0。新方案显式关闭模型默认生成参数覆盖；原 `run_latent.sh eval` 和 RL 采样入口保留。

```bash
conda activate minionerec
# 当前空闲卡；启动前按实际情况调整
export CUDA_VISIBLE_DEVICES=4,5,6,7
bash run_latent_eval_four.sh --dry-run
# 可选：先做四卡小规模验证
LATENT_EVAL_MAX_EXAMPLES=8 bash run_latent_eval_four.sh
# 完整test；用valid做调参时，设置LATENT_EVAL_SPLIT=valid
bash run_latent_eval_four.sh
```

默认复用 `outputs/latent_qwen3_0.6b_balanced15/sft/final_checkpoint`。支持 `LATENT_ROOT`、`LATENT_EVAL_MODEL`、`BALANCED_DATA_ROOT`、`PYTHON_BIN`。搜索参数可用 `LATENT_EVAL_BEAMS=200`、`LATENT_EVAL_LATENTS=4`、`LATENT_EVAL_LATENT_BEAMS=4`、`LATENT_EVAL_ITEM_BEAMS=50` 调整；打分批量为 `LATENT_EVAL_SCORE_BATCH=8`。每组latent顺序生成，避免将所有分支同时放入显存。

每次创建 `outputs/latent_qwen3_0.6b_balanced15/eval/four_<split>.<随机后缀>/`，不覆盖已有结果。每个方案有 `0..3.log`、分片JSON、`merged.json`、`merged.json.metrics.json` 和 `wall_seconds.txt`；顶层逐步更新 `comparison.json/csv/md`。异常会停止后续方案，保留日志，现阶段不支持自动续跑。

评估记录checkpoint、数据和搜索代码哈希、参数及样本数。比较包含HR/NDCG、平均唯一商品数、凑满50商品比例、路径数、latent组合数、生成打分耗时和整组耗时。各方案最多返回50个不同商品，不足时不填充；200条联合路径仍不保证凑满50个。概率聚合只覆盖已搜到的路径，不是精确边缘化。搜索预算和耗时不同，应结合原50路径结果解释提升，不能全部归因于模型能力。

### 两项评估共享四卡

`bash run_latent_eval_two.sh` 将最多两项评估同时放在同一组4卡上，每张卡最多两个分片进程。先启动联合搜索和固定latent，再按空位启动两种分层搜索。参数沿用上文环境变量，结果写入新的 `two_<split>.*` 目录。

支持 `--dry-run`。`--adopt-run <已有four_test_launch目录>` 仅接管正在运行第一项的顺序任务：核验原父子进程和输出路径，只替换旧调度shell，保留4个联合搜索worker及其进度，继续写入原结果目录。旧启动目录的 `exit_code.txt` 可能为137，`run.json` 的 `superseded_by` 标记指向新调度状态，此时以新的运行日志和状态为准。

并行状态保存在结果目录的 `parallel_state.json`。每方案wall time受共享GPU影响且相互重叠，不能将四项wall time直接相加作为整组耗时。显存充裕不保证吞吐提高，使用前应按实际GPU负载检查。调度器异常时停止它新建的评估进程，已接管的旧worker保留。

### 两组 RL 同时共享四张卡（2026-10-06）

`run_latent_rl_two.sh` 同时运行两个独立的四卡 ZeRO-2 训练进程组，默认共享 GPU4,5,6,7。两组均从现有 latent SFT 开始，4×4 独立采样、每卡16轨迹、累积16、有效batch1024轨迹（64个prompt）、2ep、lr1e-5、KL0.001、seed42。区别仅为主任务奖励：`binary` 使用完整命中；`sid_dense` 使用完整命中加0.2×前三层SID位置匹配比例；辅助任务仍使用完整命中。

```bash
# 先同时短跑两步，包括原尺寸验证batch（只限制验证prompt数）。
bash run_latent_rl_two.sh smoke
# 显存足够后运行完整训练，完成后自动评估。
bash run_latent_rl_two.sh train
# 若并发显存不足，只运行已有奖励组，仍用全部四卡。
RL_REWARDS=binary bash run_latent_rl_two.sh train
```

每次独立创建 `outputs/latent_qwen3_0.6b_balanced15/rl_two/{smoke,train}.XXXXXX/`，保存两组训练日志、运行参数、源代码快照、显存采样、退出状态及checkpoint。两组训练都退出后，对成功完成的 `final_checkpoint` 依次进行valid/test全量评估，统一使用 `hierarchical_max`、latent beam4、4条latent、每条item beam50、max路径聚合、商品去重取前50。此轮比较固定训练预算的最终checkpoint，不宣称按验证NDCG选出的最佳checkpoint。训练中的`eval_loss`依然是RL目标值，不是HR/NDCG。

评估输出位于运行目录的 `eval/{binary,sid_dense}/{valid,test}.XXXXXX/`，累计对比为 `eval/comparison.{json,md}`。可用 `bash run_latent_rl_eval.sh <运行目录>` 单独重跑评估；`LATENT_EVAL_MAX_EXAMPLES`仅用于评估链路短测，正式启动必须不设置。`RL_REWARDS`支持`binary`、`sid_dense`或默认的两组；不会覆盖已有训练结果。

采样显式禁止checkpoint默认生成参数覆盖temperature/top-p，使实际4×4采样与约束logprob一致。若登录环境的CUDA_HOME指向不存在的工具包，启动器优先使用当前Python环境已经安装的nvcc，不安装新依赖。

2026-10-10指标约定：主要报告Recall@5、Recall@10、Recall@20、NDCG@5、NDCG@10、NDCG@20，单目标时Recall=HR。@20为用户追加的项目指标，报告必须保留至少前20名并同时给出满20候选比例；旧仅存前10名产物不能用于重算@20。共享summarize同时保留旧HR/@50，便于追溯。指标约定见eval_metric_protocol.json；不意味着复现OneRec-Think的数据集/beam10设置。探索评估默认固定seed42的20%valid，小样本pilot另行标记。

2026-10-10 用户最终决定：后续每次latent eval同时运行两套协议，不只把其中一套作为消融：`single20`（兴趣beam4取最高概率完整码，再商品beam20）和`four20_marginal`（兴趣beam4四组各商品beam20，对候选并集在全部4组兴趣下重新计算联合概率后求和）。两套都报告Recall/NDCG@5/10/20、满20候选比例及速度，模型间使用同样样本，不按模型临时挑方案。`latent.eval_at20`不指定模式和方法时默认运行这两套；显式`--mode screen`仍可用于额外预算探索。无latent模型直接SID beam50作为两套共享参照。历史启动器显式指定的其他搜索协议不会自动改写，后续新任务应按本约定使用入口。

### 两层非平衡 VQ SFT（2026-10-10）

`run_two_slot_vq_sft.py --root outputs/latent_qwen3_0.6b_vq2_15ep_20261010 --gpus 4,7 --label-gpu 7`：复用原Qwen历史embedding，重新拟合两个并行槽（每槽256码、32维），不使用平衡约束。先在7卡训练VQ，随后等待4/7均空闲，实测batch16/32，在显存留有余量的成功配置里选择最快的，保持全局batch1024。SFT15轮、LR3e-4、latent平均CE权重1；固定seed42随机20%验证集（3232条），每轮验证一次、每3轮保存一次、每10步记录，关闭提前停止和最优子集checkpoint回载，使用第15轮最终模型。

量化器、标签manifest、SFT提示词/mask、生成和主要eval均从协议读取k；三槽旧checkpoint仍兼容。新label使用`--num_slots 2`，需要从预训练基座重新做SFT，不能直接剪裁旧三槽checkpoint。每个槽32维，因此两槽总连续瓶颈64维，而原三槽96维。

协调器保存运行源码快照并从快照执行，独立status/log/PID/exit_code；只有指定卡无计算进程、显存<1GiB且低利用率连续三次才进入双卡阶段。训练后自动在原固定256valid样本及固定20%valid上评估两槽、原三槽非平衡VQ、无latentSFT；两种latent协议1×20和4×20完整跨兴趣求和，同时汇报R/N@5/10/20和满20比例，无latent使用SIDbeam50共享参照。未使用测试集选方案。报告位于实验root的`eval/{sample256,valid20pct}/comparison.md`。

### 历史＋目标标题兴趣标签：只训 VQ（2026-10-10）

`run_target_conditioned_vq.py --root outputs/latent_qwen3_0.6b_target_vq3_20261010 --gpus 4,7 --batch 32` 新建独立embedding/labels，不覆盖任何旧实验。Qwen3-Embedding-4B保持冻结；输入为按时间排列的历史标题，再追加真实下一商品标题，FP16、masked mean、2560维、无L2、最大2048token。去重包含历史标题＋目标item_id＋目标标题；只裁掉最旧的完整历史标题，完整保留目标标题，并至少保留一个历史标题，否则报错。`latent.prepare encode --embedding_input history_target`实现此teacher方案；默认history模式保持原有行为。

新的recipe在embedding及labels manifest标记`contains_target: true`、`intended_use: latent_supervision_only`、`inference_input: history_only`。目标条件embedding只生成监督latent标签；历史prompt没有加入目标商品。后续SFT只能从历史预测latent和SID，正式排名eval也必须自行生成latent；使用真实teacher latent的SID loss只用于条件拟合监控，不能当成部署推荐效果。

两个分片会分配给授权4/7卡中持续空闲的卡，4卡占用时先用7卡，7卡完成一个分片后可以接未启动的另一个分片；OOM时只重试本分片并减小batch。合并后在空闲7卡训练三槽非balanced ParallelVQ（256码/槽、32维/槽、原MLP、batch512、LR.001、50轮、patience5、seed42）。仅使用train拟合，用valid重建选checkpoint，test只导出标签。状态`complete_vq_only`即完成；此脚本没有SFT或推荐eval命令，按用户要求完成后等待再次通知。

### 单卡准备新版兴趣标签（last-token + L2）

2026-10-10 首轮产物位于 `outputs/latent_qwen3_0.6b_target_lasttoken_l2_vq3_20261010`：Qwen3-Embedding-4B最后有效token提取并FP32 L2归一化，历史＋真实下一商品标题，直接绑定新版商品SID。原单位范数VQ训练塌缩为全部样本同一组三码，不能直接使用这些兴趣标签。

2026-10-11 修复后 `bash run_target_official_vq.sh` 默认只用GPU6，复用上述目录的已校验embedding，在新目录 `outputs/latent_qwen3_0.6b_target_lasttoken_l2_sqrtd_vq3_20261011` 重新训练3槽非balanced VQ，完成后停在标签阶段。默认不重新编码；可用 `TARGET_OFFICIAL_EMBEDDINGS` 指定完整embedding目录，`TARGET_OFFICIAL_VQ_ROOT` 指定新的输出目录。保留旧产物，不自动SFT/eval。

`latent.prepare vq --input_scaling auto` 会在源recipe标记L2且向量实际为单位范数时使用 `x=sqrt(dim)*e`，其他输入保持原行为。可显式选择 `sqrt_dim` 或诊断用 `none`。缩放保存在 `vq.pt` 的 `options.input_scale` 中，初始化、训练、验证、标签导出及重载后的 `quantize` 全部在模型内部一致应用；调用者应传原始unit embedding，不要提前重复缩放。原embedding文件不变，旧checkpoint缺少该选项时仍按scale1加载。

VQ仍train-only拟合、valid选模型。`labels.json.vq_input_transform` 记录变换，`validation_reconstruction` 为模型缩放尺度MSE，`validation_reconstruction_source_scale` 为原embedding尺度MSE；`training_metrics.json` 每轮记录码本使用数、perplexity及相对train-mean常量预测器的验证误差。`collapsed_all_slots` 标记完全塌缩；数值很小的重建MSE本身不代表有效兴趣标签。旧mean/noL2方案保留。

### 目标条件 mean/noL2 兴趣 SFT 对照（2026-10-11）

`python run_target_interest_sft.py --root outputs/latent_qwen3_0.6b_target_mean_sft15_20261011 --gpus 5,6`：复用 `outputs/latent_qwen3_0.6b_target_vq3_20261010/labels`，旧商品SID，历史＋目标teacher、mean pooling/无L2、三槽非balanced VQ。先验证全部split标签hash/行数/SID绑定，再等待指定两卡持续空闲；从原Qwen3-0.6B训练，双卡batch32/accum16保持global1024，原latent损失权重1、LR3e-4、warmup20、最多15轮、全量valid每半轮/最佳loss+patience3，attention不变。仅首个优化step之前发生OOM才归档失败输出并用batch16/accum32重启，不丢弃已有优化权重。源码快照固定训练/评估实现。

训练后分别对既有fixed256valid和20%valid3232行在两卡分片跑single20/four20_marginal，合并检查完整且无重叠样本。旧history-only latent与无latent对照复用已有相同样本/data/index/评估源码/协议的排名产物，历史速度不作同场对照。汇报Recall/NDCG@5/10/20与满20比例；训练中的gold teacher兴趣SIDloss仅作条件拟合监控，正式排名只从history预测latent。完成报告在root/eval/valid20pct/comparison.md；不启动RL、新版兴趣embedding或三槽并行结构。

## 新版仅历史兴趣标签（2026-10-12）

入口 `bash run_history_official_vq.sh`，独立目录 `outputs/latent_qwen3_0.6b_history_lasttoken_l2_sqrtd_vq3_20261012`。Qwen3-Embedding-4B只输入历史标题，last-valid-token pooling与FP32 L2；非平衡三槽ParallelVQ输入缩放sqrt(2560)，训练集拟合、验证集选择，输出标签绑定新版商品SID。不会接SFT、排名评估或RL。历史＋目标向量不能复用到该流程。

本次用户明确指定低利用率0卡共享，入口使用 `--gpus 0 --allow-shared-gpu`；启动需连续三次检查至少24GiB空闲且利用率低于10%，保留其他驻留进程。编码batch64，OOM自动减半；VQ最多50轮、patience5。`pipeline_status.json`、`encode0_batch64.log`、`vq.log`记录各阶段。`input_provenance_check.json`确认变更真实下一商品不影响64条实际样本的历史编码输入。

## 新商品SID＋新版仅历史兴趣：串行SFT（2026-10-13）

`python run_new_history_interest_sft.py --root outputs/latent_qwen3_0.6b_history_lasttoken_l2_newsid_serial15_20261013 --gpus 2,5`。使用新版商品SID dataset与`history_lasttoken_l2_sqrtd_vq3_20261012/labels`，标准因果串行生成三个兴趣再生成SID，从Qwen3-0.6B重新SFT；不引入query或目标兴趣。两卡batch32、accum16/global1024，15轮/LR3e-4/seed42/兴趣权重1，全量valid每半轮、最佳loss/patience3、save_total_limit1。静态单机torchrun；仅首次优化前OOM可归档失败初始化并改batch16/accum32，保留所有正式训练状态。仅启动检查磁盘，不周期检查。

自动评估固定256valid和3232valid，两种兴趣协议single20/four20_marginal，报告Recall/NDCG@5/10/20与满20比例。新版无latentSFT为主要对照；新旧SID统一映射成真实商品ID，保留原评估元数据。256样本与3232样本不嵌套，故小样本新版native需重新评估；3232新版native缓存经数据/模型路径/样本/索引/源码/协议验证后复用，旧native与旧历史latent也作背景参照。不会恢复失败的并行训练或启动RL。
