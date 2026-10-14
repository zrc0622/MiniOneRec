# 4×4 latent RL 加速说明

原版每次微批逐用户调用生成，兴趣与商品分两次生成、重复处理历史，并逐候选逐 token 计算约束概率。此改动提供显式启用的加速实现，减少这些开销，保留独立4×4采样及原分层奖励目标。

实现包含三项改进：

1. 同一微批内按主任务、商品识别、历史标题三个任务分别批量采样，处理不同历史长度时显式传入位置与 padding mask。
2. 每个用户只做一次历史 prefill。历史缓存分出4个完整兴趣，兴趣缓存再各分出4个商品；两个辅助任务直接分出16个商品。缓存复用仅用于无梯度采样，参考模型与策略模型打分仍处理完整序列。
3. 将所有候选的合法 token 分数 gather、FP32归一化与 logsumexp 向量化，合法词表索引在参考与策略打分之间复用，同时保留 autograd。

加速实现位于 `latent/rl_fast.py`，入口是 `python -m latent.train_fast rl`，沿用 `latent.train` 参数，但只支持 `rl --mode 4x4`。原入口仍使用原采样和概率实现；此提交不修改无 latent 的 `rl.py`、`minionerec_trainer.py` 或 `interest/`。启动器的显式 `--fast` 才会启用加速，默认行为保持。

当前主任务奖励为：

\[
r(y,y^*)=\mathbf{1}[y=y^*]+0.2\cdot\frac{\sum_{j\in\{a,b,c\}}\mathbf{1}[y_j=y_j^*]}{3}.
\]

完整命中包含可选第四层 d；部分匹配不包含 d、换行或 EOS。每个兴趣的奖励取其4个商品奖励的最大值，再在4个兴趣间归一化；商品优势在各自4候选内归一化。共享兴趣前缀只计一次，辅助任务仍使用完整命中的 binary 奖励。加速实现也支持已有 binary 主奖励。

保持温度1、每用户16条轨迹、原KL与参考同步、优化器、梯度检查点和原math SDP attention。没有修改attention，也没有裁剪模型输出层词表。SFT兴趣loss权重0.2仅描述初始模型；当前RL兴趣项权重为1。

数学上的条件 categorical 采样分布相同，但批量化、缓存计算形状与在合法词表内抽样会改变随机数消耗和BF16数值。同一seed不保证相同候选或最终模型。本次验证不能替代最终推荐指标对照。

当前环境为Python3.11、PyTorch2.6/cu124、Transformers4.57.1，使用项目的minionerec环境。直接运行加速入口时，传入模型、数据、items、index及新output_dir等常规参数，并使用两进程torchrun；batch32×两卡×累积16=全局1024条轨迹，每次更新64个独立用户／任务prompt。

当前实验专用启动器还执行输入和初始模型校验、两步训练/验证/checkpoint保存检查，随后丢弃短跑权重，从原SFT启动正式训练：

```sh
python run_new_latent_dense_rl.py \
  --root outputs/latent_dense_fast_new_run \
  --gpus 4,5 \
  --fast \
  --verified-fast-pilot outputs/latent_rl_speed_pilot_20261014_v2
```

root必须是新的目录，两卡需空闲。该启动器使用当前项目的固定数据和SFT路径；`--verified-fast-pilot`要求本机已完成的数值验证与真实训练短测，并校验加速源码hash。验证产物、数据、模型和outputs均不随Git提交。在新机器上可先用 `python -m latent.rl_fast_verify --model ... --data ... --index ... --output ...` 做数值检查，再用独立入口训练；不能假定本机的验证目录在新机器存在。

当前正式配置为：新版商品SID＋新版三槽仅历史兴趣，SFT权重0.2的final checkpoint初始化；2 epochs、LR1e-5、beta0.001、seed42、batch32/acc16、全局1024，完整157679条prompt/任务训练数据。训练验证固定20%（3232条）、约半epoch一次，保留3个中间checkpoint；结束后自动依次评估固定256条和3232条valid，使用single20及four20_marginal，报告Recall/NDCG@5/10/20及满20候选率。验证子集ID与数据hash写入manifest，新增RL控制项默认维持原验证规模与checkpoint数，旧SFT恢复忽略这些RL专属字段。

CPU测试覆盖不同长度历史与位置、可选第四层、EOS、4×4共享兴趣分组、约束概率及梯度、固定候选下的奖励/优势/loss/模型梯度、启动配方、全局batch预算、GPU占用/容量和完整测速证据：

```sh
python -m unittest \
  tests.test_latent_rl_fast \
  tests.test_new_latent_dense_rl \
  tests.test_latent_batch_benchmark \
  tests.test_latent_dense_reward -v
```

真实初始checkpoint的缓存采样与新鲜完整前向条件分布检查：FP32最大KL约2.14e-7，BF16最大KL约1.95e-3；原图与向量化图的概率及梯度检查通过。两卡真实训练/验证/保存与DeepSpeed清理检查通过，正式加速任务已验证有限非零梯度。

`run_latent_speed_benchmark.py`比较原版、仅概率向量化、全部加速；`run_latent_batch_benchmark.py`比较加速版batch32/64。它们冻结源码与配方、使用完整训练数据和同一SFT初始化，每组6次真实优化更新、排除前2次，取每步较慢rank的CUDA同步耗时。计时不含加载、验证、保存，阶段同步对各版本相同；短测不保存权重，不自动停止或重启正式任务。

| 版本 | 均值秒/更新 | 中位数秒/更新 | 采样秒/更新 |
|---|---:|---:|---:|
| 原版，batch32 | 26.357 | 25.367 | 10.578 |
| 仅概率向量化，batch32 | 22.457 | 21.625 | 10.763 |
| 全部加速，batch32 | 18.883 | 18.226 | 6.860 |

全部加速约1.396倍，更新时间减少28.36%。证据目录：`outputs/latent_rl_speed_pilot_20261014_v2/`，包含REPORT、results、verification、逐rank计时和训练完整性检查。

| 加速版每卡batch／累积 | 均值秒/更新 | 中位数秒/更新 | 峰值allocated／reserved GiB |
|---|---:|---:|---:|
| 32／16 | 18.913 | 18.245 | 9.83／11.98 |
| 64／8 | 19.634 | 17.974 | 13.58／16.37 |

两者都无OOM。64采样更快，但均值略慢、中位数略快，未显示稳定的整体收益，因此暂选32。证据目录：`outputs/latent_fast_batch32_64_pilot_20261014/`。只有4次稳定计时样本，不声称统计显著、长期吞吐或推荐质量提升。

实际切换于2026-10-15：旧4/5卡任务在357步停止并保留产物，新root为 `outputs/latent_qwen3_0.6b_newsid_newhistory_lw02_dense4x4_fast_rl_20261015`，从原SFT重新训练。无latent基线6/7卡保持原实现。当前入口没有新增RL完整优化器状态恢复能力，本次切换采用重新训练。
