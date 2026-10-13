> 原服务器历史快照，非实时任务状态。迁移与当前工作以 [新服务器交接文档](../NEW_SERVER_HANDOFF.md) 为准；下面的绝对路径仅用于追溯。

# MiniOneRec 阶段性实验结果

快照时间：2026-10-13T23:08:02+08:00。数值是比例，非百分数。

使用Qwen3-0.6B；原兴趣VQ是并列VQ-VAE，不是残差RQ-VAE。新版/旧版商品SID都是对应的数据版本，不能与兴趣量化器名称混淆。
1×20：兴趣beam4取最高概率完整兴趣，再商品beam20。4×20：四兴趣各beam20，候选去重后在全部四兴趣下重打分求和，取top20。无latent使用直接SID beam50取top20。
同一用户只有一个真实下一商品，因此Recall@K=HR@K。所有本报告@20表均满20个不同商品。A/C固定256验证样本，仅作探索；B为固定20%验证3232条，不能与A/C直接比较数值。
原无latent SFT在A/C两轮评估R@20分别0.06250/0.05469；保留原轮配套对照，未择优拼接。BF16批次形状等可能影响边界排序，原因未单独隔离。中间checkpoint训练轮数不同，不作最终架构结论。

## A. 八份checkpoint统一小样本：SFT与RL

| 模型 | 方式 | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 |
|---|---|---:|---:|---:|---:|---:|---:|
| 旧SID，无latent SFT | 直接beam50→top20 | 0.03125 | 0.03906 | 0.06250 | 0.02930 | 0.03186 | 0.03753 |
| 旧SID，无latent RL | 直接beam50→top20 | 0.04688 | 0.05859 | 0.07031 | 0.03691 | 0.04099 | 0.04387 |
| 旧SID，原三槽VQ SFT | 1×20 | 0.03516 | 0.03906 | 0.05859 | 0.03083 | 0.03222 | 0.03711 |
| 旧SID，原三槽VQ SFT | 4×20求和 | 0.03516 | 0.03906 | 0.05859 | 0.03032 | 0.03162 | 0.03640 |
| 原VQ RL Binary-2465 | 1×20 | 0.03125 | 0.04688 | 0.06250 | 0.02690 | 0.03178 | 0.03562 |
| 原VQ RL Binary-2465 | 4×20求和 | 0.03125 | 0.04297 | 0.06641 | 0.02690 | 0.03065 | 0.03653 |
| 原VQ RL Binary-final | 1×20 | 0.02344 | 0.04688 | 0.07422 | 0.01977 | 0.02728 | 0.03385 |
| 原VQ RL Binary-final | 4×20求和 | 0.02344 | 0.04688 | 0.07031 | 0.01977 | 0.02734 | 0.03286 |
| 原VQ RL Dense-2465 | 1×20 | 0.04688 | 0.05469 | 0.07422 | 0.03886 | 0.04122 | 0.04594 |
| 原VQ RL Dense-2465 | 4×20求和 | 0.04688 | 0.05469 | 0.07031 | 0.03886 | 0.04122 | 0.04507 |
| 旧SID，平衡三槽VQ SFT | 1×20 | 0.03125 | 0.03125 | 0.05469 | 0.02302 | 0.02302 | 0.02911 |
| 旧SID，平衡三槽VQ SFT | 4×20求和 | 0.03125 | 0.03516 | 0.05469 | 0.02302 | 0.02419 | 0.02910 |
| 旧SID，平衡RQ-kmeans兴趣 SFT | 1×20 | 0.01953 | 0.03125 | 0.04688 | 0.01347 | 0.01738 | 0.02127 |
| 旧SID，平衡RQ-kmeans兴趣 SFT | 4×20求和 | 0.01953 | 0.02734 | 0.04688 | 0.01296 | 0.01574 | 0.02073 |

## B. 后续SFT：固定20%验证

| 模型 | 方式 | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 |
|---|---|---:|---:|---:|---:|---:|---:|
| 旧SID，无latent SFT | 直接beam50→top20 | 0.04610 | 0.05693 | 0.06931 | 0.03741 | 0.04081 | 0.04391 |
| 旧SID，原三槽VQ SFT | 1×20 | 0.04363 | 0.05136 | 0.06776 | 0.03606 | 0.03855 | 0.04266 |
| 旧SID，原三槽VQ SFT | 4×20求和 | 0.04301 | 0.05167 | 0.06528 | 0.03570 | 0.03854 | 0.04195 |
| 旧SID，两槽VQ SFT | 1×20 | 0.03960 | 0.05105 | 0.06498 | 0.03258 | 0.03624 | 0.03980 |
| 旧SID，两槽VQ SFT | 4×20求和 | 0.04053 | 0.05074 | 0.06528 | 0.03274 | 0.03601 | 0.03970 |
| 旧SID，历史+目标兴趣 SFT | 1×20 | 0.02970 | 0.04053 | 0.05353 | 0.02094 | 0.02442 | 0.02768 |
| 旧SID，历史+目标兴趣 SFT | 4×20求和 | 0.02908 | 0.03899 | 0.05384 | 0.02063 | 0.02387 | 0.02755 |
| 新版SID，无latent SFT | 直接beam50→top20 | 0.04610 | 0.05848 | 0.07395 | 0.03788 | 0.04192 | 0.04571 |

## C. 新方案与中间checkpoint：固定256验证

| 模型 | 方式 | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 |
|---|---|---:|---:|---:|---:|---:|---:|
| 旧SID，无latent 配套对照 | 直接beam50→top20 | 0.03125 | 0.03906 | 0.05469 | 0.02930 | 0.03171 | 0.03568 |
| 旧SID，原三槽VQ 配套对照 | 1×20 | 0.03516 | 0.03906 | 0.05859 | 0.03083 | 0.03222 | 0.03711 |
| 旧SID，原三槽VQ 配套对照 | 4×20求和 | 0.03516 | 0.03906 | 0.05859 | 0.03032 | 0.03162 | 0.03640 |
| 旧SID，两槽VQ SFT | 1×20 | 0.03125 | 0.04297 | 0.05859 | 0.02309 | 0.02684 | 0.03101 |
| 旧SID，两槽VQ SFT | 4×20求和 | 0.03125 | 0.03906 | 0.06250 | 0.02453 | 0.02722 | 0.03342 |
| 旧SID，历史+目标兴趣 SFT | 1×20 | 0.01953 | 0.03906 | 0.04297 | 0.01569 | 0.02239 | 0.02331 |
| 旧SID，历史+目标兴趣 SFT | 4×20求和 | 0.02344 | 0.02734 | 0.04297 | 0.01977 | 0.02116 | 0.02514 |
| 新版SID，无latent SFT | 直接beam50→top20 | 0.03125 | 0.04297 | 0.05469 | 0.02680 | 0.03089 | 0.03394 |
| 新版SID+新版历史兴趣，9.5轮 | 1×20 | 0.01562 | 0.03125 | 0.04297 | 0.01562 | 0.02046 | 0.02355 |
| 新版SID+新版历史兴趣，9.5轮 | 4×20求和 | 0.01562 | 0.02734 | 0.04688 | 0.01562 | 0.01923 | 0.02432 |
| 旧SID，并行兴趣，11.5轮 | 1×20 | 0.01953 | 0.03516 | 0.04688 | 0.01953 | 0.02427 | 0.02731 |
| 旧SID，并行兴趣，11.5轮 | 4×20求和 | 0.01953 | 0.03516 | 0.05078 | 0.01953 | 0.02449 | 0.02859 |

## D. 历史RL轨迹：单独的20%测试3233条、4×50 max协议

这组是test且采用旧4×50候选预算，与上面的valid/@20协议不同。

| 模型 | HR@10 | HR@50 | NDCG@10 | NDCG@50 |
|---|---:|---:|---:|---:|
| 原三槽VQ SFT | 0.06310 | 0.09681 | 0.04925 | 0.05656 |
| Binary-1972 | 0.06836 | 0.10114 | 0.05472 | 0.06177 |
| Dense-1972 | 0.06341 | 0.09650 | 0.05430 | 0.06140 |
| Binary-final 4928 | 0.05196 | 0.08815 | 0.03880 | 0.04657 |

Binary中间1972好于最终预算4928，不能称持续训练一直有收益。Dense只保留中间结果，没有完整最终训练结果。

## E. 当前任务与排队

| 实验 | 状态 | 显卡 | 最新日志步数 | 最新保存 |
|---|---|---|---:|---|
| 新版SID+新版历史兴趣，权重1 | training_serial_sft | 2,3 | 3122/4140 | 3036 |
| 新版SID+新版历史兴趣，权重0.2 | training_serial_sft | 6,7 | 440/4140 | 414 |
| 旧SID+原兴趣，并行预测 | failed | 待2,3卡释放 | 3237/4140 | 3174 |

并行恢复队列PID 1792984，状态 waiting_for_predecessor；等待2、3卡当前任务训练及自动评估全部完成，再从3174恢复，并自动评估。当前权重0.2没有排名结果；其总loss不能与权重1直接比较。

## F. 复用模型与结果路径

环境：`/data/zuorongchang/miniconda3/envs/minionerec/bin/python`。`results.json`含绝对路径和运行状态；`metrics.csv`每行保存来源文件与实际评估模型路径。

| 实验 | 实际评估模型 | 来源 |
|---|---|---|
| 旧SID，无latent SFT | /data/zuorongchang/project/MiniOneRec/outputs/amazon23_industrial_qwen3_0.6b_sft_balanced_15ep/final_checkpoint | /data/zuorongchang/project/MiniOneRec/outputs/eval_unified_small_20261010/comparison.json |
| 旧SID，无latent RL | /data/zuorongchang/project/MiniOneRec/outputs/amazon23_industrial_qwen3_0.6b_rl_balanced_15ep/final_checkpoint | /data/zuorongchang/project/MiniOneRec/outputs/eval_unified_small_20261010/comparison.json |
| 旧SID，原三槽VQ SFT | /data/zuorongchang/project/MiniOneRec/outputs/latent_qwen3_0.6b_balanced15/sft/final_checkpoint | /data/zuorongchang/project/MiniOneRec/outputs/eval_unified_small_20261010/comparison.json |
| 原VQ RL Binary-2465 | /data/zuorongchang/project/MiniOneRec/outputs/latent_qwen3_0.6b_balanced15/rl_two/train.Trz3R6/binary/checkpoint-2465 | /data/zuorongchang/project/MiniOneRec/outputs/eval_unified_small_20261010/comparison.json |
| 原VQ RL Binary-final | /data/zuorongchang/project/MiniOneRec/outputs/latent_qwen3_0.6b_balanced15/rl_two/train.Trz3R6/binary/final_checkpoint | /data/zuorongchang/project/MiniOneRec/outputs/eval_unified_small_20261010/comparison.json |
| 原VQ RL Dense-2465 | /data/zuorongchang/project/MiniOneRec/outputs/latent_qwen3_0.6b_balanced15/rl_two/train.Trz3R6/sid_dense/checkpoint-2465 | /data/zuorongchang/project/MiniOneRec/outputs/eval_unified_small_20261010/comparison.json |
| 旧SID，平衡三槽VQ SFT | /data/zuorongchang/project/MiniOneRec/outputs/latent_qwen3_0.6b_balanced_vq15/sft/final_checkpoint | /data/zuorongchang/project/MiniOneRec/outputs/eval_unified_small_20261010/comparison.json |
| 旧SID，平衡RQ-kmeans兴趣 SFT | /data/zuorongchang/project/MiniOneRec/outputs/latent_qwen3_0.6b_balanced_rqkmeans15/sft/final_checkpoint | /data/zuorongchang/project/MiniOneRec/outputs/eval_unified_small_20261010/comparison.json |
| 旧SID，无latent SFT | /data/zuorongchang/project/MiniOneRec/outputs/amazon23_industrial_qwen3_0.6b_sft_balanced_15ep/final_checkpoint | /data/zuorongchang/project/MiniOneRec/outputs/latent_qwen3_0.6b_vq2_15ep_20261010/eval/valid20pct/comparison.json |
| 旧SID，原三槽VQ SFT | /data/zuorongchang/project/MiniOneRec/outputs/latent_qwen3_0.6b_balanced15/sft/final_checkpoint | /data/zuorongchang/project/MiniOneRec/outputs/latent_qwen3_0.6b_vq2_15ep_20261010/eval/valid20pct/comparison.json |
| 旧SID，两槽VQ SFT | /data/zuorongchang/project/MiniOneRec/outputs/latent_qwen3_0.6b_vq2_15ep_20261010/sft/final_checkpoint | /data/zuorongchang/project/MiniOneRec/outputs/latent_qwen3_0.6b_vq2_15ep_20261010/eval/valid20pct/comparison.json |
| 旧SID，历史+目标兴趣 SFT | /data/zuorongchang/project/MiniOneRec/outputs/latent_qwen3_0.6b_target_mean_sft15_20261011/sft/final_checkpoint | /data/zuorongchang/project/MiniOneRec/outputs/latent_qwen3_0.6b_target_mean_sft15_20261011/eval/valid20pct/target_latent_sft.json |
| 新版SID，无latent SFT | /data/zuorongchang/project/MiniOneRec/outputs/amazon23_industrial_qwen3_0.6b_sft_lasttoken_l2_balanced15_20261011/sft/final_checkpoint | /data/zuorongchang/project/MiniOneRec/outputs/amazon23_industrial_qwen3_0.6b_sft_lasttoken_l2_balanced15_20261011/rank_eval_20261012/new_sid.json |
| 旧SID，无latent 配套对照 | /data/zuorongchang/project/MiniOneRec/outputs/amazon23_industrial_qwen3_0.6b_sft_balanced_15ep/final_checkpoint | /data/zuorongchang/project/MiniOneRec/outputs/latent_qwen3_0.6b_vq2_15ep_20261010/eval/sample256/comparison.json |
| 旧SID，原三槽VQ 配套对照 | /data/zuorongchang/project/MiniOneRec/outputs/latent_qwen3_0.6b_balanced15/sft/final_checkpoint | /data/zuorongchang/project/MiniOneRec/outputs/latent_qwen3_0.6b_vq2_15ep_20261010/eval/sample256/comparison.json |
| 旧SID，两槽VQ SFT | /data/zuorongchang/project/MiniOneRec/outputs/latent_qwen3_0.6b_vq2_15ep_20261010/sft/final_checkpoint | /data/zuorongchang/project/MiniOneRec/outputs/latent_qwen3_0.6b_vq2_15ep_20261010/eval/sample256/comparison.json |
| 旧SID，历史+目标兴趣 SFT | /data/zuorongchang/project/MiniOneRec/outputs/latent_qwen3_0.6b_target_mean_sft15_20261011/sft/final_checkpoint | /data/zuorongchang/project/MiniOneRec/outputs/latent_qwen3_0.6b_target_mean_sft15_20261011/eval/sample256/target_latent_sft.json |
| 新版SID，无latent SFT | /data/zuorongchang/project/MiniOneRec/outputs/amazon23_industrial_qwen3_0.6b_sft_lasttoken_l2_balanced15_20261011/sft/final_checkpoint | /data/zuorongchang/project/MiniOneRec/outputs/latent_qwen3_0.6b_history_lasttoken_l2_newsid_serial15_restart23_20261013/interim_eval/sample256_gpu67_20261013/comparison.json |
| 新版SID+新版历史兴趣，9.5轮 | /data/zuorongchang/project/MiniOneRec/outputs/latent_qwen3_0.6b_history_lasttoken_l2_newsid_serial15_restart23_20261013/interim_eval/sample256_gpu67_20261013/checkpoint-2622 | /data/zuorongchang/project/MiniOneRec/outputs/latent_qwen3_0.6b_history_lasttoken_l2_newsid_serial15_restart23_20261013/interim_eval/sample256_gpu67_20261013/comparison.json |
| 旧SID，并行兴趣，11.5轮 | /data/zuorongchang/project/MiniOneRec/outputs/eval_parallel_latest_20261013/parallel_checkpoint_3174 | /data/zuorongchang/project/MiniOneRec/outputs/eval_parallel_latest_20261013/parallel.json |

## 阶段性判断

- 新版商品SID无latent的20%验证R@20从0.06931到0.07395、N@20从0.04391到0.04571；目前是较清晰的正向结果。
- 原三槽兴趣SFT在20%验证略低于无latent；两槽与历史+目标没有改善。平衡兴趣量化器在256小样本表现更差，不能据此否定新版平衡商品SID。
- RL Dense-2465在统一256小样本NDCG表现较好，但未完成全量训练；Binary最终@20Recall有提升而@5/NDCG较差，旧4×50测试轨迹也显示后期回落。不能概括为RL全面有效。
- 新版历史兴趣9.5轮和并行兴趣11.5轮仍是中间结果，当前小样本低于对应基线；权重0.2与并行续训的最终结果待补。
