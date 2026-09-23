#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

usage() {
    cat <<'EOF'
用法：bash run_qwen3_0.6b_balanced_15ep.sh <阶段>

  sft          0.6B / 3e-4 / 已有 balanced SID / 最多 15 epochs
  rl           从本实验 SFT final_checkpoint 开始，沿用 rl.sh 的参数
  eval sft     单独评估本实验 SFT（4 卡）
  eval rl      单独评估本实验 RL（4 卡）
  tensorboard  查看 balanced 10ep/15ep SFT 与本实验 RL

复用 run_qwen3_0.6b_balanced.sh prepare 生成的数据，无需重新 prepare。
从 Qwen3-0.6B 预训练权重重新训练；约每半个 epoch 验证并保存，早停 patience=3。
模型、日志和评估结果独立保存；重复运行会复用本实验目录。
共享路径/GPU 见 config/industrial.sh；自定义平衡数据路径使用 BALANCED_DATA_ROOT。
已完成 balanced 15ep SFT 时，直接运行 rl → eval rl，无需重做 prepare 或 SFT。
EOF
}

stage="${1:-help}"
case "$stage" in
    help|-h|--help) usage; exit 0 ;;
    sft|rl|tensorboard)
        if [[ $# -ne 1 ]]; then usage >&2; exit 2; fi
        ;;
    eval)
        if [[ $# -ne 2 || ( "${2:-}" != sft && "${2:-}" != rl ) ]]; then usage >&2; exit 2; fi
        ;;
    *) usage >&2; exit 2 ;;
esac

source config/industrial.sh
source_data_root="${BALANCED_SOURCE_DATA_ROOT:-$DATA_ROOT}"
baseline_output_dir="${QWEN3_06B_BALANCED_SFT_OUTPUT_DIR:-./outputs/amazon23_industrial_qwen3_0.6b_sft_balanced}"
export DATA_ROOT="${BALANCED_DATA_ROOT:-$source_data_root/variants/rqkmeans_balanced}"
export BASE_MODEL="${QWEN3_06B_MODEL:-./models/Qwen3-0.6B}"
export SFT_OUTPUT_DIR="${QWEN3_06B_BALANCED_15EP_SFT_OUTPUT_DIR:-./outputs/amazon23_industrial_qwen3_0.6b_sft_balanced_15ep}"
export RL_OUTPUT_DIR="${QWEN3_06B_BALANCED_15EP_RL_OUTPUT_DIR:-./outputs/amazon23_industrial_qwen3_0.6b_rl_balanced_15ep}"
# 固定从对应 SFT 权重开始，避免继承其他实验的 MODEL_PATH。
export MODEL_PATH="$SFT_OUTPUT_DIR/final_checkpoint"
export LOG_DIR="$LOG_DIR/qwen3_0.6b_balanced_15ep"
export RESULT_DIR="$RESULT_DIR/qwen3_0.6b_balanced_15ep"

if [[ "$stage" == rl ]]; then
    IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
    if [[ ${#gpus[@]} -ne 4 ]]; then echo 'RL需要配置4张不同GPU。' >&2; exit 2; fi
    for i in 0 1 2 3; do
        if [[ -z "${gpus[$i]}" ]]; then echo 'GPU编号不能为空。' >&2; exit 2; fi
        for ((j=0; j<i; j++)); do
            if [[ "${gpus[$i]}" == "${gpus[$j]}" ]]; then echo 'GPU编号不能重复。' >&2; exit 2; fi
        done
    done
fi

case "$stage" in
    sft)
        python prepare_balanced_sid.py --output_root "$DATA_ROOT" --check
        echo "balanced 0.6B SFT：learning_rate=3e-4，最多 15 epochs，输出=${SFT_OUTPUT_DIR}"
        # 15 × (1/30) = 0.5 epoch，保留原 balanced 10ep 的验证/保存频率。
        exec bash run.sh sft --learning_rate 3e-4 --num_epochs 15 --eval_step 0.03333333333333333
        ;;
    rl)
        python prepare_balanced_sid.py --output_root "$DATA_ROOT" --check
        if [[ ! -f "$MODEL_PATH/config.json" ]]; then
            echo "未找到 balanced 15ep SFT 模型：${MODEL_PATH}；请检查 QWEN3_06B_BALANCED_15EP_SFT_OUTPUT_DIR。" >&2
            exit 1
        fi
        echo "balanced 15ep 接 RL：SFT=${MODEL_PATH}，RL输出=${RL_OUTPUT_DIR}"
        exec bash run.sh rl
        ;;
    eval)
        python prepare_balanced_sid.py --output_root "$DATA_ROOT" --check
        exec bash run.sh eval "$2"
        ;;
    tensorboard)
        exec tensorboard \
            --logdir_spec "balanced_10ep:$baseline_output_dir/tensorboard,balanced_15ep:$SFT_OUTPUT_DIR/tensorboard,balanced_15ep_rl:$RL_OUTPUT_DIR/tensorboard" \
            --host 127.0.0.1 --port 6006
        ;;
esac
