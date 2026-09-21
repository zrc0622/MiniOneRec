#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

usage() {
    cat <<'EOF'
用法：bash run_qwen3_0.6b_25ep.sh <阶段>

  sft          0.6B / 3e-4 / 当前 RQ-Kmeans+ / 最多 25 epochs
  eval sft     单独评估本次 SFT（4 卡）
  tensorboard  对比原 0.6B 基线与本次 SFT 日志

从 Qwen3-0.6B 预训练权重重新训练，复用现有 SID/CSV，无需 prepare。
约每半个 epoch 验证并保存；保留连续 3 次验证无改善的早停。
模型、日志和评估结果独立保存；重复运行会复用本实验目录。
EOF
}

stage="${1:-help}"
case "$stage" in
    help|-h|--help) usage; exit 0 ;;
    sft|tensorboard)
        if [[ $# -ne 1 ]]; then usage >&2; exit 2; fi
        ;;
    eval)
        if [[ $# -ne 2 || "${2:-}" != sft ]]; then usage >&2; exit 2; fi
        ;;
    *) usage >&2; exit 2 ;;
esac

source config/industrial.sh
baseline_output_dir="${QWEN3_06B_SFT_OUTPUT_DIR:-./outputs/amazon23_industrial_qwen3_0.6b_sft}"
export BASE_MODEL="${QWEN3_06B_MODEL:-./models/Qwen3-0.6B}"
# 专用输出变量避免继承其他 SFT 实验的 checkpoint 目录。
export SFT_OUTPUT_DIR="${QWEN3_06B_25EP_SFT_OUTPUT_DIR:-./outputs/amazon23_industrial_qwen3_0.6b_sft_25ep}"
export LOG_DIR="$LOG_DIR/qwen3_0.6b_25ep"
export RESULT_DIR="$RESULT_DIR/qwen3_0.6b_25ep"

case "$stage" in
    sft)
        echo "0.6B SFT：learning_rate=3e-4，最多 25 epochs，输出=${SFT_OUTPUT_DIR}"
        # 25 × 0.02 = 0.5 epoch，与原 10 × 0.05 的验证/保存频率一致。
        exec bash run.sh sft --learning_rate 3e-4 --num_epochs 25 --eval_step 0.02
        ;;
    eval)
        exec bash run.sh eval sft
        ;;
    tensorboard)
        exec tensorboard \
            --logdir_spec "baseline_10ep:$baseline_output_dir/tensorboard,sft_25ep:$SFT_OUTPUT_DIR/tensorboard" \
            --host 127.0.0.1 --port 6006
        ;;
esac
