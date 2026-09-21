#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

usage() {
    cat <<'EOF'
用法：bash run_qwen3_0.6b_15ep.sh <阶段>

  sft          0.6B / 3e-4 / 当前 RQ-Kmeans+ / 最多 15 epochs
  eval sft     单独评估本次 SFT（4 卡）
  tensorboard  对比 10ep、15ep、25ep SFT 日志

从 Qwen3-0.6B 预训练权重重新训练，复用原 RQ-Kmeans+ SID/CSV，无需 prepare。
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
output_25ep_dir="${QWEN3_06B_25EP_SFT_OUTPUT_DIR:-./outputs/amazon23_industrial_qwen3_0.6b_sft_25ep}"
export BASE_MODEL="${QWEN3_06B_MODEL:-./models/Qwen3-0.6B}"
export SFT_OUTPUT_DIR="${QWEN3_06B_15EP_SFT_OUTPUT_DIR:-./outputs/amazon23_industrial_qwen3_0.6b_sft_15ep}"
export LOG_DIR="$LOG_DIR/qwen3_0.6b_15ep"
export RESULT_DIR="$RESULT_DIR/qwen3_0.6b_15ep"

case "$stage" in
    sft)
        echo "0.6B SFT：learning_rate=3e-4，最多 15 epochs，输出=${SFT_OUTPUT_DIR}"
        # 15 × (1/30) = 0.5 epoch，与原 10 × 0.05 的验证/保存频率一致。
        exec bash run.sh sft --learning_rate 3e-4 --num_epochs 15 --eval_step 0.03333333333333333
        ;;
    eval)
        exec bash run.sh eval sft
        ;;
    tensorboard)
        exec tensorboard \
            --logdir_spec "baseline_10ep:$baseline_output_dir/tensorboard,sft_15ep:$SFT_OUTPUT_DIR/tensorboard,sft_25ep:$output_25ep_dir/tensorboard" \
            --host 127.0.0.1 --port 6006
        ;;
esac
