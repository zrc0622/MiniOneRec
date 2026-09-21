#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

usage() {
    cat <<'EOF'
用法：bash run_qwen3_0.6b_balanced.sh <阶段>

  prepare      从已有 codes_constrained.npy 重建平衡 SID 与独立 CSV/info
  sft          0.6B / 3e-4 / 最多 10 epochs，4 卡 ZeRO-2
  eval sft     单独评估本实验 SFT（4 卡）
  tensorboard  对比原 RQ-Kmeans+ 10ep 与平衡 SID 实验

按 prepare → sft → eval sft 顺序运行；无需重新下载、embedding 或聚类。
沿用原 CSV 的商品 ID、标题、顺序和划分，只替换 SID。
数据默认写入 data/Amazon23/variants/rqkmeans_balanced/，不覆盖原实验。
共享路径/GPU 见 config/industrial.sh；专用路径覆盖变量见本脚本。
EOF
}

stage="${1:-help}"
case "$stage" in
    help|-h|--help) usage; exit 0 ;;
    prepare|sft|tensorboard)
        if [[ $# -ne 1 ]]; then usage >&2; exit 2; fi
        ;;
    eval)
        if [[ $# -ne 2 || "${2:-}" != sft ]]; then usage >&2; exit 2; fi
        ;;
    *) usage >&2; exit 2 ;;
esac

source config/industrial.sh
source_data_root="${BALANCED_SOURCE_DATA_ROOT:-$DATA_ROOT}"
codes_path="${BALANCED_CODES_PATH:-$source_data_root/Industrial_and_Scientific/Industrial_and_Scientific.codes_constrained.npy}"
baseline_output_dir="${QWEN3_06B_SFT_OUTPUT_DIR:-./outputs/amazon23_industrial_qwen3_0.6b_sft}"
export DATA_ROOT="${BALANCED_DATA_ROOT:-$source_data_root/variants/rqkmeans_balanced}"
export BASE_MODEL="${QWEN3_06B_MODEL:-./models/Qwen3-0.6B}"
export SFT_OUTPUT_DIR="${QWEN3_06B_BALANCED_SFT_OUTPUT_DIR:-./outputs/amazon23_industrial_qwen3_0.6b_sft_balanced}"
export LOG_DIR="$LOG_DIR/qwen3_0.6b_balanced"
export RESULT_DIR="$RESULT_DIR/qwen3_0.6b_balanced"

case "$stage" in
    prepare)
        mkdir -p "$LOG_DIR"
        python prepare_balanced_sid.py --source_root "$source_data_root" \
            --output_root "$DATA_ROOT" --codes_path "$codes_path" \
            2>&1 | tee -a "$LOG_DIR/prepare.log"
        ;;
    sft)
        python prepare_balanced_sid.py --output_root "$DATA_ROOT" --check
        exec bash run.sh sft --learning_rate 3e-4 --num_epochs 10 --eval_step 0.05
        ;;
    eval)
        python prepare_balanced_sid.py --output_root "$DATA_ROOT" --check
        exec bash run.sh eval sft
        ;;
    tensorboard)
        exec tensorboard \
            --logdir_spec "rqkmeans_plus_10ep:$baseline_output_dir/tensorboard,balanced_10ep:$SFT_OUTPUT_DIR/tensorboard" \
            --host 127.0.0.1 --port 6006
        ;;
esac
