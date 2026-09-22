#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

usage() {
    cat <<'EOF'
用法：bash run_qwen3_0.6b_emb06b_balanced.sh <阶段>

  download     从HF下载Qwen3-Embedding-0.6B，复用已有backbone和原始数据
  prepare      重新编码1024维向量 → 平衡RQ-KMeans → 独立SID/CSV/info
  sft          Qwen3-0.6B / 3e-4 / 最多10ep，4卡ZeRO-2
  eval sft     单独四卡评估本实验SFT
  tensorboard  对比Embedding-4B与0.6B的balanced 10ep

保留mean pooling、无L2、原商品文本和交互划分；不运行RQ-Kmeans+训练。
按download → prepare → sft → eval sft运行，已有本实验数据可跳过prepare。
prepare使用四卡embedding和CPU聚类，失败不发布数据；重复prepare拒绝覆盖。
路径覆盖：EMB06B_SOURCE_DATA_ROOT、EMB06B_DATA_ROOT、EMB06B_MODEL、
QWEN3_06B_MODEL、QWEN3_06B_EMB06B_BALANCED_SFT_OUTPUT_DIR。
EOF
}

stage="${1:-help}"
case "$stage" in
    help|-h|--help) usage; exit 0 ;;
    download|prepare|sft|tensorboard)
        if [[ $# -ne 1 ]]; then usage >&2; exit 2; fi ;;
    eval)
        if [[ $# -ne 2 || "${2:-}" != sft ]]; then usage >&2; exit 2; fi ;;
    *) usage >&2; exit 2 ;;
esac

source config/industrial.sh
source_data_root="${EMB06B_SOURCE_DATA_ROOT:-$DATA_ROOT}"
baseline_output_dir="${QWEN3_06B_BALANCED_SFT_OUTPUT_DIR:-./outputs/amazon23_industrial_qwen3_0.6b_sft_balanced}"
export DATA_ROOT="${EMB06B_DATA_ROOT:-$source_data_root/variants/embedding06b_1024_balanced}"
export EMB_MODEL="${EMB06B_MODEL:-./models/Qwen3-Embedding-0.6B}"
export BASE_MODEL="${QWEN3_06B_MODEL:-./models/Qwen3-0.6B}"
export SFT_OUTPUT_DIR="${QWEN3_06B_EMB06B_BALANCED_SFT_OUTPUT_DIR:-./outputs/amazon23_industrial_qwen3_0.6b_sft_emb06b_balanced}"
export LOG_DIR="$LOG_DIR/qwen3_0.6b_emb06b_balanced"
export RESULT_DIR="$RESULT_DIR/qwen3_0.6b_emb06b_balanced"
export PYTHONUNBUFFERED=1

if [[ "$stage" == prepare || "$stage" == sft || "$stage" == eval ]]; then
    IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
    if [[ ${#gpus[@]} -ne 4 ]]; then echo '请配置4张不同GPU。' >&2; exit 2; fi
    for i in 0 1 2 3; do
        if [[ -z "${gpus[$i]}" ]]; then echo 'GPU编号不能为空。' >&2; exit 2; fi
        for ((j=0; j<i; j++)); do
            if [[ "${gpus[$i]}" == "${gpus[$j]}" ]]; then echo 'GPU编号不能重复。' >&2; exit 2; fi
        done
    done
fi

case "$stage" in
    download)
        hf download Qwen/Qwen3-Embedding-0.6B --local-dir "$EMB_MODEL"
        ;;
    prepare)
        mkdir -p "$LOG_DIR"
        python prepare_embedding06b_balanced.py --source_root "$source_data_root" \
            --output_root "$DATA_ROOT" --embedding_model "$EMB_MODEL" \
            2>&1 | tee -a "$LOG_DIR/prepare.log"
        ;;
    sft)
        python prepare_embedding06b_balanced.py --output_root "$DATA_ROOT" --check
        exec bash run.sh sft --learning_rate 3e-4 --num_epochs 10 --eval_step 0.05
        ;;
    eval)
        python prepare_embedding06b_balanced.py --output_root "$DATA_ROOT" --check
        exec bash run.sh eval sft
        ;;
    tensorboard)
        exec tensorboard \
            --logdir_spec "emb4b_balanced_10ep:$baseline_output_dir/tensorboard,emb06b_balanced_10ep:$SFT_OUTPUT_DIR/tensorboard" \
            --host 127.0.0.1 --port 6006
        ;;
esac
