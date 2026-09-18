#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

usage() {
    cat <<'EOF'
用法：bash run_qwen3_1.7b_lr.sh <阶段> [学习率]

  sft [学习率]        从 Qwen3-1.7B 预训练模型开始，4 卡 SFT
  eval sft [学习率]   单独评估对应学习率的 SFT（4 卡）
  tensorboard [学习率] 对比原基线与本次实验的 TensorBoard

默认学习率 5e-4，原基线为 3e-4；其余训练参数保持不变。
训练与评估请指定相同学习率，如：
  bash run_qwen3_1.7b_lr.sh sft 5e-4
  bash run_qwen3_1.7b_lr.sh eval sft 5e-4

复用已有数据、embedding、SID 和 CSV，无需重新 prepare。
每个学习率使用独立输出目录；同一学习率重复运行会复用该目录。
EOF
}

stage="${1:-help}"
case "$stage" in
    help|-h|--help) usage; exit 0 ;;
    sft|tensorboard)
        if [[ $# -gt 2 ]]; then usage >&2; exit 2; fi
        lr="${2:-5e-4}"
        ;;
    eval)
        if [[ $# -lt 2 || $# -gt 3 || "${2:-}" != sft ]]; then
            usage >&2
            exit 2
        fi
        lr="${3:-5e-4}"
        ;;
    *) usage >&2; exit 2 ;;
esac

# 规范化目录名：0.0005、5E-4 和 5e-4 指向同一个实验。
lr="$(python - "$lr" <<'PY'
import sys
from decimal import Decimal, InvalidOperation

try:
    lr = Decimal(sys.argv[1])
except InvalidOperation:
    raise SystemExit('学习率必须是有限正数，例如 5e-4。')
if not lr.is_finite() or lr <= 0:
    raise SystemExit('学习率必须是有限正数，例如 5e-4。')
print(format(lr.normalize(), 'e'))
PY
)"

source config/industrial.sh
baseline_output_dir="$SFT_OUTPUT_DIR"
# 专用变量避免继承其他实验的模型及 checkpoint 输出目录。
export BASE_MODEL="${QWEN3_17B_LR_MODEL:-./models/Qwen3-1.7B}"
export SFT_OUTPUT_DIR="${QWEN3_17B_LR_OUTPUT_ROOT:-./outputs/amazon23_industrial_qwen3_1.7b_sft_lr}/$lr"
export LOG_DIR="$LOG_DIR/qwen3_1.7b_lr/$lr"
export RESULT_DIR="$RESULT_DIR/qwen3_1.7b_lr/$lr"

case "$stage" in
    sft)
        echo "1.7B SFT：learning_rate=${lr}，输出=${SFT_OUTPUT_DIR}"
        exec bash run.sh sft --learning_rate "$lr"
        ;;
    eval)
        exec bash run.sh eval sft
        ;;
    tensorboard)
        exec tensorboard \
            --logdir_spec "baseline:$baseline_output_dir/tensorboard,lr_$lr:$SFT_OUTPUT_DIR/tensorboard" \
            --host 127.0.0.1 --port 6006
        ;;
esac
