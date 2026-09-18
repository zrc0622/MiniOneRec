#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source config/industrial.sh

# 共享数据、embedding、SID 和 GPU 配置；0.6B 模型与训练产物单独保存。
# 使用专用变量，避免继承终端中为 1.7B 设置的模型或 checkpoint 路径。
export BASE_MODEL="${QWEN3_06B_MODEL:-./models/Qwen3-0.6B}"
export SFT_OUTPUT_DIR="${QWEN3_06B_SFT_OUTPUT_DIR:-./outputs/amazon23_industrial_qwen3_0.6b_sft}"
export RL_OUTPUT_DIR="${QWEN3_06B_RL_OUTPUT_DIR:-./outputs/amazon23_industrial_qwen3_0.6b_rl}"
export MODEL_PATH="${QWEN3_06B_MODEL_PATH:-$SFT_OUTPUT_DIR/final_checkpoint}"
export LOG_DIR="$LOG_DIR/qwen3_0.6b"
export RESULT_DIR="$RESULT_DIR/qwen3_0.6b"

case "${1:-help}" in
    help|-h|--help)
        cat <<'EOF'
用法：bash run_qwen3_0.6b.sh <阶段> [参数]

  download     仅下载 Qwen/Qwen3-0.6B
  sft [参数]   4 卡 SFT，复用 sft.sh 的训练参数
  rl [参数]    从 0.6B SFT final_checkpoint 开始，4 卡 RL
  eval sft     单独评估 0.6B SFT（4 卡）
  eval rl      单独评估 0.6B RL（4 卡）
  tensorboard  查看 0.6B SFT/RL 日志

环境与 prepare 复用 run.sh，已生成的数据、embedding 和 SID 无需重做。
0.6B 模型及输出路径可在本脚本顶部配置；GPU 与共享数据路径见 config/industrial.sh。
EOF
        ;;
    download)
        if [[ $# -ne 1 ]]; then
            echo '用法：bash run_qwen3_0.6b.sh download' >&2
            exit 2
        fi
        mkdir -p "$LOG_DIR"
        hf download Qwen/Qwen3-0.6B --local-dir "$BASE_MODEL" \
            2>&1 | tee -a "$LOG_DIR/download.log"
        ;;
    sft|rl|eval|tensorboard)
        exec bash run.sh "$@"
        ;;
    *)
        echo '未知阶段；用 bash run_qwen3_0.6b.sh help 查看用法。' >&2
        exit 2
        ;;
esac
