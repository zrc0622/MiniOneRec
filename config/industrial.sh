#!/usr/bin/env bash
# 路径均相对 MiniOneRec 根目录；也可用绝对路径或同名环境变量覆盖。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export DATA_ROOT="${DATA_ROOT:-./data/Amazon23}"
export RAW_ROOT="${RAW_ROOT:-./data/raw/Amazon23}"
export BASE_MODEL="${BASE_MODEL:-./models/Qwen3-1.7B}"
export EMB_MODEL="${EMB_MODEL:-./models/Qwen3-Embedding-4B}"
export SID_OUTPUT_DIR="${SID_OUTPUT_DIR:-./outputs/amazon23_industrial_rqkmeans_plus}"
export SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-./outputs/amazon23_industrial_qwen3_1.7b_sft}"
export RL_OUTPUT_DIR="${RL_OUTPUT_DIR:-./outputs/amazon23_industrial_qwen3_1.7b_rl}"
export LOG_DIR="${LOG_DIR:-./logs}"
export RESULT_DIR="${RESULT_DIR:-./results/amazon23_industrial}"

# HF 直连不可用时取消下一行注释。
# export HF_ENDPOINT=https://hf-mirror.com
