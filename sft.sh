#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

category=Industrial_and_Scientific
data_root="${DATA_ROOT:-./data/Amazon23}"
item_root="$data_root/$category"
file_stem="${category}_5_2016-10-2018-11"
output_dir="${SFT_OUTPUT_DIR:-./outputs/amazon23_industrial_qwen3_1.7b_sft}"

torchrun --standalone --nproc_per_node 4 sft.py \
    --base_model "${BASE_MODEL:-Qwen/Qwen3-1.7B}" \
    --batch_size 1024 \
    --micro_batch_size 16 \
    --deepspeed ./config/sft_zero2.json \
    --train_file "$data_root/train/$file_stem.csv" \
    --eval_file "$data_root/valid/$file_stem.csv" \
    --output_dir "$output_dir" \
    --logging_dir "$output_dir/tensorboard" \
    --category "$category" \
    --train_from_scratch False \
    --seed 42 \
    --sid_index_path "$item_root/$category.index.json" \
    --item_meta_path "$item_root/$category.item.json" \
    --freeze_LLM False \
    "$@"
