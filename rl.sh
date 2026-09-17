#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

category=Industrial_and_Scientific
data_root="${DATA_ROOT:-./data/Amazon23}"
item_root="$data_root/$category"
file_stem="${category}_5_2016-10-2018-11"
sft_output="${SFT_OUTPUT_DIR:-./outputs/amazon23_industrial_qwen3_1.7b_sft}"
output_dir="${RL_OUTPUT_DIR:-./outputs/amazon23_industrial_qwen3_1.7b_rl}"

accelerate launch --config_file ./config/zero2_opt.yaml \
    --num_processes 4 --main_process_port 29503 rl.py \
    --model_path "${MODEL_PATH:-$sft_output/final_checkpoint}" \
    --train_batch_size 16 \
    --eval_batch_size 128 \
    --num_train_epochs 2 \
    --gradient_accumulation_steps 16 \
    --train_file "$data_root/train/$file_stem.csv" \
    --eval_file "$data_root/valid/$file_stem.csv" \
    --info_file "$data_root/info/$file_stem.txt" \
    --category "$category" \
    --sample_train False \
    --eval_step 0.0999 \
    --reward_type ranking \
    --num_generations 16 \
    --mask_all_zero False \
    --dynamic_sampling False \
    --sync_ref_model True \
    --beam_search True \
    --test_during_training False \
    --temperature 1.0 \
    --learning_rate 1e-5 \
    --add_gt False \
    --beta 1e-3 \
    --dapo False \
    --output_dir "$output_dir" \
    --logging_dir "$output_dir/tensorboard" \
    --sid_index_path "$item_root/$category.index.json" \
    --item_meta_path "$item_root/$category.item.json" \
    "$@"
