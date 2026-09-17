#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
category=Industrial_and_Scientific
accelerate launch --multi_gpu --num_processes 4 --num_machines 1 \
    --mixed_precision no --dynamo_backend no rq/text2emb/amazon_text2emb.py \
    --dataset "$category" \
    --root "${DATA_ROOT:-./data/Amazon23}/$category" \
    --plm_checkpoint "${EMB_MODEL:-Qwen/Qwen3-Embedding-4B}" \
    --batch_size 16 \
    "$@"
