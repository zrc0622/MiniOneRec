#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
category=Industrial_and_Scientific
item_root="${DATA_ROOT:-./data/Amazon23}/$category"
python rq/rqkmeans_plus.py \
    --data_path "$item_root/$category.emb-qwen-td.npy" \
    --pretrained_codebook_path "$item_root/$category.codebooks_constrained.npz" \
    --num_emb_list 256 256 256 \
    --e_dim 2560 \
    --lr 1e-4 \
    --epochs 10000 \
    --batch_size 2048 \
    --ckpt_dir "${SID_OUTPUT_DIR:-./outputs/amazon23_industrial_rqkmeans_plus}" \
    "$@"
