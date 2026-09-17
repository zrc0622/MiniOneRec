#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${SID_CKPT:?Set SID_CKPT to the trained best_collision_model.pth path}"
category=Industrial_and_Scientific
item_root="${DATA_ROOT:-./data/Amazon23}/$category"
python rq/generate_indices_plus.py \
    --data_path "$item_root/$category.emb-qwen-td.npy" \
    --ckpt_path "$SID_CKPT" \
    --num_emb_list 256 256 256 \
    --device cuda:0 \
    "$@"
