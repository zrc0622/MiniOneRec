#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
category=Industrial_and_Scientific
raw_root="${RAW_ROOT:-./data/raw/Amazon23}"
python data/amazon23_data_process.py \
    --dataset "$category" \
    --metadata_file "$raw_root/raw/meta_categories/meta_$category.jsonl" \
    --reviews_file "$raw_root/raw/review_categories/$category.jsonl" \
    --user_k 5 \
    --st_year 2018 --st_month 10 \
    --ed_year 2023 --ed_month 9 \
    --output_path "${DATA_ROOT:-./data/Amazon23}" \
    "$@"
