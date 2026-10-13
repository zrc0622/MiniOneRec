#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec /data/zuorongchang/miniconda3/envs/minionerec/bin/python -u run_qwen_official_sid_sft.py \
  --root "${QWEN_OFFICIAL_SID_SFT_ROOT:-outputs/amazon23_industrial_qwen3_0.6b_sft_lasttoken_l2_balanced15_20261010}" \
  --gpus "${QWEN_OFFICIAL_SID_SFT_GPUS:-2,6,4,7,0,1,3,5}" "$@"
