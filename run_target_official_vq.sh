#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec /data/zuorongchang/miniconda3/envs/minionerec/bin/python -u run_target_conditioned_vq.py \
  --root "${TARGET_OFFICIAL_VQ_ROOT:-outputs/latent_qwen3_0.6b_target_lasttoken_l2_sqrtd_vq3_20261011}" \
  --gpus "${TARGET_OFFICIAL_VQ_GPU:-6}" --batch 64 \
  --data outputs/amazon23_industrial_qwen3_4b_lasttoken_l2_sid_20261010/dataset \
  --pooling last_token --l2-normalize --vq-input-scaling sqrt_dim \
  --reuse-embeddings "${TARGET_OFFICIAL_EMBEDDINGS:-outputs/latent_qwen3_0.6b_target_lasttoken_l2_vq3_20261010/embeddings}" "$@"
