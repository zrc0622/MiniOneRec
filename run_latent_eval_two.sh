#!/usr/bin/env bash
# Two experiments share four GPUs: at most two shard processes per GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
exec "${PYTHON_BIN:-python}" -u -m latent.eval_queue "$@"
