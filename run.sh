#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source config/industrial.sh
export PYTHONUNBUFFERED=1

category=Industrial_and_Scientific
item_root="$DATA_ROOT/$category"
file_stem="${category}_5_2016-10-2018-11"
single_gpu="${CUDA_VISIBLE_DEVICES%%,*}"

usage() {
    cat <<'EOF'
用法：bash run.sh <阶段> [参数]

  setup        安装依赖并检查 4 卡环境（先激活 Python 3.11 环境）
  download     下载两个 Qwen 模型与 Amazon23 Industrial 原始数据
  prepare      依次执行 preprocess → embedding → sid → convert
  sft [参数]   4 卡 SFT；参数透传给 sft.sh
  rl [参数]    4 卡 RL；参数透传给 rl.sh
  eval sft     4 卡分片评估 SFT，合并后计算指标
  eval rl      4 卡分片评估 RL，合并后计算指标
  tensorboard  启动 TensorBoard，端口 6006

可单独重跑：preprocess、embedding、sid、sid-index、convert。
sid 包含 constrained K-means 初始化、RQ-Kmeans+ 训练及 SID 生成。
sid-index 默认使用 SID_OUTPUT_DIR 中最近更新的 best_collision_model.pth，
也可用 SID_CKPT 指定本地 checkpoint。请勿在同一 SID_OUTPUT_DIR 并发训练。
路径与 GPU 配置见 config/industrial.sh；终端输出追加到 logs/<阶段>.log。
EOF
}

latest_sid_checkpoint() {
    python - <<'PY'
import os
from pathlib import Path
paths = list(Path(os.environ['SID_OUTPUT_DIR']).rglob('best_collision_model.pth'))
if not paths:
    raise SystemExit('未找到 SID checkpoint，请先运行 bash run.sh sid。')
print(max(paths, key=lambda p: p.stat().st_mtime_ns))
PY
}

generate_sid() {
    local checkpoint="${SID_CKPT:-}"
    if [[ -z "$checkpoint" ]]; then
        checkpoint="$(latest_sid_checkpoint)"
    fi
    if [[ ! -f "$checkpoint" ]]; then
        echo "SID checkpoint 不存在：$checkpoint" >&2
        return 1
    fi
    echo "生成 SID，使用 checkpoint：$checkpoint"
    # 原 checkpoint 含 argparse.Namespace；仅为本地 SID 权重开启 PyTorch 2.6 兼容加载。
    SID_CKPT="$checkpoint" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
        CUDA_VISIBLE_DEVICES="$single_gpu" bash rq/generate_indices_plus.sh
}

run_stage() {
    local stage="$1"
    shift
    echo "==> $stage"
    case "$stage" in
        setup)
            python - <<'PY'
import platform, sys
assert platform.system() == 'Linux' and platform.machine() == 'x86_64', '需要 Linux x86_64 GPU 服务器'
assert sys.version_info[:2] == (3, 11), '请先激活 Python 3.11 环境'
PY
            python -m pip install --upgrade pip setuptools wheel packaging ninja
            python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
            DS_BUILD_OPS=0 python -m pip install --no-build-isolation -r requirements-l40.txt
            python -m pip check
            nvidia-smi
            python - <<'PY'
import torch
import transformers, accelerate, trl, deepspeed, bitsandbytes
assert torch.cuda.device_count() >= 4, '需要至少 4 张可见 GPU'
assert torch.cuda.is_bf16_supported(), '需要支持 BF16 的 GPU'
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('transformers:', transformers.__version__, 'trl:', trl.__version__)
for i in range(4):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, round(p.total_memory / 2**30, 1), 'GiB')
PY
            ;;
        download)
            hf download Qwen/Qwen3-1.7B --local-dir "$BASE_MODEL"
            hf download Qwen/Qwen3-Embedding-4B --local-dir "$EMB_MODEL"
            python data/download_amazon23.py --output_dir "$RAW_ROOT"
            ;;
        prepare)
            local step
            for step in preprocess embedding sid convert; do
                run_stage "$step"
            done
            ;;
        preprocess)
            bash data/amazon23_data_process.sh
            ;;
        embedding)
            bash rq/text2emb/amazon_text2emb.sh
            python - <<'PY'
import os, json
from pathlib import Path
import numpy as np
root = Path(os.environ['DATA_ROOT']) / 'Industrial_and_Scientific'
a = np.load(root / 'Industrial_and_Scientific.emb-qwen-td.npy', mmap_mode='r')
items = json.loads((root / 'Industrial_and_Scientific.item.json').read_text())
assert a.shape == (len(items), 2560), a.shape
assert np.isfinite(a).all()
assert set(items) == {str(i) for i in range(len(items))}
print('embedding verified:', a.shape, a.dtype)
PY
            ;;
        sid)
            python rq/rqkmeans_constrained.py \
                --dataset "$category" --root "$item_root" \
                --k 256 --l 3 --max_iter 100 --seed 42 --verbose
            CUDA_VISIBLE_DEVICES="$single_gpu" bash rq/rqkmeans_plus.sh
            # 完整 sid 阶段使用刚训练的模型；SID_CKPT 仅用于单独 sid-index。
            SID_CKPT="$(latest_sid_checkpoint)" generate_sid
            ;;
        sid-index)
            generate_sid
            ;;
        convert)
            python convert_dataset.py \
                --dataset_name "$category" --category "$category" \
                --data_dir "$item_root" --output_dir "$DATA_ROOT" --seed 42
            ;;
        sft|rl)
            bash "$stage.sh" "$@"
            ;;
        eval)
            local target="$1" ckpt shard_dir i j failed=0
            local gpus=() pids=()
            IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
            if [[ ${#gpus[@]} -ne 4 ]]; then
                echo '评估需要在 CUDA_VISIBLE_DEVICES 中配置 4 张 GPU。' >&2
                return 1
            fi
            for i in 0 1 2 3; do
                if [[ -z "${gpus[$i]}" ]]; then
                    echo 'CUDA_VISIBLE_DEVICES 含空 GPU 编号。' >&2
                    return 1
                fi
                for ((j=0; j<i; j++)); do
                    if [[ "${gpus[$i]}" == "${gpus[$j]}" ]]; then
                        echo 'CUDA_VISIBLE_DEVICES 中的 GPU 不得重复。' >&2
                        return 1
                    fi
                done
            done
            if [[ "$target" == sft ]]; then
                ckpt="$SFT_OUTPUT_DIR/final_checkpoint"
            else
                ckpt="$RL_OUTPUT_DIR/final_checkpoint"
            fi
            mkdir -p "$RESULT_DIR"
            # 每次评估使用独立目录，避免混入之前运行的分片结果。
            shard_dir="$(mktemp -d "$RESULT_DIR/${target}_shards.XXXXXX")"
            python split.py --input_path "$DATA_ROOT/test/$file_stem.csv" \
                --output_path "$shard_dir" --cuda_list "0,1,2,3"
            echo "评估 ${target}：${ckpt}；分片结果及日志：$shard_dir"
            for i in 0 1 2 3; do
                CUDA_VISIBLE_DEVICES="${gpus[$i]}" python evaluate.py \
                    --base_model "$ckpt" --category "$category" \
                    --info_file "$DATA_ROOT/info/$file_stem.txt" \
                    --test_data_path "$shard_dir/$i.csv" \
                    --result_json_data "$shard_dir/$i.json" \
                    --batch_size 8 --num_beams 50 --max_new_tokens 256 --length_penalty 0.0 \
                    > "$shard_dir/$i.log" 2>&1 &
                pids+=("$!")
            done
            for i in 0 1 2 3; do
                if ! wait "${pids[$i]}"; then
                    echo "GPU ${gpus[$i]} 评估失败，见 $shard_dir/$i.log" >&2
                    failed=1
                fi
            done
            if [[ "$failed" -ne 0 ]]; then
                echo '评估失败，未合并结果或计算指标。' >&2
                return 1
            fi
            python merge.py --input_path "$shard_dir" \
                --output_path "$RESULT_DIR/$target.json" --cuda_list "0,1,2,3"
            python calc.py --path "$RESULT_DIR/$target.json" \
                --item_path "$DATA_ROOT/info/$file_stem.txt"
            ;;
        tensorboard)
            tensorboard --logdir_spec "sft:$SFT_OUTPUT_DIR/tensorboard,rl:$RL_OUTPUT_DIR/tensorboard" \
                --host 127.0.0.1 --port 6006
            ;;
    esac
}

stage="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in
    help|-h|--help) usage; exit 0 ;;
    sft|rl) ;;
    eval)
        if [[ $# -ne 1 || ( "${1:-}" != sft && "${1:-}" != rl ) ]]; then
            echo '请单独选择评估阶段：bash run.sh eval sft 或 bash run.sh eval rl' >&2
            exit 2
        fi
        ;;
    setup|download|prepare|preprocess|embedding|sid|sid-index|convert|tensorboard)
        if [[ $# -ne 0 ]]; then
            echo "阶段 $stage 不接受额外参数；路径与 GPU 请在 config/industrial.sh 配置。" >&2
            exit 2
        fi
        ;;
    *) usage >&2; exit 2 ;;
esac

mkdir -p "$LOG_DIR"
log_name="$stage"
if [[ "$stage" == eval ]]; then log_name="eval_$1"; fi
run_stage "$stage" "$@" 2>&1 | tee -a "$LOG_DIR/$log_name.log"
