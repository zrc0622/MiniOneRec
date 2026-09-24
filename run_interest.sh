#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

usage() {
    cat <<'EOF'
用法：bash run_interest.sh <阶段>
  sft                       复用原 balanced 15ep SFT（已有模型可跳过）
  mid [训练参数...]          兴趣 Mid-SFT
  rl 16x1|4x4 [训练参数...]   两种兴趣 RL，均从同一个 Mid-SFT 开始
  eval sft                  原 SFT 的独立四卡评估
  eval mid|16x1|4x4          兴趣模型独立四卡评估
  tensorboard               查看 Mid-SFT 和两种 RL

主要变量：INTEREST_SFT_DIR、INTEREST_ROOT、BALANCED_DATA_ROOT。
默认4卡、每卡16、累积16；Mid默认3ep/3e-5，RL默认2ep/1e-5。
新训练目录非空则拒绝重跑，改 INTEREST_ROOT 可新开实验。
EOF
}

stage="${1:-help}"
case "$stage" in help|-h|--help) usage; exit 0;; esac
shift
target=""
case "$stage" in
    rl) target="${1:-}"; [[ "$target" == 16x1 || "$target" == 4x4 ]] || { usage >&2; exit 2; }; shift;;
    eval) target="${1:-}"; [[ $# -eq 1 && ( "$target" == sft || "$target" == mid || "$target" == 16x1 || "$target" == 4x4 ) ]] || { usage >&2; exit 2; }; shift;;
    mid) ;;
    sft|tensorboard) [[ $# -eq 0 ]] || { usage >&2; exit 2; };;
    *) usage >&2; exit 2;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_IB_DISABLE=1
sft_dir="${INTEREST_SFT_DIR:-./outputs/amazon23_industrial_qwen3_0.6b_sft_balanced_15ep}"
root="${INTEREST_ROOT:-./outputs/interest_qwen3_0.6b_balanced}"
data_root="${BALANCED_DATA_ROOT:-./data/Amazon23/variants/rqkmeans_balanced}"
category=Industrial_and_Scientific
stem="${category}_5_2016-10-2018-11"
index="$data_root/$category/$category.index.json"
items="$data_root/$category/$category.item.json"
mid_dir="$root/mid"

if [[ "$stage" == tensorboard ]]; then
    exec tensorboard --logdir_spec "mid:$mid_dir/tensorboard,rl16x1:$root/rl_16x1/tensorboard,rl4x4:$root/rl_4x4/tensorboard" --host 127.0.0.1 --port 6006
fi
if [[ "$stage" == sft || ( "$stage" == eval && "$target" == sft ) ]]; then
    export QWEN3_06B_BALANCED_15EP_SFT_OUTPUT_DIR="$sft_dir"
    export BALANCED_DATA_ROOT="$data_root"
    if [[ "$stage" == sft ]]; then exec bash run_qwen3_0.6b_balanced_15ep.sh sft; fi
    exec bash run_qwen3_0.6b_balanced_15ep.sh eval sft
fi

IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#gpus[@]} -eq 4 ]] || { echo '需要4张不同GPU。' >&2; exit 2; }
for i in 0 1 2 3; do
    [[ -n "${gpus[$i]}" ]] || { echo 'GPU编号不能为空。' >&2; exit 2; }
    for ((j=0; j<i; j++)); do
        [[ "${gpus[$i]}" != "${gpus[$j]}" ]] || { echo 'GPU编号不能重复。' >&2; exit 2; }
    done
done
python prepare_balanced_sid.py --output_root "$data_root" --check

if [[ "$stage" == mid || "$stage" == rl ]]; then
    if [[ "$stage" == mid ]]; then
        model="$sft_dir/final_checkpoint"
        output="$mid_dir"
        mode=16x1
    else
        model="$mid_dir/final_checkpoint"
        output="$root/rl_$target"
        mode="$target"
    fi
    [[ -f "$model/config.json" ]] || { echo "缺少模型：${model}" >&2; exit 1; }
    # One ZeRO-2 launch path for both stages; rank-local batch has complete groups.
    exec torchrun --standalone --nproc_per_node 4 -m interest.train "$stage" \
        --model "$model" --output_dir "$output" --mode "$mode" \
        --train_file "$data_root/train/$stem.csv" --eval_file "$data_root/valid/$stem.csv" \
        --index "$index" --items "$items" --batch 16 --gradient_accumulation 16 \
        --deepspeed config/sft_zero2.json "$@"
fi

if [[ "$target" == mid ]]; then model="$mid_dir/final_checkpoint"; else model="$root/rl_$target/final_checkpoint"; fi
[[ -f "$model/config.json" ]] || { echo "缺少模型：${model}" >&2; exit 1; }
split="${INTEREST_EVAL_SPLIT:-test}"
[[ "$split" == test || "$split" == valid ]] || { echo 'INTEREST_EVAL_SPLIT只支持test/valid。' >&2; exit 2; }
result_root="${INTEREST_RESULT_ROOT:-./results/interest_qwen3_0.6b_balanced}"
mkdir -p "$result_root"
shard_dir="$(mktemp -d "$result_root/${target}_${split}.XXXXXX")"
echo "评估模型=${model}，分片日志=${shard_dir}"
pids=()
for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${gpus[$i]}" python -m interest.evaluate \
        --model "$model" --data "$data_root/$split/$stem.csv" --index "$index" \
        --output "$shard_dir/$i.json" --shard "$i" --shards 4 --beams 50 \
        > "$shard_dir/$i.log" 2>&1 &
    pids+=("$!")
done
failed=0
for i in 0 1 2 3; do
    if ! wait "${pids[$i]}"; then echo "GPU ${gpus[$i]} 失败，见 ${shard_dir}/$i.log" >&2; failed=1; fi
done
[[ "$failed" -eq 0 ]] || exit 1
python -m interest.evaluate --merge "$shard_dir/0.json" "$shard_dir/1.json" "$shard_dir/2.json" "$shard_dir/3.json" \
    --output "$shard_dir/merged.json"
