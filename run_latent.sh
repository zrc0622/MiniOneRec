#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
usage() {
    cat <<'HELP'
用法：bash run_latent.sh <阶段>
  encode                    四卡分别编码历史标题，然后合并
  vq [参数...]              单卡训练三并列 VQ、导出 train/valid/test 标签
  sft [参数...]             从预训练 Qwen3-0.6B 联合 SFT，最多15ep
  rl original|4x4 [参数...]  从同一新版 SFT 分别做 RL
  eval sft|original|4x4      独立四卡评估
  baseline sft|rl           调用已有 balanced15ep 基线训练入口
  baseline eval sft|rl      调用已有基线评估入口
  tensorboard               查看新实验日志
变量：CUDA_VISIBLE_DEVICES、LATENT_ROOT、BALANCED_DATA_ROOT、QWEN3_06B_MODEL、EMB_MODEL。
评估：LATENT_EVAL_SPLIT=test|valid、LATENT_EVAL_BEAMS=50。
HELP
}
stage="${1:-help}"
case "$stage" in help|-h|--help) usage; exit 0;; esac
shift
target=""
case "$stage" in
    baseline) exec bash run_qwen3_0.6b_balanced_15ep.sh "$@";;
    rl) target="${1:-}"; [[ "$target" == original || "$target" == 4x4 ]] || { usage >&2; exit 2; }; shift;;
    eval) target="${1:-}"; [[ $# -eq 1 && ( "$target" == sft || "$target" == original || "$target" == 4x4 ) ]] || { usage >&2; exit 2; }; shift;;
    encode|tensorboard) [[ $# -eq 0 ]] || { usage >&2; exit 2; };;
    vq|sft) ;;
    *) usage >&2; exit 2;;
esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_IB_DISABLE=1
root="${LATENT_ROOT:-./outputs/latent_qwen3_0.6b_balanced15}"
data_root="${BALANCED_DATA_ROOT:-./data/Amazon23/variants/rqkmeans_balanced}"
category=Industrial_and_Scientific
stem="${category}_5_2016-10-2018-11"
index="$data_root/$category/$category.index.json"
items="$data_root/$category/$category.item.json"
if [[ "$stage" == tensorboard ]]; then
    exec tensorboard --logdir "$root" --host 127.0.0.1 --port 6006
fi
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#gpus[@]} -eq 4 ]] || { echo '请配置4张不同GPU，例如3,4,5,6。' >&2; exit 2; }
for i in 0 1 2 3; do
    [[ -n "${gpus[$i]}" ]] || exit 2
    for ((j=0; j<i; j++)); do [[ "${gpus[$i]}" != "${gpus[$j]}" ]] || { echo 'GPU不能重复。' >&2; exit 2; }; done
done
python prepare_balanced_sid.py --output_root "$data_root" --check
if [[ "$stage" == encode ]]; then
    [[ ! -e "$root/embeddings" ]] || { echo '编码目录已存在，请换LATENT_ROOT。' >&2; exit 1; }
    mkdir -p "$root/embeddings"
    pids=()
    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES="${gpus[$i]}" python -m latent.prepare encode \
            --train "$data_root/train/$stem.csv" --valid "$data_root/valid/$stem.csv" --test "$data_root/test/$stem.csv" \
            --index "$index" --model "${EMB_MODEL:-./models/Qwen3-Embedding-4B}" \
            --output "$root/embeddings" --shard "$i" --shards 4 \
            > "$root/embeddings/$i.log" 2>&1 &
        pids+=("$!")
    done
    failed=0
    for i in 0 1 2 3; do if ! wait "${pids[$i]}"; then echo "编码失败：${root}/embeddings/$i.log" >&2; failed=1; fi; done
    [[ "$failed" -eq 0 ]] || exit 1
    exec python -m latent.prepare merge --output "$root/embeddings" --shards 4
fi
if [[ "$stage" == vq ]]; then
    CUDA_VISIBLE_DEVICES="${gpus[0]}" exec python -m latent.prepare vq --embeddings "$root/embeddings" --output "$root/labels" "$@"
fi
if [[ "$stage" == sft || "$stage" == rl ]]; then
    if [[ "$stage" == sft ]]; then
        model="${QWEN3_06B_MODEL:-./models/Qwen3-0.6B}"; output="$root/sft"; target=original
    else
        model="$root/sft/final_checkpoint"; output="$root/rl_$target"
    fi
    [[ -f "$model/config.json" ]] || { echo "缺少模型：${model}" >&2; exit 1; }
    exec torchrun --standalone --nproc_per_node 4 -m latent.train "$stage" \
        --model "$model" --output_dir "$output" --mode "$target" --labels "$root/labels" \
        --train_file "$data_root/train/$stem.csv" --eval_file "$data_root/valid/$stem.csv" \
        --items "$items" --index "$index" --batch 16 --gradient_accumulation 16 \
        --deepspeed config/sft_zero2.json "$@"
fi
if [[ "$target" == sft ]]; then model="$root/sft/final_checkpoint"; else model="$root/rl_$target/final_checkpoint"; fi
[[ -f "$model/config.json" ]] || { echo "缺少模型：${model}" >&2; exit 1; }
split="${LATENT_EVAL_SPLIT:-test}"
[[ "$split" == test || "$split" == valid ]] || { echo '只支持test/valid。' >&2; exit 2; }
beams="${LATENT_EVAL_BEAMS:-50}"
[[ "$beams" =~ ^[0-9]+$ && "$beams" -ge 50 ]] || { echo 'beam预算必须为不小于50的整数。' >&2; exit 2; }
mkdir -p "$root/eval"
shards="$(mktemp -d "$root/eval/${target}_${split}_b${beams}.XXXXXX")"
echo "评估模型=${model}，路径预算=${beams}，结果目录=${shards}"
pids=()
for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${gpus[$i]}" python -m latent.evaluate --model "$model" \
        --data "$data_root/$split/$stem.csv" --index "$index" --output "$shards/$i.json" \
        --shard "$i" --shards 4 --beams "$beams" > "$shards/$i.log" 2>&1 &
    pids+=("$!")
done
failed=0
for i in 0 1 2 3; do if ! wait "${pids[$i]}"; then echo "评估失败：${shards}/$i.log" >&2; failed=1; fi; done
[[ "$failed" -eq 0 ]] || exit 1
exec python -m latent.evaluate --merge "$shards/0.json" "$shards/1.json" "$shards/2.json" "$shards/3.json" --output "$shards/merged.json"
