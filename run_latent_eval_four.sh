#!/usr/bin/env bash
# Four sequential evaluation schemes; each uses four independent GPU shards.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'HELP'
用法：conda activate minionerec; bash run_latent_eval_four.sh [--dry-run]
默认：GPU=4,5,6,7，split=test，已有latent SFT，无需训练。
依次：joint_max(200)、hierarchical_max(4x50)、hierarchical_sum(4x50)、fixed_latent(1x50)。
变量：CUDA_VISIBLE_DEVICES、LATENT_ROOT、LATENT_EVAL_MODEL、BALANCED_DATA_ROOT、
LATENT_EVAL_SPLIT、LATENT_EVAL_BEAMS(200)、LATENT_EVAL_LATENTS(4)、
LATENT_EVAL_LATENT_BEAMS(4)、LATENT_EVAL_ITEM_BEAMS(50)、LATENT_EVAL_SCORE_BATCH(8)、
LATENT_EVAL_MAX_EXAMPLES(默认全量)、PYTHON_BIN(默认python)。
--dry-run 只显示配置和命令；LATENT_EVAL_MAX_EXAMPLES=8 可先做四卡小规模验证。
每次运行创建独立目录，输出每卡日志、merged.json、metrics及comparison.{json,csv,md}。
HELP
    exit 0
fi
dry_run=0
if [[ ${1:-} == --dry-run ]]; then dry_run=1; shift; fi
[[ $# -eq 0 ]] || { echo '未知参数，使用--help查看用法。' >&2; exit 2; }
python_bin="${PYTHON_BIN:-python}"
root="${LATENT_ROOT:-./outputs/latent_qwen3_0.6b_balanced15}"
model="${LATENT_EVAL_MODEL:-$root/sft/final_checkpoint}"
data_root="${BALANCED_DATA_ROOT:-./data/Amazon23/variants/rqkmeans_balanced}"
split="${LATENT_EVAL_SPLIT:-test}"
[[ $split == test || $split == valid ]] || { echo '只支持test/valid。' >&2; exit 2; }
stem=Industrial_and_Scientific_5_2016-10-2018-11
index="$data_root/Industrial_and_Scientific/Industrial_and_Scientific.index.json"
data="$data_root/$split/$stem.csv"
beams="${LATENT_EVAL_BEAMS:-200}"
latents="${LATENT_EVAL_LATENTS:-4}"
latent_beams="${LATENT_EVAL_LATENT_BEAMS:-4}"
item_beams="${LATENT_EVAL_ITEM_BEAMS:-50}"
score_batch="${LATENT_EVAL_SCORE_BATCH:-8}"
max_examples="${LATENT_EVAL_MAX_EXAMPLES:-}"
for value in "$beams" "$latents" "$latent_beams" "$item_beams" "$score_batch"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || { echo '搜索参数必须为正整数。' >&2; exit 2; }
done
[[ $beams -ge 50 && $item_beams -ge 50 ]] || { echo '联合及商品beam宽度须不小于50。' >&2; exit 2; }
limit=()
if [[ -n $max_examples ]]; then
    [[ $max_examples =~ ^[1-9][0-9]*$ ]] || { echo '样本数须为正整数。' >&2; exit 2; }
    limit=(--max-examples "$max_examples")
fi
IFS=',' read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
[[ ${#gpus[@]} -eq 4 ]] || { echo '请配置4张不同GPU。' >&2; exit 2; }
for i in 0 1 2 3; do
    [[ -n ${gpus[$i]} ]] || exit 2
    for ((j=0; j<i; j++)); do
        [[ ${gpus[$i]} != "${gpus[$j]}" ]] || { echo 'GPU不能重复。' >&2; exit 2; }
    done
done
for file in "$model/config.json" "$index" "$data"; do
    [[ -f $file ]] || { echo "缺少文件：$file" >&2; exit 1; }
done
schemes=(joint_max hierarchical_max hierarchical_sum fixed_latent)
common=(--model "$model" --data "$data" --index "$index" --shards 4
        --beams "$beams" --latent-count "$latents" --latent-beams "$latent_beams"
        --item-beams "$item_beams" --score-batch "$score_batch" "${limit[@]}")
echo "GPU=${gpus[*]}，split=$split，模型=$model，样本=${max_examples:-全量}"
if ((dry_run)); then
    for scheme in "${schemes[@]}"; do
        printf '%q ' "$python_bin" -m latent.evaluate "${common[@]}" --scheme "$scheme" --shard 0 --output "<新目录>/$scheme/0.json"
        printf '\n'
    done
    exit 0
fi
# Fail early rather than silently launching four CPU evaluations.
for gpu in "${gpus[@]}"; do
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA不可用"; print(torch.cuda.get_device_name(0))'
done
mkdir -p "$root/eval"
batch_dir="$(mktemp -d "$root/eval/four_${split}.XXXXXX")"
echo "结果目录：$batch_dir"
pids=()
cleanup() {
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for scheme in "${schemes[@]}"; do
    scheme_dir="$batch_dir/$scheme"
    mkdir "$scheme_dir"
    echo "开始：$scheme"
    started=$SECONDS
    pids=()
    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES="${gpus[$i]}" "$python_bin" -u -m latent.evaluate \
            "${common[@]}" --scheme "$scheme" --shard "$i" --output "$scheme_dir/$i.json" \
            > "$scheme_dir/$i.log" 2>&1 &
        pids+=("$!")
    done
    failed=0
    for i in 0 1 2 3; do
        if ! wait "${pids[$i]}"; then echo "评估失败：$scheme_dir/$i.log" >&2; failed=1; fi
    done
    pids=()
    [[ $failed -eq 0 ]] || exit 1
    if ! "$python_bin" -m latent.evaluate --merge "$scheme_dir/0.json" "$scheme_dir/1.json" \
        "$scheme_dir/2.json" "$scheme_dir/3.json" --output "$scheme_dir/merged.json" \
        > "$scheme_dir/merge.log" 2>&1; then
        echo "合并失败：$scheme_dir/merge.log" >&2
        exit 1
    fi
    printf '%s\n' "$((SECONDS - started))" > "$scheme_dir/wall_seconds.txt"
    "$python_bin" -m latent.eval_compare "$batch_dir"
    echo "完成：$scheme；指标：$scheme_dir/merged.json.metrics.json"
done
echo "四种方案完成，对比表：$batch_dir/comparison.md"
