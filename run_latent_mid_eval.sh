#!/usr/bin/env bash
# Evaluate two same-step RL checkpoints concurrently on one relatively idle GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
step="${1:?需要checkpoint步数，例如1479}"
[[ "$step" =~ ^[1-9][0-9]*$ ]] || exit 2
python_bin="${PYTHON_BIN:-/data/zuorongchang/miniconda3/envs/minionerec/bin/python}"
train_dir="${LATENT_RL_RUN:-./outputs/latent_qwen3_0.6b_balanced15/rl_two/train.Trz3R6}"
gpu="${LATENT_MID_EVAL_GPU:-0}"
[[ "$gpu" =~ ^[0-9]+$ ]] || exit 2
data_root="${BALANCED_DATA_ROOT:-./data/Amazon23/variants/rqkmeans_balanced}"
data="${LATENT_MID_EVAL_DATA:-$data_root/test/Industrial_and_Scientific_5_2016-10-2018-11.csv}"
index="$data_root/Industrial_and_Scientific/Industrial_and_Scientific.index.json"
sft_reference="${LATENT_SFT_EVAL_REFERENCE:-./outputs/latent_qwen3_0.6b_balanced15/eval/four_test.JWNLYi/hierarchical_max/merged.json}"
[[ -f "$sft_reference" ]] || { echo '缺少SFT参考评估文件。' >&2; exit 1; }
for reward in binary sid_dense; do
    [[ -f "$train_dir/$reward/checkpoint-$step/config.json" ]] || exit 1
done
mkdir -p "$train_dir/mid_eval"
result_dir="$(mktemp -d "$train_dir/mid_eval/step${step}.XXXXXX")"
result_dir="$(cd "$result_dir" && pwd)"
printf '%s\n' "$result_dir" > "$train_dir/mid_eval/latest.txt"
printf 'step=%s\ngpu=%s\nsplit=test\nscheme=hierarchical_max\n' "$step" "$gpu" > "$result_dir/settings.txt"
printf 'data=%s\nsft_reference=%s\n' "$data" "$sft_reference" >> "$result_dir/settings.txt"
if [[ -n ${LATENT_MID_SAMPLE_MANIFEST:-} ]]; then cp "$LATENT_MID_SAMPLE_MANIFEST" "$result_dir/sample_manifest.json"; fi
echo "中间评估：step=$step GPU=$gpu，数据=$data；输出：$result_dir"
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for reward in binary sid_dense; do
    folder="$result_dir/eval/$reward/test"
    mkdir -p "$folder"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u -m latent.evaluate \
        --model "$train_dir/$reward/checkpoint-$step" --data "$data" --index "$index" \
        --scheme hierarchical_max --latent-count 4 --latent-beams 4 --item-beams 50 \
        --score-batch 8 --shards 1 --shard 0 --output "$folder/0.json" > "$folder/0.log" 2>&1 &
    pids+=("$!")
    printf '%s\t%s\n' "$reward" "$!" >> "$result_dir/pids.tsv"
done
failed=0
for i in 0 1; do
    reward=binary; [[ "$i" -eq 0 ]] || reward=sid_dense
    folder="$result_dir/eval/$reward/test"
    if wait "${pids[$i]}"; then code=0; else code=$?; fi
    printf '%s\n' "$code" > "$result_dir/$reward.exit_code"
    if [[ "$code" -ne 0 ]]; then failed=1; continue; fi
    "$python_bin" -m latent.evaluate --merge "$folder/0.json" --output "$folder/merged.json" > "$folder/merge.log" 2>&1
    printf '%s\n' "$folder" > "$result_dir/eval/$reward/latest_test.txt"
    "$python_bin" -m latent.rl_compare "$result_dir" --checkpoint-label "$step" --sft-reference "$sft_reference"
    echo "完成：$reward checkpoint-$step；指标：$folder/merged.json.metrics.json"
done
pids=()
printf '%s\n' "$failed" > "$result_dir/exit_code"
exit "$failed"
