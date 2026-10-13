#!/usr/bin/env bash
# Evaluate completed final checkpoints with one fixed retrieval protocol.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ $# -eq 1 ]] || { echo '用法：bash run_latent_rl_eval.sh <rl_two运行目录>' >&2; exit 2; }
run_dir="$(cd "$1" && pwd)"
python_bin="${PYTHON_BIN:-/data/zuorongchang/miniconda3/envs/minionerec/bin/python}"
data_root="${BALANCED_DATA_ROOT:-./data/Amazon23/variants/rqkmeans_balanced}"
stem=Industrial_and_Scientific_5_2016-10-2018-11
index="$data_root/Industrial_and_Scientific/Industrial_and_Scientific.index.json"
IFS=',' read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
[[ ${#gpus[@]} -eq 4 ]] || { echo '评估需要4张不同GPU。' >&2; exit 2; }
for i in 0 1 2 3; do
    [[ ${gpus[$i]} =~ ^[0-9]+$ ]] || exit 2
    for ((j=0; j<i; j++)); do [[ ${gpus[$i]} != "${gpus[$j]}" ]] || exit 2; done
done
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
completed=0
for reward in binary sid_dense; do
    model="$run_dir/$reward/final_checkpoint"
    [[ -f "$model/config.json" ]] || continue
    if [[ ! -f "$run_dir/$reward.exit_code" || $(cat "$run_dir/$reward.exit_code") != 0 ]]; then
        # A detached watcher can verify saved artifacts when the original
        # coordinator exited and therefore never collected the worker exit code.
        [[ -f "$run_dir/$reward.completion_verified.json" ]] || continue
    fi
    for split in valid test; do
        mkdir -p "$run_dir/eval/$reward"
        result_dir="$(mktemp -d "$run_dir/eval/$reward/${split}.XXXXXX")"
        echo "开始评估：$reward $split hierarchical_max 4x50，输出：$result_dir"
        started=$SECONDS
        pids=()
        limit=()
        # Optional bounded evaluation for verifying the pipeline on smoke checkpoints.
        if [[ -n ${LATENT_EVAL_MAX_EXAMPLES:-} ]]; then limit=(--max-examples "$LATENT_EVAL_MAX_EXAMPLES"); fi
        for i in 0 1 2 3; do
            CUDA_VISIBLE_DEVICES="${gpus[$i]}" "$python_bin" -u -m latent.evaluate \
                --model "$model" --data "$data_root/$split/$stem.csv" --index "$index" \
                --scheme hierarchical_max --latent-count 4 --latent-beams 4 --item-beams 50 \
                --score-batch 8 --shards 4 --shard "$i" --output "$result_dir/$i.json" "${limit[@]}" \
                > "$result_dir/$i.log" 2>&1 &
            pids+=("$!")
        done
        failed=0
        for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
        pids=()
        [[ "$failed" -eq 0 ]] || { echo "评估失败，请检查 $result_dir/*.log" >&2; exit 1; }
        "$python_bin" -m latent.evaluate --merge "$result_dir/0.json" "$result_dir/1.json" \
            "$result_dir/2.json" "$result_dir/3.json" --output "$result_dir/merged.json" \
            > "$result_dir/merge.log" 2>&1
        printf '%s\n' "$((SECONDS - started))" > "$result_dir/wall_seconds.txt"
        printf '%s\n' "$result_dir" > "$run_dir/eval/$reward/latest_${split}.txt"
        completed=$((completed + 1))
        "$python_bin" -m latent.rl_compare "$run_dir"
        echo "评估完成：$reward $split；指标：$result_dir/merged.json.metrics.json"
    done
done
[[ "$completed" -gt 0 ]] || { echo '没有成功完成的训练可供评估。' >&2; exit 1; }
echo "评估结束：$run_dir/eval/comparison.md"
