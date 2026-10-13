#!/usr/bin/env bash
# Two independent four-GPU RL jobs sharing the same four physical devices.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
phase="${1:-train}"
[[ "$phase" == smoke || "$phase" == train ]] || { echo '用法：bash run_latent_rl_two.sh smoke|train' >&2; exit 2; }
python_bin="${PYTHON_BIN:-/data/zuorongchang/miniconda3/envs/minionerec/bin/python}"
python_dir="$(dirname "$python_bin")"
# A stale login CUDA_HOME can point at a removed toolkit. Use the compiler
# already installed in this Python environment when the configured one is absent.
if [[ ! -x "${CUDA_HOME:-}/bin/nvcc" && -x "$python_dir/nvcc" ]]; then
    export CUDA_HOME="$(dirname "$python_dir")"
fi
export PATH="$python_dir:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
root="${LATENT_ROOT:-./outputs/latent_qwen3_0.6b_balanced15}"
data_root="${BALANCED_DATA_ROOT:-./data/Amazon23/variants/rqkmeans_balanced}"
category=Industrial_and_Scientific
stem="${category}_5_2016-10-2018-11"
model="$root/sft/final_checkpoint"
index="$data_root/$category/$category.index.json"
items="$data_root/$category/$category.item.json"
[[ -f "$model/config.json" && -f "$index" && -f "$items" ]] || { echo '缺少模型或数据。' >&2; exit 1; }
# Check all devices once before launching either job. Neither job claims idle GPUs
# separately: sharing all four devices is intentional.
"$python_bin" - <<'PY'
import os, subprocess
gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
if len(gpus) != 4 or len(set(gpus)) != 4 or any(not x.isdigit() for x in gpus):
    raise SystemExit('需要4张不同GPU，例如4,5,6,7。')
rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
                               '--format=csv,noheader,nounits'], text=True)
memory = {p[0]: (int(p[1]), int(p[2])) for row in rows.splitlines()
          if (p := [x.strip() for x in row.split(',')])}
for gpu in gpus:
    used, total = memory[gpu]
    if used > 1024:
        raise SystemExit(f'GPU{gpu} 已占用 {used} MiB，停止启动，避免影响其他训练。')
    print(f'GPU{gpu}: {used}/{total} MiB; 两组实验共享此卡')
PY
mkdir -p "$root/rl_two"
run_dir="$(mktemp -d "$root/rl_two/${phase}.XXXXXX")"
run_dir="$(cd "$run_dir" && pwd)"
printf '%s\n' "$run_dir" > "$root/rl_two/latest_${phase}.txt"
cp latent/train.py latent/trainer.py latent/objective.py latent/generation.py "$run_dir/"
cp run_latent_rl_two.sh "$run_dir/launcher.sh"
cp run_latent_rl_eval.sh latent/rl_compare.py "$run_dir/"
printf 'phase=%s\ngpus=%s\nworld_size_per_job=4\ngradient_accumulation=16\nCUDA_HOME=%s\nrewards=%s\n' "$phase" "$CUDA_VISIBLE_DEVICES" "${CUDA_HOME:-}" "${RL_REWARDS:-binary sid_dense}" > "$run_dir/settings.txt"
echo "结果目录：$run_dir"
common=(rl --model "$model" --mode 4x4
    --train_file "$data_root/train/$stem.csv" --eval_file "$data_root/valid/$stem.csv"
    --items "$items" --index "$index" --batch 16 --eval_batch 128
    --gradient_accumulation 16 --epochs 2 --lr 1e-5 --seed 42
    --beta 0.001 --temperature 1.0 --latent_weight 1.0 --partial_weight 0.2
    --deepspeed config/sft_zero2.json)
if [[ "$phase" == smoke ]]; then
    # Exercise full train/eval batch sizes and optimizer twice; limit only the
    # number of validation prompts, never the memory-relevant batch size.
    common+=(--max_steps 2 --eval_steps 1 --save_steps 3 --eval_limit 32)
fi
pids=()
monitor_pid=''
cleanup() {
    if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; fi
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
read -r -a rewards <<< "${RL_REWARDS:-binary sid_dense}"
[[ ${#rewards[@]} -ge 1 && ${#rewards[@]} -le 2 ]] || exit 2
[[ ${#rewards[@]} -eq 1 || ${rewards[0]} != "${rewards[1]}" ]] || exit 2
for reward in "${rewards[@]}"; do
    [[ "$reward" == binary || "$reward" == sid_dense ]] || exit 2
    # --standalone picks an independent rendezvous port for each torchrun.
    "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=4 -m latent.train \
        "${common[@]}" --reward_kind "$reward" --output_dir "$run_dir/$reward" \
        > "$run_dir/$reward.log" 2>&1 &
    pids+=("$!")
    printf '%s\t%s\n' "$reward" "$!" >> "$run_dir/pids.tsv"
done
(
    while true; do
        alive=0
        for pid in "${pids[@]}"; do if kill -0 "$pid" 2>/dev/null; then alive=1; fi; done
        [[ "$alive" -eq 1 ]] || break
        nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader >> "$run_dir/gpu_memory.csv"
        sleep 5
    done
) &
monitor_pid=$!
failed=0
for i in "${!pids[@]}"; do
    reward="${rewards[$i]}"
    if wait "${pids[$i]}"; then code=0; else code=$?; failed=1; fi
    printf '%s\n' "$code" > "$run_dir/${reward}.exit_code"
    echo "$reward exit_code=$code"
done
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
printf '%s\n' "$failed" > "$run_dir/exit_code"
if [[ "$phase" == train ]]; then
    # Wait for all training workers to release memory before ranking evaluation.
    if ! bash run_latent_rl_eval.sh "$run_dir"; then failed=1; fi
    printf '%s\n' "$failed" > "$run_dir/pipeline.exit_code"
fi
exit "$failed"
