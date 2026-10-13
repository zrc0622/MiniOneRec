"""Same item budgets and marginal scoring as the AR baseline, with a parallel prior."""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import history_records, load_index, sha, validate_protocol, write_json
from .generation import Grammar
from .parallel import batch_parallel, install_attention
from .eval_batched_high_budget import capacity
from .eval_pilot import timed


def main():
    p = argparse.ArgumentParser()
    for key in ('model', 'data', 'index', 'sample', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--max-batch', type=int, default=32)
    o = p.parse_args()
    torch.set_num_threads(1)
    tk = AutoTokenizer.from_pretrained(o.model)
    tk.pad_token = tk.eos_token
    model = AutoModelForCausalLM.from_pretrained(o.model, torch_dtype=torch.bfloat16).cuda().eval()
    protocol = validate_protocol(model.config, tk, load_index(o.index), allow_parallel=True)
    install_attention(model, tk)
    index = load_index(o.index)
    grammar = Grammar(tk, index, protocol)
    ids = json.loads(Path(o.sample).read_text())['row_ids']
    records = history_records(o.data, index)
    encoded = {rid: tk.encode(records[rid]['prompt'], add_special_tokens=False)[-512:] for rid in ids}
    order = sorted(ids, key=lambda r: (len(encoded[r]), r))
    methods = ['single20', 'four20_marginal']
    metric_protocol = json.loads((Path(__file__).resolve().parents[1] / 'eval_metric_protocol.json').read_text())
    rows, batches, retries = {m: {} for m in methods}, {m: [] for m in methods}, []
    metadata = dict(model=o.model, kind='latent', prediction='parallel_bidirectional_queries',
        methods=methods, examples=len(ids), sample_sha256=sha(o.sample), data_sha256=sha(o.data),
        index_sha256=sha(o.index), source_sha256=sha(__file__),
        architecture_source_sha256=sha(Path(__file__).with_name('parallel.py')),
        serial_evaluator_sha256=sha(Path(__file__).with_name('eval_unified_small.py')),
        protocol=metric_protocol, interest_proposals='exact top4 tuples of product full-vocabulary query probabilities',
        timing='CUDA-synced batched generation/scoring; amortized per row, not single-user latency')

    def write():
        metadata['completed'] = {m: len(rows[m]) for m in methods}
        write_json(o.output, dict(rows={m: [rows[m][r] for r in ids if r in rows[m]] for m in methods},
                   batches=batches, oom_retries=retries, metadata=metadata))

    started = time.perf_counter()
    for method in methods:
        _, seconds = timed(lambda: batch_parallel(model, grammar, [encoded[order[-1]]], method))
        print(json.dumps(dict(warmup=method, seconds=seconds)), flush=True)
        cursor = 0
        while cursor < len(order):
            tentative = order[cursor:cursor + o.max_batch]
            n = 1 if method == 'single20' else 4
            count = min(len(tentative), capacity(model, grammar, max(len(encoded[r]) for r in tentative) + 3,
                                               n, 20, o.max_batch))
            while True:
                chosen = order[cursor:cursor + count]
                torch.cuda.reset_peak_memory_stats()
                try:
                    predictions, elapsed = timed(lambda: batch_parallel(model, grammar, [encoded[r] for r in chosen], method))
                    break
                except torch.cuda.OutOfMemoryError:
                    if count == 1:
                        raise
                    retries.append(dict(method=method, cursor=cursor, failed_batch=count))
                    count = max(1, count // 2)
                    gc.collect()
                    torch.cuda.empty_cache()
            batches[method].append(dict(row_ids=chosen, size=count, elapsed_seconds=elapsed,
                                       peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30))
            for rid, result in zip(chosen, predictions):
                rows[method][rid] = dict(row_id=rid, output=records[rid]['target'],
                                        elapsed_seconds=elapsed / count, **result)
            cursor += count
            write()
            print(json.dumps(dict(method=method, done=cursor, total=len(ids), wall_seconds=time.perf_counter() - started)), flush=True)


if __name__ == '__main__':main()
