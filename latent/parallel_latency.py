"""Paired single-user inference times on identical histories and the same GPU."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import history_records, load_index, validate_protocol, write_json
from .generation import Grammar
from .parallel import batch_parallel, install_attention
from .eval_unified_small import batch_latent
from .eval_pilot import timed


def main():
    p = argparse.ArgumentParser()
    for key in ('serial', 'parallel', 'data', 'index', 'sample', 'output'):
        p.add_argument('--' + key, required=True)
    o = p.parse_args()
    torch.set_num_threads(1)
    ids = json.loads(Path(o.sample).read_text())['row_ids'][:64]
    index = load_index(o.index)
    records = history_records(o.data, index)
    cases = {}
    for name in ('serial', 'parallel'):
        path = getattr(o, name)
        tk = AutoTokenizer.from_pretrained(path)
        tk.pad_token = tk.eos_token
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
        protocol = validate_protocol(model.config, tk, index, allow_parallel=name == 'parallel')
        if name == 'parallel':
            install_attention(model, tk)
        cases[name] = (model, Grammar(tk, index, protocol),
                       {rid: tk.encode(records[rid]['prompt'], add_special_tokens=False)[-512:] for rid in ids})
    methods = ['single20', 'four20_marginal']
    observations = {name: {m: [] for m in methods} for name in cases}
    for name, (model, grammar, prompts) in cases.items():
        fn = batch_latent if name == 'serial' else batch_parallel
        for method in methods:
            timed(lambda: fn(model, grammar, [prompts[ids[0]]], method))
    for i, rid in enumerate(ids):
        names = ['serial', 'parallel'] if i % 2 == 0 else ['parallel', 'serial']
        order = methods if i % 2 == 0 else list(reversed(methods))
        for name in names:
            model, grammar, prompts = cases[name]
            fn = batch_latent if name == 'serial' else batch_parallel
            for method in order:
                _, elapsed = timed(lambda: fn(model, grammar, [prompts[rid]], method))
                observations[name][method].append(dict(row_id=rid, seconds=elapsed))
        print(json.dumps(dict(done=i + 1, total=len(ids))), flush=True)
    summaries = {}
    for name, result in observations.items():
        summaries[name] = {}
        for method, values in result.items():
            times = [v['seconds'] for v in values]
            summaries[name][method] = dict(mean_seconds=float(np.mean(times)),
                median_seconds=float(np.median(times)), p95_seconds=float(np.percentile(times, 95)))
    write_json(o.output, dict(summary=summaries, observations=observations, sample_ids=ids,
        gpu=torch.cuda.get_device_name(), batch_size=1, model_load_and_tokenization_excluded=True,
        cuda_synchronized=True, alternating_model_and_method_order=True,
        scope='matched64 validation histories on same GPU; complete online generation and ranking, not stage sum'))


if __name__ == '__main__':main()
