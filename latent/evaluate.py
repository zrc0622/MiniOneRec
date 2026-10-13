"""Stage-independent deterministic evaluation with explicit path budgets."""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from interest.evaluate import summarize
from .core import history_records, load_index, rank_unique_items, sha, validate_protocol, write_json
from .generation import Grammar, full_logps, generate, pack_trajectories
from .eval_search import rank_items, search_paths


def merge(paths, output):
    parts = [json.loads(Path(p).read_text()) for p in paths]
    first = parts[0]['metadata']
    if len(parts) != first['shards'] or {p['shard'] for p in parts} != set(range(first['shards'])):
        raise ValueError('Missing or repeated shard files')
    if any(p['metadata'] != first for p in parts):
        raise ValueError('Different checkpoint/data/search settings across shards')
    rows = [r for p in parts for r in p['rows']]
    ids = [r['row_id'] for r in rows]
    if len(ids) != first['examples'] or set(ids) != set(range(first['examples'])):
        raise ValueError('Missing or overlapping evaluation rows')
    rows.sort(key=lambda r: r['row_id'])
    metrics = summarize(rows)
    metrics['mean_unique_before_top50'] = sum(r['unique_before_top50'] for r in rows) / len(rows)
    metrics['fraction_with_50_candidates'] = sum(len(r['predict']) == 50 for r in rows) / len(rows)
    metrics['mean_valid_paths'] = sum(r['valid_paths'] for r in rows) / len(rows)
    if all('elapsed_seconds' in r for r in rows):
        metrics['mean_generation_and_scoring_seconds'] = sum(r['elapsed_seconds'] for r in rows) / len(rows)
        metrics['mean_unique_latents'] = sum(len(set(r['latents'])) for r in rows) / len(rows)
        metrics['fraction_below_50_candidates'] = 1 - metrics['fraction_with_50_candidates']
    write_json(output, dict(metadata=first, metrics=metrics, rows=rows))
    write_json(output + '.metrics.json', metrics)
    print(json.dumps(metrics, indent=2))


def main():
    p = argparse.ArgumentParser()
    for key in ('model', 'data', 'index'):
        p.add_argument('--' + key)
    p.add_argument('--output', required=True)
    p.add_argument('--merge', nargs='+')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=4)
    p.add_argument('--beams', type=int, default=50)
    p.add_argument('--scheme', choices=('legacy', 'joint_max', 'hierarchical_max',
                                       'hierarchical_sum', 'fixed_latent'), default='legacy')
    p.add_argument('--latent-count', type=int, default=4)
    p.add_argument('--latent-beams', type=int, default=4)
    p.add_argument('--item-beams', type=int, default=50)
    p.add_argument('--score-batch', type=int, default=8)
    p.add_argument('--max-examples', type=int, help='Evaluate first N rows, then shard those rows')
    o = p.parse_args()
    if o.merge:
        merge(o.merge, o.output)
        return
    if (not 0 <= o.shard < o.shards or o.beams < 50 or o.item_beams < 50
            or min(o.latent_count, o.latent_beams, o.score_batch) < 1
            or (o.max_examples is not None and o.max_examples < 1)):
        raise ValueError('Invalid shard or search budget')
    index = load_index(o.index)
    tokenizer = AutoTokenizer.from_pretrained(o.model)
    tokenizer.pad_token = tokenizer.eos_token
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(o.model, torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32).to(device).eval()
    protocol = validate_protocol(model.config, tokenizer, index)
    grammar = Grammar(tokenizer, index, protocol)
    records = history_records(o.data, index, grammar.k)
    if o.max_examples is not None:
        records = records[:o.max_examples]
    rows = []
    for row_id in tqdm(range(o.shard, len(records), o.shards)):
        record = records[row_id]
        prompt = tokenizer.encode(record['prompt'], add_special_tokens=False)[-512:]
        started = time.perf_counter()
        if o.scheme == 'legacy':
            samples = generate(model, prompt, grammar, o.beams, sampling='beam')
        else:
            samples = search_paths(model, prompt, grammar, o.scheme, o.beams,
                                   o.latent_count, o.item_beams, o.latent_beams)
        scores = []
        for start in range(0, len(samples), o.score_batch):
            chunk = samples[start:start + o.score_batch]
            packed = pack_trajectories([prompt] * len(chunk), chunk, [grammar.k] * len(chunk), [False] * len(chunk), grammar, device)
            with torch.no_grad():
                logps = full_logps(model, packed)
            scores.extend((logps * (packed['interest_mask'] + packed['item_mask'])).sum(1).tolist())
        paths = [(grammar.item(s), score) for s, score in zip(samples, scores)]
        aggregation = 'logsumexp' if o.scheme == 'hierarchical_sum' else 'max'
        ranked = rank_unique_items(paths, 50) if o.scheme == 'legacy' else rank_items(paths, aggregation)
        rows.append(dict(row_id=row_id, output=record['target'], predict=ranked,
            valid_paths=len(paths), unique_before_top50=len({item for item, _ in paths}),
            elapsed_seconds=time.perf_counter() - started,
            latents=[tokenizer.decode(s[:grammar.k], skip_special_tokens=False) for s in samples]))
    model_files = list(Path(o.model).glob('*.safetensors')) + list(Path(o.model).glob('pytorch_model*.bin'))
    metadata = dict(model=str(Path(o.model).resolve()), model_hashes={p.name: sha(p) for p in model_files},
        config_sha256=sha(Path(o.model) / 'config.json'), data_sha256=sha(o.data), index_sha256=sha(o.index),
        protocol=protocol, shards=o.shards, examples=len(records), beams=o.beams,
        search='joint_deterministic_beam_full_vocab_logprob_length_penalty0', dedup='max_path_score')
    if o.scheme != 'legacy':
        metadata.update(scheme=o.scheme, search='deterministic_beam_full_vocab_logprob_length_penalty0',
            dedup='logsumexp_observed_unique_paths' if o.scheme == 'hierarchical_sum' else 'max_path_score',
            latent_count=1 if o.scheme == 'fixed_latent' else o.latent_count,
            latent_beams=max(o.latent_beams, 1 if o.scheme == 'fixed_latent' else o.latent_count),
            item_beams=o.item_beams, score_batch=o.score_batch, max_examples=o.max_examples,
            code_hashes={name: sha(Path(__file__).parent / name)
                         for name in ('evaluate.py', 'eval_search.py', 'generation.py')})
    write_json(o.output, dict(shard=o.shard, metadata=metadata, rows=rows))


if __name__ == '__main__':
    main()
