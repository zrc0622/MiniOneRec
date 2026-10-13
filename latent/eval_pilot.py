"""Fixed-sample inference pilot; leaves production evaluation unchanged."""
import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from interest.evaluate import summarize
from .core import history_records, load_index, validate_protocol, sha, write_json
from .generation import Grammar, full_logps, pack_trajectories
from .eval_search import beam_generate, rank_items


class PrefixScorer:
    """Full-vocabulary path likelihood with prompt KV reused between paths."""
    @torch.no_grad()
    def __init__(self, model, prompt):
        self.model = model
        self.device = next(model.parameters()).device
        self.width = len(prompt)
        ids = torch.tensor([prompt], device=self.device)
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    use_cache=True, logits_to_keep=1)
        self.cache = out.past_key_values
        logits = out.logits[0, -1].float()
        self.first = logits - torch.logsumexp(logits, -1)

    @torch.no_grad()
    def score(self, sequences, batch=8):
        result = [None] * len(sequences)
        groups = defaultdict(list)
        for i, seq in enumerate(sequences):
            groups[len(seq)].append((i, seq))
        for length, group in groups.items():
            for start in range(0, len(group), batch):
                chunk = group[start:start + batch]
                ids = torch.tensor([seq for _, seq in chunk], device=self.device)
                cache = copy.deepcopy(self.cache)
                cache.batch_repeat_interleave(len(chunk))
                if length > 1:
                    out = self.model(input_ids=ids[:, :-1], past_key_values=cache,
                        attention_mask=torch.ones((len(chunk), self.width+length-1),
                            dtype=torch.long, device=self.device), use_cache=True)
                    logits = out.logits.float()
                    logps = logits.gather(-1, ids[:, 1:, None]).squeeze(-1) - torch.logsumexp(logits, -1)
                    scores = logps.sum(1) + self.first[ids[:, 0]]
                else:
                    scores = self.first[ids[:, 0]]
                for (i, _), score in zip(chunk, scores.tolist()):
                    result[i] = score
        return result


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync()
    start = time.perf_counter()
    value = fn()
    sync()
    return value, time.perf_counter()-start


@torch.no_grad()
def original_scores(model, prompt, samples, grammar, batch=8):
    scores = []
    for start in range(0, len(samples), batch):
        chunk = samples[start:start+batch]
        packed = pack_trajectories([prompt]*len(chunk), chunk, [grammar.k]*len(chunk),
            [False]*len(chunk), grammar, next(model.parameters()).device)
        logps = full_logps(model, packed)
        scores.extend((logps*(packed['interest_mask']+packed['item_mask'])).sum(1).tolist())
    return scores


def trial(model, tokenizer, grammar, record, row_id, batch=8):
    prompt = tokenizer.encode(record['prompt'], add_special_tokens=False)[-512:]
    prefixes, prefix_time = timed(lambda: beam_generate(model, prompt, grammar, 4, 'interest'))
    full, full_times, small, small_times = [], [], [], []
    for i, prefix in enumerate(prefixes):
        paths, elapsed = timed(lambda: beam_generate(model, prompt+prefix, grammar, 50, 'item'))
        full.append([prefix+item for item in paths]); full_times.append(elapsed)
        if i:
            paths, elapsed = timed(lambda: beam_generate(model, prompt+prefix, grammar, 10, 'item'))
            small.append([prefix+item for item in paths]); small_times.append(elapsed)
    paths200 = sum(full, [])
    old_scores, old_time = timed(lambda: original_scores(model, prompt, paths200, grammar, batch))
    scorer, prefill_time = timed(lambda: PrefixScorer(model, prompt))
    results = {}
    def save(name, paths, scores, elapsed, **extra):
        items = [grammar.item(s) for s in paths]
        results[name] = dict(row_id=row_id, output=record['target'],
            predict=rank_items(list(zip(items,scores)), extra.pop('aggregation','max')),
            elapsed_seconds=elapsed, valid_paths=len(paths),
            unique_before_top50=len(set(items)), **extra)
    save('reference_4x50', paths200, old_scores, prefix_time+sum(full_times)+old_time)
    for name, n in [('cached_1x50',1), ('cached_2x50',2), ('cached_4x50',4)]:
        paths = sum(full[:n], [])
        scores, elapsed = timed(lambda: scorer.score(paths,batch))
        save(name, paths, scores, prefix_time+sum(full_times[:n])+prefill_time+elapsed,
             max_original_score_error=max(abs(x-y) for x,y in zip(scores,old_scores[:len(paths)])))
    paths80 = full[0]+sum(small, [])
    scores80, score_time = timed(lambda: scorer.score(paths80,batch))
    generation_time = prefix_time+full_times[0]+sum(small_times)
    save('budget80_max', paths80, scores80,generation_time+prefill_time+score_time)
    save('budget80_observed_sum', paths80, scores80,generation_time+prefill_time+score_time,
         aggregation='logsumexp')
    candidates = {}
    for seq in paths80:
        candidates.setdefault(grammar.item(seq),seq[grammar.k:])
    cross = [prefix+item for item in candidates.values() for prefix in prefixes]
    cross_scores, cross_time = timed(lambda: scorer.score(cross,batch))
    save('budget80_marginal', cross,cross_scores,generation_time+prefill_time+cross_time,
         aggregation='logsumexp',proposal_paths=len(paths80))
    # Compute probability mass of selected complete latent codes under original full softmax.
    latent_scores, _ = timed(lambda: scorer.score(prefixes,batch))
    mass = float(np.exp(latent_scores).sum())
    for result in results.values():
        result['selected_latent_mass_full_vocab'] = mass
        result['prompt_tokens'] = len(prompt)
    return results


def main():
    p=argparse.ArgumentParser()
    for key in ['model','data','index','sample','output']: p.add_argument('--'+key,required=True)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=2)
    p.add_argument('--score-batch',type=int,default=8)
    o=p.parse_args()
    torch.set_num_threads(1)
    index=load_index(o.index)
    tk=AutoTokenizer.from_pretrained(o.model); tk.pad_token=tk.eos_token
    model=AutoModelForCausalLM.from_pretrained(o.model,torch_dtype=torch.bfloat16).cuda().eval()
    grammar=Grammar(tk,index,validate_protocol(model.config,tk,index))
    records=history_records(o.data,index,grammar.k)
    selected=json.loads(Path(o.sample).read_text())['row_ids']
    # Excluded warmup; real model, complete generation and scoring pipeline.
    print('warmup',flush=True)
    trial(model,tk,grammar,records[selected[0]],selected[0],o.score_batch)
    torch.cuda.reset_peak_memory_stats()
    rows=defaultdict(list)
    started=time.perf_counter()
    for n,row_id in enumerate(selected[o.shard::o.shards]):
        result=trial(model,tk,grammar,records[row_id],row_id,o.score_batch)
        for name,row in result.items(): rows[name].append(row)
        elapsed=time.perf_counter()-started
        write_json(o.output,dict(shard=o.shard, rows=dict(rows),elapsed_seconds=elapsed,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            metadata=dict(model=o.model,data_sha256=sha(o.data),index_sha256=sha(o.index),
                sample_sha256=sha(o.sample),score_batch=o.score_batch,
                attention=model.config._attn_implementation,
                timing='CUDA-synchronized, warmup excluded, shared generation stages composed into per-method standalone latency; sequential batch1 requests',
                caveat='Full-vocabulary likelihood assigns mass outside legal grammar; selected latent mass is not normalized over legal codes.')))
        print(json.dumps(dict(done=n+1,total=len(selected[o.shard::o.shards]),row_id=row_id,
            wall_seconds=elapsed,latency={name:round(r['elapsed_seconds'],3) for name,r in result.items()})),flush=True)

if __name__=='__main__': main()
