"""Bidirectional fixed queries predict parallel VQ labels; SID decoding stays causal."""
import copy
import itertools
import math
from functools import wraps
from types import MethodType

import torch
import torch.nn.functional as F

from .core import atomic_id
from .eval_search import rank_items
from .trainer import LatentSFTTrainer

QUERY_TOKENS = ['<interest_query_1>', '<interest_query_2>', '<interest_query_3>']


def initialize_queries(model, tokenizer):
    protocol = model.config.latent_protocol
    if protocol['k'] != 3:
        raise ValueError('This controlled parallel experiment requires three slots')
    if any(t in tokenizer.get_vocab() for t in QUERY_TOKENS):
        raise ValueError('Query tokens already present in a fresh initialization')
    tokenizer.add_tokens(QUERY_TOKENS)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    protocol.update(prediction='parallel_bidirectional_queries', query_tokens=QUERY_TOKENS,
                    prior='product_of_three_full_vocabulary_query_distributions',
                    response='three_queries_then_three_actual_codes_then_causal_sid')
    install_attention(model, tokenizer)
    return protocol


def query_ids(model, tokenizer=None):
    if model.config.latent_protocol.get('prediction') != 'parallel_bidirectional_queries':
        raise ValueError('Not a parallel-query checkpoint')
    if tokenizer is not None:
        ids = [atomic_id(tokenizer, t) for t in model.config.latent_protocol['query_tokens']]
        model.config.latent_protocol['query_ids'] = ids
    return model.config.latent_protocol['query_ids']


def query_attention_mask(ids, padding, queries, dtype):
    """Causal everywhere, with a symmetric 3x3 query block; no future-label edges."""
    if padding.shape != ids.shape:
        raise ValueError('Query prefill/training requires aligned 2D padding mask')
    length = ids.shape[1]
    positions = torch.arange(length, device=ids.device)
    causal = positions[None, :] <= positions[:, None]
    is_query = (ids[..., None] == torch.tensor(queries, device=ids.device)).any(-1)
    allowed = causal[None, :, :] | (is_query[:, :, None] & is_query[:, None, :])
    allowed = allowed & padding[:, None, :].bool()
    result = torch.zeros((len(ids), length, length), dtype=dtype, device=ids.device)
    return result.masked_fill(~allowed, torch.finfo(dtype).min)[:, None]


def install_attention(model, tokenizer):
    queries = query_ids(model, tokenizer)
    if getattr(model, '_parallel_attention_installed', False):
        return model
    original = model.forward

    @wraps(original.__func__)
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # SID decode tokens and ordinary auxiliary tasks use Qwen's usual path.
        if input_ids is not None and input_ids.shape[1] >= 3:
            present = (input_ids[..., None] == torch.tensor(queries, device=input_ids.device)).any()
            if present:
                cache = kwargs.get('past_key_values')
                if cache is not None and cache.get_seq_length():
                    raise ValueError('All three queries must be prefetched together, before cached SID decoding')
                padding = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
                kwargs.setdefault('position_ids', (padding.cumsum(-1) - 1).clamp_min(0))
                attention_mask = query_attention_mask(input_ids, padding, queries, next(self.parameters()).dtype)
        return original(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    model.forward = MethodType(forward, model)
    model._parallel_attention_installed = True
    return model


def parallel_example(example, queries):
    """Keep exactly the old history and SID targets, adding only three fixed queries."""
    positions = [i for i, flag in enumerate(example['interest_mask']) if flag]
    if len(positions) != 3 or positions != list(range(positions[0], positions[0] + 3)):
        raise ValueError('Expected three contiguous original latent targets')
    start = positions[0]
    codes = example['input_ids'][start:start + 3]
    inputs = example['input_ids'][:start] + list(queries) + example['input_ids'][start:]
    labels = [-100] * start + codes + [-100] * 3 + example['labels'][start + 3:]
    mask = [0] * start + [1] * 3 + [0] * (len(inputs) - start - 3)
    return dict(input_ids=inputs, attention_mask=[1] * len(inputs), labels=labels, interest_mask=mask)


def component_loss(logits, labels, interest_mask, latent_weight=1.):
    latent = interest_mask.bool() & (labels != -100)
    main = latent.any(1)
    shifted = labels[:, 1:].clone()
    shifted[interest_mask[:, 1:].bool()] = -100
    valid = shifted != -100
    item, auxiliary = valid & main[:, None], valid & ~main[:, None]
    ce = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), shifted,
                         ignore_index=-100, reduction='none')
    latent_values = F.cross_entropy(logits[latent].float(), labels[latent], reduction='none')
    latent_ce = torch.zeros_like(labels, dtype=ce.dtype).masked_scatter(latent, latent_values)
    main_loss = ((ce * item).sum(1) / item.sum(1).clamp_min(1)
                 + latent_weight * latent_ce.sum(1) / latent.sum(1).clamp_min(1)).sum()
    aux_loss = (ce * auxiliary).sum() / auxiliary.sum().clamp_min(1)
    loss = (main_loss + (~main).sum() * aux_loss) / len(labels)
    stats = {'latent_ce': (latent_values.detach().sum(), latent.sum()),
             'item_ce': ((ce.detach() * item).sum(), item.sum()),
             'auxiliary_ce': ((ce.detach() * auxiliary).sum(), auxiliary.sum())}
    return loss, stats


class ParallelSFTTrainer(LatentSFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        out = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], use_cache=False)
        loss, stats = component_loss(out.logits, inputs['labels'], inputs['interest_mask'], self.latent_weight)
        for name, (total, count) in stats.items():
            pair = self.accelerator.reduce(torch.stack([total, count.float()]), reduction='sum')
            if pair[1] > 0:
                self._metrics[name].append((pair[0] / pair[1]).item())
        return (loss, out) if return_outputs else loss


def joint_proposals(logps, slots, count=4):
    """Exact top-count tuples of the declared factorized prior, not greedy slot mixing."""
    values, tokens = [], []
    for slot, ids in enumerate(slots):
        ids = torch.tensor(ids, device=logps.device)
        v, j = logps[:, slot, ids].topk(min(count, len(ids)), dim=-1)
        values.append(v)
        tokens.append(ids[j])
    combinations = list(itertools.product(*(range(v.shape[1]) for v in values)))
    scores = torch.stack([sum(values[k][:, c[k]] for k in range(3)) for c in combinations], 1)
    scores, chosen = scores.topk(min(count, len(combinations)), dim=1)
    prefixes = [[tokens[k][u, combinations[j][k]].item() for k in range(3)]
                for u in range(len(logps)) for j in chosen[u].tolist()]
    return prefixes, scores


class ConditionalScorer:
    def __init__(self, model, cache, mask, k=3):
        self.model, self.cache, self.mask, self.k = model, cache, mask, k

    @torch.no_grad()
    def score(self, sequences, users, batch=32):
        result = [None] * len(sequences)
        groups = {}
        for i, path in enumerate(sequences):
            groups.setdefault(len(path), []).append(i)
        for indices in groups.values():
            for start in range(0, len(indices), batch):
                chosen = indices[start:start + batch]
                uid = torch.tensor([users[i] for i in chosen], device=self.mask.device)
                tokens = torch.tensor([sequences[i] for i in chosen], device=self.mask.device)
                cache = copy.deepcopy(self.cache)
                cache.batch_select_indices(uid)
                mask = torch.cat([self.mask[uid], torch.ones_like(tokens[:, :-1])], 1)
                positions = (mask.cumsum(-1) - 1).clamp_min(0)[:, self.mask.shape[1]:]
                output = self.model(input_ids=tokens[:, :-1], past_key_values=cache,
                                    attention_mask=mask, position_ids=positions, use_cache=True)
                logits = output.logits[:, self.k - 1:].float()
                target = tokens[:, self.k:]
                scores = (logits.gather(-1, target[..., None]).squeeze(-1)
                          - torch.logsumexp(logits, -1)).sum(1)
                for i, value in zip(chosen, scores.tolist()):
                    result[i] = value
        return result


@torch.no_grad()
def batch_parallel(model, grammar, prompts, method):
    from .eval_unified_small import padded
    from .eval_batched_high_budget import cfg
    if method not in ('single20', 'four20_marginal'):
        raise ValueError(method)
    n = 1 if method == 'single20' else 4
    queries = query_ids(model)
    ids, mask = padded([p + queries for p in prompts], grammar.eos, next(model.parameters()).device)
    size, plen = ids.shape
    out = model(input_ids=ids, attention_mask=mask, position_ids=(mask.cumsum(-1) - 1).clamp_min(0),
                use_cache=True, logits_to_keep=3)
    logits = out.logits.float()
    logps = logits - torch.logsumexp(logits, -1, keepdim=True)
    codes, priors = joint_proposals(logps, grammar.slots, 4)
    base = out.past_key_values
    del out, logits, logps
    prefixes = [codes[u * 4 + b] for u in range(size) for b in range(n)]
    prior = [priors[u, b].item() for u in range(size) for b in range(n)]
    pref = torch.tensor(prefixes, device=ids.device)
    cache = copy.deepcopy(base)
    cache.batch_repeat_interleave(n * 20)
    inputs = torch.cat([ids.repeat_interleave(n, 0), pref], 1)
    masks = torch.cat([mask.repeat_interleave(n, 0), torch.ones_like(pref)], 1)
    start = plen + 3
    generated = model.generate(input_ids=inputs, attention_mask=masks, past_key_values=cache,
        generation_config=cfg(grammar, 20, grammar.item_length), use_model_defaults=False,
        do_sample=False, prefix_allowed_tokens_fn=lambda _, t: grammar.allowed(t[start:].tolist(), 'item'),
        synced_gpus=False)
    sequences, scores = generated.sequences[:, start:].tolist(), generated.sequences_scores.tolist()
    del generated, cache, inputs, masks
    candidates, pairs, counts = [], [], []
    for u in range(size):
        proposed, ranked, valid = {}, [], 0
        for b in range(n):
            branch = u * n + b
            seen = set()
            for sequence, score in zip(sequences[branch * 20:(branch + 1) * 20], scores[branch * 20:(branch + 1) * 20]):
                if grammar.eos in sequence:
                    sequence = sequence[:sequence.index(grammar.eos) + 1]
                key = tuple(sequence)
                if key in seen or not math.isfinite(score) or not sequence or sequence[-1] != grammar.eos:
                    continue
                if not all(t in grammar.allowed(sequence[:i], 'item') for i, t in enumerate(sequence)):
                    continue
                seen.add(key)
                item = grammar.item(prefixes[branch] + sequence)
                proposed.setdefault(item, sequence)
                ranked.append((item, prior[branch] + score))
                valid += 1
        if not proposed:
            raise ValueError('No valid SID proposals')
        candidates.append(proposed)
        pairs.append(ranked)
        counts.append(valid)
    if n == 4:
        paths, users, targets, pri = [], [], [], []
        for u, proposed in enumerate(candidates):
            for item, sid in proposed.items():
                for b in range(4):
                    paths.append(prefixes[u * 4 + b] + sid)
                    users.append(u)
                    targets.append(item)
                    pri.append(prior[u * 4 + b])
        conditional = ConditionalScorer(model, base, mask).score(paths, users, 32)
        pairs = [[] for _ in range(size)]
        for u, item, score, p in zip(users, targets, conditional, pri):
            pairs[u].append((item, p + score))
    return [dict(predict=rank_items(pairs[u], 'logsumexp' if n == 4 else 'max', 20),
                 unique_before_top20=len(candidates[u]), generated_paths=counts[u], scored_paths=len(pairs[u]),
                 selected_latent_mass=sum(math.exp(p) for p in prior[u * n:(u + 1) * n])) for u in range(size)]
