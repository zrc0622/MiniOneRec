"""Evaluation-only deterministic search; no changes to RL sampling."""
import math

import torch
from transformers import GenerationConfig


@torch.no_grad()
def beam_generate(model, prompt, grammar, count, phase='joint', beam_width=None):
    width = len(prompt)
    beams = count if beam_width is None else beam_width
    if not 1 <= count <= beams:
        raise ValueError('Return count must be between 1 and beam width')
    length = grammar.k if phase == 'interest' else grammar.item_length
    if phase == 'joint':
        length += grammar.k
    ids = torch.tensor([prompt], device=next(model.parameters()).device)
    config = GenerationConfig(
        max_new_tokens=length, do_sample=False, num_beams=beams,
        num_return_sequences=count, length_penalty=0.0, top_k=None, top_p=None,
        pad_token_id=grammar.eos, eos_token_id=grammar.eos,
        forced_eos_token_id=None, renormalize_logits=False, use_cache=True)
    sequences = model.generate(
        input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=config,
        use_model_defaults=False, do_sample=False,
        prefix_allowed_tokens_fn=lambda _, tokens: grammar.allowed(tokens[width:].tolist(), phase),
        synced_gpus=False)[:, width:].tolist()
    result, seen = [], set()
    for seq in sequences:
        if grammar.eos in seq:
            seq = seq[:seq.index(grammar.eos) + 1]
        try:
            legal = all(t in grammar.allowed(seq[:i], phase) for i, t in enumerate(seq))
        except ValueError:
            legal = False
        legal = legal and (len(seq) == grammar.k if phase == 'interest' else bool(seq) and seq[-1] == grammar.eos)
        # Tiny catalogs may yield -inf placeholders. Never rank those or count
        # the same complete token sequence twice when aggregating probability.
        key = tuple(seq)
        if legal and key not in seen:
            result.append(seq)
            seen.add(key)
    if not result:
        raise ValueError('No valid beam completions')
    return result


def search_paths(model, prompt, grammar, scheme, beams=200, latent_count=4,
                 item_beams=50, latent_beams=4):
    if scheme == 'joint_max':
        return beam_generate(model, prompt, grammar, beams)
    count = 1 if scheme == 'fixed_latent' else latent_count
    prefixes = beam_generate(model, prompt, grammar, count, phase='interest',
                             beam_width=max(latent_beams, count))
    samples = []
    # Sequential branches bound peak KV-cache memory to item_beams, rather
    # than multiplying it by the number of latent branches.
    for prefix in prefixes:
        items = beam_generate(model, prompt + prefix, grammar, item_beams, phase='item')
        samples.extend(prefix + item for item in items)
    return samples


def rank_items(paths, aggregation='max', limit=50):
    scores = {}
    for item, score in paths:
        score = float(score)
        if not math.isfinite(score):
            raise ValueError('Non-finite path score')
        if item not in scores:
            scores[item] = score
        elif aggregation == 'max':
            scores[item] = max(scores[item], score)
        elif aggregation == 'logsumexp':
            high, low = sorted((scores[item], score), reverse=True)
            scores[item] = high + math.log1p(math.exp(low - high))
        else:
            raise ValueError(aggregation)
    return sorted(scores, key=lambda item: (-scores[item], item))[:limit]
