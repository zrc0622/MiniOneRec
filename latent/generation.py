"""Fixed-position latent grammar and separate original/hierarchical samplers."""
import torch
from transformers import GenerationConfig

from interest.generation import pack_trajectories, action_logps as constrained_logps
from .core import atomic_id


class Grammar:
    def __init__(self, tokenizer, index, protocol):
        self.k, self.eos, self.tokenizer = 3, tokenizer.eos_token_id, tokenizer
        self.slots = [[atomic_id(tokenizer, t) for t in group] for group in protocol['tokens']]
        self.tries, self.sid_by_ids = {}, {}
        suffix = tuple(tokenizer.encode('\n', add_special_tokens=False)) + (self.eos,)
        # Auxiliary identification uses 3 levels; history-title still full SID.
        for auxiliary in (False, 'identification', 'title'):
            trie, sid_map = {}, {}
            for tokens in index.values():
                tokens = tokens[:3] if auxiliary == 'identification' else tokens
                sid_ids = tuple(atomic_id(tokenizer, t) for t in tokens)
                sequence = sid_ids + suffix
                sid_map[sequence] = ''.join(tokens)
                for i, token in enumerate(sequence):
                    trie.setdefault(sequence[:i], set()).add(token)
            self.tries[auxiliary] = {p: sorted(v) for p, v in trie.items()}
            self.sid_by_ids[auxiliary] = sid_map
        self.item_length = max(len(s) for s in self.sid_by_ids[False])

    def allowed(self, prefix, phase='joint', auxiliary=False):
        prefix = list(prefix)
        if phase == 'interest' or (phase == 'joint' and not auxiliary):
            if len(prefix) < self.k:
                # Position-specific codebooks; repeated numerical code is legal.
                return self.slots[len(prefix)]
            prefix = prefix[self.k:]
        if self.eos in prefix:
            return [self.eos]
        allowed = self.tries[auxiliary].get(tuple(prefix))
        if allowed is None:
            raise ValueError(f'Illegal SID prefix: {prefix}')
        return allowed

    def item(self, completion, auxiliary=False):
        ids = list(completion)[0 if auxiliary else self.k:]
        if self.eos not in ids:
            raise ValueError('Incomplete item response')
        sequence = tuple(ids[:ids.index(self.eos) + 1])
        if sequence not in self.sid_by_ids[auxiliary]:
            raise ValueError('Unknown SID response')
        return self.sid_by_ids[auxiliary][sequence]


@torch.no_grad()
def generate(model, prompt, grammar, count, phase='joint', auxiliary=False,
             sampling='independent', temperature=1.0):
    if sampling not in ('independent', 'original', 'beam'):
        raise ValueError(sampling)
    prompts = [prompt] if isinstance(prompt[0], int) else prompt
    if len({len(p) for p in prompts}) != 1:
        raise ValueError('Unequal prompt lengths')
    width = len(prompts[0])
    ids = torch.tensor(prompts, device=next(model.parameters()).device)
    sample = sampling != 'beam'
    beams = 1 if sampling == 'independent' else count
    length = grammar.k if phase == 'interest' else grammar.item_length + (grammar.k if phase == 'joint' and not auxiliary else 0)
    cfg = GenerationConfig(max_new_tokens=length, do_sample=sample, num_beams=beams,
        num_return_sequences=count, length_penalty=1.0 if beams == 1 else 0.0, temperature=temperature if sample else 1.0,
        top_k=0 if sample else 50, top_p=1.0, repetition_penalty=1.0, pad_token_id=grammar.eos, eos_token_id=grammar.eos,
        forced_eos_token_id=None, renormalize_logits=sampling == 'independent', use_cache=True)
    sequences = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=cfg,
        do_sample=sample, num_beams=beams, num_return_sequences=count,
        prefix_allowed_tokens_fn=lambda batch_id, tokens: grammar.allowed(tokens[width:].tolist(), phase, auxiliary),
        synced_gpus=False)[:, width:].tolist()
    result = []
    for seq in sequences:
        if grammar.eos in seq:
            seq = seq[:seq.index(grammar.eos) + 1]
        legal = all(t in grammar.allowed(seq[:i], phase, auxiliary) for i, t in enumerate(seq))
        legal = legal and (len(seq) == grammar.k if phase == 'interest' else seq[-1] == grammar.eos)
        if not legal:
            if sampling != 'beam':
                raise ValueError('Sampler returned an invalid or unfinished path')
            continue  # HF can return -inf placeholders for tiny candidate catalogs.
        result.append(seq)
    if not result or (sampling != 'beam' and len(result) != count * len(prompts)):
        raise ValueError('Insufficient valid trajectories')
    return result


def rollout(model, prompt, grammar, mode, auxiliary=False, temperature=1.0):
    if mode == 'original':
        return generate(model, prompt, grammar, 16, auxiliary=auxiliary, sampling='original', temperature=temperature)
    if mode != '4x4':
        raise ValueError(mode)
    if auxiliary:
        return generate(model, prompt, grammar, 16, auxiliary=auxiliary, temperature=temperature)
    prefixes = generate(model, prompt, grammar, 4, phase='interest', temperature=temperature)
    items = generate(model, [prompt + prefix for prefix in prefixes], grammar, 4, phase='item', temperature=temperature)
    return [prefixes[i // 4] + item for i, item in enumerate(items)]


def full_logps(model, packed, temperature=1.0):
    # Original ReReTrainer: full vocabulary softmax, no temperature division.
    length = packed['completion_ids'].shape[1]
    logits = model(input_ids=packed['input_ids'], attention_mask=packed['attention_mask'],
                   logits_to_keep=length + 1, use_cache=False).logits[:, -length - 1:-1].float()
    return logits.gather(-1, packed['completion_ids'][..., None]).squeeze(-1) - torch.logsumexp(logits, -1)
