"""Opt-in latent-only 4x4 rollout and constrained scoring.

No native or shared trainer is patched. Sampling keeps the original constrained
categorical distribution, temperature, 4x4 grouping and auxiliary SID grammar.
Different batching consumes RNG differently; BF16 is not bitwise equivalent.
"""
import math

import torch
from trl.models import unwrap_model_for_generation

from .trainer import LatentGRPOTrainer


def restricted_logps(logits, packed, temperature=1.0):
    """Vectorize only the allowed-token normalization, retaining autograd."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive finite temperature required')
    if '_fast_allowed' not in packed:
        sets = packed['allowed_sets']
        width = max(len(s) for row in sets for s in row)
        # Build metadata once per rollout, reuse for reference and policy.
        rows = [[s + [0] * (width - len(s)) for s in row] for row in sets]
        lengths = torch.tensor([[len(s) for s in row] for row in sets], device=logits.device)
        ids = torch.tensor(rows, device=logits.device)
        mask = torch.arange(width, device=logits.device)[None, None] < lengths[..., None]
        positions = torch.tensor(packed['action_positions'], device=logits.device)
        packed['_fast_allowed'] = ids, mask, positions
    ids, mask, positions = packed['_fast_allowed']
    values = (logits.gather(-1, ids).float() / temperature).masked_fill(~mask, -torch.inf)
    return values.gather(-1, positions[..., None]).squeeze(-1) - torch.logsumexp(values, -1)


def vectorized_logps(model, packed, temperature=1.0):
    length = packed['completion_ids'].shape[1]
    logits = model(input_ids=packed['input_ids'], attention_mask=packed['attention_mask'],
                   logits_to_keep=length + 1, use_cache=False).logits[:, -length - 1:-1]
    return restricted_logps(logits, packed, temperature)


@torch.no_grad()
def cached_rollouts(model, prompts, grammar, auxiliary=False, temperature=1.0, audit=None):
    """One history prefill per user, then branch caches 4x4 (auxiliaries:16).

    `audit` is an optional verification callback receiving logits, allowed sets,
    original user indices and generated prefixes before each sampled action.
    """
    if not prompts or any(not p for p in prompts):
        raise ValueError('Nonempty prompts required')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive finite temperature required')
    device = next(model.parameters()).device
    width = max(map(len, prompts))
    ids = torch.tensor([[grammar.eos] * (width - len(p)) + p for p in prompts], device=device)
    mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=device)
    out = model(input_ids=ids, attention_mask=mask,
                position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=True, logits_to_keep=1)
    cache, logits = out.past_key_values, out.logits[:, -1]
    del out
    branches = 16 if auxiliary else 4
    cache.batch_repeat_interleave(branches)
    logits = logits.repeat_interleave(branches, 0)
    mask = mask.repeat_interleave(branches, 0)
    users = [u for u in range(len(prompts)) for _ in range(branches)]
    prefixes = [[] for _ in users]

    def draw(allowed):
        if audit is not None:
            audit(logits, allowed, users, prefixes)
        maximum = max(map(len, allowed))
        tokens = torch.tensor([a + [0] * (maximum - len(a)) for a in allowed], device=device)
        lengths = torch.tensor([len(a) for a in allowed], device=device)
        valid = torch.arange(maximum, device=device)[None] < lengths[:, None]
        scores = (logits.gather(-1, tokens).float() / temperature).masked_fill(~valid, -torch.inf)
        choices = torch.multinomial(scores.softmax(-1), 1)
        selected = tokens.gather(-1, choices)
        for prefix, token in zip(prefixes, selected[:, 0].tolist()):
            prefix.append(token)
        return selected

    def advance(selected):
        nonlocal cache, logits, mask
        mask = torch.cat([mask, torch.ones_like(selected)], -1)
        positions = (mask.sum(-1) - 1)[:, None]
        out = model(input_ids=selected, attention_mask=mask, position_ids=positions,
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache, logits = out.past_key_values, out.logits[:, -1]

    if not auxiliary:
        for slot in range(grammar.k):
            selected = draw([grammar.slots[slot]] * len(users))
            advance(selected)
        # Interest KV already contains all three sampled tokens. Reuse for the
        # four conditional item samples rather than prefilling their histories.
        cache.batch_repeat_interleave(4)
        logits = logits.repeat_interleave(4, 0)
        mask = mask.repeat_interleave(4, 0)
        users = [u for u in users for _ in range(4)]
        prefixes = [list(p) for p in prefixes for _ in range(4)]
    offset = 0 if auxiliary else grammar.k
    for step in range(grammar.item_length):
        allowed = [grammar.allowed(p[offset:], 'item', auxiliary) for p in prefixes]
        selected = draw(allowed)
        if all(grammar.eos in p[offset:] for p in prefixes):
            break
        if step + 1 < grammar.item_length:
            advance(selected)
    else:
        if any(grammar.eos not in p[offset:] for p in prefixes):
            raise ValueError('Sampler returned unfinished SID')
    result = []
    for prefix in prefixes:
        stop = prefix.index(grammar.eos, offset) + 1
        sequence = prefix[:stop]
        grammar.item(sequence, auxiliary)  # Catalog/EOS check, including d suffix.
        result.append(sequence)
    return [result[u * 16:(u + 1) * 16] for u in range(len(prompts))]


class FastLatentGRPOTrainer(LatentGRPOTrainer):
    """Reuse original metrics/rewards/packing; replace rollout and scoring only."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mode != '4x4':
            raise ValueError('Fast latent trainer only supports independent4x4')
        self.logps = vectorized_logps

    def _prepare_inputs(self, inputs):
        # The original trainer consumes one group at a time. Precompute batched
        # groups, then pass them through its exact existing reward/loss code.
        if len(inputs) % 16:
            raise ValueError('Incomplete16trajectorygroup')
        groups, buckets = [], {}
        for start in range(0, len(inputs), 16):
            record = inputs[start]
            if any(r != record for r in inputs[start:start + 16]):
                raise ValueError('Sampler split a prompt group')
            prompt = self.processing_class.encode(record['prompt'], add_special_tokens=False)[-self.args.max_prompt_length:]
            aux = ('identification' if record['task'] == 'item_identification' else 'title') if record['auxiliary'] else False
            groups.append((prompt, aux))
            buckets.setdefault(aux, []).append(len(groups) - 1)
        prepared = [None] * len(groups)
        with unwrap_model_for_generation(self.model, self.accelerator) as model:
            training = model.training
            model.eval()
            try:
                for aux, indices in buckets.items():
                    samples = cached_rollouts(model, [groups[i][0] for i in indices], self.grammar, aux, self.temperature)
                    for i, sample in zip(indices, samples):
                        prepared[i] = sample
            finally:
                model.train(training)
        # Process-local substitution is bounded to this synchronous super call;
        # only this opt-in entrypoint loads FastLatentGRPOTrainer. No files or
        # native/shared functions are modified. Always restore even on failure.
        from . import trainer as original
        saved = original.rollout
        cursor = 0

        def ready(model, prompt, grammar, mode, auxiliary=False, temperature=1.0):
            nonlocal cursor
            if mode != '4x4' or (prompt, auxiliary) != groups[cursor]:
                raise ValueError('Precomputed group ordering changed')
            value = prepared[cursor]
            cursor += 1
            return value

        original.rollout = ready
        try:
            packed = super()._prepare_inputs(inputs)
            if cursor != len(groups):
                raise ValueError('Unused rollout group')
            return packed
        finally:
            original.rollout = saved
