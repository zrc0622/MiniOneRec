"""Original ranking reward and the independent 4×4 hierarchical objective."""
import math
import re

import torch
from interest.objective import advantages as hierarchical_advantages, grpo_loss


def ranking_rewards(hits):
    if hits.shape != (16,):
        raise ValueError('Original ranking expects 16 ordered beam samples')
    discount = 1 / torch.log2(torch.arange(2, 18, device=hits.device, dtype=hits.dtype))
    negatives = -discount / discount.sum()
    return torch.where(hits.bool(), hits, negatives) if hits.any() else torch.zeros_like(hits)


def advantages(hits, mode, auxiliary=False):
    if mode == 'original':
        reward = ranking_rewards(hits)
        # Preserve original correction=1, unlike hierarchical population std.
        advantage = (reward - reward.mean()) / (reward.std(correction=1) + 1e-4)
        return advantage, advantage, reward
    return hierarchical_advantages(hits, '4x4', bool(auxiliary))


def sid_rewards(items, target, *, kind='binary', partial_weight=0.2, device=None):
    """Exact hit plus optional positional matching of a/b/c, never d or EOS."""
    if kind not in ('binary', 'sid_dense') or not math.isfinite(partial_weight) or partial_weight < 0:
        raise ValueError('Invalid SID reward settings')
    hits = torch.tensor([float(item == target) for item in items], device=device)
    if kind == 'binary':
        return hits
    def semantic_tokens(sid):
        match = re.fullmatch(r'(<a_\d+>)(<b_\d+>)(<c_\d+>)(?:<d_\d+>)?', sid)
        if match is None:
            raise ValueError(f'Invalid reward SID: {sid}')
        return match.groups()
    gold = semantic_tokens(target)
    fractions = [sum(a == b for a, b in zip(semantic_tokens(item), gold)) / 3 for item in items]
    return hits + partial_weight * torch.tensor(fractions, device=device)
