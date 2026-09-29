"""Original ranking reward and the independent 4×4 hierarchical objective."""
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
