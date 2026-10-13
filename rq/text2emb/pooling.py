"""Embedding pooling with padding-aware last-token extraction."""
import torch
import torch.nn.functional as F


def pool_embedding(hidden, mask, pooling="masked_mean", l2_normalize=False):
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
        raise ValueError("Hidden states and attention mask have incompatible shapes")
    if not mask.bool().any(dim=1).all():
        raise ValueError("Cannot pool an entirely padded sequence")
    if pooling == "masked_mean":
        weights = mask.unsqueeze(-1).expand_as(hidden).float()
        result = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-9)
    elif pooling == "last_token":
        positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        last = positions.masked_fill(~mask.bool(), -1).max(dim=1).values
        result = hidden[torch.arange(hidden.shape[0], device=hidden.device), last].float()
    else:
        raise ValueError(f"Unknown pooling: {pooling}")
    if l2_normalize:
        if (result.norm(dim=1) <= 1e-12).any():
            raise ValueError("Cannot normalize a zero embedding")
        result = F.normalize(result, p=2, dim=1)
    return result
