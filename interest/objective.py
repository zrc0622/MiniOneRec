"""Two GRPO objectives with explicit group and token masks."""
import torch


def relative_advantage(rewards, dim=-1):
    # Population std handles singleton/constant groups without NaN.
    return (rewards - rewards.mean(dim, keepdim=True)) / (rewards.std(dim, correction=0, keepdim=True) + 1e-4)


def advantages(hits, mode, auxiliary=False):
    """hits has 16 entries for one prompt; never normalize across users/tasks."""
    if hits.shape != (16,):
        raise ValueError("Each prompt requires exactly 16 item trajectories")
    if mode == "16x1" or auxiliary:
        adv = relative_advantage(hits)
        return adv, adv, hits
    if mode != "4x4":
        raise ValueError(mode)
    matrix = hits.reshape(4, 4)
    interest_reward = matrix.max(dim=1).values
    interest_adv = relative_advantage(interest_reward).repeat_interleave(4)
    item_adv = relative_advantage(matrix, dim=1).flatten()
    return interest_adv, item_adv, interest_reward


def sequence_mean(values, mask):
    return (values * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def grpo_loss(logps, ref_logps, packed, group_specs, beta, interest_weight=1.0):
    # One on-policy update per rollout; forward ratio is 1, derivative is not 0.
    ratio = torch.exp(logps - logps.detach())
    delta = ref_logps - logps
    kl = torch.expm1(delta) - delta
    losses = []
    for start, mode, auxiliary, int_adv, item_adv in group_specs:
        sl = slice(start, start + 16)
        im, ym = packed["interest_mask"][sl], packed["item_mask"][sl]
        if mode == "16x1" or auxiliary:
            mask = im + ym
            losses.append(sequence_mean(-ratio[sl] * item_adv[:, None] + beta * kl[sl], mask).mean())
        else:
            # Shared prefix appears four times in storage. Taking one representative
            # per branch counts each interest decision once (mean over four branches).
            interest_loss = sequence_mean(-ratio[sl] * int_adv[:, None] + beta * kl[sl], im)[::4].mean()
            item_loss = sequence_mean(-ratio[sl] * item_adv[:, None] + beta * kl[sl], ym).mean()
            losses.append(interest_weight * interest_loss + item_loss)
    return torch.stack(losses).mean(), sequence_mean(kl, packed["interest_mask"] + packed["item_mask"]).mean()
