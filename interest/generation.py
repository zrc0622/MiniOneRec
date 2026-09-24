"""Staged constrained sampling and the matching action log probabilities."""
import torch
from transformers import GenerationConfig

from .core import ITEM, NONE, atomic_id, interest_token


class Grammar:
    def __init__(self, tokenizer, index, k):
        self.tokenizer, self.k = tokenizer, k
        self.eos = tokenizer.eos_token_id
        self.none = atomic_id(tokenizer, NONE)
        self.marker = atomic_id(tokenizer, ITEM)
        self.interests = sorted(atomic_id(tokenizer, interest_token(x)) for x in {v[0] for v in index.values()})
        self.tries = {}
        self.sid_by_ids = {}
        for auxiliary in (False, True):
            trie, sid_map = {}, {}
            for tokens in index.values():
                tokens = tokens[:3] if auxiliary else tokens
                ids = tuple(atomic_id(tokenizer, t) for t in tokens)
                sid_map[ids] = "".join(tokens)
                sequence = ids + (self.eos,)
                for i, token in enumerate(sequence):
                    trie.setdefault(sequence[:i], set()).add(token)
            self.tries[auxiliary] = {p: sorted(v) for p, v in trie.items()}
            self.sid_by_ids[auxiliary] = sid_map

    def allowed(self, prefix, phase="joint", auxiliary=False):
        prefix = list(prefix)
        if phase == "interest" or (phase == "joint" and not auxiliary):
            if len(prefix) < self.k:
                if self.none in prefix:
                    return [self.none]
                candidates = [x for x in self.interests if x not in prefix]
                return candidates + ([self.none] if prefix else [])
            if len(prefix) == self.k:
                return [self.marker]
            if prefix[self.k] != self.marker:
                raise ValueError("Missing interest/item boundary")
            prefix = prefix[self.k + 1:]
        if self.eos in prefix:
            return [self.eos]  # finished beams may be padded with EOS
        allowed = self.tries[auxiliary].get(tuple(prefix))
        if allowed is None:
            raise ValueError(f"Invalid SID prefix: {prefix}")
        return allowed

    def item(self, completion, auxiliary=False):
        ids = list(completion)
        if not auxiliary:
            if len(ids) <= self.k or ids[self.k] != self.marker:
                raise ValueError("Incomplete interest generation")
            ids = ids[self.k + 1:]
        if self.eos not in ids:
            raise ValueError("Item generation did not finish")
        ids = tuple(ids[:ids.index(self.eos)])
        if ids not in self.sid_by_ids[auxiliary]:
            raise ValueError("Generated unknown item SID")
        return self.sid_by_ids[auxiliary][ids]


@torch.no_grad()
def generate(model, prompt, grammar, count, phase="joint", auxiliary=False, sample=True, temperature=1.0):
    """One prompt (or equal-length prompts), count samples/beams per prompt."""
    device = next(model.parameters()).device
    prompts = [prompt] if isinstance(prompt[0], int) else prompt
    if len({len(p) for p in prompts}) != 1:
        raise ValueError("Batched generation expects equal-length prompts")
    input_ids = torch.tensor(prompts, dtype=torch.long, device=device)
    width = len(prompts[0])
    max_new = grammar.k + 1 if phase == "interest" else 5 + (grammar.k + 1 if phase == "joint" and not auxiliary else 0)
    config = GenerationConfig(
        max_new_tokens=max_new, do_sample=sample,
        num_beams=1 if sample else count, num_return_sequences=count,
        temperature=temperature if sample else 1.0,
        top_k=0 if sample else 50, top_p=1.0, repetition_penalty=1.0, length_penalty=1.0 if sample else 0.0,
        pad_token_id=grammar.eos, eos_token_id=grammar.eos,
        forced_eos_token_id=None, use_cache=True,
        renormalize_logits=True,
    )
    sequences = model.generate(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids), generation_config=config,
        # Explicit overrides prevent checkpoint generation_config defaults leaking in.
        do_sample=sample, num_beams=1 if sample else count, num_return_sequences=count,
        prefix_allowed_tokens_fn=lambda batch_id, ids: grammar.allowed(ids[width:].tolist(), phase, auxiliary),
        synced_gpus=False,
    )[:, width:].tolist()
    result = []
    for seq in sequences:
        if phase != "interest" and grammar.eos in seq:
            seq = seq[:seq.index(grammar.eos) + 1]
        if phase == "interest" and (len(seq) != grammar.k + 1 or seq[-1] != grammar.marker):
            raise ValueError("Incomplete interest prefix")
        legal = all(token in grammar.allowed(seq[:i], phase, auxiliary) for i, token in enumerate(seq))
        legal = legal and (phase == "interest" or seq[-1] == grammar.eos)
        if not legal:
            if sample:
                raise ValueError("Sampler generated a token outside the grammar")
            # When there are fewer legal paths than beams, HF can return dummy
            # paths from -inf beam slots. Never rank these as actual products.
            continue
        result.append(seq)
    if not result:
        raise ValueError("Generation returned no legal paths")
    return result


def rollout(model, prompt, grammar, mode="16x1", auxiliary=False, temperature=1.0):
    if auxiliary:
        sequences = generate(model, prompt, grammar, 16, "item", True, temperature=temperature)
        return sequences, [0] * 16, 16, 1
    n, m = {"16x1": (16, 1), "4x4": (4, 4)}[mode]
    interests = generate(model, prompt, grammar, n, "interest", temperature=temperature)
    items = generate(model, [prompt + interest for interest in interests], grammar, m, "item", temperature=temperature)
    branches = [i for i in range(n) for _ in range(m)]
    sequences = [interests[i] + item for i, item in zip(branches, items)]
    return sequences, branches, n, m


def pack_trajectories(prompts, completions, interest_lengths, auxiliary, grammar, device):
    p_len, c_len = max(map(len, prompts)), max(map(len, completions))
    ids, masks, interest_masks, item_masks = [], [], [], []
    allowed_sets, action_positions = [], []
    for prompt, comp, interest_len, aux in zip(prompts, completions, interest_lengths, auxiliary):
        ids.append([grammar.eos] * (p_len - len(prompt)) + prompt + comp + [grammar.eos] * (c_len - len(comp)))
        masks.append([0] * (p_len - len(prompt)) + [1] * (len(prompt) + len(comp)) + [0] * (c_len - len(comp)))
        interest_masks.append([1] * interest_len + [0] * (c_len - interest_len))
        item_masks.append([0] * interest_len + [1] * (len(comp) - interest_len) + [0] * (c_len - len(comp)))
        allowed_sets.append([
            grammar.allowed(comp[:i], "joint", aux) if i < len(comp) else [grammar.eos]
            for i in range(c_len)
        ])
        action_positions.append([allowed.index(comp[i] if i < len(comp) else grammar.eos)
                                 for i, allowed in enumerate(allowed_sets[-1])])
    return dict(input_ids=torch.tensor(ids, device=device), attention_mask=torch.tensor(masks, device=device),
                completion_ids=torch.tensor([x + [grammar.eos] * (c_len - len(x)) for x in completions], device=device),
                interest_mask=torch.tensor(interest_masks, device=device, dtype=torch.float32),
                item_mask=torch.tensor(item_masks, device=device, dtype=torch.float32),
                allowed_sets=allowed_sets, action_positions=action_positions)


def action_logps(model, packed, temperature=1.0):
    """Match the constrained sampling distribution, including temperature."""
    length = packed["completion_ids"].shape[1]
    logits = model(input_ids=packed["input_ids"], attention_mask=packed["attention_mask"],
                   logits_to_keep=length + 1, use_cache=False).logits[:, -length - 1:-1]
    rows = []
    for b, sets in enumerate(packed["allowed_sets"]):
        values = []
        for t, allowed in enumerate(sets):
            restricted = logits[b, t, allowed].float() / temperature
            values.append(restricted[packed["action_positions"][b][t]] - torch.logsumexp(restricted, 0))
        rows.append(torch.stack(values))
    return torch.stack(rows)
