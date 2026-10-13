"""Evaluation-only exact beam search reusing the historical prompt KV."""
import copy
import torch
from transformers import GenerationConfig
from .eval_pilot import PrefixScorer


@torch.no_grad()
def cached_item_beam(model, prompt, prefix, grammar, count, prompt_cache):
    width=len(prompt)+len(prefix)
    cache=copy.deepcopy(prompt_cache)
    cache.batch_repeat_interleave(count)
    ids=torch.tensor([prompt+prefix],device=next(model.parameters()).device)
    cfg=GenerationConfig(max_new_tokens=grammar.item_length,do_sample=False,num_beams=count,
        num_return_sequences=count,length_penalty=0.0,top_k=None,top_p=None,
        pad_token_id=grammar.eos,eos_token_id=grammar.eos,forced_eos_token_id=None,
        renormalize_logits=False,use_cache=True,return_dict_in_generate=True,output_scores=True)
    out=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=cache,
        generation_config=cfg,use_model_defaults=False,do_sample=False,
        prefix_allowed_tokens_fn=lambda _,tokens:grammar.allowed(tokens[width:].tolist(),'item'),synced_gpus=False)
    result=[]; scores=[];seen=set()
    for sequence,score in zip(out.sequences[:,width:].tolist(),out.sequences_scores.tolist()):
        if grammar.eos in sequence: sequence=sequence[:sequence.index(grammar.eos)+1]
        key=tuple(sequence)
        if key in seen or not sequence or sequence[-1]!=grammar.eos: continue
        if not all(t in grammar.allowed(sequence[:i],'item') for i,t in enumerate(sequence)): continue
        seen.add(key);result.append(sequence);scores.append(score)
    return result,scores


@torch.no_grad()
def cached_item_beam_batch(model, prompt, prefixes, grammar, count, prompt_cache):
    """Run equal-width item beams for multiple latent prefixes in one batch."""
    if not prefixes or len({len(p) for p in prefixes}) != 1:
        raise ValueError('Nonempty equal-length prefixes required')
    width = len(prompt) + len(prefixes[0])
    cache = copy.deepcopy(prompt_cache)
    cache.batch_repeat_interleave(len(prefixes) * count)
    ids = torch.tensor([prompt + p for p in prefixes], device=next(model.parameters()).device)
    cfg = GenerationConfig(max_new_tokens=grammar.item_length, do_sample=False, num_beams=count,
        num_return_sequences=count, length_penalty=0.0, top_k=None, top_p=None,
        pad_token_id=grammar.eos, eos_token_id=grammar.eos, forced_eos_token_id=None,
        renormalize_logits=False, use_cache=True, return_dict_in_generate=True, output_scores=True)
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), past_key_values=cache,
        generation_config=cfg, use_model_defaults=False, do_sample=False,
        prefix_allowed_tokens_fn=lambda _, tokens: grammar.allowed(tokens[width:].tolist(), 'item'),
        synced_gpus=False)
    sequences = out.sequences[:, width:].tolist()
    scores = out.sequences_scores.tolist()
    groups = []
    for j in range(len(prefixes)):
        items, probs, seen = [], [], set()
        for seq, score in zip(sequences[j*count:(j+1)*count], scores[j*count:(j+1)*count]):
            if grammar.eos in seq:
                seq = seq[:seq.index(grammar.eos)+1]
            key = tuple(seq)
            if key in seen or not seq or seq[-1] != grammar.eos:
                continue
            if not all(t in grammar.allowed(seq[:i], 'item') for i, t in enumerate(seq)):
                continue
            seen.add(key); items.append(seq); probs.append(score)
        if not items:
            raise ValueError('No valid batch beam completions')
        groups.append((items, probs))
    return groups
