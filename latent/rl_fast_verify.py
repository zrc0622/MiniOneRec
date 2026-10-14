"""Real-checkpoint conditional-distribution checks before speed benchmarking."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from .core import history_records, load_index, validate_protocol, write_json
from .generation import Grammar, pack_trajectories
from .rl_fast import cached_rollouts, restricted_logps


def main():
    p = argparse.ArgumentParser()
    for key in ('model', 'index', 'data', 'output'):
        p.add_argument('--' + key, required=True)
    o = p.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    tk = AutoTokenizer.from_pretrained(o.model)
    index = load_index(o.index)
    report = {}
    for label, dtype in (('fp32', torch.float32), ('bf16', torch.bfloat16)):
        model = AutoModelForCausalLM.from_pretrained(o.model, torch_dtype=dtype).cuda().eval()
        grammar = Grammar(tk, index, validate_protocol(model.config, tk, index))
        records = history_records(o.data, index, grammar.k)
        stride = max(1, len(records) // 64)
        prompts = [tk.encode(r['prompt'], add_special_tokens=False)[-512:] for r in records[::stride][:64]]
        prompts = [min(prompts, key=len), max(prompts, key=len)]
        audits = []

        @torch.no_grad()
        def audit(logits, allowed, users, prefixes):
            sequences = [prompts[u] + prefix for u, prefix in zip(users, prefixes)]
            width = max(map(len, sequences))
            ids = torch.tensor([[grammar.eos] * (width - len(s)) + s for s in sequences], device='cuda')
            mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in sequences], device='cuda')
            ref = model(input_ids=ids, attention_mask=mask, position_ids=(mask.cumsum(-1) - 1).clamp_min(0),
                        use_cache=False, logits_to_keep=1).logits[:, -1]
            maximum = max(map(len, allowed))
            aid = torch.tensor([s + [0] * (maximum - len(s)) for s in allowed], device='cuda')
            valid = torch.arange(maximum, device='cuda')[None] < torch.tensor(list(map(len, allowed)), device='cuda')[:, None]
            x = logits.gather(-1, aid).float().masked_fill(~valid, -torch.inf).log_softmax(-1)
            y = ref.gather(-1, aid).float().masked_fill(~valid, -torch.inf).log_softmax(-1)
            delta = (x - y).masked_fill(~valid, 0)
            kl = (x.exp() * delta).sum(-1).clamp_min(0)
            audits.append(dict(rows=len(users), max_logprob_difference=delta.abs().max().item(),
                               max_kl=kl.max().item(), mean_kl=kl.mean().item()))

        set_seed(42)
        for auxiliary in (False, 'identification', 'title'):
            groups = cached_rollouts(model, prompts, grammar, auxiliary, 1., audit)
            if not auxiliary:
                main_samples = [s for group in groups for s in group]
            for group in groups:
                if len(group) != 16:
                    raise ValueError('Incorrect16candidatebudget')
                if not auxiliary and any(group[b][:3] != group[b + j][:3] for b in range(0, 16, 4) for j in range(4)):
                    raise ValueError('Incorrect4x4grouping')
        packed = pack_trajectories([s for s in prompts for _ in range(16)], main_samples,
                                  [3] * 32, [False] * 32, grammar, 'cuda')
        length = packed['completion_ids'].shape[1]
        with torch.no_grad():
            values = model(input_ids=packed['input_ids'], attention_mask=packed['attention_mask'],
                           logits_to_keep=length + 1, use_cache=False).logits[:, -length - 1:-1]
        # Actual-checkpoint logits: identical input values, verify normalization
        # and its gradients independently of nondeterministic sampled actions.
        values = values.detach().requires_grad_()
        original_rows = []
        for b, row in enumerate(packed['allowed_sets']):
            original_tokens = []
            for t, allowed in enumerate(row):
                restricted = values[b, t, allowed].float()
                original_tokens.append(restricted[packed['action_positions'][b][t]] - torch.logsumexp(restricted, 0))
            original_rows.append(torch.stack(original_tokens))
        old = torch.stack(original_rows)
        fast = restricted_logps(values, packed)
        weights = torch.randn_like(old)
        g1 = torch.autograd.grad((old * weights).sum(), values, retain_graph=True)[0]
        g2 = torch.autograd.grad((fast * weights).sum(), values)[0]
        torch.testing.assert_close(fast, old, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(g1, g2, atol=.0001 if dtype == torch.bfloat16 else 2e-6, rtol=.02 if dtype == torch.bfloat16 else 2e-5)
        max_kl = max(r['max_kl'] for r in audits)
        if max_kl > (5e-6 if dtype == torch.float32 else .002):
            raise ValueError(f'Conditional distribution drift too large: {label} KL{max_kl}')
        report[label] = dict(audits=audits, max_kl=max_kl, prompt_lengths=list(map(len, prompts)),
                             logprob_gradient_check='pass', grammar_and_4x4_check='pass')
        print(json.dumps({label: report[label]}), flush=True)
        del model, values, old, fast, g1, g2, packed, original_rows, original_tokens, restricted
        torch.cuda.empty_cache()
    write_json(o.output, dict(status='pass', model=o.model, checks=report,
        semantics='same constrained categorical law; RNG order differs; BF16 not bitwise equivalent',
        attention='original mathSDP unchanged'))


if __name__ == '__main__':
    main()
