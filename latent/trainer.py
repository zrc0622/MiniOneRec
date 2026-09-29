"""Trainer integration reuses tested grouping, reference sync and DS cleanup."""
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl.models import unwrap_model_for_generation

from interest.trainer import InterestCollator, InterestGRPOTrainer
from .generation import constrained_logps, full_logps, pack_trajectories, rollout
from .objective import advantages, grpo_loss


class LatentSFTTrainer(Trainer):
    def __init__(self, *args, latent_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_weight = latent_weight
        self.model_accepts_loss_kwargs = False
        self._metrics = defaultdict(list)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs['labels'][:, 1:]
        valid = labels != -100
        latent = inputs['interest_mask'][:, 1:].bool() & valid
        main = latent.any(1)
        item = valid & ~latent & main[:, None]
        auxiliary = valid & ~main[:, None]
        out = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], use_cache=False)
        ce = F.cross_entropy(out.logits[:, :-1].float().transpose(1, 2), labels, ignore_index=-100, reduction='none')
        main_loss = ((ce * item).sum(1) / item.sum(1).clamp_min(1)
                     + self.latent_weight * (ce * latent).sum(1) / latent.sum(1).clamp_min(1)).sum()
        # Original auxiliary CE remains token averaged; combine tasks by example
        # count. On an all-auxiliary batch this equals the original HF causal CE.
        aux_loss = (ce * auxiliary).sum() / auxiliary.sum().clamp_min(1)
        loss = (main_loss + (~main).sum() * aux_loss) / len(labels)
        for name, mask in (('latent_ce', latent), ('item_ce', item), ('auxiliary_ce', auxiliary)):
            stats = torch.stack([(ce.detach() * mask).sum(), mask.sum().float()])
            stats = self.accelerator.reduce(stats, reduction='sum')
            if stats[1] > 0:
                self._metrics[name].append((stats[0] / stats[1]).item())
        return (loss, out) if return_outputs else loss

    def log(self, logs, start_time=None):
        prefix = 'eval_' if any(k.startswith('eval_') for k in logs) else ''
        logs = dict(logs, **{prefix + k: sum(v) / len(v) for k, v in self._metrics.items() if v})
        super().log(logs, start_time)
        self._metrics.clear()


class LatentGRPOTrainer(InterestGRPOTrainer):
    def __init__(self, *, mode, **kwargs):
        if mode not in ('original', '4x4'):
            raise ValueError(mode)
        super().__init__(mode='16x1' if mode == 'original' else mode, **kwargs)
        self.mode = mode
        self.logps = full_logps if mode == 'original' else constrained_logps

    def _prepare_inputs(self, inputs):
        if len(inputs) % 16:
            raise ValueError('Incomplete 16-trajectory prompt group')
        prompts, completions, lengths, flags, specs = [], [], [], [], []
        metrics = defaultdict(list)
        with unwrap_model_for_generation(self.model, self.accelerator) as model:
            training = model.training
            model.eval()
            try:
                for start in range(0, len(inputs), 16):
                    record = inputs[start]
                    if any(r != record for r in inputs[start:start + 16]):
                        raise ValueError('Sampler split a prompt group')
                    prompt = self.processing_class.encode(record['prompt'], add_special_tokens=False)[-self.args.max_prompt_length:]
                    aux = ('identification' if record['task'] == 'item_identification' else 'title') if record['auxiliary'] else False
                    samples = rollout(model, prompt, self.grammar, self.mode, aux, self.temperature)
                    items = [self.grammar.item(s, aux) for s in samples]
                    hits = torch.tensor([float(x == record['target']) for x in items], device=self.accelerator.device)
                    ia, ya, rewards = advantages(hits, self.mode, aux)
                    specs.append((start, '16x1' if self.mode == 'original' else '4x4', bool(aux), ia, ya))
                    prompts.extend([prompt] * 16)
                    completions.extend(samples)
                    lengths.extend([0 if aux else self.grammar.k] * 16)
                    flags.extend([aux] * 16)
                    task = record['task']
                    metrics[f'{task}/item_hit'].append(hits.mean())
                    metrics[f'{task}/reward'].append(rewards.mean())
                    metrics[f'{task}/zero_latent_advantage'].append((ia.abs().sum() == 0).float())
                    metrics[f'{task}/zero_item_advantage'].append((ya.abs().sum() == 0).float())
                    metrics[f'{task}/unique_items'].append(torch.tensor(float(len(set(items))), device=hits.device))
                    if not aux:
                        step = 4 if self.mode == '4x4' else 1
                        count = len({tuple(s[:3]) for s in samples[::step]})
                        metrics[f'{task}/unique_latents'].append(torch.tensor(float(count), device=hits.device))
            finally:
                model.train(training)
        packed = pack_trajectories(prompts, completions, lengths, flags, self.grammar, self.accelerator.device)
        with torch.no_grad():
            packed['ref_logps'] = self.logps(self.ref_model, packed, self.temperature)
        packed['group_specs'] = specs
        # Identical reduction order across ranks even for different tasks.
        for task in ('history_sid', 'history_title', 'item_identification'):
            for name in ('item_hit', 'reward', 'zero_latent_advantage', 'zero_item_advantage', 'unique_items', 'unique_latents'):
                if name == 'unique_latents' and task != 'history_sid':
                    continue
                key = f'{task}/{name}'
                values = metrics[key]
                total = sum(values, torch.tensor(0., device=self.accelerator.device))
                stats = torch.stack([total, torch.tensor(float(len(values)), device=total.device)])
                stats = self.accelerator.reduce(stats, reduction='sum')
                if stats[1] > 0:
                    self._metrics[key].append((stats[0] / stats[1]).item())
        return packed

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError('GRPO prediction returns loss only')
        logps = self.logps(model, inputs, self.temperature)
        loss, kl = grpo_loss(logps, inputs['ref_logps'], inputs, inputs['group_specs'], self.beta, self.interest_weight)
        self._metrics['kl'].append(self.accelerator.reduce(kl.detach(), reduction='mean').item())
        return loss
