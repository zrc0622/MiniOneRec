"""Independent Trainer integration; original SFT/RL trainers are untouched."""
from collections import defaultdict

import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from torch.utils.data import Dataset, Sampler
from transformers import DataCollatorForSeq2Seq, Trainer
from trl import SyncRefModelCallback
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation

from deepspeed_cleanup import patch_reference_bf16_cleanup
from .generation import action_logps, pack_trajectories, rollout
from .objective import advantages, grpo_loss


def passthrough(features):
    return features


class InterestCollator:
    def __init__(self, tokenizer):
        self.base = DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8, return_tensors="pt")
        self.left = tokenizer.padding_side == "left"

    def __call__(self, features):
        masks = [x["interest_mask"] for x in features]
        batch = self.base([{k: v for k, v in x.items() if k != "interest_mask"} for x in features])
        width = batch["input_ids"].shape[1]
        batch["interest_mask"] = torch.tensor([
            ([0] * (width - len(m)) + m) if self.left else (m + [0] * (width - len(m))) for m in masks
        ], dtype=torch.float32)
        return batch


class MidTrainer(Trainer):
    def __init__(self, *args, interest_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.interest_weight = interest_weight
        self.model_accepts_loss_kwargs = False
        self._mid_metrics = defaultdict(list)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"][:, 1:]
        mask = inputs["interest_mask"][:, 1:].bool() & (labels != -100)
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False)
        ce = F.cross_entropy(out.logits[:, :-1].float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
        item_mask = (labels != -100) & ~mask
        # Each response component has its own normalization; k does not silently
        # multiply the interest supervision weight.
        loss = (self.interest_weight * (ce * mask).sum(1) / mask.sum(1).clamp_min(1)
                + (ce * item_mask).sum(1) / item_mask.sum(1).clamp_min(1)).mean()
        for name, component in (("interest_ce", mask), ("response_ce", item_mask)):
            stats = torch.stack([(ce.detach() * component).sum(), component.sum().float()])
            stats = self.accelerator.reduce(stats, reduction="sum")
            if stats[1] > 0:
                self._mid_metrics[name].append((stats[0] / stats[1]).item())
        return (loss, out) if return_outputs else loss

    def log(self, logs, start_time=None):
        prefix = "eval_" if any(k.startswith("eval_") for k in logs) else ""
        logs = dict(logs, **{prefix + k: sum(v) / len(v) for k, v in self._mid_metrics.items() if v})
        super().log(logs, start_time)
        self._mid_metrics.clear()


class GroupDataset(Dataset):
    """Expose trajectory-row cardinality to Trainer's eval gather/truncation."""
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records) * 16

    def __getitem__(self, index):
        return self.records[index // 16]


class GroupSampler(Sampler):
    """Sixteen consecutive rows per prompt; each rank holds whole groups."""
    def __init__(self, data, seed=42):
        self.grouped = isinstance(data, GroupDataset)
        self.size = len(data) // 16 if self.grouped else len(data)
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter([i * 16 + j if self.grouped else i
                     for i in torch.randperm(self.size, generator=generator).tolist() for j in range(16)])

    def __len__(self):
        return self.size * 16


class InterestGRPOTrainer(Trainer):
    def __init__(self, *, grammar, mode, beta=0.001, temperature=1.0, interest_weight=1.0, **kwargs):
        self.grammar, self.mode = grammar, mode
        self.beta, self.temperature, self.interest_weight = beta, temperature, interest_weight
        self._metrics = defaultdict(list)
        for key in ("train_dataset", "eval_dataset"):
            kwargs[key] = GroupDataset(kwargs[key])
        super().__init__(data_collator=passthrough, **kwargs)
        self.model.warnings_issued["estimate_tokens"] = True
        self.model_accepts_loss_kwargs = False
        if mode not in ("16x1", "4x4"):
            raise ValueError(mode)
        for batch in (self.args.per_device_train_batch_size, self.args.per_device_eval_batch_size):
            if batch % 16:
                raise ValueError("Each device's batch must be a multiple of 16 trajectory rows")
        if self.args.dataloader_drop_last:
            raise ValueError("Do not drop prompt groups")
        # Repeated prefixes must yield identical policies within one group.
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        self.ref_model = create_reference_model(self.model)
        if self.is_deepspeed_enabled:
            self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            patch_reference_bf16_cleanup(self.ref_model)
        else:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))
        set_seed(self.args.seed, device_specific=True)

    def _get_train_sampler(self, train_dataset=None):
        return GroupSampler(self.train_dataset if train_dataset is None else train_dataset, self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return GroupSampler(eval_dataset, self.args.seed)

    def _prepare_inputs(self, inputs):
        # All trajectories for each prompt live on one rank. No cross-rank reward
        # reshape can accidentally combine different users or task types.
        if len(inputs) % 16:
            raise ValueError("Incomplete prompt group")
        prompts, completions, lengths, auxiliary_flags, specs = [], [], [], [], []
        metrics = defaultdict(list)
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped:
            was_training = unwrapped.training
            unwrapped.eval()
            try:
                for start in range(0, len(inputs), 16):
                    records = inputs[start:start + 16]
                    if any(r != records[0] for r in records[1:]):
                        raise ValueError("Sampler split a prompt group across examples")
                    record = records[0]
                    prompt = self.processing_class.encode(record["prompt"], add_special_tokens=False)[-self.args.max_prompt_length:]
                    auxiliary = record["auxiliary"]
                    samples, branches, n, m = rollout(unwrapped, prompt, self.grammar, self.mode, auxiliary, self.temperature)
                    items = [self.grammar.item(s, auxiliary) for s in samples]
                    hits = torch.tensor([float(s == record["target"]) for s in items], device=self.accelerator.device)
                    int_adv, item_adv, rewards = advantages(hits, self.mode, auxiliary)
                    specs.append((start, self.mode, auxiliary, int_adv, item_adv))
                    prompts.extend([prompt] * 16)
                    completions.extend(samples)
                    lengths.extend([0 if auxiliary else self.grammar.k + 1] * 16)
                    auxiliary_flags.extend([auxiliary] * 16)
                    task = record["task"]
                    metrics[f"{task}/item_hit"].append(hits.mean())
                    metrics[f"{task}/group_hit"].append(rewards.mean())
                    metrics[f"{task}/zero_advantage"].append((rewards.std(correction=0) == 0).float())
                    if not auxiliary:
                        unique = len({tuple(samples[i * m][:self.grammar.k]) for i in range(n)})
                        metrics[f"{task}/interest_diversity"].append(torch.tensor(unique / n, device=hits.device))
            finally:
                unwrapped.train(was_training)
        packed = pack_trajectories(prompts, completions, lengths, auxiliary_flags, self.grammar, self.accelerator.device)
        with torch.no_grad():
            packed["ref_logps"] = action_logps(self.ref_model, packed, self.temperature)
        packed["group_specs"] = specs
        # Fixed metric key order on every rank, even when their task types differ.
        for task in ("history_sid", "history_title", "item_identification"):
            names = ["item_hit", "group_hit", "zero_advantage"]
            if task != "item_identification":
                names.append("interest_diversity")
            for name in names:
                key = f"{task}/{name}"
                vals = metrics[key]
                total = sum(vals, torch.tensor(0.0, device=self.accelerator.device))
                stats = torch.stack([total, torch.tensor(float(len(vals)), device=total.device)])
                stats = self.accelerator.reduce(stats, reduction="sum")
                if stats[1] > 0:
                    self._metrics[key].append((stats[0] / stats[1]).item())
        return packed

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GRPO does not return logits through Trainer prediction")
        logps = action_logps(model, inputs, self.temperature)
        loss, kl = grpo_loss(logps, inputs["ref_logps"], inputs, inputs["group_specs"], self.beta, self.interest_weight)
        self._metrics["kl"].append(self.accelerator.reduce(kl.detach(), reduction="mean").item())
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None

    def log(self, logs, start_time=None):
        prefix = "eval_" if any(k.startswith("eval_") for k in logs) else ""
        logs = dict(logs, **{prefix + k: sum(v) / len(v) for k, v in self._metrics.items() if v})
        super().log(logs, start_time)
        self._metrics.clear()
