"""Measure actual two-rank SFT updates, retaining effective batch 1024."""
import argparse
import json
import os
from pathlib import Path
import time

from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainingArguments, set_seed

from interest.trainer import InterestCollator
from .core import initialize, load_index
from .data import load_labels, sft_datasets
from .trainer import LatentSFTTrainer


class Timing(TrainerCallback):
    def __init__(self):
        self.durations = []

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.durations.append(time.perf_counter() - self.start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--steps', type=int, default=5)
    o = parser.parse_args()
    world = int(os.environ.get('WORLD_SIZE', 1))
    if 1024 % (world * o.batch) or o.steps < 4:
        parser.error('Microbatch must divide 1024/world; at least four steps')
    set_seed(42)
    data = Path('data/Amazon23/variants/rqkmeans_balanced')
    stem = 'Industrial_and_Scientific_5_2016-10-2018-11'
    train_file, valid_file = [str(data / split / f'{stem}.csv') for split in ('train', 'valid')]
    index_file = str(data / 'Industrial_and_Scientific/Industrial_and_Scientific.index.json')
    items = str(data / 'Industrial_and_Scientific/Industrial_and_Scientific.item.json')
    index = load_index(index_file)
    _, manifest = load_labels(o.labels, 'train', train_file, index)
    tokenizer = AutoTokenizer.from_pretrained('./models/Qwen3-0.6B')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained('./models/Qwen3-0.6B', torch_dtype=torch.bfloat16)
    protocol = initialize(model, tokenizer, index, manifest)
    model.config.use_cache = False
    train, _ = sft_datasets(train_file, valid_file, items, index_file, index,
                            tokenizer, protocol, o.labels, 42)
    args = TrainingArguments(output_dir=o.output, per_device_train_batch_size=o.batch,
        gradient_accumulation_steps=1024 // (world * o.batch), max_steps=o.steps,
        learning_rate=3e-4, warmup_steps=20, lr_scheduler_type='linear', optim='adamw_torch',
        bf16=True, deepspeed='config/sft_zero2.json', report_to=[], logging_steps=1,
        save_strategy='no', eval_strategy='no', remove_unused_columns=False,
        prediction_loss_only=True, seed=42, ddp_find_unused_parameters=False)
    timing = Timing()
    trainer = LatentSFTTrainer(model=model, args=args, processing_class=tokenizer,
        data_collator=InterestCollator(tokenizer),
        train_dataset=Dataset.from_list(train).shuffle(seed=42), callbacks=[timing])
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    measurements = torch.tensor([sum(timing.durations[2:]) / len(timing.durations[2:]),
        torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30],
        device=trainer.accelerator.device)
    ranks = trainer.accelerator.gather(measurements).reshape(world, 3).cpu().tolist()
    if trainer.is_world_process_zero():
        seconds = max(row[0] for row in ranks)
        result = dict(batch_per_device=o.batch, world_size=world,
            accumulation=args.gradient_accumulation_steps, global_batch=1024,
            steps=o.steps, warmup_steps_excluded=2, step_seconds=seconds,
            examples_per_second=1024 / seconds, rank_measurements=ranks,
            local_step_durations=timing.durations,
            note='Real mixed-task dataset, original loss, attention and ZeRO2; no benchmark weights reused.')
        Path(o.output).mkdir(parents=True, exist_ok=True)
        (Path(o.output) / 'benchmark.json').write_text(json.dumps(result, indent=2) + '\n')
        print('BENCHMARK', json.dumps(result), flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
