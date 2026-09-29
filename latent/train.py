"""python -m latent.train {sft,rl}; separate outputs for all experiments."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, TrainingArguments, set_seed

from deepspeed_cleanup import finish_rl_training
from interest.train import InterestRLArguments
from .core import history_records, initialize, load_index, load_labels, sha, validate_protocol, write_json
from .data import rl_records, sft_datasets
from .generation import Grammar
from .trainer import InterestCollator, LatentGRPOTrainer, LatentSFTTrainer


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['sft', 'rl'])
    for key in ('model', 'train_file', 'eval_file', 'items', 'index', 'output_dir'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--labels')
    p.add_argument('--mode', choices=['original', '4x4'], default='original')
    p.add_argument('--epochs', type=float)
    p.add_argument('--lr', type=float)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--eval_batch', type=int)
    p.add_argument('--gradient_accumulation', type=int, default=16)
    p.add_argument('--latent_weight', type=float, default=1.0)
    p.add_argument('--beta', type=float, default=0.001)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--title_sample', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--deepspeed', default='config/sft_zero2.json')
    p.add_argument('--max_steps', type=int, default=-1)
    p.add_argument('--cpu', action='store_true')
    return p


def main():
    o = parser().parse_args()
    if o.batch != 16 or o.gradient_accumulation <= 0 or o.latent_weight <= 0 or o.beta < 0 or o.temperature <= 0:
        raise ValueError('Invalid batch/loss/generation settings')
    if (o.epochs is not None and o.epochs <= 0) or (o.lr is not None and o.lr <= 0):
        raise ValueError('Epochs and LR must be positive')
    output = Path(o.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output directory must be empty: {output}')
    index = load_index(o.index)
    set_seed(o.seed)
    cuda = torch.cuda.is_available() and not o.cpu
    tokenizer = AutoTokenizer.from_pretrained(o.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(o.model, torch_dtype=torch.bfloat16 if cuda else torch.float32)
    epochs = o.epochs or (15 if o.stage == 'sft' else 2)
    common = dict(output_dir=str(output), per_device_train_batch_size=o.batch,
        per_device_eval_batch_size=o.eval_batch or (16 if o.stage == 'sft' else 128),
        gradient_accumulation_steps=o.gradient_accumulation, bf16=cuda, use_cpu=o.cpu,
        deepspeed=o.deepspeed or None, logging_steps=1, report_to='tensorboard',
        logging_dir=str(output / 'tensorboard'), eval_strategy='steps', save_strategy='steps',
        remove_unused_columns=False, prediction_loss_only=True, seed=o.seed,
        ddp_find_unused_parameters=False, max_steps=o.max_steps, num_train_epochs=epochs)
    if o.stage == 'sft':
        if not o.labels:
            raise ValueError('SFT requires --labels')
        _, manifest = load_labels(o.labels, 'train', o.train_file, index)
        protocol = initialize(model, tokenizer, index, manifest)
        train, valid = sft_datasets(o.train_file, o.eval_file, o.items, o.index, index, tokenizer, protocol, o.labels, o.seed)
        args = TrainingArguments(**common, learning_rate=o.lr or 3e-4, warmup_steps=20,
            lr_scheduler_type='linear', optim='adamw_torch', save_total_limit=1,
            eval_steps=min(0.5 / epochs, 0.999), save_steps=min(0.5 / epochs, 0.999),
            load_best_model_at_end=True, metric_for_best_model='eval_loss')
        trainer = LatentSFTTrainer(model=model, args=args, processing_class=tokenizer,
            data_collator=InterestCollator(tokenizer), latent_weight=o.latent_weight,
            train_dataset=Dataset.from_list(train).shuffle(seed=o.seed), eval_dataset=Dataset.from_list(valid).shuffle(seed=o.seed),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
        task_counts = dict(main=len(history_records(o.train_file, index)), auxiliary=len(train) - len(history_records(o.train_file, index)))
    else:
        # Match baseline kernel selection and original optimization settings.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        protocol = validate_protocol(model.config, tokenizer, index)
        train = rl_records(o.train_file, o.items, o.index, index, o.title_sample)
        valid = history_records(o.eval_file, index)
        args = InterestRLArguments(**common, learning_rate=o.lr or 1e-5, warmup_ratio=0.03,
            lr_scheduler_type='cosine', optim='paged_adamw_32bit' if cuda else 'adamw_torch',
            max_grad_norm=0.3, save_total_limit=20, eval_steps=0.0999, save_steps=0.1,
            load_best_model_at_end=False, gradient_checkpointing=True)
        trainer = LatentGRPOTrainer(model=model, args=args, processing_class=tokenizer,
            grammar=Grammar(tokenizer, index, protocol), mode=o.mode, beta=o.beta,
            temperature=o.temperature, interest_weight=o.latent_weight,
            train_dataset=Dataset.from_list(train).shuffle(seed=o.seed), eval_dataset=Dataset.from_list(valid).shuffle(seed=o.seed))
        task_counts = dict(Counter(r['task'] for r in train))
    model.config.use_cache = False
    if trainer.is_world_process_zero():
        output.mkdir(parents=True, exist_ok=True)
        paths = [o.train_file, o.eval_file, o.items, o.index]
        paths += [str(Path(o.model) / name) for name in ('config.json', 'tokenizer.json') if (Path(o.model) / name).is_file()]
        if o.labels and o.stage == 'sft':
            paths.append(str(Path(o.labels) / 'labels.json'))
        write_json(output / 'latent_run.json', dict(options=vars(o), training_args=args.to_dict(), protocol=protocol,
            source_hashes={str(Path(p).resolve()): sha(p) for p in paths}, task_counts=task_counts,
            train_rows=len(train), validation_rows=len(valid)))
        tokenizer.save_pretrained(output)
    trainer.accelerator.wait_for_everyone()
    trainer.train()
    trainer.save_model(str(output / 'final_checkpoint'))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output / 'final_checkpoint')
    if o.stage == 'rl':
        finish_rl_training(trainer)
    else:
        trainer.accelerator.wait_for_everyone()
        trainer.accelerator.end_training()


if __name__ == '__main__':
    main()
