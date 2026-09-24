"""python -m interest.train {mid,rl}; isolated from the baseline training entrypoints."""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, TrainingArguments, set_seed

from deepspeed_cleanup import finish_rl_training
from .core import initialize_interests, load_index, validate_protocol
from .data import history_records, mid_datasets, rl_records
from .generation import Grammar
from .trainer import InterestCollator, InterestGRPOTrainer, MidTrainer


@dataclass
class InterestRLArguments(TrainingArguments):
    max_prompt_length: int = 512
    ref_model_sync_steps: int = 512
    ref_model_mixup_alpha: float = 0.6


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["mid", "rl"])
    for key in ("model", "train_file", "eval_file", "index", "items", "output_dir"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--mode", choices=["16x1", "4x4"], default="16x1")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--eval_batch", type=int, default=16)
    p.add_argument("--gradient_accumulation", type=int, default=16)
    p.add_argument("--beta", type=float, default=0.001)
    p.add_argument("--interest_weight", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--title_sample", type=int, default=10000)
    p.add_argument("--deepspeed", default="config/sft_zero2.json")
    p.add_argument("--max_steps", type=int, default=-1, help="Override epochs for a smoke run; use a separate output dir")
    p.add_argument("--cpu", action="store_true", help="CPU smoke tests only; also pass --deepspeed ''")
    return p


def main():
    options = parser().parse_args()
    if options.batch != 16 or options.gradient_accumulation < 1:
        raise ValueError("Use batch=16 trajectory rows and positive gradient accumulation")
    if options.interest_weight <= 0 or options.temperature <= 0 or options.beta < 0:
        raise ValueError("Invalid loss weight, temperature or beta")
    if (options.epochs is not None and options.epochs <= 0) or (options.lr is not None and options.lr <= 0):
        raise ValueError("Epochs and learning rate must be positive")
    output = Path(options.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty; use a new experiment directory: {output}")
    index = load_index(options.index)
    set_seed(options.seed)
    cuda = torch.cuda.is_available() and not options.cpu
    dtype = torch.bfloat16 if cuda else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(options.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(options.model, torch_dtype=dtype)
    common = dict(
        output_dir=str(output), per_device_train_batch_size=options.batch,
        per_device_eval_batch_size=options.eval_batch,
        gradient_accumulation_steps=options.gradient_accumulation,
        bf16=cuda, use_cpu=options.cpu, deepspeed=options.deepspeed or None,
        logging_steps=1, report_to="tensorboard", logging_dir=str(output / "tensorboard"),
        eval_strategy="steps", save_strategy="steps", eval_steps=0.1, save_steps=0.1,
        remove_unused_columns=False, prediction_loss_only=True, seed=options.seed,
        ddp_find_unused_parameters=False, max_steps=options.max_steps,
    )
    if options.stage == "mid":
        initialize_interests(model, tokenizer, index, options.k)
        train, valid = mid_datasets(options.train_file, options.eval_file, options.items, options.index,
                                    index, tokenizer, options.k, options.seed, title_sample=options.title_sample)
        args = TrainingArguments(**common, num_train_epochs=options.epochs or 3,
                                  learning_rate=options.lr or 3e-5, warmup_steps=20,
                                  lr_scheduler_type="linear", optim="adamw_torch", save_total_limit=2,
                                  load_best_model_at_end=True, metric_for_best_model="eval_loss")
        model.config.use_cache = False
        trainer = MidTrainer(model=model, args=args, processing_class=tokenizer,
                             data_collator=InterestCollator(tokenizer), interest_weight=options.interest_weight,
                             train_dataset=Dataset.from_list(train), eval_dataset=Dataset.from_list(valid),
                             callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    else:
        protocol = validate_protocol(model.config, tokenizer, index)
        if options.k != protocol["k"]:
            raise ValueError("k must match the Mid-SFT checkpoint")
        train = rl_records(options.train_file, options.items, options.index, index, options.seed, options.title_sample)
        valid = history_records(options.eval_file, index)
        args = InterestRLArguments(**common, num_train_epochs=options.epochs or 2,
                                    learning_rate=options.lr or 1e-5, warmup_ratio=0.03,
                                    lr_scheduler_type="cosine", optim="paged_adamw_32bit" if cuda else "adamw_torch",
                                    max_grad_norm=0.3, save_total_limit=20, load_best_model_at_end=False,
                                    gradient_checkpointing=True)
        trainer = InterestGRPOTrainer(model=model, args=args, processing_class=tokenizer,
                                      grammar=Grammar(tokenizer, index, protocol["k"]), mode=options.mode,
                                      beta=options.beta, temperature=options.temperature,
                                      interest_weight=options.interest_weight,
                                      train_dataset=Dataset.from_list(train), eval_dataset=Dataset.from_list(valid))
    if trainer.is_world_process_zero():
        output.mkdir(parents=True, exist_ok=True)
        source_paths = [options.train_file, options.eval_file, options.index, options.items]
        source_paths += [str(Path(options.model) / name) for name in ("config.json", "tokenizer.json")
                         if (Path(options.model) / name).is_file()]
        manifest = dict(options=vars(options), training_args=args.to_dict(),
                        protocol=model.config.interest_protocol,
                        source_hashes={str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths},
                        task_counts=dict(Counter(r.get("task", "supervised") for r in train)),
                        train_rows=len(train), validation_rows=len(valid))
        (output / "interest_run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        tokenizer.save_pretrained(output)
    trainer.accelerator.wait_for_everyone()
    trainer.train()
    trainer.save_model(str(output / "final_checkpoint"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output / "final_checkpoint")
    if options.stage == "rl":
        finish_rl_training(trainer)
    else:
        trainer.accelerator.wait_for_everyone()
        trainer.accelerator.end_training()


if __name__ == "__main__":
    main()
