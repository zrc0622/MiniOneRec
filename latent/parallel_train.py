"""Prepare once, measure two-rank batch speed, then train the controlled parallel SFT."""
import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, TrainingArguments, set_seed

from .core import initialize, load_index, load_labels, sha, write_json
from .data import sft_datasets
from .parallel import initialize_queries, install_attention, ParallelSFTTrainer
from .sft_benchmark import Timing
from .trainer import InterestCollator


def validate_resume(o, output, prepared_metadata):
    if o.stage != 'train':
        raise ValueError('Only production parallel SFT supports checkpoint recovery')
    checkpoint = Path(o.resume_from_checkpoint).resolve()
    if checkpoint.parent != output.resolve():
        raise ValueError('Resume checkpoint must belong to this training output')
    for name in ('trainer_state.json', 'model.safetensors', 'scheduler.pt', 'rng_state_0.pth', 'rng_state_1.pth'):
        if not (checkpoint / name).is_file():
            raise ValueError(f'Incomplete checkpoint: missing {name}')
    for rank in (0, 1):
        if not list(checkpoint.glob(f'global_step*/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt')):
            raise ValueError(f'Missing optimizer state for rank{rank}')
    original = json.loads((output / 'parallel_run.json').read_text())
    options = dict(vars(o))
    options.pop('resume_from_checkpoint', None)
    expected = dict(original['options'])
    expected.pop('resume_from_checkpoint', None)
    if options != expected or prepared_metadata != original['prepared_metadata']:
        raise ValueError('Recovery must retain all original options and prepared inputs')
    config = json.loads((checkpoint / 'config.json').read_text())
    if config['latent_protocol'] != original['protocol']:
        raise ValueError('Resume checkpoint has a different parallel protocol')
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    if checkpoint.name != f'checkpoint-{state["global_step"]}':
        raise ValueError('Checkpoint step and Trainer state disagree')
    return checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'benchmark', 'train'])
    for key in ('model', 'labels', 'train_file', 'eval_file', 'items', 'index', 'prepared', 'output_dir'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--eval_batch', type=int, default=16)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--latent_weight', type=float, default=1.)
    p.add_argument('--benchmark_steps', type=int, default=5)
    p.add_argument('--deepspeed', default='config/sft_zero2.json')
    p.add_argument('--resume_from_checkpoint')
    o = p.parse_args()
    torch.set_num_threads(1)
    set_seed(o.seed)
    prepared, output = Path(o.prepared), Path(o.output_dir)
    files = [o.labels + '/labels.json', o.train_file, o.eval_file, o.items, o.index,
             o.model + '/config.json', o.model + '/tokenizer.json']
    hashes = {str(Path(f).resolve()): sha(f) for f in files}
    if o.stage == 'prepare':
        if o.resume_from_checkpoint:
            raise ValueError('Preparation cannot resume training checkpoints')
        if prepared.exists():
            raise ValueError('Prepared data directory must be new')
        index = load_index(o.index)
        _, manifest = load_labels(o.labels, 'train', o.train_file, index)
        recipe = manifest['embedding_recipe']
        if (manifest['k'] != 3 or manifest['vq_options'].get('balance_epsilon', 0)
                or recipe.get('contains_target', False) or recipe.get('normalization', 'none') != 'none'):
            raise ValueError('Use original history-only, nonbalanced three-slot labels')
        tk = AutoTokenizer.from_pretrained(o.model)
        tk.pad_token = tk.eos_token
        tk.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(o.model, torch_dtype=torch.bfloat16)
        protocol = initialize(model, tk, index, manifest)
        initialize_queries(model, tk)
        train, valid = sft_datasets(o.train_file, o.eval_file, o.items, o.index, index, tk, protocol, o.labels, o.seed)
        prepared.mkdir()
        Dataset.from_list(train).shuffle(seed=o.seed).save_to_disk(str(prepared / 'train'))
        Dataset.from_list(valid).shuffle(seed=o.seed).save_to_disk(str(prepared / 'valid'))
        # This artifact is freshly initialized, never optimized or selected using labels.
        model.save_pretrained(prepared / 'initialized_model')
        tk.save_pretrained(prepared / 'initialized_model')
        write_json(prepared / 'manifest.json', dict(source_hashes=hashes, seed=o.seed, protocol=protocol,
             train_rows=len(train), validation_rows=len(valid), initialized_only=True,
             history_budget='same exact retained history as original512-token SFT; +3 fixed queries',
             max_input_tokens=max(len(x['input_ids']) for x in train)))
        print('PREPARED', len(train), len(valid), flush=True)
        return
    meta = json.loads((prepared / 'manifest.json').read_text())
    if meta['source_hashes'] != hashes or meta['seed'] != o.seed or not meta['initialized_only']:
        raise ValueError('Prepared initialization/data belongs to different inputs')
    resume = validate_resume(o, output, meta) if o.resume_from_checkpoint else None
    if resume is None and output.exists() and any(output.iterdir()):
        raise ValueError('Training/benchmark output must be empty')
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world != 2 or o.batch not in (16, 32):
        raise ValueError('Production comparison requires two ranks with batch16 or32')
    accumulation = 1024 // (world * o.batch)
    tk = AutoTokenizer.from_pretrained(prepared / 'initialized_model')
    tk.pad_token = tk.eos_token
    tk.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(prepared / 'initialized_model', torch_dtype=torch.bfloat16)
    install_attention(model, tk)
    model.config.use_cache = False
    train, valid = load_from_disk(str(prepared / 'train')), load_from_disk(str(prepared / 'valid'))
    benchmark = o.stage == 'benchmark'
    args = TrainingArguments(output_dir=str(output), per_device_train_batch_size=o.batch,
        per_device_eval_batch_size=o.eval_batch, gradient_accumulation_steps=accumulation,
        num_train_epochs=o.epochs, max_steps=o.benchmark_steps if benchmark else -1,
        bf16=True, deepspeed=o.deepspeed, learning_rate=o.lr, warmup_steps=20,
        lr_scheduler_type='linear', optim='adamw_torch', logging_steps=1,
        seed=o.seed, remove_unused_columns=False, prediction_loss_only=True,
        ddp_find_unused_parameters=False, report_to=[] if benchmark else 'tensorboard',
        logging_dir=str(output / 'tensorboard'),
        eval_strategy='no' if benchmark else 'steps', save_strategy='no' if benchmark else 'steps',
        eval_steps=.5 / o.epochs, save_steps=.5 / o.epochs, save_total_limit=1,
        load_best_model_at_end=not benchmark, metric_for_best_model='eval_loss')
    timing = Timing() if benchmark else None
    trainer = ParallelSFTTrainer(model=model, args=args, processing_class=tk, latent_weight=o.latent_weight,
        train_dataset=train, eval_dataset=valid, data_collator=InterestCollator(tk),
        callbacks=[timing] if benchmark else [EarlyStoppingCallback(early_stopping_patience=3)])
    if trainer.is_world_process_zero():
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / ('parallel_resume_run.json' if resume else 'parallel_run.json'), dict(options=vars(o), training_args=args.to_dict(),
                   prepared_metadata=meta, protocol=model.config.latent_protocol,
                   original_attention_backend=model.config._attn_implementation,
                   attention_change='only three-query visibility; original Qwen attention kernels'))
    torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    if benchmark:
        durations = timing.durations[2:]
        if not durations:
            raise ValueError('Benchmark needs at least three post-warmup measurements')
        values = torch.tensor([sum(durations) / len(durations),
             torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30],
             device=trainer.accelerator.device)
        ranks = trainer.accelerator.gather(values).reshape(world, 3).cpu().tolist()
        if trainer.is_world_process_zero():
            write_json(output / 'benchmark.json', dict(batch_per_device=o.batch, accumulation=accumulation,
                 global_batch=1024, step_seconds=max(r[0] for r in ranks), rank_measurements=ranks,
                 local_step_durations=timing.durations, trained_weights_reused=False))
    else:
        trainer.save_model(str(output / 'final_checkpoint'))
        if trainer.is_world_process_zero():
            tk.save_pretrained(output / 'final_checkpoint')
    trainer.accelerator.wait_for_everyone()
    trainer.accelerator.end_training()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':main()
