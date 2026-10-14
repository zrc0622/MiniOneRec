"""Isolated full optimizer-update benchmark; no checkpoints or formal restart."""
import json
import math
import os
from pathlib import Path
import time

import torch
from transformers import TrainerCallback
from . import train as entry
from . import trainer as original
from . import rl_fast


variant = os.environ['LATENT_SPEED_VARIANT']
if variant not in ('old', 'vectorized', 'cached'):
    raise ValueError(variant)
timings = {}


def measure(name, function):
    def wrapped(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = function(*args, **kwargs)
        torch.cuda.synchronize()
        timings[name] = timings.get(name, 0.) + time.perf_counter() - start
        return result
    return wrapped


original.rollout = measure('rollout_seconds', original.rollout)
rl_fast.cached_rollouts = measure('rollout_seconds', rl_fast.cached_rollouts)
base = rl_fast.FastLatentGRPOTrainer if variant == 'cached' else original.LatentGRPOTrainer


class Profile(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings.clear()
        self.started = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        row = dict(step=state.global_step, seconds=time.perf_counter() - self.started,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            rank=int(os.environ['RANK']), variant=variant, components=dict(timings))
        root = Path(os.environ['LATENT_SPEED_DEST'])
        with (root / f'rank{row["rank"]}.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        control.should_save = False


class BenchTrainer(base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if variant == 'vectorized':
            self.logps = rl_fast.vectorized_logps
        logps = self.logps

        def timed_logps(model, packed, temperature):
            key = 'policy_forward_seconds' if torch.is_grad_enabled() else 'reference_forward_seconds'
            return measure(key, logps)(model, packed, temperature)

        self.logps = timed_logps
        self.add_callback(Profile())
        self._save_checkpoint = lambda *a, **k: None
        self.save_model = lambda *a, **k: None

    def train(self, *args, **kwargs):
        result = super().train(*args, **kwargs)
        checked = [{k: row[k] for k in ('loss', 'grad_norm', 'kl') if k in row}
                   for row in self.state.log_history if 'loss' in row]
        if not checked or any(not math.isfinite(value) for row in checked for value in row.values()):
            raise ValueError('Nonfinite training statistics')
        evidence = dict(status='pass', statistics=checked, steps=self.state.global_step,
            prompt_task_rows=len(self.train_dataset.records), batch=self.args.per_device_train_batch_size,
            accumulation=self.args.gradient_accumulation_steps, world_size=self.accelerator.num_processes)
        dest = Path(os.environ['LATENT_SPEED_DEST']) / f'checks.rank{os.environ["RANK"]}.json'
        dest.write_text(json.dumps(evidence, indent=2) + '\n')
        return result


entry.LatentGRPOTrainer = BenchTrainer
if __name__ == '__main__':
    entry.main()
