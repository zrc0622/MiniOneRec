"""Freeze a current serial checkpoint and evaluate matched 256 validation rows."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

from run_new_history_interest_sft import (
    PROJECT, PYTHON, STEM, DATA, INDEX, NATIVE, OLD_DATA, OLD_INDEX,
    OLD_NATIVE, OLD_LATENT, CONTROLS, METHODS, canonical_result,
    validate_cache, report, gpu_snapshot, save, load_index, sha,
)
from run_new_sid_ranking_eval import validate_alignment


def freeze_latest(training, root):
    checkpoints = []
    for state_file in (training / 'sft').glob('checkpoint-*/trainer_state.json'):
        try:
            state = json.loads(state_file.read_text())
            if all((state_file.parent / n).is_file() for n in
                   ('model.safetensors', 'config.json', 'tokenizer.json')):
                checkpoints.append((state['global_step'], state_file.parent, state))
        except FileNotFoundError:
            continue
    step, checkpoint, state = max(checkpoints)
    frozen = root / f'checkpoint-{step}'
    frozen.mkdir()
    # Trainer may rotate the original directory; this link preserves the model inode.
    os.link(checkpoint / 'model.safetensors', frozen / 'model.safetensors')
    for source in checkpoint.iterdir():
        if source.is_file() and source.suffix in ('.json', '.txt', '.jinja'):
            shutil.copy2(source, frozen / source.name)
    return frozen, dict(original_checkpoint=str(checkpoint), step=step, epoch=state['epoch'],
                        reference_epoch=15, matched_training_epochs=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--gpus', default='4,5')
    p.add_argument('--seconds', type=int, default=1200)
    o = p.parse_args()
    training, root = Path(o.training_root).resolve(), Path(o.root).resolve()
    gpus = o.gpus.split(',')
    if len(gpus) != 2 or len(set(gpus)) != 2 or not all(g.isdigit() for g in gpus):
        raise ValueError('Require two distinct idle GPUs')
    active_training = json.loads((training / 'pipeline_status.json').read_text())
    if set(gpus) & set(active_training['gpus']):
        raise ValueError('Do not use active training GPUs')
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + o.seconds
    active, streams = [], []

    def status(phase, **extra):
        save(root / 'pipeline_status.json', dict(phase=phase, coordinator_pid=os.getpid(),
             gpus=gpus, elapsed_seconds=time.monotonic() - started, **extra))

    def interrupted(signum, frame):
        raise TimeoutError(f'Interim evaluator deadline/termination signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(o.seconds)
    try:
        status('preparing')
        frozen, progress = freeze_latest(training, root)
        snapshot = root / 'source_snapshot'
        for name in ('latent', 'interest'):
            shutil.copytree(training / 'source_snapshot' / name, snapshot / name,
                            ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('data.py', 'eval_metric_protocol.json'):
            shutil.copy2(training / 'source_snapshot' / name, snapshot / name)
        sample_file = root / 'sample.json'
        shutil.copy2(CONTROLS / 'sample256/sample.json', sample_file)
        sample = json.loads(sample_file.read_text())
        with (DATA / 'valid' / f'{STEM}.csv').open() as stream:
            source = list(csv.DictReader(stream))
        with (OLD_DATA / 'valid' / f'{STEM}.csv').open() as stream:
            old_source = list(csv.DictReader(stream))
        validate_alignment(old_source, source)
        index, old_index = load_index(INDEX), load_index(OLD_INDEX)
        results = {}
        for name, filename, model, kind in (
            ('old_history_latent', 'three_slot_vq_sft.json', OLD_LATENT, 'latent'),
            ('old_sid_no_latent', 'no_latent_sft.json', OLD_NATIVE, 'sid'),
        ):
            cached = json.loads((CONTROLS / 'sample256' / filename).read_text())
            validate_cache(cached, model, kind, sample_file,
                           OLD_DATA / 'valid' / f'{STEM}.csv', OLD_INDEX, snapshot)
            results[name] = canonical_result(cached, old_index, source, sample['row_ids'])
        save(root / 'manifest.json', dict(**progress, examples=len(sample['row_ids']),
             methods=METHODS, sample_sha256=sha(sample_file), gpus=gpus,
             data=str(DATA), reference=str(NATIVE), frozen_checkpoint=str(frozen),
             checkpoint_weights_sha256=sha(frozen / 'model.safetensors'),
             inference_uses_history_only=True, training_untouched=True,
             source_sha256=sha(snapshot / 'latent/eval_unified_small.py')))
        for check in range(3):
            measurements = gpu_snapshot(gpus)
            if not all(v['used_mib'] < 1024 and v['utilization'] < 10 and not v['pids']
                       for v in measurements.values()):
                raise RuntimeError('Selected evaluation GPU became occupied')
            if check < 2:
                time.sleep(5)
        env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
             TOKENIZERS_PARALLELISM='false', PYTHONPATH=str(snapshot) + ':' + str(PROJECT),
             PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        jobs = []
        for gpu, name, model, kind in (
            (gpus[0], 'new_history_latent', frozen, 'latent'),
            (gpus[1], 'new_sid_no_latent', NATIVE, 'sid'),
        ):
            command = [PYTHON, '-u', '-m', 'latent.eval_unified_small', '--model', str(model),
                 '--kind', kind, '--data', str(DATA / 'valid' / f'{STEM}.csv'),
                 '--index', str(INDEX), '--sample', str(sample_file),
                 '--output', str(root / f'{name}.json'), '--max-batch', '32']
            save(root / f'{name}.command.json', command)
            stream = (root / f'{name}.log').open('w')
            streams.append(stream)
            worker = subprocess.Popen(command, cwd=snapshot, env=dict(env, CUDA_VISIBLE_DEVICES=gpu),
                 stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            active.append(worker)
            jobs.append((name, model, kind, worker))
        status('evaluating', **progress, worker_pids={name: worker.pid for name, _, _, worker in jobs})
        for name, model, kind, worker in jobs:
            code = worker.wait(timeout=max(1, deadline - time.monotonic()))
            save(root / f'{name}.exit_code.json', dict(exit_code=code))
            if code:
                raise RuntimeError(f'{name} evaluator exited {code}')
            result = json.loads((root / f'{name}.json').read_text())
            validate_cache(result, model, kind, sample_file, DATA / 'valid' / f'{STEM}.csv', INDEX, snapshot)
            expected = METHODS if kind == 'latent' else ['direct50']
            if any(result['metadata']['completed'][m] != len(sample['row_ids']) for m in expected):
                raise ValueError('Evaluation output incomplete')
            results[name] = canonical_result(result, index, source, sample['row_ids'])
        report(root, sample, results)
        path = root / 'comparison.md'
        text = path.read_text()
        path.write_text(text + f'\n当前 latent 为中间 checkpoint {progress["step"]}，'
             f'{progress["epoch"]:.2f} 轮；基线为 15 轮。256 条验证仅作初步诊断，训练继续运行。\n')
        status('complete', **progress, comparison=str(path))
        print(path.read_text(), flush=True)
    except Exception:
        status('failed_or_deadline', error=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        for worker in active:
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGTERM)
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, signal.SIGKILL)
                    worker.wait()
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    main()
