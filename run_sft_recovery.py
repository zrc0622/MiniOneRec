#!/usr/bin/env python3
"""Recover one interrupted SFT using its original snapshot and training state."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

from run_two_slot_vq_sft import PROJECT, PYTHON, gpu_snapshot, save
from latent.core import sha


def checkpoint_state(root, checkpoint):
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.parent != (root / 'sft').resolve():
        raise ValueError('Checkpoint must belong to this interrupted experiment')
    for name in ('config.json', 'tokenizer_config.json', 'model.safetensors', 'trainer_state.json', 'scheduler.pt'):
        if not (checkpoint / name).is_file():
            raise ValueError(f'Missing recovery file: {name}')
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    if state['max_steps'] != 4140 or state['global_step'] != int(checkpoint.name.split('-')[-1]):
        raise ValueError('Checkpoint progress differs from the original 15-epoch run')
    folder = checkpoint / f'global_step{state["global_step"]}'
    for name in ('bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt',
                 'bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt', 'mp_rank_00_model_states.pt'):
        if not (folder / name).is_file():
            raise ValueError('Both original rank optimizer states are required')
    return state


def recovery_command(root, snapshot, checkpoint, kind):
    command = json.loads((root / 'sft.command.json').read_text())
    if kind == 'native':
        if not any(Path(s).name == 'sft.py' for s in command):
            raise ValueError('Not an original native SFT command')
        command = [str(snapshot / 'sft.py') if Path(s).name == 'sft.py' else s for s in command]
    elif kind == 'target':
        if 'latent.train' not in command or command[command.index('latent.train') + 1] != 'sft':
            raise ValueError('Not an original latent SFT command')
    else:
        raise ValueError('Unknown experiment kind')
    if '--resume_from_checkpoint' in command:
        raise ValueError('Original command already contains a resume option')
    return command + ['--resume_from_checkpoint', str(checkpoint)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--kind', required=True, choices=['native', 'target'])
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--monitor-disk', action='store_true', help='Optional periodic disk guard; disabled by default')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    gpus = args.gpus.split(',')
    if len(gpus) != 2 or len(set(gpus)) != 2 or not set(gpus) <= {'4', '5', '6', '7'}:
        raise ValueError('Recovery uses two distinct user-authorized GPUs from 4–7')
    lock = (root / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'sft/final_checkpoint/config.json').exists():
        raise ValueError('Experiment already completed; do not retrain')
    checkpoint = Path(args.checkpoint).resolve()
    state = checkpoint_state(root, checkpoint)
    original = root / 'source_snapshot'
    for relative, digest in json.loads((original / 'hashes.json').read_text()).items():
        if sha(original / relative) != digest:
            raise ValueError(f'Original source snapshot changed: {relative}')
    config = json.loads((root / 'pipeline_config.json').read_text())
    if args.kind == 'target':
        from run_target_interest_sft import check_labels
        check_labels()
        if sha(Path(config['labels']) / 'labels.json') != config['labels_sha256']:
            raise ValueError('Original interest labels changed')
    else:
        from prepare_qwen_official_sid import check_experiment, RECIPE_FILE
        dataset = Path(config['dataset'])
        check_experiment(dataset)
        if sha(dataset / RECIPE_FILE) != config['sid_recipe_sha256']:
            raise ValueError('Original item SID changed')
    attempt = root / 'recovery_20261012'
    attempt.mkdir(exist_ok=False)
    for name in ('pipeline_status.json', 'latent_run.json', 'sft.command.json'):
        source = root / ('sft/latent_run.json' if name == 'latent_run.json' else name)
        if source.exists():
            shutil.copy2(source, attempt / name)
    snapshot = attempt / 'source_snapshot'
    shutil.copytree(original, snapshot, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
    if args.kind == 'target':
        shutil.copy2(PROJECT / 'latent/train.py', snapshot / 'latent/train.py')
    shutil.copy2(Path(__file__), attempt / Path(__file__).name)
    save(attempt / 'recovery_source_hashes.json', {str(p.relative_to(snapshot)): sha(p)
         for p in snapshot.rglob('*') if p.is_file() and not p.is_symlink()})
    command = recovery_command(root, snapshot, checkpoint, args.kind)
    save(attempt / 'sft.command.json', command)
    save(attempt / 'resume_manifest.json', dict(original_checkpoint=str(checkpoint),
         resume_step=state['global_step'], max_steps=state['max_steps'], gpus=gpus,
         batch_per_rank=32, accumulation=16, global_batch=1024,
         original_settings_preserved=True, original_snapshot_preserved=True))
    (root / 'pipeline.pid').write_text(str(os.getpid()) + '\n')

    def status(phase, **details):
        value = dict(phase=phase, updated_at=time.time(), coordinator_pid=os.getpid(),
                     gpus=gpus, recovery_directory=str(attempt), resumed_from_step=state['global_step'], **details)
        save(root / 'pipeline_status.json', value)
        print(json.dumps(value), flush=True)

    consecutive = 0
    while consecutive < 3:
        free = shutil.disk_usage(root).free
        if free < 100 * 2**30:
            raise RuntimeError('Require 100 GiB free output storage before recovery')
        measurements = gpu_snapshot(gpus)
        idle = all(v['used_mib'] < 1024 and v['utilization'] < 10 and not v['pids'] for v in measurements.values())
        if not idle:
            raise RuntimeError('Selected GPU pair became occupied; recovery deferred')
        consecutive += 1
        status('recovery_preflight', consecutive_idle_checks=consecutive, free_disk_GiB=free / 2**30)
        if consecutive < 3:
            time.sleep(5)
    temp = attempt / 'tmp'
    temp.mkdir()
    probe = temp / 'write_probe'
    with probe.open('wb') as stream:
        stream.write(b'\0' * 2**20)
        stream.flush()
        os.fsync(stream.fileno())
    probe.unlink()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, PYTHONUNBUFFERED='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', NCCL_IB_DISABLE='1',
               PYTHONPATH=str(snapshot) + ':' + str(PROJECT), TMPDIR=str(temp),
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env['PATH'] = str(Path(PYTHON).parent) + ':' + env.get('PATH', '')
    if not Path(env.get('CUDA_HOME', '') + '/bin/nvcc').is_file():
        env['CUDA_HOME'] = str(Path(PYTHON).parent.parent)
    with (attempt / 'sft.log').open('w') as log:
        worker = subprocess.Popen(command, cwd=snapshot, env=env, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (root / 'sft.pid').write_text(str(worker.pid) + '\n')
        (attempt / 'sft.pid').write_text(str(worker.pid) + '\n')
        try:
            while worker.poll() is None:
                free = shutil.disk_usage(root).free if args.monitor_disk else None
                status('training_sft_recovery', worker_pid=worker.pid,
                       periodic_disk_check=args.monitor_disk, free_disk_GiB=free / 2**30 if free is not None else None,
                       batch_per_device=32, accumulation=16, global_batch=1024)
                if args.monitor_disk and free < 100 * 2**30:
                    raise RuntimeError('Free storage fell below 100 GiB; preserve latest checkpoint and stop')
                time.sleep(30)
            code = worker.returncode
        finally:
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGTERM)
                try:
                    worker.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, signal.SIGKILL)
                    worker.wait()
    (attempt / 'sft.exit_code').write_text(str(code) + '\n')
    if code:
        raise RuntimeError(f'Recovery failed exit={code}; inspect {attempt / "sft.log"}')
    final = root / 'sft/final_checkpoint'
    for name in ('config.json', 'tokenizer_config.json', 'model.safetensors'):
        if not (final / name).is_file():
            raise ValueError('Recovery did not export complete final weights')
    save(root / 'sft_complete.json', dict(checkpoint=str(final), completed_at=time.time(),
         recovered_from=str(checkpoint), exit_code=0, config=config))
    if args.kind == 'target':
        status('starting_ranking_eval', methods=['single20', 'four20_marginal'])
        # Release the same pipeline lock before the existing eval coordinator takes it.
        fcntl.flock(lock, fcntl.LOCK_UN)
        command = [PYTHON, str(PROJECT / 'run_target_interest_sft.py'), '--root', str(root),
                   '--gpus', args.gpus, '--eval-only']
        with (attempt / 'eval_pipeline.log').open('w') as log:
            subprocess.run(command, cwd=PROJECT, env=env, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    else:
        status('complete_sft_only', checkpoint=str(final))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        import sys
        if '--root' in sys.argv:
            root = Path(sys.argv[sys.argv.index('--root') + 1]).resolve()
            save(root / 'pipeline_status.json', dict(phase='failed_recovery',
                 updated_at=time.time(), error=traceback.format_exc()))
        raise
