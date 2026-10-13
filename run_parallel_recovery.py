"""Recover an interrupted two-card parallel experiment with original training state."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import traceback
import zipfile

from run_two_slot_vq_sft import PROJECT, PYTHON, gpu_snapshot, save
from latent.core import sha

OWNED_WORKERS = []


def resume_command(original, checkpoint, port):
    if original[1:3] != ['-m', 'torch.distributed.run'] or 'latent.parallel_train' not in original:
        raise ValueError('Not the original parallel training command')
    stage = original.index('train')
    return [PYTHON, '-m', 'torch.distributed.run', '--nnodes=1', '--nproc_per_node=2',
            '--rdzv_backend=static', '--master_addr=127.0.0.1', f'--master_port={port}',
            '-m', 'latent.parallel_train'] + original[stage:] + ['--resume_from_checkpoint', str(checkpoint)]


def evaluation_command(root, runner, gpus):
    return [PYTHON, '-u', str(Path(runner).resolve()), '--root', str(root),
            '--gpus', ','.join(gpus), '--eval-only']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--gpus', default='2,5')
    p.add_argument('--resume-source', default=str(PROJECT / 'latent/parallel_train.py'))
    p.add_argument('--eval-runner', default=str(PROJECT / 'run_parallel_interest_sft.py'))
    o = p.parse_args()
    gpus = o.gpus.split(',')
    if len(gpus) != 2 or len(set(gpus)) != 2 or not all(g.isdigit() for g in gpus):
        raise ValueError('Require two distinct GPU indices')
    root = Path(o.root).resolve()
    lock = (root / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    recovery = root / 'recovery_20261013'
    recovery.mkdir(exist_ok=False)
    for name in ('pipeline_status.json', 'sft.command.json', 'sft.exit_code'):
        shutil.copy2(root / name, recovery / name)
    snapshot = recovery / 'source_snapshot'
    for name, digest in json.loads((root / 'source_snapshot/hashes.json').read_text()).items():
        if sha(root / 'source_snapshot' / name) != digest:
            raise ValueError('Original source snapshot changed: ' + name)
    shutil.copytree(root / 'source_snapshot', snapshot, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(o.resume_source, snapshot / 'latent/parallel_train.py')
    save(recovery / 'source_hashes.json', {str(f.relative_to(snapshot)): sha(f)
         for f in snapshot.rglob('*') if f.is_file()})
    options = json.loads((root / 'sft/parallel_run.json').read_text())['options']
    checkpoint = root / 'sft/checkpoint-3174'
    from latent.parallel_train import validate_resume
    prepared = json.loads((Path(options['prepared']) / 'manifest.json').read_text())
    validate_resume(argparse.Namespace(**dict(options, resume_from_checkpoint=str(checkpoint))), root / 'sft', prepared)
    for path, digest in prepared['source_hashes'].items():
        if sha(path) != digest:
            raise ValueError('Original input has changed: ' + path)

    def status(phase, **extra):
        value = dict(phase=phase, updated_at=time.time(), coordinator_pid=os.getpid(),
                     gpus=gpus, resumed_checkpoint_step=3174, periodic_disk_check=False, **extra)
        save(root / 'pipeline_status.json', value)
        print(json.dumps(value), flush=True)

    status('verifying_parallel_checkpoint')
    checks = {}
    for f in [checkpoint / 'scheduler.pt', checkpoint / 'rng_state_0.pth', checkpoint / 'rng_state_1.pth',
              *checkpoint.glob('global_step3174/*.pt')]:
        with zipfile.ZipFile(f) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError('Corrupt archive member: ' + bad)
        checks[str(f.relative_to(checkpoint))] = 'zip_crc_pass'
    with (checkpoint / 'model.safetensors').open('rb') as stream:
        size = int.from_bytes(stream.read(8), 'little')
        header = json.loads(stream.read(size))
        body = (checkpoint / 'model.safetensors').stat().st_size - size - 8
        for name, value in header.items():
            if name != '__metadata__' and not 0 <= value['data_offsets'][0] <= value['data_offsets'][1] <= body:
                raise ValueError('Invalid safetensors offset')
    checks['model.safetensors'] = 'header_and_all_offsets_pass'
    save(recovery / 'checkpoint_integrity.json', checks)
    # Only a startup storage check; no recurring disk polling.
    if shutil.disk_usage(root).free < 100 * 2**30:
        raise RuntimeError('Insufficient storage before resumed training')
    ready = 0
    while ready < 3:
        values = gpu_snapshot(gpus)
        ready = ready + 1 if all(v['used_mib'] < 1024 and v['utilization'] < 10 and not v['pids'] for v in values.values()) else 0
        status('waiting_for_idle_gpus', measurements=values, consecutive_idle_checks=ready)
        if ready < 3:
            time.sleep(5 if ready else 30)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    original = json.loads((root / 'sft.command.json').read_text())
    command = resume_command(original, checkpoint, port)
    save(recovery / 'resume_manifest.json', dict(checkpoint=str(checkpoint), step=3174,
         max_steps=4140, original_options=options, source_hashes_verified=True,
         launcher='single-node static rendezvous; no elastic heartbeat',
         attention_and_loss_unchanged=True, global_batch=1024, batch_per_rank=32, accumulation=16,
         gpus=gpus, resume_source_sha256=sha(o.resume_source), eval_runner_sha256=sha(o.eval_runner)))
    save(recovery / 'sft.command.json', command)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=o.gpus, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false', NCCL_IB_DISABLE='1', TMPDIR=str(root / 'tmp'),
        PYTHONPATH=str(snapshot) + ':' + str(PROJECT), PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env['PATH'] = str(Path(PYTHON).parent) + ':' + env.get('PATH', '')
    env['CUDA_HOME'] = str(Path(PYTHON).parent.parent)
    (root / 'pipeline.pid').write_text(str(os.getpid()) + '\n')
    with (recovery / 'sft.log').open('w') as log:
        worker = subprocess.Popen(command, cwd=snapshot, env=env, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        OWNED_WORKERS.append(worker)
        save(recovery / 'worker.json', dict(pid=worker.pid))
        status('training_parallel_sft_resumed', worker_pid=worker.pid, log=str(recovery / 'sft.log'))
        code = worker.wait()
        OWNED_WORKERS.remove(worker)
    save(recovery / 'sft.exit_code.json', dict(exit_code=code))
    if code:
        raise RuntimeError('Recovered parallel SFT failed; inspect recovery log')
    if not (root / 'sft/final_checkpoint/model.safetensors').is_file():
        raise ValueError('Training exited without final export')
    # The evaluator takes the same experiment lock, after this coordinator releases it.
    lock.close()
    evaluation = evaluation_command(root, o.eval_runner, gpus)
    save(recovery / 'eval.command.json', evaluation)
    with (recovery / 'eval.log').open('w') as log:
        worker = subprocess.Popen(evaluation, cwd=PROJECT, env=dict(env, PYTHONPATH=str(PROJECT)),
                                  stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        OWNED_WORKERS.append(worker)
        code = worker.wait()
        OWNED_WORKERS.remove(worker)
    save(recovery / 'eval.exit_code.json', dict(exit_code=code))
    if code:
        raise RuntimeError('Recovered model evaluation failed')


if __name__ == '__main__':
    def stopped(signum, frame):
        raise RuntimeError(f'Recovery coordinator received signal {signum}')
    signal.signal(signal.SIGTERM, stopped)
    try:
        main()
    except Exception:
        traceback.print_exc()
        import sys
        if '--root' in sys.argv:
            root = Path(sys.argv[sys.argv.index('--root') + 1]).resolve()
            save(root / 'pipeline_status.json', dict(phase='failed_recovery', updated_at=time.time(),
                 error=traceback.format_exc()))
        raise
    finally:
        for worker in OWNED_WORKERS:
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGTERM)
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, signal.SIGKILL)
                    worker.wait()
