"""Wait for a serial experiment and its eval, then recover parallel SFT on GPU2/3."""
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


def coordinator_alive(pid, root):
    try:
        process = Path(f'/proc/{int(pid)}')
        state = (process / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        command = (process / 'cmdline').read_bytes().split(b'\0')
        return state != 'Z' and str(root).encode() in command
    except (FileNotFoundError, ProcessLookupError):
        return False


def predecessor_complete(root, alive=coordinator_alive):
    state = json.loads((root / 'pipeline_status.json').read_text())
    phase = state['phase']
    if phase.startswith(('failed', 'interrupted')):
        raise RuntimeError('Predecessor failed or was interrupted; do not skip its evaluation')
    if phase != 'complete':
        if not alive(state.get('coordinator_pid', -1), root):
            raise RuntimeError('Predecessor coordinator exited without a complete result')
        return False
    for name in ('model.safetensors', 'config.json', 'tokenizer_config.json'):
        if not (root / 'sft/final_checkpoint' / name).is_file():
            raise ValueError('Predecessor final export incomplete')
    for scope, count in (('sample256', 256), ('valid20pct', 3232)):
        result = json.loads((root / 'eval' / scope / 'comparison.json').read_text())
        if result['examples'] != count or result.get('verified_same_ids_and_targets') is not True:
            raise ValueError('Predecessor ranking report incomplete or unverified')
        if set(result['results']['new_history_latent']) != {'single20', 'four20_marginal'}:
            raise ValueError('Predecessor has not finished both evaluation methods')
    return True


def idle_pair(measurements, gpus):
    return set(measurements) == set(gpus) and all(
        v['used_mib'] < 1024 and v['utilization'] < 10 and not v['pids']
        for v in measurements.values())


def recovery_command(root, snapshot, gpus):
    return [PYTHON, '-u', str(snapshot / 'run_parallel_recovery.py'), '--root', str(root),
            '--gpus', ','.join(gpus), '--resume-source', str(snapshot / 'parallel_train.py'),
            '--eval-runner', str(snapshot / 'run_parallel_interest_sft.py')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predecessor', required=True)
    p.add_argument('--parallel-root', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--gpus', default='2,3')
    p.add_argument('--poll-seconds', type=int, default=60)
    o = p.parse_args()
    previous, target, root = [Path(s).resolve() for s in (o.predecessor, o.parallel_root, o.root)]
    gpus = o.gpus.split(',')
    if gpus != ['2', '3'] or o.poll_seconds < 5:
        raise ValueError('This queue waits for the authorized GPU2/3 pair; polling must be >=5s')
    state = json.loads((previous / 'pipeline_status.json').read_text())
    if state['gpus'] != gpus or previous == target:
        raise ValueError('Predecessor GPU placement or target experiment differs')
    root.mkdir(parents=True, exist_ok=False)
    # A separate target lock prevents duplicate pending queues without blocking its future training lock.
    lock = (target / 'resume_queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot = root / 'source_snapshot'
    snapshot.mkdir()
    for source in (PROJECT / 'run_parallel_recovery.py', PROJECT / 'run_parallel_interest_sft.py',
                   PROJECT / 'latent/parallel_train.py'):
        shutil.copy2(source, snapshot / source.name)
    shutil.copy2(Path(__file__), snapshot / Path(__file__).name)
    command = recovery_command(target, snapshot, gpus)
    save(root / 'manifest.json', dict(predecessor=str(previous), parallel_root=str(target),
         gpus=gpus, checkpoint=str(target / 'sft/checkpoint-3174'), checkpoint_step=3174,
         wait_for='complete training AND both ranking evaluation scopes, then actual GPU release',
         poll_seconds=o.poll_seconds, periodic_disk_check=False,
         source_hashes={p.name: sha(p) for p in snapshot.iterdir() if p.is_file()}))
    save(root / 'recovery.command.json', command)
    (root / 'queue.pid').write_text(str(os.getpid()) + '\n')
    child = None

    def status(phase, **extra):
        save(root / 'pipeline_status.json', dict(phase=phase, updated_at=time.time(),
             coordinator_pid=os.getpid(), predecessor=str(previous), parallel_root=str(target),
             gpus=gpus, **extra))

    def stopped(signum, frame):
        raise RuntimeError(f'Queue coordinator received signal {signum}')

    signal.signal(signal.SIGTERM, stopped)
    try:
        while not predecessor_complete(previous):
            state = json.loads((previous / 'pipeline_status.json').read_text())
            status('waiting_for_predecessor', predecessor_phase=state['phase'])
            time.sleep(o.poll_seconds)
        ready = 0
        while ready < 3:
            values = gpu_snapshot(gpus)
            ready = ready + 1 if idle_pair(values, gpus) else 0
            status('waiting_for_gpu_release', measurements=values, consecutive_idle_checks=ready)
            if ready < 3:
                time.sleep(5 if ready else o.poll_seconds)
        # Recheck full completion immediately before launching the frozen recovery runner.
        if not predecessor_complete(previous):
            raise RuntimeError('Predecessor state changed before launch')
        env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
                   TOKENIZERS_PARALLELISM='false', PYTHONPATH=str(PROJECT))
        with (root / 'recovery.log').open('w') as stream:
            child = subprocess.Popen(command, cwd=PROJECT, env=env, stdin=subprocess.DEVNULL,
                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            status('running_parallel_recovery', worker_pid=child.pid)
            code = child.wait()
        save(root / 'recovery.exit_code.json', dict(exit_code=code))
        if code:
            raise RuntimeError(f'Parallel recovery/evaluation exited {code}')
        if json.loads((target / 'pipeline_status.json').read_text())['phase'] != 'complete':
            raise ValueError('Parallel runner exited without its final evaluation')
        status('complete', report=str(target / 'eval/valid20pct/comparison.md'))
    except Exception:
        status('failed_queue', error=traceback.format_exc())
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


if __name__ == '__main__':
    main()
