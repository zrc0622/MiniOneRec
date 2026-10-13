#!/usr/bin/env python3
"""Replace a disk-monitoring coordinator while leaving its training workers alive."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from run_two_slot_vq_sft import PROJECT, PYTHON, save


def worker_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--kind', required=True, choices=['native', 'target'])
    args = parser.parse_args()
    root = Path(args.root).resolve()
    old = json.loads((root / 'pipeline_status.json').read_text())
    if old['phase'] != 'training_sft_recovery':
        raise ValueError('Only a running recovery coordinator may be replaced')
    attempt = Path(old['recovery_directory'])
    worker_pid = old['worker_pid']
    previous_pid = old['coordinator_pid']
    actual = Path(f'/proc/{worker_pid}/cmdline').read_bytes().split(b'\0')[:-1]
    expected = json.loads((attempt / 'sft.command.json').read_text())
    if [p.decode() for p in actual] != expected:
        raise ValueError('Worker identity differs from the authorized recovery command')
    coordinator = Path(f'/proc/{previous_pid}/cmdline').read_bytes().split(b'\0')[:-1]
    if (str(PROJECT / 'run_sft_recovery.py').encode() not in coordinator
            or str(root).encode() not in coordinator or args.kind.encode() not in coordinator):
        raise ValueError('Old coordinator identity is not this experiment')
    # Process start time pins the worker identity even if its PID is later reused.
    identity = worker_identity(worker_pid)
    if identity is None:
        raise ValueError('Worker has already exited')
    save(attempt / 'coordinator_replacement.json', dict(previous_coordinator_pid=previous_pid,
         replacement_coordinator_pid=os.getpid(), unchanged_worker_pid=worker_pid,
         periodic_disk_check=False, reason='User requested no periodic disk checks'))
    # SIGTERM has its default action in the old coordinator; the worker has its own
    # session and survives. Do not signal the torchrun process group.
    os.kill(previous_pid, signal.SIGTERM)
    lock = (root / 'pipeline.lock').open('a')
    deadline = time.monotonic() + 15
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() > deadline:
                raise RuntimeError('Old coordinator did not release its pipeline lock')
            time.sleep(.1)
    (root / 'pipeline.pid').write_text(str(os.getpid()) + '\n')

    def status(phase, **details):
        value = dict(old, phase=phase, updated_at=time.time(), coordinator_pid=os.getpid(),
                     periodic_disk_check=False, free_disk_GiB=None, **details)
        save(root / 'pipeline_status.json', value)
        print(json.dumps(value), flush=True)

    while True:
        status('training_sft_recovery')
        if worker_identity(worker_pid) != identity:
            break
        time.sleep(30)
    final = root / 'sft/final_checkpoint'
    for name in ('config.json', 'tokenizer_config.json', 'model.safetensors'):
        if not (final / name).is_file():
            raise RuntimeError('Recovered worker exited without complete final weights; inspect sft.log')
    save(root / 'sft_complete.json', dict(checkpoint=str(final), completed_at=time.time(),
         recovered_from_step=old['resumed_from_step'], worker_exit_code=None,
         completion_verified='worker process exited and final checkpoint files exist',
         config=json.loads((root / 'pipeline_config.json').read_text())))
    if args.kind == 'target':
        status('starting_ranking_eval', methods=['single20', 'four20_marginal'])
        fcntl.flock(lock, fcntl.LOCK_UN)
        command = [PYTHON, str(PROJECT / 'run_target_interest_sft.py'), '--root', str(root),
                   '--gpus', ','.join(old['gpus']), '--eval-only']
        with (attempt / 'eval_pipeline.log').open('w') as log:
            subprocess.run(command, cwd=PROJECT, stdin=subprocess.DEVNULL,
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
