"""Run at most two four-GPU evaluation groups, optionally adopting a live run."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import psutil

from .eval_compare import compare


def option(command, name):
    return command[command.index(name) + 1]


def adopt_spec(launch):
    """Read and validate everything before sending any process a signal."""
    launch = Path(launch).resolve()
    record = json.loads((launch / 'run.json').read_text())
    root = Path(record['results_directory']).resolve()
    wrapper = psutil.Process(record['pid'])
    if launch.as_posix() not in wrapper.cmdline():
        raise ValueError('Launch record does not match running wrapper')
    workers = []
    for process in wrapper.children(recursive=True):
        command = process.cmdline()
        if 'latent.evaluate' in command and '--shard' in command:
            workers.append((int(option(command, '--shard')), process, command))
    workers.sort(key=lambda entry: entry[0])
    if [entry[0] for entry in workers] != [0, 1, 2, 3]:
        raise ValueError('Expected four running evaluation workers')
    common = []
    for shard, process, command in workers:
        if option(command, '--scheme') != 'joint_max':
            raise ValueError('Only a running first joint_max stage can be adopted')
        if Path(option(command, '--output')).resolve() != root / 'joint_max' / f'{shard}.json':
            raise ValueError('Worker output path differs from launch record')
        args = command[command.index('latent.evaluate') + 1:]
        filtered = []
        for i in range(0, len(args), 2):
            if args[i] not in ('--scheme', '--shard', '--output'):
                filtered.extend(args[i:i + 2])
        if common and common != filtered:
            raise ValueError('Worker settings differ')
        common = filtered
    parents = {process.ppid() for _, process, _ in workers}
    if len(parents) != 1:
        raise ValueError('Workers have different coordinators')
    coordinator = psutil.Process(parents.pop())
    if coordinator.ppid() != wrapper.pid or 'run_latent_eval_four.sh' not in coordinator.cmdline():
        raise ValueError('Unexpected coordinator; refusing takeover')
    return dict(root=root, launch=launch, record=record, common=common,
                coordinator=coordinator, workers=[p for _, p, _ in workers])


def alive(process):
    if isinstance(process, subprocess.Popen):
        return process.poll() is None
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def run(options):
    gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '4,5,6,7').split(',')
    if len(gpus) != 4 or len(set(gpus)) != 4 or any(not g for g in gpus):
        raise ValueError('Configure four different GPUs')
    active, completed = {}, []
    if options.adopt_run:
        spec = adopt_spec(options.adopt_run)
        if gpus != [str(g) for g in spec['record']['gpus']]:
            raise ValueError('GPU selection differs from the existing run')
        root, common = spec['root'], spec['common']
        pending = ['fixed_latent', 'hierarchical_max', 'hierarchical_sum']
        if any((root / name).exists() for name in pending):
            raise ValueError('Additional scheme directories already exist')
        if options.dry_run:
            print(json.dumps(dict(root=str(root), adopt_pids=[p.pid for p in spec['workers']],
                                  replace_coordinator=spec['coordinator'].pid, pending=pending)))
            return
        # SIGKILL deliberately bypasses the old shell's EXIT trap, which would
        # terminate its evaluation workers. Signal only the verified shell.
        spec['coordinator'].send_signal(signal.SIGKILL)
        spec['record']['superseded_by'] = str(root / 'parallel_state.json')
        (spec['launch'] / 'run.json').write_text(json.dumps(spec['record'], indent=2) + '\n')
        active['joint_max'] = dict(processes=spec['workers'], adopted=True,
            started=datetime.fromisoformat(spec['record']['started']).timestamp())
    else:
        base = Path(os.environ.get('LATENT_ROOT', 'outputs/latent_qwen3_0.6b_balanced15')).resolve()
        data = Path(os.environ.get('BALANCED_DATA_ROOT', 'data/Amazon23/variants/rqkmeans_balanced')).resolve()
        split = os.environ.get('LATENT_EVAL_SPLIT', 'test')
        if split not in ('test', 'valid'):
            raise ValueError('Use test or valid split')
        common = ['--model', os.environ.get('LATENT_EVAL_MODEL', str(base / 'sft/final_checkpoint')),
                  '--data', str(data / split / 'Industrial_and_Scientific_5_2016-10-2018-11.csv'),
                  '--index', str(data / 'Industrial_and_Scientific/Industrial_and_Scientific.index.json'),
                  '--shards', '4']
        for flag, variable, default in (
            ('beams', 'LATENT_EVAL_BEAMS', '200'), ('latent-count', 'LATENT_EVAL_LATENTS', '4'),
            ('latent-beams', 'LATENT_EVAL_LATENT_BEAMS', '4'),
            ('item-beams', 'LATENT_EVAL_ITEM_BEAMS', '50'),
            ('score-batch', 'LATENT_EVAL_SCORE_BATCH', '8')):
            common.extend(['--' + flag, os.environ.get(variable, default)])
        if os.environ.get('LATENT_EVAL_MAX_EXAMPLES'):
            common.extend(['--max-examples', os.environ['LATENT_EVAL_MAX_EXAMPLES']])
        if options.dry_run:
            print(json.dumps(dict(gpus=gpus, common=common, max_parallel=2)))
            return
        (base / 'eval').mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix=f'two_{split}.', dir=base / 'eval'))
        pending = ['joint_max', 'fixed_latent', 'hierarchical_max', 'hierarchical_sum']
    print(f'Results: {root}; GPUs: {gpus}; max concurrent schemes: 2', flush=True)

    def save_state():
        state = dict(pid=os.getpid(), max_parallel=2, gpus=gpus, pending=pending, completed=completed,
                     active={name: dict(pids=[p.pid for p in job['processes']], adopted=job['adopted'],
                                        started=job['started']) for name, job in active.items()})
        temporary = root / 'parallel_state.json.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(root / 'parallel_state.json')

    save_state()
    try:
        while pending or active:
            while pending and len(active) < 2:
                name = pending.pop(0)
                folder = root / name
                folder.mkdir()
                job = dict(processes=[], adopted=False, started=time.time())
                active[name] = job
                for shard, gpu in enumerate(gpus):
                    env = os.environ.copy()
                    env['CUDA_VISIBLE_DEVICES'] = gpu
                    with (folder / f'{shard}.log').open('w') as log:
                        job['processes'].append(subprocess.Popen(
                            [sys.executable, '-u', '-m', 'latent.evaluate', *common,
                             '--scheme', name, '--shard', str(shard), '--output', str(folder / f'{shard}.json')],
                            env=env, stdout=log, stderr=subprocess.STDOUT))
                print(f'Started {name}', flush=True)
                save_state()
            for name, job in list(active.items()):
                if not job['adopted'] and any(p.poll() not in (None, 0) for p in job['processes']):
                    raise RuntimeError(f'{name} failed; inspect shard logs in {root / name}')
                if any(alive(p) for p in job['processes']):
                    continue
                folder = root / name
                paths = [str(folder / f'{i}.json') for i in range(4)]
                if not all(Path(p).is_file() for p in paths):
                    raise RuntimeError(f'{name} missing output shards; inspect {folder}')
                with (folder / 'merge.log').open('w') as log:
                    subprocess.run([sys.executable, '-m', 'latent.evaluate', '--merge', *paths,
                                    '--output', str(folder / 'merged.json')], stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
                (folder / 'wall_seconds.txt').write_text(str(time.time() - job['started']) + '\n')
                completed.append(name)
                del active[name]
                compare(root)
                save_state()
                print(f'Completed {name}', flush=True)
            if active:
                time.sleep(5)
        print(f'All four schemes completed: {root / "comparison.md"}', flush=True)
    except BaseException:
        # Let an adopted long-running job retain its progress if a newly
        # started job fails. New subprocesses are owned by this supervisor.
        for job in active.values():
            if not job['adopted']:
                for process in job['processes']:
                    if alive(process):
                        process.terminate()
        save_state()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adopt-run', help='Launch directory of a live sequential first stage')
    parser.add_argument('--dry-run', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
