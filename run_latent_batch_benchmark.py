"""Matched fast-latent32/64 benchmark; keep formal jobs and native code unchanged."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import statistics
import subprocess
import time
import traceback

from run_two_slot_vq_sft import PROJECT, PYTHON, gpu_snapshot, save
from latent.rl_launch import check_gpu_capacity
from latent.core import sha


def batch_command(base, batch, destination, port, steps):
    if batch not in (32, 64) or steps < 5:
        raise ValueError('Use32/64 and at least2warmup+3measured updates')
    command = list(base)
    if command.count('latent.rl_speed_benchmark') != 1:
        raise ValueError('Require the isolated benchmark entrypoint')
    for key, value in {'--batch': str(batch), '--gradient_accumulation': str(1024 // (2 * batch)),
                       '--max_steps': str(steps), '--output_dir': str(destination / 'model'),
                       '--eval_steps': '1000', '--save_steps': '1000'}.items():
        command[command.index(key) + 1] = value
    return [f'--master_port={port}' if s.startswith('--master_port=') else s for s in command]


def summarize(destination, steps, batch):
    ranks = [[json.loads(s) for s in (destination / f'rank{r}.jsonl').read_text().splitlines()] for r in (0, 1)]
    for rank, rows in enumerate(ranks):
        if [r['step'] for r in rows] != list(range(1, steps + 1)):
            raise ValueError('Incomplete optimizer update timing')
        checks = json.loads((destination / f'checks.rank{rank}.json').read_text())
        if (checks['status'] != 'pass' or checks['steps'] != steps or checks['prompt_task_rows'] != 157679
                or checks['batch'] != batch or checks['accumulation'] != 1024 // (2 * batch) or checks['world_size'] != 2):
            raise ValueError('Training evidence does not match fixedglobal1024/full data')
        if not any(row.get('grad_norm', 0) > 0 for row in checks['statistics']):
            raise ValueError('No actual gradient updates')
    times = [max(ranks[0][i]['seconds'], ranks[1][i]['seconds']) for i in range(2, steps)]
    if not all(math.isfinite(t) and t > 0 for t in times):
        raise ValueError('Nonfinite timing')
    if list(destination.rglob('*.safetensors')):
        raise ValueError('Benchmark unexpectedly saved model weights')
    return dict(steady_seconds=times, mean_seconds=statistics.mean(times), median_seconds=statistics.median(times),
        components={k: statistics.mean(max(ranks[0][i]['components'].get(k, 0), ranks[1][i]['components'].get(k, 0))
            for i in range(2, steps)) for k in ('rollout_seconds', 'reference_forward_seconds', 'policy_forward_seconds')},
        peak_allocated_gib=max(row['peak_allocated_gib'] for rows in ranks for row in rows),
        peak_reserved_gib=max(row['peak_reserved_gib'] for rows in ranks for row in rows),
        global_candidates=1024, per_device_batch=batch, accumulation=1024 // (2 * batch),
        full_train_rows=157679, finite_statistics_nonzero_gradients=True, exit_code=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--verified-pilot', required=True)
    parser.add_argument('--gpus', default='1,3')
    parser.add_argument('--steps', type=int, default=6)
    o = parser.parse_args()
    gpus = o.gpus.split(',')
    if len(gpus) != 2 or len(set(gpus)) != 2 or any(not g.isdigit() for g in gpus) or o.steps < 5:
        raise ValueError('Two distinctGPUs, at least5updates required')
    root = Path(o.root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    lock = (root / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / 'pipeline.pid').write_text(str(os.getpid()) + '\n')
    pilot = Path(o.verified_pilot).resolve()
    if json.loads((pilot / 'pipeline_status.json').read_text())['phase'] != 'complete':
        raise ValueError('Previous pilot must be complete')
    if json.loads((pilot / 'verification.json').read_text())['status'] != 'pass':
        raise ValueError('Require passed distribution/gradient checks')
    frozen = pilot / 'source_snapshot'
    for name, digest in json.loads((pilot / 'source_hashes.json').read_text()).items():
        if sha(frozen / name) != digest:
            raise ValueError('Verified source changed')
    snapshot = root / 'source_snapshot'
    shutil.copytree(frozen, snapshot, ignore=shutil.ignore_patterns('__pycache__'))
    # Same timing hooks; newest benchmark adds post-train evidence only.
    shutil.copy2(PROJECT / 'latent/rl_speed_benchmark.py', snapshot / 'latent/rl_speed_benchmark.py')
    save(root / 'source_hashes.json', {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*') if p.is_file()})
    shutil.copy2(pilot / 'verification.json', root / 'distribution_verification.json')
    base = json.loads((pilot / 'cached/command.json').read_text())
    save(root / 'protocol.json', dict(gpus=gpus, verified_pilot=str(pilot), variant='cached',
        batches=[32, 64], accumulations=[16, 8], global_candidates=1024, full_train_rows=157679,
        optimizer_updates=o.steps, discard_first_updates=2, same_sft_initialization=True,
        no_attention_change=True, native_source_unchanged=True, formal_jobs_unchanged=True,
        no_formal_restart=True, no_saved_benchmark_weights=True,
        timing='CUDA synced full updates and components; no loading/eval/save I/O',
        randomness='same seed and prompt budget; microbatch/sampling RNG order differs'))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=o.gpus, PYTHONPATH=str(snapshot),
        PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', NCCL_IB_DISABLE='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', CUDA_HOME=str(Path(PYTHON).parent.parent),
        LATENT_SPEED_VARIANT='cached')
    env['PATH'] = str(Path(PYTHON).parent) + ':' + env.get('PATH', '')
    active = None

    def status(phase, **extra):
        save(root / 'pipeline_status.json', dict(phase=phase, coordinator_pid=os.getpid(),
            gpus=gpus, updated_at=time.time(), **extra))

    def interrupted(signum, frame):
        raise RuntimeError(f'Batchpilot receivedsignal{signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        results = {}
        for batch in (32, 64):
            check_gpu_capacity(gpu_snapshot(gpus), gpus, False, 35)
            destination = root / f'batch{batch}'
            destination.mkdir()
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            command = batch_command(base, batch, destination, port, o.steps)
            save(destination / 'command.json', command)
            with (destination / 'train.log').open('w') as log:
                active = subprocess.Popen(command, cwd=snapshot, env=dict(env, LATENT_SPEED_DEST=str(destination)),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                status('benchmarking', batch_per_device=batch, accumulation=1024 // (2 * batch), worker_pid=active.pid)
                code = active.wait()
            active = None
            (destination / 'exit_code').write_text(str(code) + '\n')
            if code:
                if batch == 64 and 'out of memory' in (destination / 'train.log').read_text(errors='replace').lower():
                    results['64'] = dict(oom=True, exit_code=code)
                    break
                raise RuntimeError(f'Batch{batch}benchmark failed; inspect train.log')
            results[str(batch)] = summarize(destination, o.steps, batch)
            save(root / 'results.json', results)
        available64 = 'mean_seconds' in results.get('64', {})
        speedup = results['32']['mean_seconds'] / results['64']['mean_seconds'] if available64 else None
        recommended = 64 if available64 and speedup > 1.05 and results['64']['peak_reserved_gib'] < 40 else 32
        save(root / 'results.json', dict(batches=results, speedup=speedup, recommended_batch=recommended,
                                       formal_jobs_unchanged=True))
        rows = ['| 每卡batch | 累积 | 均值秒/更新 | 中位数 | 采样秒 | 峰值allocated/reserved GiB |',
                '|---|---:|---:|---:|---:|---:|']
        for batch, r in results.items():
            if r.get('oom'):
                rows.append('|64|8|OOM|—|—|—|')
            else:
                rows.append(f'|{batch}|{r["accumulation"]}|{r["mean_seconds"]:.3f}|{r["median_seconds"]:.3f}|'
                            f'{r["components"]["rollout_seconds"]:.3f}|{r["peak_allocated_gib"]:.2f}/{r["peak_reserved_gib"]:.2f}|')
        (root / 'REPORT.md').write_text('加速版latent4×4 dense RL，两卡global1024，全157679任务行，'
            f'同一SFT初始化。各{o.steps}真实更新、丢前2次，CUDA同步计时，无验证/保存开销。\n\n' +
            '\n'.join(rows) + f'\n\n建议每卡batch{recommended}，未自动切换任何正式任务。'
            '短测不保证长期吞吐或推荐指标；微批改变也改变采样随机数顺序。\n')
        status('complete', recommended_batch=recommended, speedup=speedup, report=str(root / 'REPORT.md'))
    except Exception:
        status('failed', error=traceback.format_exc())
        raise
    finally:
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait()


if __name__ == '__main__':
    main()
