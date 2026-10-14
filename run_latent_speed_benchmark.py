"""Check isolated latent optimizations then benchmark on explicitly assigned GPUs."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--gpus', default='1,3')
    parser.add_argument('--steps', type=int, default=6)
    o = parser.parse_args()
    gpus = o.gpus.split(',')
    if len(gpus) != 2 or len(set(gpus)) != 2 or any(not g.isdigit() for g in gpus):
        raise ValueError('Two distinct GPUs required')
    if o.steps < 5:
        raise ValueError('At least2warmup+3measured updates')
    root = Path(o.root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    lock = (root / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / 'pipeline.pid').write_text(str(os.getpid()) + '\n')
    source = Path(o.source_run).resolve()
    frozen = source / 'source_snapshot'
    for name, digest in json.loads((frozen / 'hashes.json').read_text()).items():
        if sha(frozen / name) != digest:
            raise ValueError('Original frozen source changed')
    snapshot = root / 'source_snapshot'
    shutil.copytree(frozen, snapshot)
    for name in ('rl_fast.py', 'rl_fast_verify.py', 'rl_speed_benchmark.py', 'train_fast.py'):
        shutil.copy2(PROJECT / 'latent' / name, snapshot / 'latent' / name)
    save(root / 'source_hashes.json', {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*') if p.is_file()})
    base = json.loads((source / 'rl.command.json').read_text())
    model, data, index = [base[base.index('--' + key) + 1] for key in ('model', 'train_file', 'index')]
    save(root / 'protocol.json', dict(source_run=str(source), model=model, steps=o.steps,
        global_candidates=1024, batch_per_rank=32, accumulation=16, gpus=gpus,
        variants=['old', 'vectorized', 'cached'], discard_first_updates=2,
        no_formal_restart=True, no_attention_change=True, no_native_source_change=True,
        no_checkpoint_weights_saved=True, same_reward_and4x4=True,
        timing='CUDA synced; component synchronization applied equally to variants; no evaluation/save I/O'))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=o.gpus, PYTHONPATH=str(snapshot),
        PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', NCCL_IB_DISABLE='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', CUDA_HOME=str(Path(PYTHON).parent.parent))
    env['PATH'] = str(Path(PYTHON).parent) + ':' + env.get('PATH', '')
    active = None

    def status(phase, **extra):
        save(root / 'pipeline_status.json', dict(phase=phase, coordinator_pid=os.getpid(),
             updated_at=time.time(), gpus=gpus, **extra))

    def interrupted(signum, frame):
        raise RuntimeError(f'Benchmark receivedsignal{signum}')

    def run(command, dest, job_env, phase):
        nonlocal active
        check_gpu_capacity(gpu_snapshot(gpus), gpus, False, 35)
        dest.mkdir(exist_ok=True)
        save(dest / 'command.json', command)
        with (dest / 'train.log').open('w') as log:
            active = subprocess.Popen(command, cwd=snapshot, env=job_env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status(phase, worker_pid=active.pid)
            code = active.wait()
        active = None
        (dest / 'exit_code').write_text(str(code) + '\n')
        if code:
            raise RuntimeError(f'{phase} failed; inspect {dest}/train.log')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        status('verifying')
        verification = root / 'verification'
        run([PYTHON, '-u', '-m', 'latent.rl_fast_verify', '--model', model, '--data', data,
             '--index', index, '--output', str(root / 'verification.json')], verification,
            dict(env, CUDA_VISIBLE_DEVICES=gpus[0]), 'checking_real_model_distributions')
        if json.loads((root / 'verification.json').read_text())['status'] != 'pass':
            raise ValueError('Verification failed')
        results = {}
        for variant in ('old', 'vectorized', 'cached'):
            dest = root / variant
            command = list(base)
            command[command.index('latent.train')] = 'latent.rl_speed_benchmark'
            for key, value in {'--output_dir': str(dest / 'model'), '--max_steps': str(o.steps),
                              '--eval_steps': '1000', '--save_steps': '1000'}.items():
                if key in command:
                    command[command.index(key) + 1] = value
                else:
                    command.extend([key, value])
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            command = [f'--master_port={port}' if s.startswith('--master_port=') else s for s in command]
            run(command, dest, dict(env, LATENT_SPEED_VARIANT=variant, LATENT_SPEED_DEST=str(dest)),
                'benchmarking_' + variant)
            ranks = [[json.loads(s) for s in (dest / f'rank{r}.jsonl').read_text().splitlines()] for r in (0, 1)]
            if any([row['step'] for row in rows] != list(range(1, o.steps + 1)) for rows in ranks):
                raise ValueError('Incomplete optimizer update profile')
            times = [max(ranks[0][i]['seconds'], ranks[1][i]['seconds']) for i in range(2, o.steps)]
            if not all(math.isfinite(t) and t > 0 for t in times):
                raise ValueError('Invalid timing')
            results[variant] = dict(steady_seconds=times, mean_seconds=statistics.mean(times),
                median_seconds=statistics.median(times),
                components={k: statistics.mean(max(ranks[0][i]['components'].get(k, 0),
                    ranks[1][i]['components'].get(k, 0)) for i in range(2, o.steps))
                    for k in ('rollout_seconds', 'reference_forward_seconds', 'policy_forward_seconds')},
                peak_allocated_gib=max(row['peak_allocated_gib'] for rows in ranks for row in rows),
                peak_reserved_gib=max(row['peak_reserved_gib'] for rows in ranks for row in rows))
            save(root / 'results.json', results)
        speedup = results['old']['mean_seconds'] / results['cached']['mean_seconds']
        save(root / 'results.json', dict(variants=results, speedup=speedup, formal_jobs_unchanged=True))
        rows = ['| 版本 | 秒/更新（均值） | 中位数 | 采样秒 | 参考前向秒 | 策略前向秒 | 峰值显存GiB |',
                '|---|---:|---:|---:|---:|---:|---:|']
        for name, r in results.items():
            c = r['components']
            rows.append(f'| {name} | {r["mean_seconds"]:.3f} | {r["median_seconds"]:.3f} | '
                        f'{c["rollout_seconds"]:.3f} | {c["reference_forward_seconds"]:.3f} | '
                        f'{c["policy_forward_seconds"]:.3f} | {r["peak_allocated_gib"]:.2f} |')
        (root / 'REPORT.md').write_text('同一SFT初始化、全训练数据、4×4 dense奖励、两卡/global1024，'
            f'每组{o.steps}次真实优化更新，丢弃前2次。正式任务未改变。\n\n' + '\n'.join(rows) +
            f'\n\n加速比 {speedup:.3f}×。采样RNG顺序不同，BF16批量计算不保证逐位相同；'
            '不宣称训练结果完全一致。概率/梯度校验见verification.json。'
            '这是带等量阶段同步的短测，不含验证/保存，长期吞吐需另确认。\n')
        status('complete', speedup=speedup, report=str(root / 'REPORT.md'))
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
