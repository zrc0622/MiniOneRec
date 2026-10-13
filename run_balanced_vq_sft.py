#!/usr/bin/env python3
"""Durable balanced VQ -> measured two-GPU SFT -> fixed ranking evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


PYTHON = '/data/zuorongchang/miniconda3/envs/minionerec/bin/python'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--gpus', default='6,7')
    parser.add_argument('--reuse-benchmark', action='store_true',
                        help='Reuse an existing measured selection when moving an uncheckpointed SFT')
    parser.add_argument('--label-kind', choices=('vq', 'rq_kmeans'), default='vq')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    lock = (root / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Older running coordinators may predate the lock; do not duplicate them.
    pid_file = root / 'pipeline.pid'
    if pid_file.exists():
        previous_pid = int(pid_file.read_text())
        if previous_pid != os.getpid():
            try:
                previous = (Path('/proc') / str(previous_pid) / 'cmdline').read_bytes()
            except OSError:
                previous = b''
            if b'run_balanced_vq_sft.py' in previous and str(root).encode() in previous:
                raise RuntimeError(f'Pipeline already running: PID{previous_pid}')
    os.chdir(Path(__file__).resolve().parent)
    gpus = args.gpus.split(',')
    if len(gpus) != 2 or len(set(gpus)) != 2 or not all(g.isdigit() for g in gpus):
        parser.error('Two distinct GPU indices are required')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpus, PYTHONUNBUFFERED='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', NCCL_IB_DISABLE='1')
    prefix = str(Path(PYTHON).parent.parent)
    env['PATH'] = str(Path(PYTHON).parent) + ':' + env.get('PATH', '')
    if not Path(env.get('CUDA_HOME', '') + '/bin/nvcc').is_file():
        env['CUDA_HOME'] = prefix

    def status(phase, **extra):
        info = dict(phase=phase, updated_at=time.time(), **extra)
        (root / 'pipeline_status.json').write_text(json.dumps(info, indent=2) + '\n')
        print(json.dumps(info), flush=True)

    def run(command, log_path, run_env=None):
        with log_path.open('w') as log:
            proc = subprocess.Popen(command, env=run_env or env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (log_path.with_suffix('.pid')).write_text(str(proc.pid) + '\n')
            return proc.wait()

    label_pid = int((root / ('vq.pid' if args.label_kind == 'vq' else 'rq_kmeans.pid')).read_text())
    label_module = b'latent.prepare' if args.label_kind == 'vq' else b'latent.prepare_rq'
    status('waiting_for_labels', label_kind=args.label_kind, label_pid=label_pid)
    # Wait for export to finish and the verified VQ process to release its GPU.
    while True:
        try:
            cmd = (Path('/proc') / str(label_pid) / 'cmdline').read_bytes()
            active = label_module in cmd and str(root / 'labels').encode() in cmd
        except OSError:
            active = False
        if not active:
            break
        time.sleep(10)
    if not (root / 'labels/labels.json').exists():
        raise RuntimeError('Label fitting ended without complete labels; inspect label-fit log')
    labels = json.loads((root / 'labels/labels.json').read_text())
    print('Labels completed', json.dumps(dict(kind=args.label_kind, best_epoch=labels.get('best_epoch'),
        reconstruction=labels.get('validation_reconstruction'), codes=labels['code_statistics'])), flush=True)
    while True:
        output = subprocess.check_output(['nvidia-smi', '-i', args.gpus,
            '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
        if all(int(x.strip()) < 1024 for x in output.splitlines()):
            break
        status('waiting_for_idle_selected_gpus', memory=output.strip())
        time.sleep(30)

    torchrun = [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2']
    results = []
    for batch in (() if args.reuse_benchmark else (16, 32)):
        status('benchmarking', batch_per_device=batch, accumulation=512 // batch)
        dest = root / 'benchmarks' / f'batch{batch}'
        dest.mkdir(parents=True)
        code = run(torchrun + ['-m', 'latent.sft_benchmark', '--labels', str(root / 'labels'),
            '--batch', str(batch), '--output', str(dest)], dest / 'train.log')
        if code:
            text = (dest / 'train.log').read_text(errors='replace')
            if 'out of memory' in text.lower():
                print(f'batch{batch}: OOM, retaining smaller successful batch', flush=True)
                (dest / 'failure.json').write_text(json.dumps(dict(exit_code=code, reason='OOM')))
                continue
            raise RuntimeError(f'Benchmark batch{batch} failed: exit{code}; inspect {dest}/train.log')
        results.append(json.loads((dest / 'benchmark.json').read_text()))
    if args.reuse_benchmark:
        previous = json.loads((root / 'batch_selection.json').read_text())
        results = previous['benchmarks']
        if any(r['world_size'] != 2 or r['global_batch'] != 1024 for r in results):
            raise ValueError('Incompatible saved two-GPU benchmark')
    if not results:
        raise RuntimeError('No successful benchmark; smaller microbatch needed')
    safe = [r for r in results if max(row[2] for row in r['rank_measurements']) < 41.0]
    if not safe:
        raise RuntimeError('Benchmarks leave too little GPU headroom for formal training')
    fastest = min(safe, key=lambda r: r['step_seconds'])
    small = min(safe, key=lambda r: r['batch_per_device'])
    # Prefer the original microbatch unless a larger one measurably wins (>5%).
    chosen = fastest if fastest['step_seconds'] < .95 * small['step_seconds'] else small
    selection = dict(chosen=chosen, benchmarks=results,
        rule='Reserved VRAM <41GiB; choose larger batch only if >5% faster; global batch1024')
    (root / 'batch_selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    status('training_sft', batch_per_device=chosen['batch_per_device'],
           accumulation=chosen['accumulation'], benchmark_step_seconds=chosen['step_seconds'])
    data = Path('data/Amazon23/variants/rqkmeans_balanced').resolve()
    stem = 'Industrial_and_Scientific_5_2016-10-2018-11'
    index = data / 'Industrial_and_Scientific/Industrial_and_Scientific.index.json'
    command = torchrun + ['-m', 'latent.train', 'sft', '--model', './models/Qwen3-0.6B',
        '--output_dir', str(root / 'sft'), '--labels', str(root / 'labels'),
        '--train_file', str(data / 'train' / f'{stem}.csv'),
        '--eval_file', str(data / 'valid' / f'{stem}.csv'),
        '--items', str(data / 'Industrial_and_Scientific/Industrial_and_Scientific.item.json'),
        '--index', str(index), '--batch', str(chosen['batch_per_device']),
        '--gradient_accumulation', str(chosen['accumulation']), '--eval_batch', '16',
        '--epochs', '15', '--lr', '0.0003', '--seed', '42', '--deepspeed', 'config/sft_zero2.json']
    (root / 'sft_command.json').write_text(json.dumps(command, indent=2) + '\n')
    code = run(command, root / 'sft.log')
    (root / 'sft.exit_code').write_text(str(code) + '\n')
    if code:
        raise RuntimeError(f'SFT failed exit{code}; inspect sft.log')
    for split in ('valid', 'test'):
        status('ranking_eval', split=split, scheme='hierarchical_max', latent_beams=4, item_beams=50)
        dest = root / 'eval' / split
        dest.mkdir(parents=True)
        workers = []
        for shard, gpu in enumerate(gpus):
            worker_env = dict(env, CUDA_VISIBLE_DEVICES=gpu)
            log = (dest / f'{shard}.log').open('w')
            command = [PYTHON, '-u', '-m', 'latent.evaluate', '--model', str(root / 'sft/final_checkpoint'),
                '--data', str(data / split / f'{stem}.csv'), '--index', str(index),
                '--scheme', 'hierarchical_max', '--latent-count', '4', '--latent-beams', '4',
                '--item-beams', '50', '--score-batch', '8', '--shards', '2', '--shard', str(shard),
                '--output', str(dest / f'{shard}.json')]
            proc = subprocess.Popen(command, env=worker_env, stdout=log,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            workers.append((proc, log))
        codes = []
        for proc, log in workers:
            codes.append(proc.wait())
            log.close()
        if any(codes):
            raise RuntimeError(f'Ranking eval {split} failed: {codes}')
        code = run([PYTHON, '-m', 'latent.evaluate', '--merge', str(dest / '0.json'),
            str(dest / '1.json'), '--output', str(dest / 'merged.json')], dest / 'merge.log')
        if code:
            raise RuntimeError(f'Merge failed for {split}')
    old_path = Path('outputs/latent_qwen3_0.6b_balanced15/eval/four_test.JWNLYi/hierarchical_max/merged.json')
    old = json.loads(old_path.read_text())
    new = json.loads((root / 'eval/test/merged.json').read_text())
    for key in ('data_sha256', 'index_sha256', 'examples', 'scheme', 'latent_count', 'latent_beams', 'item_beams', 'score_batch'):
        if old['metadata'].get(key) != new['metadata'].get(key):
            raise ValueError(f'Baseline eval identity mismatch: {key}')
    assert {x['row_id']:x['output'] for x in old['rows']} == {x['row_id']:x['output'] for x in new['rows']}
    summary = dict(baseline=str(old_path), balanced=str(root / 'eval/test/merged.json'),
                   baseline_metrics=old['metrics'], balanced_metrics=new['metrics'])
    (root / 'eval/sft_comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    status('complete', comparison=str(root / 'eval/sft_comparison.json'))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import sys
        import traceback
        traceback.print_exc()
        # Leave completed stages intact and make failures visible on recovery.
        if '--root' in sys.argv:
            root = Path(sys.argv[sys.argv.index('--root') + 1]).resolve()
            (root / f'pipeline_error_{os.getpid()}.json').write_text(json.dumps(
                dict(time=time.time(), error=traceback.format_exc()), indent=2) + '\n')
            pid_file = root / 'pipeline.pid'
            if pid_file.exists() and int(pid_file.read_text()) == os.getpid():
                (root / 'pipeline_status.json').write_text(json.dumps(dict(
                    phase='failed', updated_at=time.time(), error=traceback.format_exc()), indent=2) + '\n')
        raise
