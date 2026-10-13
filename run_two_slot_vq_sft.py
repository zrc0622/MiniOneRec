#!/usr/bin/env python3
"""Two-slot nonbalanced VQ -> measured two-GPU SFT -> matched dual-protocol eval."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

PROJECT = Path(__file__).resolve().parent
PYTHON = '/data/zuorongchang/miniconda3/envs/minionerec/bin/python'
DATA = PROJECT / 'data/Amazon23/variants/rqkmeans_balanced'
STEM = 'Industrial_and_Scientific_5_2016-10-2018-11'
INDEX = DATA / 'Industrial_and_Scientific/Industrial_and_Scientific.index.json'
ITEMS = DATA / 'Industrial_and_Scientific/Industrial_and_Scientific.item.json'
REFERENCE = PROJECT / 'outputs/latent_qwen3_0.6b_balanced15/sft/final_checkpoint'
NATIVE = PROJECT / 'outputs/amazon23_industrial_qwen3_0.6b_sft_balanced_15ep/final_checkpoint'


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def gpu_snapshot(gpus):
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    rows = {}
    for line in result.splitlines():
        idx, uuid, total, used, util = [s.strip() for s in line.split(',')]
        if idx in gpus:
            rows[idx] = dict(uuid=uuid, total_mib=int(total), used_mib=int(used), utilization=int(util), pids=[])
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    for line in apps.splitlines():
        uuid, pid = [s.strip() for s in line.split(',')]
        for info in rows.values():
            if info['uuid'] == uuid:
                info['pids'].append(int(pid))
    if set(rows) != set(gpus):
        raise RuntimeError('Selected GPU not available')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--gpus', default='4,7')
    parser.add_argument('--label-gpu', default='7')
    args = parser.parse_args()
    os.chdir(PROJECT)
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpus = args.gpus.split(',')
    if len(gpus) != 2 or len(set(gpus)) != 2 or args.label_gpu not in gpus:
        parser.error('Two distinct training GPUs and one of them for VQ are required')
    save(root / 'pipeline_config.json', dict(gpus=gpus, label_gpu=args.label_gpu, num_slots=2,
        balanced=False, codebook_size=256, slot_dim=32, epochs=15, global_batch=1024,
        lr=3e-4, latent_weight=1, validation_fraction=.2, validation_every_epochs=1,
        save_every_epochs=3, final_checkpoint_selection='final_epoch_not_best_subset',
        methods=['single20','four20_marginal'], attention='unchanged', seed=42))
    (root / 'pipeline.pid').write_text(str(os.getpid()) + '\n')
    env = dict(os.environ, PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='1', NCCL_IB_DISABLE='1', CUDA_VISIBLE_DEVICES=args.gpus)
    env['PATH'] = str(Path(PYTHON).parent) + ':' + env.get('PATH', '')
    if not Path(env.get('CUDA_HOME', '') + '/bin/nvcc').is_file():
        env['CUDA_HOME'] = str(Path(PYTHON).parent.parent)
    # Expand allocations instead of keeping separate large pools for each shape.
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    def status(phase, **extra):
        info = dict(phase=phase, updated_at=time.time(), coordinator_pid=os.getpid(), gpus=gpus, **extra)
        save(root / 'pipeline_status.json', info)
        print(json.dumps(info, ensure_ascii=False), flush=True)
    def wait_idle(selected):
        good = 0
        while good < 3:
            snap = gpu_snapshot(selected)
            idle = all(x['used_mib'] < 1024 and x['utilization'] < 10 and not x['pids'] for x in snap.values())
            good = good + 1 if idle else 0
            status('waiting_for_idle_gpus', selected_gpus=selected, measurements=snap, consecutive_idle_checks=good)
            if good < 3:
                time.sleep(10 if idle else 30)
        return snap
    def start(command, log, run_env):
        log.parent.mkdir(parents=True, exist_ok=True)
        save(log.with_suffix('.command.json'), command)
        stream = log.open('w')
        proc = subprocess.Popen(command, env=run_env, stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True, cwd=snapshot)
        log.with_suffix('.pid').write_text(str(proc.pid) + '\n')
        return proc, stream
    def finish(proc, stream, log):
        code = proc.wait(); stream.close(); log.with_suffix('.exit_code').write_text(str(code) + '\n')
        return code
    def run(command, log, run_env=None):
        proc, stream = start(command, log, run_env or env)
        code = finish(proc, stream, log)
        if code:
            raise RuntimeError(f'Command failed exit={code}; inspect {log}')
    snapshot = root / 'source_snapshot'
    if not snapshot.exists():
        snapshot.mkdir()
        shutil.copytree(PROJECT / 'latent', snapshot / 'latent', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(__file__, snapshot / Path(__file__).name)
        for name in ('models', 'data', 'config'):
            (snapshot / name).symlink_to(PROJECT / name, target_is_directory=True)
        shutil.copy2(PROJECT / 'eval_metric_protocol.json', snapshot / 'eval_metric_protocol.json')
        save(snapshot / 'hashes.json', {str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (PROJECT / 'latent').glob('*.py')})
    # Execute the pinned latent package so later workspace edits cannot change the queued run.
    env['PYTHONPATH'] = str(snapshot) + ':' + str(PROJECT)
    label_env = dict(env, CUDA_VISIBLE_DEVICES=args.label_gpu)
    if not (root / 'labels/labels.json').exists():
        wait_idle([args.label_gpu]); status('training_two_slot_vq', label_gpu=args.label_gpu)
        run([PYTHON, '-u', '-m', 'latent.prepare', 'vq', '--embeddings',
            str(PROJECT / 'outputs/latent_qwen3_0.6b_balanced15/embeddings'),
            '--output', str(root / 'labels'), '--num_slots', '2', '--codebook_size', '256',
            '--latent_dim', '32', '--balance_epsilon', '0', '--epochs', '50', '--patience', '5',
            '--batch', '512', '--lr', '.001', '--seed', '42'], root / 'vq.log', label_env)
    labels = json.loads((root / 'labels/labels.json').read_text())
    if labels['k'] != 2 or labels['codebook_size'] != 256 or labels['vq_options'].get('balance_epsilon', 0) != 0:
        raise ValueError('Wrong quantizer for this experiment')
    status('labels_complete', best_epoch=labels['best_epoch'], reconstruction=labels['validation_reconstruction'], code_statistics=labels['code_statistics'])
    torchrun = [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2']
    if not (root / 'sft/final_checkpoint/config.json').exists():
        idle = wait_idle(gpus)
        results = []
        for batch in (16, 32):
            dest = root / 'benchmarks' / f'batch{batch}'; dest.mkdir(parents=True, exist_ok=True)
            status('benchmarking', batch_per_device=batch, global_batch=1024)
            proc, stream = start(torchrun + ['-m', 'latent.sft_benchmark', '--labels', str(root / 'labels'),
                '--batch', str(batch), '--output', str(dest)], dest / 'train.log', env)
            code = finish(proc, stream, dest / 'train.log')
            if code:
                if 'out of memory' in (dest / 'train.log').read_text(errors='replace').lower():
                    save(dest / 'failure.json', dict(exit_code=code, reason='OOM'));continue
                raise RuntimeError(f'Benchmark failed: {dest}/train.log')
            results.append(json.loads((dest / 'benchmark.json').read_text()))
        # Actual reserved memory includes ZeRO communication buffers; retain 1GiB headroom.
        limit = min((v['total_mib'] - v['used_mib']) / 1024 - 1 for v in idle.values())
        safe = [r for r in results if max(m[2] for m in r['rank_measurements']) < limit]
        if not safe:
            raise RuntimeError('No measured microbatch leaves sufficient GPU headroom')
        fastest = min(safe, key=lambda r:r['step_seconds'])
        save(root / 'batch_selection.json', dict(chosen=fastest, benchmarks=results, reserved_gib_limit=limit,
            rule='Fastest measured successful batch; at least1GiB memory headroom; globalbatch1024'))
        wait_idle(gpus)
        command = torchrun + ['-m', 'latent.train', 'sft', '--model', str(PROJECT / 'models/Qwen3-0.6B'),
            '--output_dir', str(root / 'sft'), '--labels', str(root / 'labels'),
            '--train_file', str(DATA / 'train' / f'{STEM}.csv'), '--eval_file', str(DATA / 'valid' / f'{STEM}.csv'),
            '--items', str(ITEMS), '--index', str(INDEX), '--batch', str(fastest['batch_per_device']),
            '--gradient_accumulation', str(fastest['accumulation']), '--eval_batch', '16',
            '--epochs', '15', '--lr', '.0003', '--seed', '42', '--latent_weight', '1',
            '--sft_eval_fraction', '.2', '--sft_eval_every_epochs', '1', '--sft_save_every_epochs', '3',
            '--no-sft_load_best', '--sft_early_stopping_patience', '0', '--logging_steps', '10',
            '--deepspeed', str(PROJECT / 'config/sft_zero2.json')]
        status('training_sft', batch_per_device=fastest['batch_per_device'], accumulation=fastest['accumulation'],
            benchmark_step_seconds=fastest['step_seconds'], estimated_training_hours=4140*fastest['step_seconds']/3600,
            validation_rows=3232, validation_every_epochs=1, saving_every_epochs=3)
        run(command, root / 'sft.log')
    # First fixed256 pilot; then shared20% validation with original3slot and nativeSFT controls.
    subset = json.loads((root / 'sft/validation_subset.json').read_text())
    sample20 = root / 'sample20pct.json';save(sample20, subset)
    small = PROJECT / 'outputs/eval_search_compare_20261009/sample256.json'
    models = {'two_slot_vq_sft':(root / 'sft/final_checkpoint','latent'),
              'three_slot_vq_sft':(REFERENCE,'latent'), 'no_latent_sft':(NATIVE,'sid')}
    eval_root = root / 'eval';eval_root.mkdir(exist_ok=True)
    for scope, sample in [('sample256', small), ('valid20pct', sample20)]:
        dest=eval_root / scope;dest.mkdir(exist_ok=True)
        save(dest / 'sample.json', json.loads(sample.read_text()))
        queue=list(models)
        while queue:
            selected=queue[:2];queue=queue[2:];wait_idle(gpus);workers=[]
            for name,gpu in zip(selected,gpus):
                model,kind=models[name];log=dest/f'{name}.log'
                status('ranking_eval', scope=scope, model=name, gpu=gpu, methods=['single20','four20_marginal'] if kind=='latent' else ['direct50'])
                proc,stream=start([PYTHON,'-u','-m','latent.eval_unified_small','--model',str(model),'--kind',kind,
                    '--data',str(DATA/'valid'/f'{STEM}.csv'),'--index',str(INDEX),'--sample',str(sample),
                    '--output',str(dest/f'{name}.json'),'--max-batch','32'],log,dict(env,CUDA_VISIBLE_DEVICES=gpu))
                workers.append((proc,stream,log))
            codes=[finish(*worker) for worker in workers]
            if any(codes):raise RuntimeError(f'Ranking evaluation failed: {codes}')
        # Summarize in a separate process that uses the same project dependency environment.
        run([PYTHON,str(PROJECT/'report_two_slot_vq.py'),'--root',str(dest)],dest/'report.log',dict(env,CUDA_VISIBLE_DEVICES=''))
    status('complete', comparison=str(eval_root/'valid20pct/comparison.md'))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        import sys
        if '--root' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();root.mkdir(parents=True,exist_ok=True)
            save(root/'pipeline_error.json',dict(time=time.time(),error=traceback.format_exc()))
            save(root/'pipeline_status.json',dict(phase='failed',updated_at=time.time(),error=traceback.format_exc()))
        raise
