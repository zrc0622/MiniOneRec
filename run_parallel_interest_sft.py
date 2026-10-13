#!/usr/bin/env python3
"""Original history VQ -> controlled parallel SFT on GPUs2/5 -> both evals."""
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

from run_two_slot_vq_sft import PROJECT, PYTHON, DATA, INDEX, ITEMS, STEM, REFERENCE, NATIVE, gpu_snapshot, save
from run_target_interest_sft import CONTROLS, METHODS, validate_result
from latent.core import load_index, load_labels, sha
from interest.evaluate import summarize

LABELS = PROJECT / 'outputs/latent_qwen3_0.6b_balanced15/labels'


def training_command(root, snapshot, stage, batch=16):
    command = [PYTHON, '-u', '-m', 'latent.parallel_train', stage, '--model', str(PROJECT / 'models/Qwen3-0.6B'),
        '--labels', str(LABELS), '--train_file', str(DATA / 'train' / f'{STEM}.csv'),
        '--eval_file', str(DATA / 'valid' / f'{STEM}.csv'), '--items', str(ITEMS), '--index', str(INDEX),
        '--prepared', str(root / 'prepared'), '--output_dir', str(root / ('sft' if stage == 'train' else f'benchmark{batch}')),
        '--batch', str(batch), '--epochs', '15', '--lr', '.0003', '--seed', '42',
        '--latent_weight', '1', '--eval_batch', '16', '--deepspeed', str(snapshot / 'config/sft_zero2.json')]
    return command if stage == 'prepare' else [PYTHON, '-m', 'torch.distributed.run', '--standalone',
        '--nproc_per_node=2', '-m', 'latent.parallel_train'] + command[4:]


def report(dest, sample, results):
    metrics = [f'{m}@{k}' for m in ('Recall', 'NDCG') for k in (5, 10, 20)]
    targets = None
    summaries = {}
    for name, result in results.items():
        current = validate_result(result, sample)
        if targets is not None and targets != current:
            raise ValueError('Comparison target items differ')
        targets = current
        summaries[name] = {}
        for method, rows in result['rows'].items():
            values = summarize(rows)
            total = sum(b['elapsed_seconds'] for b in result['batches'][method])
            summaries[name][method] = {k: values[k] for k in metrics}
            summaries[name][method].update(full20=sum(len(r['predict']) >= 20 for r in rows) / len(rows),
                mean_amortized_seconds=total / len(rows), mean_unique_proposals=sum(r['unique_before_top20'] for r in rows) / len(rows))
    save(dest / 'comparison.json', dict(examples=len(sample['row_ids']), results=summaries,
         paired_targets_verified=True, same_sid_and_vq_labels=True,
         prior_change='AR chain rule vs product of three bidirectional-query heads',
         baseline_timings='historical batch throughput, not matched hardware speed comparison'))
    lines = [f'# 三槽双向并行预测对照（{len(sample["row_ids"])}条验证样本）', '',
        '旧商品SID和原版仅历史、非balanced三槽VQ标签固定。并行模型一次前向预测三槽，SID继续自回归。',
        '并行先验为三个全词表槽分布的乘积，精确选top4完整组合；串行对照为interest beam4。SID分支预算一致。',
        '两者均使用15轮上限、fullvalid半轮验证、最佳loss/earlystop3、globalbatch1024；并行输入仅额外增加3固定查询，历史截断不变。',
        '吞吐基线为历史测量，不能作为同硬件端到端延迟对照。', '']
    for method, label in zip(METHODS, ('1×20', '4×20完整跨兴趣求和')):
        lines += [f'## {label}', '', '| 模型 | R5 | R10 | R20 | N5 | N10 | N20 | Full20 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name, label in [('no_latent', '无latent'), ('serial', '三槽串行'), ('parallel', '三槽双向并行')]:
            row = summaries[name]['direct50' if name == 'no_latent' else method]
            lines.append('| ' + label + ' | ' + ' | '.join(f'{row[k]:.5f}' for k in metrics) + f" | {row['full20']:.1%} |")
        lines.append('')
    (dest / 'comparison.md').write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--gpus', default='2,5')
    p.add_argument('--eval-only', action='store_true', help='Evaluate the completed final model after checkpoint recovery')
    o = p.parse_args()
    root = Path(o.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpus = o.gpus.split(',')
    if len(gpus) != 2 or len(set(gpus)) != 2 or not all(g.isdigit() for g in gpus):
        raise ValueError('Require two distinct GPU indices')
    if not o.eval_only and gpus != ['2', '5']:
        raise ValueError('Fresh parallel training uses the original GPU2/5 pair')
    if (root / 'source_snapshot').exists() and not o.eval_only:
        raise ValueError('Fresh experiment root required; preserve existing artifacts')
    index = load_index(INDEX)
    for split in ('train', 'valid', 'test'):
        _, meta = load_labels(LABELS, split, DATA / split / f'{STEM}.csv', index)
    if meta['k'] != 3 or meta['vq_options'].get('balance_epsilon', 0) or meta['embedding_recipe'].get('contains_target', False):
        raise ValueError('Original history-only three-slot NONbalanced labels required')
    if not o.eval_only:
        save(root / 'pipeline_config.json', dict(gpus=gpus, labels=str(LABELS), labels_sha256=sha(LABELS / 'labels.json'),
         sid_index=str(INDEX), global_batch=1024, epochs=15, lr=.0003, seed=42, latent_weight=1,
         full_validation_half_epoch=True, checkpoint_selection='best loss, early stopping3',
         parallel_queries_mutually_visible=True, queries_cannot_read_gold_codes_or_sid=True,
         no_new_vq=True, no_new_sid=True, no_target_teacher_labels=True, periodic_disk_check=False,
             eval_methods=METHODS, eval_samples=[256, 3232], rl_authorized=False))
    snapshot = root / 'source_snapshot'
    if not o.eval_only:
        snapshot.mkdir()
        for folder in ('latent', 'interest', 'config'):
            shutil.copytree(PROJECT / folder, snapshot / folder, ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('data.py', 'deepspeed_cleanup.py', 'eval_metric_protocol.json', Path(__file__).name):
            shutil.copy2(PROJECT / name, snapshot / name)
        save(snapshot / 'hashes.json', {str(f.relative_to(snapshot)): sha(f) for f in snapshot.rglob('*') if f.is_file()})
    elif not (root / 'sft/final_checkpoint/model.safetensors').is_file():
        raise ValueError('Evaluation-only requires a complete final model')
    temp = root / 'tmp'
    temp.mkdir(exist_ok=o.eval_only)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=o.gpus, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', NCCL_IB_DISABLE='1', TMPDIR=str(temp),
               PYTHONPATH=str(snapshot) + ':' + str(PROJECT), PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env['PATH'] = str(Path(PYTHON).parent) + ':' + env.get('PATH', '')
    if not Path(env.get('CUDA_HOME', '') + '/bin/nvcc').is_file():
        env['CUDA_HOME'] = str(Path(PYTHON).parent.parent)
    (root / 'pipeline.pid').write_text(str(os.getpid()) + '\n')

    def status(phase, **details):
        value = dict(phase=phase, updated_at=time.time(), coordinator_pid=os.getpid(), gpus=gpus, **details)
        save(root / 'pipeline_status.json', value)
        print(json.dumps(value), flush=True)

    def wait_idle():
        consecutive = 0
        while consecutive < 3:
            values = gpu_snapshot(gpus)
            consecutive = consecutive + 1 if all(v['used_mib'] < 1024 and v['utilization'] < 10 and not v['pids'] for v in values.values()) else 0
            status('waiting_for_idle_gpus', measurements=values, consecutive_idle_checks=consecutive)
            if consecutive < 3:
                time.sleep(5 if consecutive else 30)

    def run(command, name, phase, gpu=None):
        save(root / f'{name}.command.json', command)
        with (root / f'{name}.log').open('w') as log:
            worker = subprocess.Popen(command, cwd=snapshot, env=dict(env, CUDA_VISIBLE_DEVICES=gpu) if gpu else env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (root / f'{name}.pid').write_text(str(worker.pid) + '\n')
            status(phase, worker_pid=worker.pid, log=str(root / f'{name}.log'))
            try:
                code = worker.wait()
            finally:
                if worker.poll() is None:
                    os.killpg(worker.pid, signal.SIGTERM)
                    worker.wait()
        (root / f'{name}.exit_code').write_text(str(code) + '\n')
        return code

    # No recurring storage checks; this one startup check follows the user's preference.
    if shutil.disk_usage(root).free < 100 * 2**30:
        raise RuntimeError('Insufficient storage before launching')
    if not o.eval_only:
        wait_idle()
    if not o.eval_only and run(training_command(root, snapshot, 'prepare'), 'prepare', 'preparing_parallel_data', gpus[0]):
        raise RuntimeError('Parallel data preparation failed')
    measured = {}
    for batch in (() if o.eval_only else (16, 32)):
        wait_idle()
        code = run(training_command(root, snapshot, 'benchmark', batch), f'benchmark{batch}', f'benchmark_batch{batch}')
        result = root / f'benchmark{batch}/benchmark.json'
        if not code and result.exists():
            measured[batch] = json.loads(result.read_text())
        elif 'out of memory' not in (root / f'benchmark{batch}.log').read_text(errors='replace').lower():
            raise RuntimeError(f'Benchmark{batch} failed for a reason other than OOM')
    if not measured and not o.eval_only:
        raise RuntimeError('Neither batch16 nor32 fits; preserve diagnostics before changing settings')
    safe = {b: r for b, r in measured.items() if max(v[1] for v in r['rank_measurements']) <= 42.5}
    if not safe and not o.eval_only:
        raise RuntimeError('Benchmarks leave insufficient GPU headroom; preserve results before tuning')
    if not o.eval_only:
        batch = min(safe, key=lambda b: safe[b]['step_seconds'])
        save(root / 'batch_selection.json', dict(chosen_batch=batch, accumulation=1024 // (2 * batch),
             measured=measured, global_batch=1024, reason='fastest successful two-rank benchmark with >=2GiB device headroom; fresh initialized weights for production'))
        wait_idle()
        if run(training_command(root, snapshot, 'train', batch), 'sft', 'training_parallel_sft'):
            raise RuntimeError('Parallel SFT failed')
    final = root / 'sft/final_checkpoint'
    if not (final / 'tokenizer_config.json').exists() or not (final / 'model.safetensors').exists():
        raise ValueError('Final model incomplete')
    save(root / 'sft_complete.json', dict(checkpoint=str(final), completed_at=time.time()))
    protocol = json.loads((snapshot / 'eval_metric_protocol.json').read_text())
    for scope in ('sample256', 'valid20pct'):
        dest = root / 'eval' / scope
        dest.mkdir(parents=True)
        shutil.copy2(CONTROLS / scope / 'sample.json', dest / 'sample.json')
        sample = json.loads((dest / 'sample.json').read_text())
        results = {}
        for name, filename, model, kind in [('serial', 'three_slot_vq_sft.json', REFERENCE, 'latent'),
                                             ('no_latent', 'no_latent_sft.json', NATIVE, 'sid')]:
            control = json.loads((CONTROLS / scope / filename).read_text())
            validate_result(control, sample, model, kind)
            expected = dict(sample_sha256=sha(dest / 'sample.json'), data_sha256=sha(DATA / 'valid' / f'{STEM}.csv'),
                            index_sha256=sha(INDEX), source_sha256=sha(snapshot / 'latent/eval_unified_small.py'), protocol=protocol)
            if any(control['metadata'][k] != v for k, v in expected.items()):
                raise ValueError('Historical evaluation provenance differs')
            results[name] = control
            save(dest / f'{name}.json', control)
        wait_idle()
        command = [PYTHON, '-u', '-m', 'latent.eval_parallel', '--model', str(final),
            '--data', str(DATA / 'valid' / f'{STEM}.csv'), '--index', str(INDEX),
            '--sample', str(dest / 'sample.json'), '--output', str(dest / 'parallel.json'), '--max-batch', '32']
        if run(command, f'eval_{scope}', f'evaluating_{scope}', gpus[0]):
            raise RuntimeError('Parallel ranking evaluation failed')
        results['parallel'] = json.loads((dest / 'parallel.json').read_text())
        report(dest, sample, results)
    wait_idle()
    command = [PYTHON, '-u', '-m', 'latent.parallel_latency', '--serial', str(REFERENCE), '--parallel', str(final),
        '--data', str(DATA / 'valid' / f'{STEM}.csv'), '--index', str(INDEX),
        '--sample', str(root / 'eval/sample256/sample.json'), '--output', str(root / 'eval/paired_latency64.json')]
    if run(command, 'paired_latency64', 'paired_single_user_latency', gpus[0]):
        raise RuntimeError('Paired latency benchmark failed')
    status('complete', checkpoint=str(final), report=str(root / 'eval/valid20pct/comparison.md'))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        import sys
        if '--root' in sys.argv:
            root = Path(sys.argv[sys.argv.index('--root') + 1]).resolve()
            save(root / 'pipeline_status.json', dict(phase='failed', updated_at=time.time(), error=traceback.format_exc()))
        raise
