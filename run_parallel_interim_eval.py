"""Bounded GPU4 interim comparison; retain training and foreign GPU processes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


PROJECT = Path(__file__).resolve().parent
PYTHON = '/data/zuorongchang/miniconda3/envs/minionerec/bin/python'
PARALLEL = PROJECT / 'outputs/latent_qwen3_0.6b_parallel_queries15_20261012'
SERIAL = PROJECT / 'outputs/latent_qwen3_0.6b_balanced15/sft/final_checkpoint'
DATA = PROJECT / 'data/Amazon23/variants/rqkmeans_balanced'
STEM = 'Industrial_and_Scientific_5_2016-10-2018-11'


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def summarize(rows):
    ranks = [r['predict'].index(r['output']) + 1 if r['output'] in r['predict'] else None for r in rows]
    return {**{f'Recall@{k}': sum(r is not None and r <= k for r in ranks) / len(rows) for k in (5, 10, 20)},
            **{f'NDCG@{k}': sum(1 / math.log2(r + 1) for r in ranks if r is not None and r <= k) / len(rows)
               for k in (5, 10, 20)},
            'fraction_with20': sum(len(set(r['predict'])) >= 20 for r in rows) / len(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--seconds', type=int, default=840)
    parser.add_argument('--gpus', default='4,4', help='Parallel and serial evaluator GPU IDs; distinct IDs run concurrently')
    o = parser.parse_args()
    gpus = o.gpus.split(',')
    if len(gpus) != 2:
        parser.error('Provide one GPU ID for each of the two models')
    concurrent = gpus[0] != gpus[1]
    started = time.monotonic()
    deadline = started + o.seconds
    root = Path(o.root).resolve()
    root.mkdir(exist_ok=False)
    active = []
    streams = []

    def interrupted(signum, frame):
        raise TimeoutError('Diagnostic deadline or termination signal')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(o.seconds)

    def status(phase, **extra):
        save(root / 'pipeline_status.json', dict(phase=phase, coordinator_pid=os.getpid(), gpus=gpus,
             elapsed_seconds=time.monotonic() - started, deadline_seconds=o.seconds, **extra))

    try:
        status('preparing')
        # Freeze completed weights before Trainer deletes an older checkpoint.
        checkpoints = []
        for state in (PARALLEL / 'sft').glob('checkpoint-*/trainer_state.json'):
            cp = state.parent
            if all((cp / name).is_file() for name in ('model.safetensors', 'config.json', 'tokenizer.json')):
                checkpoints.append((json.loads(state.read_text())['global_step'], cp))
        step, checkpoint = max(checkpoints)
        frozen = root / f'parallel_checkpoint_{step}'
        frozen.mkdir()
        for source in checkpoint.iterdir():
            if source.is_file() and (source.name == 'model.safetensors' or source.suffix in ('.json', '.txt', '.jinja')):
                if source.name == 'model.safetensors':
                    os.link(source, frozen / source.name)
                else:
                    shutil.copy2(source, frozen / source.name)
        serial_protocol = json.loads((SERIAL / 'config.json').read_text())['latent_protocol']
        parallel_protocol = json.loads((frozen / 'config.json').read_text())['latent_protocol']
        assert serial_protocol['label_model_sha256'] == parallel_protocol['label_model_sha256']
        assert serial_protocol['sid_sha256'] == parallel_protocol['sid_sha256']
        snapshot = root / 'source_snapshot'
        shutil.copytree(PARALLEL / 'source_snapshot/latent', snapshot / 'latent',
                        ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(PARALLEL / 'source_snapshot/eval_metric_protocol.json', snapshot / 'eval_metric_protocol.json')
        sample = root / 'sample.json'
        shutil.copy2(PROJECT / 'outputs/latent_qwen3_0.6b_vq2_15ep_20261010/eval/sample256/sample.json', sample)
        ids = json.loads(sample.read_text())['row_ids']
        manifest = dict(gpus=gpus, concurrent=concurrent, examples=len(ids),
            parallel_original_checkpoint=str(checkpoint), parallel_step=step,
            parallel_epoch=json.loads((frozen / 'trainer_state.json').read_text())['epoch'],
            serial_model=str(SERIAL), serial_step=4140, serial_epoch=15,
            matched_training_steps=False, parallel_weights_frozen_by_hardlink=True,
            sample_sha256=digest(sample), model_hashes={'parallel': digest(frozen / 'model.safetensors'),
            'serial': digest(SERIAL / 'model.safetensors')}, methods=['single20', 'four20_marginal'],
            max_batch=16, deadline_seconds=o.seconds,
            timing_note='Batched throughput; different model GPU IDs may differ, no matched inference speed conclusion')
        save(root / 'manifest.json', manifest)
        env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
            TOKENIZERS_PARALLELISM='false', PYTHONPATH=str(snapshot) + ':' + str(PROJECT),
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        results = {}
        # Parallel first ensures that the new checkpoint is assessed promptly.
        jobs = []
        for gpu, (name, module, model) in zip(gpus, [('parallel', 'latent.eval_parallel', frozen),
                                    ('serial', 'latent.eval_unified_small', SERIAL)]):
            command = [PYTHON, '-u', '-m', module, '--model', str(model),
                '--data', str(DATA / 'valid' / f'{STEM}.csv'),
                '--index', str(DATA / 'Industrial_and_Scientific/Industrial_and_Scientific.index.json'),
                '--sample', str(sample), '--output', str(root / f'{name}.json'), '--max-batch', '16']
            if name == 'serial':
                command += ['--kind', 'latent']
            save(root / f'{name}.command.json', command)
            stream = (root / f'{name}.log').open('w')
            streams.append(stream)
            worker = subprocess.Popen(command, cwd=snapshot, env=dict(env, CUDA_VISIBLE_DEVICES=gpu),
                                      stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            active.append(worker)
            jobs.append((name, worker, stream))
            status('evaluating_dual_models' if concurrent else 'evaluating_' + name,
                   worker_pids={n: w.pid for n, w, _ in jobs}, parallel_step=step)
            if not concurrent:
                worker.wait(timeout=max(1, deadline - time.monotonic()))
        for name, worker, stream in jobs:
            code = worker.wait(timeout=max(1, deadline - time.monotonic()))
            stream.close()
            save(root / f'{name}.exit_code.json', {'exit_code': code})
            if code:
                raise RuntimeError(f'{name} evaluator failed exit {code}')
            result = json.loads((root / f'{name}.json').read_text())
            assert result['metadata']['sample_sha256'] == manifest['sample_sha256']
            assert set(result['rows']) == set(manifest['methods'])
            for rows in result['rows'].values():
                assert len(rows) == len(ids) and {r['row_id'] for r in rows} == set(ids)
                assert all(len(r['predict']) == len(set(r['predict'])) for r in rows)
            results[name] = result
        metrics = {}
        for method in manifest['methods']:
            a = {r['row_id']: r['output'] for r in results['serial']['rows'][method]}
            b = {r['row_id']: r['output'] for r in results['parallel']['rows'][method]}
            assert a == b
            metrics[method] = {name: summarize(result['rows'][method]) for name, result in results.items()}
        for key in ('data_sha256', 'index_sha256', 'protocol'):
            assert results['serial']['metadata'][key] == results['parallel']['metadata'][key]
        save(root / 'comparison.json', dict(examples=len(ids), metrics=metrics, manifest=manifest,
             independently_recomputed_six_metrics=True, verified_same_ids_and_targets=True))
        columns = [f'{name}@{k}' for name in ('Recall', 'NDCG') for k in (5, 10, 20)]
        lines = ['# 并行兴趣中间checkpoint与串行最终模型小样本比较', '',
                 f'相同{len(ids)}条验证样本；并行step{step}、{manifest["parallel_epoch"]:.1f}轮，串行4140步、15轮。训练轮数不同，不能据此给出架构最终胜负。', '',
                 '| 模型 | 协议 | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 | 满20比例 |',
                 '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for method in manifest['methods']:
            for name in ('serial', 'parallel'):
                m = metrics[method][name]
                lines.append('| ' + name + ' | ' + method + ' | ' +
                    ' | '.join(f'{m[k]:.5f}' for k in columns) + f' | {m["fraction_with20"]:.1%} |')
        lines += ['', f'本轮使用GPU{gpus[0]}/{gpus[1]}；不报告独立请求速度结论。BF16批次形状可改变边缘排序，256样本仅为诊断。']
        (root / 'comparison.md').write_text('\n'.join(lines) + '\n')
        status('complete', parallel_step=step, report=str(root / 'comparison.md'))
        print((root / 'comparison.md').read_text(), flush=True)
    except Exception as error:
        status('failed_or_deadline', error=str(error))
        raise
    finally:
        signal.alarm(0)
        for worker in active:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    main()
