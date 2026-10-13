"""Train-only capacity-balanced residual kmeans for history interest labels.

No encoder, PCA, normalization, or collision suffix. Fit assignments are saved
for diagnosis; every exported split uses the same independent nearest rule.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from k_means_constrained import KMeansConstrained
from threadpoolctl import threadpool_limits

from .core import sha, write_json
from .vq import code_stats


def capacities(n, k):
    low, high = max(1, n // k - 1), n // k + 1
    if n < k or low * k > n or high * k < n:
        raise ValueError('Invalid capacity constraints')
    return low, high


def quantize(x, centers):
    residual = x.float()
    codes = []
    for center in centers:
        d = residual.square().sum(1, keepdim=True) + center.square().sum(1)[None] - 2 * residual @ center.T
        code = d.argmin(1)
        residual = residual - center[code]
        codes.append(code)
    return torch.stack(codes, dim=1), residual


def fit(o):
    root, source = Path(o.output), Path(o.embeddings)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'labels.json').exists():
        raise ValueError('Labels already complete; choose a new output directory')
    manifest = json.loads((source / 'embeddings.json').read_text())
    arrays = {}
    for split in ('train', 'valid', 'test'):
        path = source / f'{split}.npy'
        if sha(path) != manifest['embedding_hashes'][split]:
            raise ValueError(f'Embedding hash mismatch: {split}')
        array = np.load(path, mmap_mode='r')
        if array.shape != (manifest['counts'][split], manifest['recipe']['dim']):
            raise ValueError('Invalid embedding shape')
        arrays[split] = array
    signature = dict(embedding_manifest_sha256=sha(source / 'embeddings.json'),
        codebook_size=o.codebook_size, levels=3, seed=o.seed, max_iter=o.max_iter,
        tol=o.tol, n_init=o.n_init, size_limits=capacities(len(arrays['train']), o.codebook_size))
    config_file = root / 'fit_config.json'
    if config_file.exists() and json.loads(config_file.read_text()) != json.loads(json.dumps(signature)):
        raise ValueError('Cannot resume a different clustering configuration')
    write_json(config_file, signature)
    residual = np.array(arrays['train'], dtype=np.float32, copy=True)
    if not np.isfinite(residual).all():
        raise ValueError('Nonfinite train embeddings')
    centers, fit_codes, stats = [], [], []
    low, high = signature['size_limits']
    for level in range(3):
        checkpoint = root / f'fit_level{level}.npz'
        if checkpoint.exists():
            with np.load(checkpoint) as saved:
                center, codes = saved['centers'], saved['codes']
            print(f'Resuming saved level {level + 1}', flush=True)
            elapsed = None
        else:
            seed = int(np.random.RandomState(o.seed + level).randint(0, 2**31 - 1))
            model = KMeansConstrained(n_clusters=o.codebook_size, size_min=low, size_max=high,
                max_iter=o.max_iter, tol=o.tol, random_state=seed, n_init=o.n_init,
                verbose=True, n_jobs=o.jobs)
            print(f'FIT level={level + 1}/3 shape={residual.shape} capacities={low}..{high}', flush=True)
            started = time.perf_counter()
            with threadpool_limits(limits=o.threads):
                codes = model.fit_predict(residual).astype(np.int32)
            center = model.cluster_centers_.astype(np.float32)
            elapsed = time.perf_counter() - started
            temp = checkpoint.with_suffix('.tmp')
            with temp.open('wb') as file:
                np.savez_compressed(file, centers=center, codes=codes)
            os.replace(temp, checkpoint)
        if center.shape != (o.codebook_size, residual.shape[1]) or codes.shape != (len(residual),):
            raise ValueError('Invalid saved clustering stage')
        counts = np.bincount(codes, minlength=o.codebook_size)
        if counts.min() < low or counts.max() > high:
            raise ValueError('Fit assignment violates balance capacities')
        residual -= center[codes]
        info = dict(level=level + 1, seconds=elapsed, minimum_count=int(counts.min()),
            maximum_count=int(counts.max()), residual_mse=float(np.square(residual).mean()))
        print('FIT_COMPLETE', json.dumps(info), flush=True)
        centers.append(center)
        fit_codes.append(codes)
        stats.append(info)
        write_json(root / 'fit_progress.json', dict(completed_levels=len(centers), stages=stats))
    codebooks = np.stack(centers)
    np.savez_compressed(root / 'rq_kmeans.npz', codebooks=codebooks)
    constrained = np.stack(fit_codes, axis=1)
    np.save(root / 'train_capacity_codes.npy', constrained)
    device = 'cuda' if torch.cuda.is_available() and not o.cpu else 'cpu'
    tensors = torch.from_numpy(codebooks).to(device)
    hashes, usage, mse = {}, {}, {}
    for split, array in arrays.items():
        collected, error = [], 0.
        with torch.no_grad():
            for start in range(0, len(array), o.batch):
                x = torch.from_numpy(np.array(array[start:start + o.batch], copy=True)).to(device)
                if not torch.isfinite(x).all():
                    raise ValueError(f'Nonfinite {split} embeddings')
                codes, remainder = quantize(x, tensors)
                collected.append(codes.cpu())
                error += remainder.square().sum().item()
        codes = torch.cat(collected)
        np.save(root / f'{split}_codes.npy', codes.numpy())
        hashes[split] = sha(root / f'{split}_codes.npy')
        usage[split] = code_stats(codes, o.codebook_size)
        mse[split] = error / array.size
        print('EXPORT', split, json.dumps(dict(mse=mse[split], usage=usage[split])), flush=True)
    nearest = np.load(root / 'train_codes.npy')
    disagreement = (nearest != constrained).mean(axis=0).tolist()
    write_json(root / 'labels.json', dict(version=1, k=3, codebook_size=o.codebook_size,
        tokenizer_type='balanced_residual_kmeans', source_hashes=manifest['source_hashes'],
        sid_sha256=manifest['sid_sha256'], embedding_recipe=manifest['recipe'],
        embedding_manifest_sha256=sha(source / 'embeddings.json'), fit_split='train',
        selection_split=None, label_model_file='rq_kmeans.npz',
        label_model_sha256=sha(root / 'rq_kmeans.npz'), label_hashes=hashes,
        code_statistics=usage, reconstruction_mse=mse, validation_reconstruction=mse['valid'],
        fit_options=vars(o), fit_statistics=stats, train_fit_vs_export_disagreement=disagreement,
        assignment_recipe=dict(fitting='full_train_capacity_constrained',
            export='independent_nearest_residual_all_splits', exported_labels_exactly_balanced=False)))
    print('LABELS_COMPLETE', str(root / 'labels.json'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embeddings', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--codebook_size', type=int, default=256)
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--tol', type=float, default=1e-7)
    parser.add_argument('--n_init', type=int, default=3)
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    o = parser.parse_args()
    if min(o.codebook_size, o.max_iter, o.n_init, o.jobs, o.threads, o.batch) < 1 or o.tol <= 0:
        parser.error('All counts and tolerance must be positive')
    fit(o)


if __name__ == '__main__':
    main()
