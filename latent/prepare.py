"""Frozen history encoding, train-only parallel VQ fitting, and label export."""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoTokenizer, set_seed
from tqdm import tqdm

from .core import bounded_history, history_key, index_hash, load_index, rows, sha, write_json
from .vq import DEFAULT_LAYERS, DEFAULT_LATENT_DIM, ParallelVQ, code_stats

SPLITS = ('train', 'valid', 'test')


def sources(o):
    index = load_index(o.index)
    paths = {s: Path(getattr(o, s)) for s in SPLITS}
    return index, paths, {s: sha(p) for s, p in paths.items()}


def mean_pool(hidden, mask, dim):
    if dim > hidden.shape[-1]:
        raise ValueError('Requested embedding dimension exceeds model output')
    # Match the existing encoder's FP32 accumulation even with FP16 weights.
    # Summing up to 2048 FP16 activations can otherwise overflow.
    hidden = hidden.float()
    mask = mask[..., None].float()
    # Preserve the existing embedding recipe: masked mean, truncate, no L2.
    return ((hidden * mask).sum(1) / mask.sum(1).clamp_min(1))[:, :dim].float()


def encode(o):
    index, paths, hashes = sources(o)
    root = Path(o.output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'embeddings.json').exists() or (root / f'shard{o.shard}.json').exists():
        raise ValueError('Embedding output already exists; choose a fresh LATENT_ROOT')
    if not 0 <= o.shard < o.shards:
        raise ValueError('Invalid shard')
    device = 'cuda' if torch.cuda.is_available() and not o.cpu else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(o.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(o.model, torch_dtype=torch.float16 if device == 'cuda' else torch.float32).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    counts, outputs = {}, {}
    for split, path in paths.items():
        records = rows(path, index)
        selected = list(range(o.shard, len(records), o.shards))
        vectors = np.empty((len(selected), o.dim), dtype=np.float32)
        kept = []
        # Dedup identical histories inside each batch; labels remain row-aligned.
        for start in tqdm(range(0, len(selected), o.batch), desc=split):
            ids = selected[start:start + o.batch]
            features, positions, unique = [], [], {}
            for i in ids:
                key = history_key(records[i])
                encoded, count = bounded_history(ast.literal_eval(records[i]['history_item_title']), tokenizer, o.max_length)
                kept.append(count)
                if key not in unique:
                    unique[key] = len(features)
                    features.append(dict(input_ids=encoded, attention_mask=[1] * len(encoded)))
                positions.append(unique[key])
            batch = tokenizer.pad(features, padding=True, return_tensors='pt').to(device)
            with torch.inference_mode():
                vector = mean_pool(model(**batch).last_hidden_state, batch['attention_mask'], o.dim).cpu().numpy()
            if not np.isfinite(vector).all():
                raise ValueError('Non-finite history embedding')
            vectors[start:start + len(ids)] = vector[positions]
        dest = root / f'{split}.{o.shard}.npz'
        np.savez(dest, row_ids=np.asarray(selected, dtype=np.int64), embeddings=vectors, kept_titles=np.asarray(kept))
        counts[split], outputs[dest.name] = len(records), sha(dest)
    model_hashes = {name: sha(Path(o.model) / name) for name in ('config.json', 'tokenizer.json') if (Path(o.model) / name).is_file()}
    write_json(root / f'shard{o.shard}.json', dict(source_hashes=hashes, sid_sha256=index_hash(index),
        counts=counts, shards=o.shards, shard=o.shard, outputs=outputs,
        recipe=dict(model=str(Path(o.model).resolve()), model_hashes=model_hashes, dim=o.dim,
                    pooling='masked_mean', normalization='none', max_length=o.max_length,
                    text='chronological_numbered_titles_v1', dtype=str(next(model.parameters()).dtype))))


def merge(o):
    root = Path(o.output)
    if (root / 'embeddings.json').exists():
        raise ValueError('Merged embeddings already exist')
    parts = [json.loads((root / f'shard{i}.json').read_text()) for i in range(o.shards)]
    first = parts[0]
    for i, part in enumerate(parts):
        if part['shard'] != i or part['shards'] != o.shards or any(part[k] != first[k] for k in ('source_hashes', 'sid_sha256', 'recipe', 'counts')):
            raise ValueError('Embedding shard provenance mismatch')
        for filename, digest in part['outputs'].items():
            if sha(root / filename) != digest:
                raise ValueError('Changed embedding shard')
    files, retained = {}, {}
    for split in SPLITS:
        count, dim = first['counts'][split], first['recipe']['dim']
        target = root / f'{split}.npy'
        array = np.lib.format.open_memmap(target, mode='w+', dtype='float32', shape=(count, dim))
        seen = np.zeros(count, dtype=bool)
        total_kept = 0
        for i in range(o.shards):
            with np.load(root / f'{split}.{i}.npz') as part:
                ids, vectors = part['row_ids'], part['embeddings']
                if len(ids) != len(set(ids.tolist())) or (ids < 0).any() or (ids >= count).any() or seen[ids].any() or vectors.shape != (len(ids), dim) or not np.isfinite(vectors).all():
                    raise ValueError('Invalid/overlapping embedding rows')
                array[ids] = vectors
                seen[ids] = True
                total_kept += int(part['kept_titles'].sum())
        if not seen.all():
            raise ValueError('Missing embedding rows')
        array.flush()
        del array
        files[split], retained[split] = sha(target), total_kept / count
    write_json(root / 'embeddings.json', {**{k: first[k] for k in ('source_hashes', 'sid_sha256', 'recipe', 'counts')},
                                        'embedding_hashes': files, 'mean_retained_titles': retained})


def fit(o):
    root, source = Path(o.output), Path(o.embeddings)
    if (root / 'vq.pt').exists() or (root / 'labels.json').exists():
        raise ValueError('VQ/labels already exist; choose a fresh LATENT_ROOT')
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / 'embeddings.json').read_text())
    arrays = {}
    for split in SPLITS:
        path = source / f'{split}.npy'
        if sha(path) != manifest['embedding_hashes'][split]:
            raise ValueError('Embedding changed after merge')
        arrays[split] = torch.from_numpy(np.load(path))
        if arrays[split].shape != (manifest['counts'][split], manifest['recipe']['dim']) or not torch.isfinite(arrays[split]).all() or not len(arrays[split]):
            raise ValueError('Invalid embedding array')
    set_seed(o.seed)
    device = 'cuda' if torch.cuda.is_available() and not o.cpu else 'cpu'
    model = ParallelVQ(arrays['train'].shape[1], o.codebook_size, o.latent_dim, o.layers, o.commitment).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f'VQ layers={model.options["layers"]}, parameters={parameter_count:,}', flush=True)
    generator = torch.Generator().manual_seed(o.seed)
    init_ids = torch.randperm(len(arrays['train']), generator=generator)[:min(8192, len(arrays['train']))]
    model.initialize_from_train(arrays['train'][init_ids].to(device), generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=o.lr, weight_decay=0)
    loader = DataLoader(TensorDataset(arrays['train']), batch_size=o.batch, shuffle=True, generator=generator)
    writer = SummaryWriter(root / 'tensorboard_vq')
    best, stale, best_epoch = float('inf'), 0, None
    try:
        for epoch in range(1, o.epochs + 1):
            model.train()
            total = 0
            for (x,) in loader:
                optimizer.zero_grad(set_to_none=True)
                loss, _, _ = model(x.to(device))
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite VQ loss')
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(x)
            model.eval()
            val_rec, val_loss = 0., 0.
            with torch.no_grad():
                for x in arrays['valid'].split(o.batch):
                    loss, components, _ = model(x.to(device))
                    val_loss += float(loss) * len(x)
                    val_rec += float(components['reconstruction']) * len(x)
            val_rec /= len(arrays['valid'])
            writer.add_scalar('train/loss', total / len(arrays['train']), epoch)
            writer.add_scalar('valid/loss', val_loss / len(arrays['valid']), epoch)
            writer.add_scalar('valid/reconstruction', val_rec, epoch)
            print(f'VQ epoch={epoch} valid_reconstruction={val_rec:.6f}', flush=True)
            if val_rec < best:
                best, stale, best_epoch = val_rec, 0, epoch
                torch.save(dict(options=model.options, state=model.state_dict()), root / 'vq.pt')
            else:
                stale += 1
                if stale >= o.patience:
                    break
    finally:
        writer.close()
    saved = torch.load(root / 'vq.pt', map_location=device, weights_only=True)
    model.load_state_dict(saved['state'])
    model.eval()
    hashes, stats = {}, {}
    with torch.no_grad():
        for split, array in arrays.items():
            codes = torch.cat([model.quantize(x.to(device))[2].cpu() for x in array.split(o.batch)])
            np.save(root / f'{split}_codes.npy', codes.numpy())
            hashes[split] = sha(root / f'{split}_codes.npy')
            stats[split] = code_stats(codes, o.codebook_size)
    write_json(root / 'labels.json', dict(version=1, k=3, codebook_size=o.codebook_size,
        source_hashes=manifest['source_hashes'], sid_sha256=manifest['sid_sha256'],
        embedding_recipe=manifest['recipe'], embedding_manifest_sha256=sha(source / 'embeddings.json'),
        vq_sha256=sha(root / 'vq.pt'), label_hashes=hashes, fit_split='train', selection_split='valid',
        best_epoch=best_epoch, validation_reconstruction=best, code_statistics=stats,
        vq_options=model.options, parameter_count=parameter_count, fit_options=vars(o)))
    print(json.dumps(stats, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='stage', required=True)
    e = sub.add_parser('encode')
    for key in ('train', 'valid', 'test', 'index', 'model'):
        e.add_argument('--' + key, required=True)
    e.add_argument('--shard', type=int, default=0)
    e.add_argument('--shards', type=int, default=4)
    e.add_argument('--batch', type=int, default=16)
    e.add_argument('--max_length', type=int, default=2048)
    e.add_argument('--dim', type=int, default=2560)
    m = sub.add_parser('merge')
    m.add_argument('--shards', type=int, default=4)
    f = sub.add_parser('vq')
    f.add_argument('--embeddings', required=True)
    f.add_argument('--codebook_size', type=int, default=256)
    f.add_argument('--latent_dim', type=int, default=DEFAULT_LATENT_DIM)
    f.add_argument('--layers', type=int, nargs='+', default=list(DEFAULT_LAYERS),
                   help='Encoder hidden widths before the 3 * latent_dim output; decoder reverses them')
    f.add_argument('--commitment', type=float, default=0.25)
    f.add_argument('--epochs', type=int, default=50)
    f.add_argument('--patience', type=int, default=5)
    f.add_argument('--batch', type=int, default=512)
    f.add_argument('--lr', type=float, default=1e-3)
    f.add_argument('--seed', type=int, default=42)
    for sp in (e, m, f):
        sp.add_argument('--output', required=True)
        sp.add_argument('--cpu', action='store_true')
    o = p.parse_args()
    for name in ('batch', 'max_length', 'dim', 'shards', 'epochs', 'patience', 'codebook_size', 'latent_dim', 'lr'):
        if hasattr(o, name) and getattr(o, name) <= 0:
            p.error(f'{name} must be positive')
    if o.stage == 'vq' and any(n <= 0 for n in o.layers):
        p.error('layers must be positive integers')
    {'encode': encode, 'merge': merge, 'vq': fit}[o.stage](o)


if __name__ == '__main__':
    main()
