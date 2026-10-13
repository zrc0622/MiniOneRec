"""Versioned data and token protocol, independent of training libraries."""
import ast
import csv
import hashlib
import json
from pathlib import Path

from interest.core import atomic_id, index_hash, load_index, rank_unique_items

K = 3


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def token_names(size, k=K):
    if k not in (2, 3):
        raise ValueError('Supported latent slot counts are 2 and 3')
    return [[f'<latent{slot + 1}_{code}>' for code in range(size)] for slot in range(k)]


def rows(path, index):
    with open(path, newline='') as f:
        result = list(csv.DictReader(f))
    for row in result:
        ids = ast.literal_eval(row['history_item_id'])
        history = ast.literal_eval(row['history_item_sid'])
        titles = ast.literal_eval(row['history_item_title'])
        if not ids or not isinstance(titles, list) or len(ids) != len(titles):
            raise ValueError('Missing or misaligned history titles')
        if history != [''.join(index[str(i)]) for i in ids]:
            raise ValueError('CSV history and SID index differ')
        if row['item_sid'] != ''.join(index[str(row['item_id'])]):
            raise ValueError('CSV target and SID index differ')
        if not all(isinstance(t, str) and t.strip() for t in titles):
            raise ValueError('History title must be nonempty text')
    return result


def history_key(row):
    # Deliberately independent of user ID, target item, and its title.
    titles = ast.literal_eval(row['history_item_title'])
    return hashlib.sha256(json.dumps(titles, ensure_ascii=False).encode()).hexdigest()


def history_text(titles):
    return ('The user interacted with the following items, from oldest to newest:\n'
            + '\n'.join(f'{i + 1}. {json.dumps(t, ensure_ascii=False)}' for i, t in enumerate(titles)))


def bounded_history(titles, tokenizer, max_length):
    """Keep the most recent complete titles. A single oversized title is rejected."""
    titles = list(titles)
    while titles:
        ids = tokenizer.encode(history_text(titles), add_special_tokens=True)
        if len(ids) <= max_length:
            return ids, len(titles)
        titles.pop(0)
    raise ValueError('The latest title alone exceeds the embedding token budget')


def history_target_key(row):
    """Teacher labels may depend on the next item; never use as inference input."""
    title = row.get('item_title')
    if not isinstance(title, str) or not title.strip():
        raise ValueError('Target-conditioned embedding requires a nonempty item_title')
    payload = (ast.literal_eval(row['history_item_title']), str(row['item_id']), title)
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def history_target_text(titles, target_title):
    return history_text(titles) + '\nThe next item the user interacted with is:\n' + json.dumps(target_title, ensure_ascii=False)


def bounded_history_target(titles, target_title, tokenizer, max_length):
    """Retain the complete target and most recent complete historical titles."""
    if not isinstance(target_title, str) or not target_title.strip():
        raise ValueError('Target-conditioned embedding requires a nonempty item_title')
    titles = list(titles)
    while titles:
        ids = tokenizer.encode(history_target_text(titles, target_title), add_special_tokens=True)
        if len(ids) <= max_length:
            return ids, len(titles)
        titles.pop(0)
    raise ValueError('Latest history title plus complete target exceeds the embedding token budget')


def history_prompt(row, k=K):
    count = {2: 'two', 3: 'three'}[k]
    history = ', '.join(ast.literal_eval(row['history_item_sid']))
    return (
        'Below is an instruction that describes a task, paired with an input that provides further context. '
        'Write a response that appropriately completes the request. \n\n'
        '### Instruction:\nCan you predict the next possible item that the user may expect?\n\n'
        f'### User Input: \nThe user has interacted with items {history} in chronological order. '
        f'First generate {count} latent interest tokens, then predict the semantic ID of the next item.\n\n'
        '### Response:\n'
    )


def history_records(path, index, k=K):
    return [dict(prompt=history_prompt(row, k), target=row['item_sid'], task='history_sid', auxiliary=False)
            for row in rows(path, index)]


def initialize(model, tokenizer, index, manifest):
    if getattr(model.config, 'latent_protocol', None) or getattr(model.config, 'interest_protocol', None):
        raise ValueError('New SFT must start from pretrained backbone weights')
    k = manifest.get('k', K)
    names = token_names(manifest['codebook_size'], k)
    vocabulary = tokenizer.get_vocab()
    if any(t in vocabulary for group in names for t in group):
        raise ValueError('Latent tokens already exist')
    # SID rows use the same HF mean initialization as the baseline. Latent rows
    # get independent parameters, not copies of matching item SID code numbers.
    tokenizer.add_tokens(sorted({t for sid in index.values() for t in sid}))
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.add_tokens([t for group in names for t in group])
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.latent_protocol = dict(version=1, k=k, codebook_size=manifest['codebook_size'],
        sid_sha256=index_hash(index), label_model_sha256=(manifest.get('label_model_sha256') or manifest['vq_sha256']),
        tokens=names, response={2: 'two', 3: 'three'}[k] + '_latents_then_full_sid_newline_eos')
    return model.config.latent_protocol


def validate_protocol(config, tokenizer, index, *, allow_parallel=False):
    p = getattr(config, 'latent_protocol', None)
    if not p or p.get('version') != 1 or p.get('k') not in (2, 3):
        raise ValueError('Checkpoint has no compatible latent protocol')
    if p.get('prediction') == 'parallel_bidirectional_queries' and not allow_parallel:
        raise ValueError('Parallel-query checkpoints require the parallel evaluator, not autoregressive latent generation')
    if p['sid_sha256'] != index_hash(index) or p['tokens'] != token_names(p['codebook_size'], p['k']):
        raise ValueError('Checkpoint/index/token vocabulary mismatch')
    for group in p['tokens']:
        for t in group:
            atomic_id(tokenizer, t)
    return p


def load_labels(root, split, csv_file, index):
    import numpy as np
    root = Path(root)
    manifest = json.loads((root / 'labels.json').read_text())
    if manifest.get('version') != 1 or manifest.get('k') not in (2, 3):
        raise ValueError('Unsupported latent label protocol')
    if manifest['sid_sha256'] != index_hash(index) or manifest['source_hashes'][split] != sha(csv_file):
        raise ValueError('Latent labels belong to different data/SID index')
    model_file = manifest.get('label_model_file', 'vq.pt')
    if Path(model_file).name != model_file:
        raise ValueError('Label model must be a file inside the labels directory')
    model_hash = manifest.get('label_model_sha256') or manifest['vq_sha256']
    if sha(root / model_file) != model_hash:
        raise ValueError('Label model changed after labels were generated')
    path = root / f'{split}_codes.npy'
    if sha(path) != manifest['label_hashes'][split]:
        raise ValueError('Label file changed')
    codes = np.load(path)
    expected_rows = len(rows(csv_file, index))
    if codes.shape != (expected_rows, manifest['k']) or codes.dtype.kind not in 'iu' or codes.min() < 0 or codes.max() >= manifest['codebook_size']:
        raise ValueError('Invalid label shape/range/row alignment')
    return codes, manifest
