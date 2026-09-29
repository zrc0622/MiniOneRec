"""Only replace the original primary SFT task; retain auxiliary datasets."""
from .core import atomic_id, history_records, load_labels


def supervised_example(record, codes, tokenizer, names, max_length=512):
    latent = [atomic_id(tokenizer, names[k][int(c)]) for k, c in enumerate(codes)]
    target = latent + tokenizer.encode(record['target'] + '\n', add_special_tokens=False) + [tokenizer.eos_token_id]
    prompt = tokenizer.encode(record['prompt'], add_special_tokens=False)
    if len(target) >= max_length:
        raise ValueError('Response exceeds sequence budget')
    prompt = prompt[-(max_length - len(target)):]
    ids = prompt + target
    return dict(input_ids=ids, attention_mask=[1] * len(ids), labels=[-100] * len(prompt) + target,
                interest_mask=[0] * len(prompt) + [1] * 3 + [0] * (len(target) - 3))


def sft_datasets(train_file, eval_file, items, index_file, index, tokenizer, protocol, label_root, seed=42):
    from data import SidItemFeatDataset, FusionSeqRecDataset
    result = []
    for split, path in (('train', train_file), ('valid', eval_file)):
        codes, manifest = load_labels(label_root, split, path, index)
        if manifest['vq_sha256'] != protocol['label_model_sha256']:
            raise ValueError('Tokenizer and label generator differ')
        records = history_records(path, index)
        result.append([supervised_example(r, c, tokenizer, protocol['tokens']) for r, c in zip(records, codes)])
    for ds in (SidItemFeatDataset(items, index_file, tokenizer, max_len=512, seed=seed),
               FusionSeqRecDataset(train_file, items, index_file, tokenizer, max_len=512, seed=seed)):
        result[0].extend(dict(x, interest_mask=[0] * len(x['input_ids'])) for x in ds)
    return result


def rl_records(train_file, items, index_file, index, title_sample=10000):
    from data import RLTitle2SidDataset, RLSeqTitle2SidDataset
    records = history_records(train_file, index)
    # Exact original auxiliary constructors, including original seed=0 sampling.
    # These tasks retain SID-only responses and original prompts in BOTH RL modes.
    for ds, task in ((RLTitle2SidDataset(items, index_file, sample=-1), 'item_identification'),
                     (RLSeqTitle2SidDataset(train_file, sample=title_sample), 'history_title')):
        records.extend(dict(prompt=x['prompt'], target=x['completion'].strip(),
                            task=task, auxiliary=True) for x in ds)
    return records
