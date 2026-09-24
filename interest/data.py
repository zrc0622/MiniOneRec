"""Shared prompt construction for Mid-SFT, both RL variants and evaluation."""
import ast
import csv

from .core import ITEM, history_prompt, pseudo_interests


def history_records(path, index, kind="sid", sample=-1, seed=42):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if sample > 0 and len(rows) > sample:
        import pandas as pd
        rows = pd.DataFrame(rows).sample(sample, random_state=seed).to_dict("records")
    result = []
    for row in rows:
        history = ast.literal_eval(row["history_item_sid"])
        ids = ast.literal_eval(row["history_item_id"])
        if history != ["".join(index[str(i)]) for i in ids]:
            raise ValueError("CSV history and SID index mismatch")
        target = "".join(index[str(row["item_id"])])
        if target != row["item_sid"]:
            raise ValueError("CSV target and SID index mismatch")
        values = history if kind == "sid" else ast.literal_eval(row["history_item_title"])
        if len(values) != len(history) or not history:
            raise ValueError("Malformed history")
        result.append(dict(prompt=history_prompt(values, kind), target=target,
                           history=history, task=f"history_{kind}", auxiliary=False))
    return result


def rl_records(train_file, item_file, index_file, index, seed=42, title_sample=10000):
    # Preserve the existing title/description identification examples, including
    # their three-level SID targets, separately from full recommendation SIDs.
    from data import RLTitle2SidDataset

    result = history_records(train_file, index)
    auxiliary = RLTitle2SidDataset(item_file, index_file, sample=-1, seed=seed)
    result.extend(dict(prompt=x["prompt"], target=x["completion"].strip(),
                       history=[], task="item_identification", auxiliary=True) for x in auxiliary)
    result.extend(history_records(train_file, index, "title", title_sample, seed))
    return result


def supervised_example(record, tokenizer, k, max_length=512):
    from .core import atomic_id

    # Explicit token assembly avoids boundary tokenization differences between
    # teacher forcing and staged generation.
    interests = pseudo_interests(record["history"], k)
    prefix = [atomic_id(tokenizer, x) for x in interests] + [atomic_id(tokenizer, ITEM)]
    target = prefix + tokenizer.encode(record["target"], add_special_tokens=False) + [tokenizer.eos_token_id]
    prompt = tokenizer.encode(record["prompt"], add_special_tokens=False)
    if len(target) >= max_length:
        raise ValueError("Target exceeds the sequence budget")
    prompt = prompt[-(max_length - len(target)):]
    tokens = prompt + target
    return dict(input_ids=tokens, attention_mask=[1] * len(tokens),
                labels=[-100] * len(prompt) + target,
                interest_mask=[0] * len(prompt) + [1] * len(prefix) + [0] * (len(target) - len(prefix)))


def mid_datasets(train_file, eval_file, item_file, index_file, index, tokenizer, k,
                 seed=42, max_length=512, title_sample=10000):
    from data import SidItemFeatDataset, FusionSeqRecDataset

    records = history_records(train_file, index)
    records += history_records(train_file, index, "title", title_sample, seed)
    train = [supervised_example(r, tokenizer, k, max_length) for r in records]
    # Retain both original SFT auxiliary datasets without introducing interest labels.
    for ds in (
        SidItemFeatDataset(item_file, index_file, tokenizer, max_len=max_length, seed=seed),
        FusionSeqRecDataset(train_file, item_file, index_file, tokenizer, max_len=max_length, seed=seed),
    ):
        train.extend(dict(x, interest_mask=[0] * len(x["input_ids"])) for x in ds)
    valid = [supervised_example(r, tokenizer, k, max_length) for r in history_records(eval_file, index)]
    return train, valid
