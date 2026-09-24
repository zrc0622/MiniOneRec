"""Interest labels and checkpoint protocol; no training dependencies required."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

NONE = "<interest_none>"
ITEM = "<interest_item>"
PROTOCOL_VERSION = 1


def load_index(path):
    index = json.loads(Path(path).read_text())
    if not index:
        raise ValueError("SID index is empty")
    for sid in index.values():
        if len(sid) not in (3, 4) or any(
            re.fullmatch(rf"<{letter}_\d+>", token) is None
            for letter, token in zip("abcd", sid)
        ):
            raise ValueError(f"Invalid SID: {sid}")
    if len({"".join(x) for x in index.values()}) != len(index):
        raise ValueError("Full item SIDs must be unique")
    return index


def index_hash(index):
    return hashlib.sha256(json.dumps(index, sort_keys=True).encode()).hexdigest()


def interest_token(first_sid):
    if re.fullmatch(r"<a_\d+>", first_sid) is None:
        raise ValueError(first_sid)
    return first_sid.replace("<a_", "<interest_")


def pseudo_interests(history, k=2):
    """Frequency first; ties use latest occurrence. Only history is an input."""
    if k < 1 or not history:
        raise ValueError("Need k>=1 and a nonempty history")
    first = []
    for sid in history:
        match = re.match(r"<a_\d+>", sid)
        if not match:
            raise ValueError(f"History contains invalid SID: {sid}")
        first.append(match[0])
    counts = Counter(first)
    latest = {sid: i for i, sid in enumerate(first)}
    chosen = sorted(counts, key=lambda sid: (-counts[sid], -latest[sid]))[:k]
    return [interest_token(x) for x in chosen] + [NONE] * (k - len(chosen))


def history_prompt(values, kind="sid"):
    if kind not in ("sid", "title"):
        raise ValueError(kind)
    history = ", ".join(values) if kind == "sid" else ", ".join(json.dumps(x, ensure_ascii=False) for x in values)
    return (
        "### User Input:\n"
        f"The user's historical item {kind} sequence in chronological order is: {history}.\n"
        "Predict the user's interests, then recommend the next item by its semantic ID.\n"
        "### Response:\n"
    )


def atomic_id(tokenizer, token):
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1 or tokenizer.convert_tokens_to_ids(token) != ids[0]:
        raise ValueError(f"Token is not in the trained atomic vocabulary: {token}")
    return ids[0]


def initialize_interests(model, tokenizer, index, k):
    """Copy rows, never alias SID and interest parameters. Call before ZeRO wrap."""
    import torch

    if getattr(model.config, "interest_protocol", None):
        raise ValueError("Mid-train must start from the original SFT checkpoint")
    first = sorted({x[0] for x in index.values()})
    if not 1 <= k <= len(first):
        raise ValueError("Invalid number of interests")
    source_ids = [atomic_id(tokenizer, x) for x in first]
    names = [interest_token(x) for x in first] + [NONE, ITEM]
    if any(x in tokenizer.get_vocab() for x in names):
        raise ValueError("Interest tokens already exist; refusing to reset learned rows")
    tokenizer.add_tokens(names)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    target_ids = [atomic_id(tokenizer, x) for x in names[:len(first)]]
    with torch.no_grad():
        matrices = [model.get_input_embeddings().weight, model.get_output_embeddings().weight]
        seen = set()
        for matrix in matrices:
            if id(matrix) in seen:
                continue
            seen.add(id(matrix))
            matrix[target_ids] = matrix[source_ids].clone()
            matrix[atomic_id(tokenizer, NONE)] = matrix[source_ids].mean(0)
            matrix[atomic_id(tokenizer, ITEM)] = matrix[tokenizer.eos_token_id].clone()
    model.config.interest_protocol = {
        "version": PROTOCOL_VERSION, "k": k, "sid_sha256": index_hash(index),
        "interest_tokens": names[:len(first)], "none": NONE, "item_separator": ITEM,
        "pseudo_label_rule": "frequency_then_recency", "history_limit_from_preprocessing": 10,
    }


def validate_protocol(config, tokenizer, index):
    protocol = getattr(config, "interest_protocol", None)
    if not protocol or protocol.get("version") != PROTOCOL_VERSION:
        raise ValueError("Checkpoint has no compatible interest protocol; run mid-train first")
    if protocol["sid_sha256"] != index_hash(index):
        raise ValueError("Checkpoint and SID index do not match")
    for token in protocol["interest_tokens"] + [NONE, ITEM]:
        atomic_id(tokenizer, token)
    return protocol


def rank_unique_items(paths, limit=50):
    """paths: (full SID, joint log probability); retain the best path per item."""
    scores = {}
    for sid, score in paths:
        scores[sid] = max(scores.get(sid, float("-inf")), float(score))
    return sorted(scores, key=lambda sid: (-scores[sid], sid))[:limit]
