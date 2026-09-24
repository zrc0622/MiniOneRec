"""Shared deterministic beam evaluation for Mid-SFT and either RL policy."""
import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from .core import load_index, rank_unique_items, validate_protocol
from .data import history_records
from .generation import Grammar, action_logps, generate, pack_trajectories


def summarize(rows):
    if not rows:
        raise ValueError("No evaluation records")
    values = {}
    for k in (1, 3, 5, 10, 20, 50):
        hr = ndcg = 0
        for row in rows:
            predictions = row["predict"][:k]
            if row["output"] in predictions:
                rank = predictions.index(row["output"])
                hr += 1
                ndcg += 1 / math.log2(rank + 2)
        values[f"HR@{k}"] = hr / len(rows)
        values[f"NDCG@{k}"] = ndcg / len(rows)
    values["mean_unique_candidates"] = sum(len(r["predict"]) for r in rows) / len(rows)
    values["examples"] = len(rows)
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--data")
    p.add_argument("--index")
    p.add_argument("--output", required=True)
    p.add_argument("--merge", nargs="+")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--beams", type=int, default=50)
    options = p.parse_args()
    if options.merge:
        rows = [r for path in options.merge for r in json.loads(Path(path).read_text())]
        ids = [r["row_id"] for r in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(range(len(ids))):
            raise ValueError("Missing or overlapping evaluation shards")
        rows.sort(key=lambda r: r["row_id"])
        Path(options.output).write_text(json.dumps(rows, ensure_ascii=False) + "\n")
        metrics = summarize(rows)
        Path(options.output + ".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))
        return
    if not (0 <= options.shard < options.shards) or options.beams < 50:
        raise ValueError("Invalid shard or insufficient beam budget for HR@50")
    index = load_index(options.index)
    tokenizer = AutoTokenizer.from_pretrained(options.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(options.model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
    protocol = validate_protocol(model.config, tokenizer, index)
    grammar = Grammar(tokenizer, index, protocol["k"])
    records = history_records(options.data, index)
    rows = []
    for row_id in tqdm(range(options.shard, len(records), options.shards)):
        record = records[row_id]
        prompt = tokenizer.encode(record["prompt"], add_special_tokens=False)[-512:]
        # Identical search for all interest checkpoints. A beam is a full
        # interest+item path, not a distinct item; duplicate products are removed.
        samples = generate(model, prompt, grammar, options.beams, sample=False)
        scores = []
        for start in range(0, len(samples), 8):
            chunk = samples[start:start + 8]
            packed = pack_trajectories([prompt] * len(chunk), chunk, [grammar.k + 1] * len(chunk),
                                       [False] * len(chunk), grammar, device)
            with torch.no_grad():
                logps = action_logps(model, packed)
            scores.extend((logps * (packed["interest_mask"] + packed["item_mask"])).sum(1).tolist())
        paths = [(grammar.item(seq), score) for seq, score in zip(samples, scores)]
        rows.append(dict(row_id=row_id, output=record["target"], predict=rank_unique_items(paths),
                         beam_paths=options.beams,
                         interests=[tokenizer.decode(s[:grammar.k], skip_special_tokens=False) for s in samples]))
    Path(options.output).write_text(json.dumps(rows, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
