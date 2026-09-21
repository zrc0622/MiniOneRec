#!/usr/bin/env python3
"""Build an isolated SID ablation from saved assignments and existing CSV rows."""

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from convert_dataset import create_item_info_file


CATEGORY = "Industrial_and_Scientific"
STEM = f"{CATEGORY}_5_2016-10-2018-11"
VARIANT = "rqkmeans_balanced_initialization"
MANIFEST = "balanced_sid_manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_index(codes, items):
    if codes.shape != (len(items), 3) or not np.issubdtype(codes.dtype, np.integer):
        raise ValueError("codes 必须是 (商品数, 3) 整数数组。")
    if not len(codes) or codes.min() < 0 or codes.max() >= 256:
        raise ValueError("codes 必须使用原始 0–255 编号。")
    if set(items) != {str(i) for i in range(len(codes))}:
        raise ValueError("商品 ID 必须与 codes 行号对应：0 到 N-1。")

    paths = [tuple(int(v) + 1 for v in row) for row in codes]
    counts, ordinal = Counter(paths), Counter()
    index = {}
    for item_id, path in enumerate(paths):
        # Same +1 offset and ordinal collision suffix as rqkmeans_constrained.py.
        ordinal[path] += 1
        values = path + (ordinal[path],) if counts[path] > 1 else path
        index[str(item_id)] = [f"<{chr(97 + level)}_{v}>" for level, v in enumerate(values)]
    return index


def remap_csv(source, destination, old_index, new_index):
    old_sid = {key: "".join(tokens) for key, tokens in old_index.items()}
    new_sid = {key: "".join(tokens) for key, tokens in new_index.items()}
    count = 0
    with source.open(newline="", encoding="utf-8") as src, destination.open("w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        required = {"history_item_id", "item_id", "history_item_sid", "item_sid"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{source} 缺少商品 ID 或 SID 列。")
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for line, row in enumerate(reader, start=2):
            try:
                history = ast.literal_eval(row["history_item_id"])
                if not isinstance(history, list) or any(type(v) is not int for v in history):
                    raise ValueError("history_item_id 必须是整数列表")
                history_keys = [str(v) for v in history]
                target = str(int(row["item_id"]))
                if row["item_sid"] != old_sid[target] or ast.literal_eval(row["history_item_sid"]) != [old_sid[k] for k in history_keys]:
                    raise ValueError("原 CSV 的 SID 与原 index 不一致")
                row["history_item_sid"] = repr([new_sid[k] for k in history_keys])
                row["item_sid"] = new_sid[target]
                writer.writerow(row)
            except (KeyError, TypeError, ValueError, SyntaxError) as exc:
                raise ValueError(f"{source} 第 {line} 条记录解析失败：{exc}") from exc
            count += 1
    if count == 0:
        raise ValueError(f"{source} 没有样本。")
    return count


def output_files():
    return [
        f"{CATEGORY}/{CATEGORY}.index.json",
        f"{CATEGORY}/{CATEGORY}.item.json",
        *(f"{split}/{STEM}.csv" for split in ("train", "valid", "test")),
        f"info/{STEM}.txt",
    ]


def check_prepared(output_root):
    output_root = Path(output_root)
    manifest = json.loads((output_root / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("variant") != VARIANT or set(manifest.get("outputs", {})) != set(output_files()):
        raise ValueError("平衡 SID 数据清单不完整，请重新 prepare 到新目录。")
    for relative, expected in manifest["outputs"].items():
        if sha256(output_root / relative) != expected:
            raise ValueError(f"{relative} 与 prepare 清单不一致，请勿混用其他 SID 数据。")
    print(f"平衡 SID 数据校验通过：{output_root}")
    return manifest


def prepare(source_root, output_root, codes_path):
    source_root, output_root, codes_path = (Path(p).resolve() for p in (source_root, output_root, codes_path))
    if output_root.exists():
        raise ValueError(f"输出目录已存在：{output_root}；已有数据可直接训练，或用 BALANCED_DATA_ROOT 指定新目录。")
    item_path = source_root / CATEGORY / f"{CATEGORY}.item.json"
    index_path = source_root / CATEGORY / f"{CATEGORY}.index.json"
    csv_paths = {split: source_root / split / f"{STEM}.csv" for split in ("train", "valid", "test")}
    inputs = [codes_path, item_path, index_path, *csv_paths.values()]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"缺少原实验文件：{path}")
    fingerprints = {str(path): sha256(path) for path in inputs}
    items = json.loads(item_path.read_text(encoding="utf-8"))
    old_index = json.loads(index_path.read_text(encoding="utf-8"))
    if set(old_index) != set(items) or len({"".join(v) for v in old_index.values()}) != len(items):
        raise ValueError("原 index 的商品集合或完整 SID 唯一性不正确。")
    codes = np.load(codes_path, allow_pickle=False)
    index = build_index(codes, items)

    # Publish only after all three splits pass; a failure leaves no usable partial dataset.
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".balanced-sid-", dir=output_root.parent) as temp:
        staging = Path(temp) / "dataset"
        for subdir in (CATEGORY, "train", "valid", "test", "info"):
            (staging / subdir).mkdir(parents=True)
        (staging / CATEGORY / f"{CATEGORY}.index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        shutil.copyfile(item_path, staging / CATEGORY / item_path.name)
        rows = {split: remap_csv(path, staging / split / path.name, old_index, index) for split, path in csv_paths.items()}
        create_item_info_file(items, index, str(staging / "info" / f"{STEM}.txt"))
        for path in inputs:
            if sha256(path) != fingerprints[str(path)]:
                raise ValueError(f"准备过程中源文件发生变化：{path}，请在源文件稳定后重试。")
        manifest = {
            "variant": VARIANT,
            "source_root": str(source_root),
            "codes_path": str(codes_path),
            "inputs": fingerprints,
            "rows": rows,
            "items": len(items),
            "unique_three_level_sids": len({tuple(v[:3]) for v in index.values()}),
            "items_with_fourth_token": sum(len(v) == 4 for v in index.values()),
            "outputs": {relative: sha256(staging / relative) for relative in output_files()},
        }
        (staging / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staging.rename(output_root)
    print(f"平衡初始化 SID 已生成：{len(items)} 商品，划分样本数 {rows}")
    check_prepared(output_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, default=Path("./data/Amazon23"))
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--codes_path", type=Path)
    parser.add_argument("--check", action="store_true", help="Verify prepared files before training/evaluation")
    args = parser.parse_args()
    if args.check:
        check_prepared(args.output_root)
    else:
        codes_path = args.codes_path or args.source_root / CATEGORY / f"{CATEGORY}.codes_constrained.npy"
        prepare(args.source_root, args.output_root, codes_path)


if __name__ == "__main__":
    main()
