#!/usr/bin/env python3
"""Prepare the isolated 1024-dimensional mean-embedding balanced SID experiment."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from prepare_balanced_sid import CATEGORY, STEM, MANIFEST, check_prepared, prepare, sha256


REPO = Path(__file__).resolve().parent
RECIPE_FILE = "embedding06b_balanced_manifest.json"
RECIPE = {
    "embedding_model_id": "Qwen/Qwen3-Embedding-0.6B",
    "embedding_dim": 1024,
    "pooling": "masked_mean",
    "l2_normalize": False,
    "features": ["title", "description"],
    "max_sent_len": 2048,
    "embedding_batch_size": 16,
    "embedding_dtype": "float16",
    "sid": "rqkmeans_balanced_initialization",
    "k": 256, "l": 3, "max_iter": 100, "seed": 42,
}
ARTIFACTS = [f"{CATEGORY}/{CATEGORY}.{suffix}" for suffix in (
    "emb-qwen-td.npy", "codes_constrained.npy", "codebooks_constrained.npz",
)]


def check_experiment(output_root):
    root = Path(output_root)
    recipe = json.loads((root / RECIPE_FILE).read_text(encoding="utf-8"))
    if recipe.get("recipe") != RECIPE or set(recipe.get("artifacts", {})) != {MANIFEST, *ARTIFACTS}:
        raise ValueError("缺少本实验的1024维编码清单，请勿使用原4B balanced数据目录。")
    for relative, expected in recipe["artifacts"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"实验产物与清单不一致：{relative}")
    check_prepared(root)
    print(f"Embedding-0.6B / 1024 / mean / balanced 数据校验通过：{root}")
    return recipe


def prepare_experiment(source_root, output_root, embedding_model):
    source, output, model = (Path(p).resolve() for p in (source_root, output_root, embedding_model))
    if output.exists():
        raise ValueError(f"输出目录已存在：{output}；请直接训练，或用 EMB06B_DATA_ROOT 指定新目录。")
    item_file = source / CATEGORY / f"{CATEGORY}.item.json"
    inputs = [item_file, source / CATEGORY / f"{CATEGORY}.index.json",
              *(source / split / f"{STEM}.csv" for split in ("train", "valid", "test"))]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"缺少原实验输入：{path}")
    fingerprints = {str(path): sha256(path) for path in inputs}
    items = json.loads(item_file.read_text(encoding="utf-8"))
    if not items or set(items) != {str(i) for i in range(len(items))}:
        raise ValueError("商品ID必须连续为0到N-1，以保证embedding行号一致。")
    config_file = model / "config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("hidden_size") != 1024:
        raise ValueError("本实验需要Qwen3-Embedding-0.6B（hidden_size=1024），请先运行download。")
    config_hash = sha256(config_file)

    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish the dataset only once all GPU/CPU stages and remapping have succeeded.
    with tempfile.TemporaryDirectory(prefix=".embedding06b-balanced-", dir=output.parent) as tmp:
        workspace, dataset = Path(tmp) / "embedding", Path(tmp) / "dataset"
        item_root = workspace / CATEGORY
        item_root.mkdir(parents=True)
        shutil.copyfile(item_file, item_root / item_file.name)
        env = os.environ | {"DATA_ROOT": str(workspace), "EMB_MODEL": str(model), "PYTHONUNBUFFERED": "1"}
        subprocess.run(["bash", "rq/text2emb/amazon_text2emb.sh"], cwd=REPO, env=env, check=True)
        embedding_path = item_root / f"{CATEGORY}.emb-qwen-td.npy"
        embeddings = np.load(embedding_path, allow_pickle=False, mmap_mode="r")
        if embeddings.shape != (len(items), 1024) or not np.issubdtype(embeddings.dtype, np.floating):
            raise ValueError(f"embedding应为({len(items)}, 1024)浮点数组，实际{embeddings.shape}/{embeddings.dtype}")
        if not np.isfinite(embeddings).all():
            raise ValueError("embedding包含NaN或Inf。")
        print(f"1024维embedding校验通过：{embeddings.shape}", flush=True)
        del embeddings
        subprocess.run([
            sys.executable, "rq/rqkmeans_constrained.py", "--dataset", CATEGORY,
            "--root", str(item_root), "--k", "256", "--l", "3",
            "--max_iter", "100", "--seed", "42", "--verbose",
        ], cwd=REPO, env=env, check=True)
        codes_path = item_root / f"{CATEGORY}.codes_constrained.npy"
        prepare(source, dataset, codes_path)
        generated_index = json.loads((item_root / f"{CATEGORY}.index.json").read_text())
        remapped_index = json.loads((dataset / CATEGORY / f"{CATEGORY}.index.json").read_text())
        if generated_index != remapped_index:
            raise ValueError("聚类输出SID与CSV转换SID不一致。")
        with np.load(item_root / f"{CATEGORY}.codebooks_constrained.npz", allow_pickle=False) as books:
            if set(books.files) != {f"codebook_{i}" for i in range(3)}:
                raise ValueError("应生成3层平衡码本。")
            for key in books.files:
                if books[key].shape != (256, 1024) or not np.isfinite(books[key]).all():
                    raise ValueError(f"码本维度或数值异常：{key}")
        for relative in ARTIFACTS:
            shutil.move(str(workspace / relative), dataset / relative)
        # Keep the existing data manifest usable after the staging directory is removed.
        manifest_path = dataset / MANIFEST
        manifest = json.loads(manifest_path.read_text())
        final_codes_path = str(output / CATEGORY / codes_path.name)
        manifest["inputs"][final_codes_path] = manifest["inputs"].pop(str(codes_path))
        manifest["codes_path"] = final_codes_path
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if any(sha256(path) != fingerprints[str(path)] for path in inputs) or sha256(config_file) != config_hash:
            raise ValueError("prepare期间源数据或模型配置发生变化，请在文件稳定后重试。")
        recipe = {
            "recipe": RECIPE, "embedding_model_path": str(model),
            "model_config": config, "model_config_sha256": config_hash,
            "source_inputs": fingerprints,
            "artifacts": {relative: sha256(dataset / relative) for relative in [MANIFEST, *ARTIFACTS]},
        }
        (dataset / RECIPE_FILE).write_text(json.dumps(recipe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        check_experiment(dataset)
        dataset.rename(output)
    print(f"本实验数据已生成：{output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, default=Path("./data/Amazon23"))
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--embedding_model", type=Path, default=Path("./models/Qwen3-Embedding-0.6B"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check_experiment(args.output_root)
    else:
        prepare_experiment(args.source_root, args.output_root, args.embedding_model)


if __name__ == "__main__":
    main()
