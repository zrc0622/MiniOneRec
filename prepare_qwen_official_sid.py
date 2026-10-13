#!/usr/bin/env python3
"""Build an isolated balanced SID dataset from last-token, L2-normalized item vectors."""
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
RECIPE_FILE = "qwen_official_sid_manifest.json"
RECIPE = dict(embedding_model_id="Qwen/Qwen3-Embedding-4B", embedding_dim=2560,
              pooling="last_valid_token", l2_normalize=True,
              normalization_dtype="float32", embedding_dtype="float16",
              features=["title", "description"], text_cleaning="MiniOneRec_generate_text",
              max_sent_len=2048, sid="rqkmeans_balanced_initialization",
              k=256, l=3, max_iter=100, seed=42,
              residual_normalization=False)
ARTIFACTS = [f"{CATEGORY}/{CATEGORY}.{suffix}" for suffix in
             ("emb-qwen-td.npy", "codes_constrained.npy", "codebooks_constrained.npz")]


def validate_embeddings(path, count, dim=2560):
    vectors = np.load(path, allow_pickle=False, mmap_mode="r")
    if vectors.shape != (count, dim) or vectors.dtype != np.float32:
        raise ValueError(f"Expected float32 embeddings {(count, dim)}, got {vectors.shape}/{vectors.dtype}")
    if not np.isfinite(vectors).all():
        raise ValueError("Non-finite item embedding")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1, atol=1e-5, rtol=0):
        raise ValueError("Item embeddings are not L2-normalized")
    return dict(shape=list(vectors.shape), dtype=str(vectors.dtype),
                min_norm=float(norms.min()), max_norm=float(norms.max()))


def check_experiment(root):
    root = Path(root)
    manifest = json.loads((root / RECIPE_FILE).read_text())
    if manifest["recipe"] != RECIPE or set(manifest["artifacts"]) != {MANIFEST, *ARTIFACTS}:
        raise ValueError("Wrong item embedding/SID recipe")
    for name, digest in manifest["artifacts"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Changed item embedding/SID artifact: {name}")
    balanced = check_prepared(root)
    validate_embeddings(root / ARTIFACTS[0], balanced["items"])
    return manifest


def prepare_experiment(source_root, embedding_root, output_root, embedding_model):
    source, encoded, output, model = map(lambda p: Path(p).resolve(),
                                       (source_root, embedding_root, output_root, embedding_model))
    if output.exists():
        raise ValueError(f"Output exists; refusing to overwrite: {output}")
    item_file = source / CATEGORY / f"{CATEGORY}.item.json"
    inputs = [item_file, source / CATEGORY / f"{CATEGORY}.index.json",
              *(source / split / f"{STEM}.csv" for split in ("train", "valid", "test"))]
    fingerprints = {str(p): sha256(p) for p in inputs}
    items = json.loads(item_file.read_text())
    if set(items) != {str(i) for i in range(len(items))}:
        raise ValueError("Item IDs must correspond to contiguous embedding row indices")
    config = model / "config.json"
    config_hash = sha256(config)
    if json.loads(config.read_text())["hidden_size"] != 2560:
        raise ValueError("Expected Qwen3-Embedding-4B with hidden_size 2560")
    encoded_items = encoded / CATEGORY / item_file.name
    embedding_path = encoded / ARTIFACTS[0]
    if sha256(encoded_items) != fingerprints[str(item_file)]:
        raise ValueError("Encoded item metadata differs from source")
    encoding = json.loads((encoded / "encoding_manifest.json").read_text())
    if encoding["recipe"] != RECIPE or encoding["embedding_sha256"] != sha256(embedding_path):
        raise ValueError("Missing or inconsistent last-token encoding provenance")
    if encoding["model_config_sha256"] != config_hash or encoding["source_inputs"] != fingerprints:
        raise ValueError("Model or source data changed since encoding")
    diagnostics = validate_embeddings(embedding_path, len(items))
    env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    subprocess.run([sys.executable, "rq/rqkmeans_constrained.py", "--dataset", CATEGORY,
                    "--root", str(encoded / CATEGORY), "--k", "256", "--l", "3",
                    "--max_iter", "100", "--seed", "42", "--verbose"],
                   cwd=REPO, env=env, check=True)
    codes = encoded / ARTIFACTS[1]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".qwen-official-sid-", dir=output.parent) as temporary:
        dataset = Path(temporary) / "dataset"
        prepare(source, dataset, codes)
        if json.loads((encoded / CATEGORY / f"{CATEGORY}.index.json").read_text()) != json.loads(
                (dataset / CATEGORY / f"{CATEGORY}.index.json").read_text()):
            raise ValueError("Clustering and CSV remapping produced different SID indices")
        with np.load(encoded / ARTIFACTS[2], allow_pickle=False) as books:
            if set(books.files) != {f"codebook_{i}" for i in range(3)}:
                raise ValueError("Expected three residual codebooks")
            if any(books[k].shape != (256, 2560) or not np.isfinite(books[k]).all() for k in books.files):
                raise ValueError("Invalid residual codebook")
        for name in ARTIFACTS:
            shutil.copy2(encoded / name, dataset / name)
        manifest_path = dataset / MANIFEST
        manifest = json.loads(manifest_path.read_text())
        final_codes = str(output / ARTIFACTS[1])
        manifest["inputs"][final_codes] = manifest["inputs"].pop(str(codes))
        manifest["codes_path"] = final_codes
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        if fingerprints != {str(p): sha256(p) for p in inputs} or sha256(config) != config_hash:
            raise ValueError("Source inputs changed during SID construction")
        provenance = dict(recipe=RECIPE, embedding_model_path=str(model),
                          model_config_sha256=config_hash, source_inputs=fingerprints,
                          encoding=encoding, embedding_diagnostics=diagnostics,
                          artifacts={name: sha256(dataset / name) for name in [MANIFEST, *ARTIFACTS]})
        (dataset / RECIPE_FILE).write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
        check_experiment(dataset)
        dataset.rename(output)
    print(f"New last-token/L2 balanced SID dataset published: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check_experiment(args.output_root)
    else:
        prepare_experiment(args.source_root, args.embedding_root, args.output_root, args.embedding_model)


if __name__ == "__main__":
    main()
