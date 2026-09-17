"""Download the pinned Amazon23 Industrial JSONL files from Hugging Face."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="./data/raw/Amazon23")
    args = parser.parse_args()
    manifest = json.loads(
        Path(__file__).with_name("amazon23_industrial_manifest.json").read_text()
    )
    output_dir = Path(args.output_dir)
    for entry in manifest["files"]:
        destination = output_dir / entry["path"]
        if (destination.is_file() and destination.stat().st_size == entry["size"]
                and sha256_file(destination) == entry["sha256"]):
            print(f"Already verified: {destination}")
            continue
        downloaded = Path(hf_hub_download(
            repo_id=manifest["repo_id"],
            repo_type="dataset",
            revision=manifest["revision"],
            filename=entry["path"],
            local_dir=output_dir,
            force_download=destination.exists(),
        ))
        if (downloaded.stat().st_size != entry["size"]
                or sha256_file(downloaded) != entry["sha256"]):
            raise RuntimeError(f"Size/SHA-256 verification failed: {downloaded}")
        print(f"Downloaded and verified: {downloaded}")


if __name__ == "__main__":
    main()
