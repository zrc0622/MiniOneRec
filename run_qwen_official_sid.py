#!/usr/bin/env python3
"""Durable two-GPU official item extraction -> balanced SID, then stop."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

from prepare_balanced_sid import CATEGORY, STEM, sha256
from prepare_qwen_official_sid import RECIPE, RECIPE_FILE, ARTIFACTS, validate_embeddings, check_experiment
from run_two_slot_vq_sft import gpu_snapshot, save, PYTHON, PROJECT, DATA


def idle_pair(measurements, consecutive, preferences):
    """Require sustained idle memory, utilization and absence of compute processes."""
    for gpu in preferences:
        info = measurements[gpu]
        idle = info["used_mib"] < 1024 and info["utilization"] < 10 and not info["pids"]
        consecutive[gpu] = consecutive.get(gpu, 0) + 1 if idle else 0
    selected = [gpu for gpu in preferences if consecutive[gpu] >= 3]
    return selected[:2] if len(selected) >= 2 else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--gpus", default="2,6,4,7,0,1,3,5", help="Allowed idle GPU pool, in preference order")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "pipeline.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpus = args.gpus.split(",")
    if len(gpus) < 2 or len(set(gpus)) != len(gpus):
        raise ValueError("At least two distinct candidate GPUs required")
    (root / "pipeline.pid").write_text(str(os.getpid()) + "\n")
    config = dict(recipe=RECIPE, allowed_gpu_pool=gpus, max_concurrent_gpus=2, sft_authorized=False,
                  ranking_eval_authorized=False, rl_authorized=False,
                  embedding_batch_per_device=32, attention="unchanged",
                  stop_after="item_embeddings_and_balanced_sid_dataset")
    config_path = root / "pipeline_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Cannot resume using changed experiment configuration")
    save(config_path, config)
    source_inputs = [DATA / CATEGORY / f"{CATEGORY}.{suffix}.json" for suffix in ("item", "index")]
    source_inputs += [DATA / split / f"{STEM}.csv" for split in ("train", "valid", "test")]
    fingerprints = {str(p): sha256(p) for p in source_inputs}
    source_path = root / "source_inputs.json"
    if source_path.exists() and json.loads(source_path.read_text()) != fingerprints:
        raise ValueError("Cannot resume after source dataset changed")
    save(source_path, fingerprints)
    snapshot = root / "source_snapshot"
    if not snapshot.exists():
        snapshot.mkdir()
        for name in ("rq", "latent", "interest", "config"):
            shutil.copytree(PROJECT / name, snapshot / name, ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("data.py", "convert_dataset.py", "prepare_balanced_sid.py",
                     "prepare_qwen_official_sid.py", "run_qwen_official_sid.py",
                     "run_two_slot_vq_sft.py", "eval_metric_protocol.json"):
            shutil.copy2(PROJECT / name, snapshot / name)
        save(snapshot / "hashes.json", {str(p.relative_to(snapshot)): sha256(p)
                                      for p in snapshot.rglob("*") if p.is_file()})
    env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="1", NCCL_IB_DISABLE="1", PYTHONPATH=str(snapshot) + ":" + str(PROJECT),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    env["PATH"] = str(Path(PYTHON).parent) + ":" + env.get("PATH", "")
    if not Path(env.get("CUDA_HOME", "") + "/bin/nvcc").is_file():
        env["CUDA_HOME"] = str(Path(PYTHON).parent.parent)
    active = []

    def status(phase, **extra):
        payload = dict(phase=phase, updated_at=time.time(), coordinator_pid=os.getpid(), allowed_gpu_pool=gpus, **extra)
        save(root / "pipeline_status.json", payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    def wait_idle():
        consecutive = {}
        while True:
            snap = gpu_snapshot(gpus)
            selected = idle_pair(snap, consecutive, gpus)
            status("waiting_for_idle_gpus", selected_gpus=selected, measurements=snap,
                   consecutive_idle_checks=consecutive)
            if selected:
                return selected
            time.sleep(10 if any(consecutive.values()) else 30)

    def start(command, log, selected=()):
        log.parent.mkdir(parents=True, exist_ok=True)
        save(log.with_suffix(".command.json"), command)
        stream = log.open("w")
        process = subprocess.Popen(command, cwd=snapshot, env=dict(env, CUDA_VISIBLE_DEVICES=",".join(selected)),
                                   stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        log.with_suffix(".pid").write_text(str(process.pid) + "\n")
        worker = (process, stream, log)
        active.append(worker)
        return worker

    def finish(worker):
        process, stream, log = worker
        code = process.wait()
        stream.close()
        active.remove(worker)
        log.with_suffix(".exit_code").write_text(str(code) + "\n")
        if code:
            raise RuntimeError(f"Stage failed exit={code}; inspect {log}")

    torchrun = [PYTHON, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
    encoded = root / "embeddings"
    dataset = root / "dataset"
    embedding_model = PROJECT / "models/Qwen3-Embedding-4B"
    try:
        if not (encoded / "encoding_manifest.json").exists():
            item_root = encoded / CATEGORY
            item_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_inputs[0], item_root / source_inputs[0].name)
            selected_gpus = wait_idle()
            status("encoding_items", selected_gpus=selected_gpus, pooling="last_valid_token", l2_normalize=True,
                   item_count=len(json.loads(source_inputs[0].read_text())))
            finish(start(torchrun + ["rq/text2emb/amazon_text2emb.py", "--dataset", CATEGORY,
                         "--root", str(item_root), "--plm_checkpoint", str(embedding_model),
                         "--batch_size", "32", "--max_sent_len", "2048",
                         "--pooling", "last_token", "--l2_normalize"], root / "embedding.log", selected_gpus))
            if fingerprints != {str(p): sha256(p) for p in source_inputs}:
                raise ValueError("Source data changed during item encoding")
            count = len(json.loads(source_inputs[0].read_text()))
            diagnostics = validate_embeddings(encoded / ARTIFACTS[0], count)
            save(encoded / "encoding_manifest.json", dict(recipe=RECIPE, source_inputs=fingerprints,
                 embedding_model=str(embedding_model), model_config_sha256=sha256(embedding_model / "config.json"),
                 embedding_sha256=sha256(encoded / ARTIFACTS[0]), diagnostics=diagnostics,
                 command=json.loads((root / "embedding.command.json").read_text()), gpus=selected_gpus))
            status("item_embeddings_complete", **diagnostics)
        if not (dataset / RECIPE_FILE).exists():
            status("clustering_and_remapping", GPU_required=False)
            finish(start([PYTHON, "-u", "prepare_qwen_official_sid.py", "--source-root", str(DATA),
                          "--embedding-root", str(encoded), "--output-root", str(dataset),
                          "--embedding-model", str(embedding_model)], root / "sid.log"))
        check_experiment(dataset)
        status("complete_sid_only", dataset=str(dataset),
               embeddings=str(dataset / ARTIFACTS[0]),
               next_action="Wait for explicit SFT/eval authorization; neither launched")
    finally:
        for process, stream, log in active:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            stream.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        import sys
        if "--root" in sys.argv:
            root = Path(sys.argv[sys.argv.index("--root") + 1]).resolve()
            root.mkdir(parents=True, exist_ok=True)
            failure = dict(phase="failed", updated_at=time.time(), error=traceback.format_exc())
            save(root / "pipeline_status.json", failure)
            save(root / "pipeline_error.json", failure)
        raise
