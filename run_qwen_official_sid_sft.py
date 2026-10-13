#!/usr/bin/env python3
"""Wait for two idle GPUs, then train native SFT on the checked last-token/L2 SID."""
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
from prepare_qwen_official_sid import check_experiment, RECIPE_FILE
from run_qwen_official_sid import idle_pair
from run_two_slot_vq_sft import PROJECT, PYTHON, gpu_snapshot, save

DEFAULT_DATA = PROJECT / "outputs/amazon23_industrial_qwen3_4b_lasttoken_l2_sid_20261010/dataset"


def training_command(snapshot, dataset, output):
    return [PYTHON, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
            str(snapshot / "sft.py"), "--base_model", str(PROJECT / "models/Qwen3-0.6B"),
            "--batch_size", "1024", "--micro_batch_size", "32",
            "--deepspeed", str(snapshot / "config/sft_zero2.json"),
            "--train_file", str(dataset / "train" / f"{STEM}.csv"),
            "--eval_file", str(dataset / "valid" / f"{STEM}.csv"),
            "--output_dir", str(output), "--logging_dir", str(output / "tensorboard"),
            "--category", CATEGORY, "--train_from_scratch", "False", "--seed", "42",
            "--sid_index_path", str(dataset / CATEGORY / f"{CATEGORY}.index.json"),
            "--item_meta_path", str(dataset / CATEGORY / f"{CATEGORY}.item.json"),
            "--freeze_LLM", "False", "--learning_rate", "3e-4", "--num_epochs", "15",
            "--eval_step", "0.03333333333333333"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--gpus", default="2,6,4,7,0,1,3,5", help="Idle GPU pool in preference order")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    dataset = Path(args.data).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "pipeline.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_experiment(dataset)
    allowed = args.gpus.split(",")
    if len(allowed) < 2 or len(set(allowed)) != len(allowed):
        raise ValueError("At least two distinct candidate GPUs required")
    config = dict(dataset=str(dataset), sid_recipe_sha256=sha256(dataset / RECIPE_FILE),
                  pretrained_model=str(PROJECT / "models/Qwen3-0.6B"), native_sft_only=True,
                  num_epochs=15, learning_rate=3e-4, global_batch=1024,
                  world_size=2, per_device_batch=32, accumulation=16,
                  cutoff_len=512, seed=42, warmup_steps=20,
                  validation="original full validation split approximately every half epoch",
                  checkpoint_selection="best validation loss, early stopping patience3",
                  attention="unchanged", allowed_gpu_pool=allowed,
                  target_latent_sft_authorized=False, ranking_eval_authorized=False,
                  rl_authorized=False, stop_after="native_sft")
    config_path = root / "pipeline_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Refusing to reuse output with different training inputs/configuration")
    save(config_path, config)
    (root / "pipeline.pid").write_text(str(os.getpid()) + "\n")
    snapshot = root / "source_snapshot"
    if not snapshot.exists():
        snapshot.mkdir()
        for name in ("sft.py", "data.py", "run_qwen_official_sid_sft.py"):
            shutil.copy2(PROJECT / name, snapshot / name)
        shutil.copytree(PROJECT / "config", snapshot / "config")
        save(snapshot / "hashes.json", {str(p.relative_to(snapshot)): sha256(p)
                                      for p in snapshot.rglob("*") if p.is_file()})
    command = training_command(snapshot, dataset, root / "sft")
    save(root / "sft.command.json", command)

    def status(phase, **details):
        payload = dict(phase=phase, updated_at=time.time(), coordinator_pid=os.getpid(), **details)
        save(root / "pipeline_status.json", payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    if (root / "sft_complete.json").exists():
        status("complete_sft_only", checkpoint=str(root / "sft/final_checkpoint"))
        return
    if (root / "sft").exists():
        raise ValueError("Partial SFT directory exists; preserve it and use a fresh output for rerun")
    consecutive = {}
    while True:
        measurements = gpu_snapshot(allowed)
        selected = idle_pair(measurements, consecutive, allowed)
        status("waiting_for_idle_gpus", allowed_gpu_pool=allowed, selected_gpus=selected,
               measurements=measurements, consecutive_idle_checks=consecutive)
        if selected:
            break
        time.sleep(10 if any(consecutive.values()) else 30)
    check_experiment(dataset)
    if sha256(dataset / RECIPE_FILE) != config["sid_recipe_sha256"]:
        raise ValueError("SID recipe changed while queued")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(selected), PYTHONUNBUFFERED="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1", NCCL_IB_DISABLE="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONPATH=str(snapshot))
    env["PATH"] = str(Path(PYTHON).parent) + ":" + env.get("PATH", "")
    if not Path(env.get("CUDA_HOME", "") + "/bin/nvcc").is_file():
        env["CUDA_HOME"] = str(Path(PYTHON).parent.parent)
    save(root / "sft_launch.json", dict(gpus=selected, launched_at=time.time(), config=config))
    with (root / "sft.log").open("w") as stream:
        worker = subprocess.Popen(command, cwd=snapshot, env=env, stdin=subprocess.DEVNULL,
                                  stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        (root / "sft.pid").write_text(str(worker.pid) + "\n")
        status("training_no_latent_sft", selected_gpus=selected, worker_pid=worker.pid,
               per_device_batch=32, accumulation=16, global_batch=1024)
        try:
            code = worker.wait()
        finally:
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGTERM)
                try:
                    worker.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, signal.SIGKILL)
                    worker.wait()
    (root / "sft.exit_code").write_text(str(code) + "\n")
    if code:
        raise RuntimeError(f"SFT failed exit={code}; inspect {root / 'sft.log'}")
    checkpoint = root / "sft/final_checkpoint"
    if not (checkpoint / "config.json").exists() or not (checkpoint / "tokenizer_config.json").exists():
        raise ValueError("Training exited without a complete checkpoint")
    save(root / "sft_complete.json", dict(checkpoint=str(checkpoint), dataset=str(dataset),
                                        completed_at=time.time(), exit_code=0, config=config))
    status("complete_sft_only", selected_gpus=selected, checkpoint=str(checkpoint),
           next_action="SFT complete; no latent SFT, ranking evaluation or RL launched")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        import sys
        if "--root" in sys.argv:
            root = Path(sys.argv[sys.argv.index("--root") + 1]).resolve()
            root.mkdir(parents=True, exist_ok=True)
            save(root / "pipeline_status.json", dict(phase="failed", updated_at=time.time(), error=traceback.format_exc()))
        raise
