"""Detached evaluation recovery for already-running orphaned RL workers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def training_workers(root):
    targets = {str(root / reward) for reward in ('binary', 'sid_dense')}
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().decode().split('\0')
            if 'latent.train' in args and '--output_dir' in args:
                if args[args.index('--output_dir') + 1] in targets:
                    found.append(int(entry.name))
        except (OSError, UnicodeError, IndexError):
            continue
    return found


def verify_final(root, reward):
    """Check finished training and complete saved weights; never invent exit codes."""
    model = root / reward / 'final_checkpoint'
    log = (root / f'{reward}.log').read_text(errors='replace')
    if "'train_runtime':" not in log:
        raise ValueError('Training completion summary is absent')
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
        json.loads((model / name).read_text())
    index = model / 'model.safetensors.index.json'
    weights = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
    # Validate each header and declared data length without loading weights/GPU.
    for name in weights:
        path = model / name
        with path.open('rb') as file:
            size = int.from_bytes(file.read(8), 'little')
            if not 0 < size < path.stat().st_size - 8:
                raise ValueError('Incomplete safetensors header')
            header = json.loads(file.read(size))
        offsets = [v['data_offsets'][1] for k, v in header.items() if k != '__metadata__']
        if not offsets or path.stat().st_size != 8 + size + max(offsets):
            raise ValueError('Incomplete safetensors data')
    return dict(model=str(model), weights=weights, verification='training_summary_and_complete_saved_artifacts',
                note='Worker exit status is unknown because original coordinator exited')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    with (root / 'eval_watch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root / 'eval_watch.pid').write_text(str(os.getpid()) + '\n')
        print('Waiting for running RL workers:', training_workers(root), flush=True)
        while training_workers(root):
            time.sleep(30)
        verified = []
        for reward in ('binary', 'sid_dense'):
            try:
                info = verify_final(root, reward)
            except (OSError, ValueError, KeyError) as exc:
                print(f'{reward}: final model verification failed: {exc}', flush=True)
                continue
            (root / f'{reward}.completion_verified.json').write_text(json.dumps(info, indent=2) + '\n')
            verified.append(reward)
        if not verified:
            raise RuntimeError('No verified completed model to evaluate')
        # Other users may acquire these GPUs before training ends. Wait for idle
        # capacity rather than interfering with their newly-started work.
        gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '4,5,6,7')
        while True:
            output = subprocess.check_output(['nvidia-smi', '-i', gpus, '--query-gpu=memory.used',
                                              '--format=csv,noheader,nounits'], text=True)
            if all(int(value.strip()) < 1024 for value in output.splitlines()):
                break
            time.sleep(30)
        print('Starting full final-checkpoint evaluation:', verified, flush=True)
        code = subprocess.call(['bash', 'run_latent_rl_eval.sh', str(root)])
        (root / 'recovered_eval.exit_code').write_text(str(code) + '\n')
        raise SystemExit(code)


if __name__ == '__main__':
    main()
