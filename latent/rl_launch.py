"""GPU capacity checks for the independent latent RL launchers."""


def check_gpu_capacity(values, gpus, allow_shared, min_free_gib):
    if set(values) != set(gpus):
        raise ValueError('GPU inventory differs from requested pool')
    for gpu in gpus:
        info = values[gpu]
        if info['pids'] and not allow_shared:
            raise RuntimeError(f'GPU {gpu} has existing processes; shared use requires explicit authorization')
        if info['total_mib'] - info['used_mib'] < min_free_gib * 1024:
            raise RuntimeError(f'GPU {gpu} has less than {min_free_gib:g} GiB free')
