"""Explicit opt-in entrypoint for latent-only cached4x4 RL.

Usage: python -m latent.train_fast rl [the usual latent.train arguments]
Existing latent.train and native rl.py keep their original implementations.
"""
from . import train as entry
from .rl_fast import FastLatentGRPOTrainer


def main():
    options = entry.parser().parse_args()
    if options.stage != 'rl' or options.mode != '4x4':
        raise ValueError('Fast entrypoint is only for independent4x4 latent RL')
    entry.LatentGRPOTrainer = FastLatentGRPOTrainer
    entry.main()


if __name__ == '__main__':
    main()
