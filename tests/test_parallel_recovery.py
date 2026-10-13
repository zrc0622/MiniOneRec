import argparse
import json
from pathlib import Path
import tempfile
import unittest

from latent.parallel_train import validate_resume
from run_parallel_recovery import resume_command


class ParallelRecoveryTests(unittest.TestCase):
    def test_static_launcher_preserves_original_training_options(self):
        original = ['python', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                    '-m', 'latent.parallel_train', 'train', '--batch', '32', '--epochs', '15']
        command = resume_command(original, Path('/tmp/checkpoint-3174'), 12345)
        self.assertNotIn('--standalone', command)
        self.assertIn('--rdzv_backend=static', command)
        self.assertEqual(command[command.index('train'):], original[original.index('train'):] +
                         ['--resume_from_checkpoint', '/tmp/checkpoint-3174'])

    def test_resume_requires_both_rank_states_and_unchanged_options_protocol_and_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'checkpoint-3174'
            checkpoint.mkdir()
            options = dict(stage='train', batch=32, epochs=15, lr=.0003)
            prepared = dict(source_hashes={'data': 'unchanged'}, initialized_only=True)
            protocol = dict(prediction='parallel_bidirectional_queries', k=3)
            (root / 'parallel_run.json').write_text(json.dumps(dict(options=options,
                 prepared_metadata=prepared, protocol=protocol)))
            for name in ('model.safetensors', 'scheduler.pt', 'rng_state_0.pth', 'rng_state_1.pth'):
                (checkpoint / name).touch()
            (checkpoint / 'trainer_state.json').write_text(json.dumps(dict(global_step=3174)))
            (checkpoint / 'config.json').write_text(json.dumps(dict(latent_protocol=protocol)))
            state = checkpoint / 'global_step3174'
            state.mkdir()
            for rank in (0, 1):
                (state / f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt').touch()
            o = argparse.Namespace(**options, resume_from_checkpoint=str(checkpoint))
            self.assertEqual(validate_resume(o, root, prepared), checkpoint)
            o.batch = 16
            with self.assertRaises(ValueError):validate_resume(o, root, prepared)
            o.batch = 32
            with self.assertRaises(ValueError):validate_resume(o, root, dict(prepared, initialized_only=False))
            (state / 'bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt').unlink()
            with self.assertRaises(ValueError):validate_resume(o, root, prepared)


if __name__ == '__main__':unittest.main()
