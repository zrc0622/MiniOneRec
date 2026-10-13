import argparse
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

from latent.train import validate_sft_resume
from run_sft_recovery import checkpoint_state, recovery_command
from adopt_sft_recovery import worker_identity


class SFTRecoveryTests(unittest.TestCase):
    def test_worker_identity_handles_process_exit_without_pidfd_api(self):
        worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.2)'])
        self.assertIsNotNone(worker_identity(worker.pid))
        worker.wait()
        self.assertIsNone(worker_identity(worker.pid))

    def checkpoint(self, root):
        cp = root / 'sft/checkpoint-138'
        cp.mkdir(parents=True)
        for name in ('config.json', 'tokenizer_config.json', 'model.safetensors', 'scheduler.pt'):
            (cp / name).write_text('{}')
        (cp / 'trainer_state.json').write_text(json.dumps(dict(global_step=138, max_steps=4140)))
        state = cp / 'global_step138'
        state.mkdir()
        for name in ('bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt',
                     'bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt', 'mp_rank_00_model_states.pt'):
            (state / name).write_text('state')
        return cp

    def test_both_rank_optimizer_states_and_original_step_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cp = self.checkpoint(root)
            self.assertEqual(checkpoint_state(root, cp)['global_step'], 138)
            (cp / 'global_step138/bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt').unlink()
            with self.assertRaises(ValueError):
                checkpoint_state(root, cp)

    def test_resume_does_not_accept_changed_options_or_input(self):
        from latent.core import sha
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cp = self.checkpoint(root)
            data = root / 'train.csv'
            data.write_text('original')
            original = dict(stage='sft', batch=32, lr=.0003)
            (root / 'sft/latent_run.json').write_text(json.dumps(dict(
                options=original, source_hashes={str(data): sha(data)})))
            o = argparse.Namespace(**original, resume_from_checkpoint=str(cp))
            self.assertEqual(validate_sft_resume(o, root / 'sft'), cp)
            o.batch = 16
            with self.assertRaises(ValueError):
                validate_sft_resume(o, root / 'sft')
            o.batch = 32
            data.write_text('different')
            with self.assertRaises(ValueError):
                validate_sft_resume(o, root / 'sft')

    def test_original_commands_preserve_training_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = ['python', '-m', 'torch.distributed.run', '-m', 'latent.train',
                       'sft', '--batch', '32', '--gradient_accumulation', '16']
            (root / 'sft.command.json').write_text(json.dumps(command))
            cp = root / 'sft/checkpoint-138'
            self.assertEqual(recovery_command(root, root / 'snapshot', cp, 'target'),
                             command + ['--resume_from_checkpoint', str(cp)])
            command = ['python', '/original/sft.py', '--batch_size', '1024', '--micro_batch_size', '32']
            (root / 'sft.command.json').write_text(json.dumps(command))
            recovered = recovery_command(root, root / 'snapshot', cp, 'native')
            self.assertEqual(recovered[1], str(root / 'snapshot/sft.py'))
            self.assertEqual(recovered[2:-2], command[2:])


if __name__ == '__main__':
    unittest.main()
