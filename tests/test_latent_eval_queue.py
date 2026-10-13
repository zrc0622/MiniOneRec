"""Exercise queue limits and completion without allocating GPUs."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from latent.eval_queue import run


class QueueTests(unittest.TestCase):
    def test_two_groups_finish_and_update_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            snapshots = []

            def launch(command, **kwargs):
                output = Path(command[command.index('--output') + 1])
                output.write_text('{}')
                # Capture state before each new shard is spawned. The third
                # experiment must not start until an active slot is released.
                state = output.parent.parent / 'parallel_state.json'
                snapshots.append(json.loads(state.read_text()))
                return SimpleNamespace(pid=100 + len(snapshots), poll=lambda: 0)

            def merge(command, **kwargs):
                output = Path(command[command.index('--output') + 1])
                identity = dict(model='fixture', model_hashes={}, config_sha256='c',
                                data_sha256='d', index_sha256='i', examples=4)
                metrics = {'examples': 4, 'HR@1': 0, 'HR@10': 0, 'HR@50': 0,
                           'NDCG@50': 0, 'mean_unique_candidates': 50,
                           'fraction_with_50_candidates': 1}
                output.write_text(json.dumps(dict(metadata=identity, metrics=metrics)))

            with patch.dict('os.environ', {'LATENT_ROOT': str(base), 'CUDA_VISIBLE_DEVICES': '4,5,6,7'}, clear=True), \
                    patch('latent.eval_queue.subprocess.Popen', side_effect=launch), \
                    patch('latent.eval_queue.alive', side_effect=lambda process: process.poll() is None), \
                    patch('latent.eval_queue.subprocess.run', side_effect=merge), \
                    patch('latent.eval_queue.time.sleep'):
                run(SimpleNamespace(adopt_run=None, dry_run=False))
            roots = list((base / 'eval').iterdir())
            self.assertEqual(len(roots), 1)
            state = json.loads((roots[0] / 'parallel_state.json').read_text())
            self.assertEqual(len(state['completed']), 4)
            self.assertEqual(state['active'], {})
            self.assertTrue(all(len(s['active']) <= 2 for s in snapshots))
            self.assertTrue((roots[0] / 'comparison.csv').is_file())


if __name__ == '__main__':
    unittest.main()
