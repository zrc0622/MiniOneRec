"""Guard candidate/prompt budget and failure checks in isolated batch32/64 pilot."""
import json
from pathlib import Path
import tempfile
import unittest

from latent.train import parser
from run_latent_batch_benchmark import batch_command, summarize


class BatchBenchmarkTests(unittest.TestCase):
    def test_only_batch_accumulation_change_in_matched_training_recipe(self):
        base = ['python', '-m', 'torch.distributed.run', '--master_port=9999', '-m', 'latent.rl_speed_benchmark', 'rl',
                '--model', 'fixture', '--train_file', 'train', '--eval_file', 'valid', '--items', 'items',
                '--index', 'index', '--output_dir', 'unused', '--batch', '32', '--gradient_accumulation', '16',
                '--max_steps', '6', '--eval_steps', '1000', '--save_steps', '1000', '--mode', '4x4',
                '--reward_kind', 'sid_dense']
        commands = [batch_command(base, b, Path('/tmp/pilot'), 12345, 6) for b in (32, 64)]
        opts = [vars(parser().parse_args(c[c.index('latent.rl_speed_benchmark') + 1:])) for c in commands]
        for o in opts:
            self.assertEqual(o['batch'] * 2 * o['gradient_accumulation'], 1024)
            self.assertEqual(o['mode'], '4x4')
            self.assertEqual(o['reward_kind'], 'sid_dense')
            self.assertEqual(o['max_steps'], 6)
        for key in ('batch', 'gradient_accumulation'):
            for o in opts:
                o.pop(key)
        self.assertEqual(opts[0], opts[1])

    def test_summary_requires_complete_ranks_and_training_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp)
            for rank in (0, 1):
                rows = [dict(step=i, seconds=i + rank / 10, peak_allocated_gib=9., peak_reserved_gib=12.,
                             components={}) for i in range(1, 7)]
                (dest / f'rank{rank}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
                (dest / f'checks.rank{rank}.json').write_text(json.dumps(dict(status='pass', steps=6,
                    prompt_task_rows=157679, batch=64, accumulation=8, world_size=2, statistics=[dict(grad_norm=.1)])))
            result = summarize(dest, 6, 64)
            self.assertAlmostEqual(result['mean_seconds'], 4.6)
            (dest / 'rank1.jsonl').write_text('')
            with self.assertRaises(ValueError):
                summarize(dest, 6, 64)


if __name__ == '__main__':
    unittest.main()
