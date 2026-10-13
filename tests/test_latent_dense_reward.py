"""Regression checks for dense SID rewards and controlled rollout sampling."""
import unittest
from unittest.mock import patch

import torch
from datasets import Dataset

from interest.train import InterestRLArguments
from latent.generation import generate
from latent.objective import advantages, sid_rewards
from latent.trainer import LatentGRPOTrainer
from tests.test_latent import fixture, INDEX


class DenseRewardTests(unittest.TestCase):
    def test_exact_hit_dominates_partial_and_disambiguation_is_excluded(self):
        gold = '<a_1><b_2><c_3><d_0>'
        items = [gold, '<a_1><b_2><c_3><d_1>', '<a_1><b_2><c_8>',
                 '<a_1><b_7><c_8>', '<a_9><b_7><c_8>']
        torch.testing.assert_close(sid_rewards(items, gold, kind='sid_dense'),
                                   torch.tensor([1.2, .2, .2 * 2 / 3, .2 / 3, 0.]))
        torch.testing.assert_close(sid_rewards(items, gold), torch.tensor([1., 0., 0., 0., 0.]))
        for bad in ('<a_1><b_2><c_3>\n', '<b_2><a_1><c_3>', '<a_1>'):
            with self.assertRaises(ValueError):
                sid_rewards([bad], gold, kind='sid_dense')

    def test_all_misses_can_have_nonzero_dense_advantages(self):
        gold = '<a_1><b_2><c_3><d_0>'
        branch = ['<a_9><b_8><c_7>', '<a_1><b_8><c_7>',
                  '<a_1><b_2><c_7>', '<a_1><b_2><c_3><d_1>']
        items = [branch[0]] * 4 + branch + [branch[1]] * 4 + [branch[2]] * 4
        binary = advantages(sid_rewards(items, gold), '4x4')
        self.assertEqual(float(binary[0].abs().sum() + binary[1].abs().sum()), 0.)
        ia, ya, group_rewards = advantages(sid_rewards(items, gold, kind='sid_dense'), '4x4')
        self.assertGreater(float(ia.abs().sum()), 0.)
        self.assertGreater(float(ya[4:8].abs().sum()), 0.)
        self.assertEqual(float(ya[:4].abs().sum()), 0.)
        torch.testing.assert_close(group_rewards, torch.tensor([0., .2, .2 / 3, .2 * 2 / 3]))

    def test_checkpoint_defaults_do_not_change_sampling(self):
        torch.set_num_threads(1)
        _, model, _, grammar = fixture()
        model.eval()
        model.generation_config.temperature = .6
        model.generation_config.top_p = .95
        torch.manual_seed(77)
        first = generate(model, [2], grammar, 16)
        model.generation_config.temperature = .2
        model.generation_config.top_p = .1
        torch.manual_seed(77)
        second = generate(model, [2], grammar, 16)
        self.assertEqual(first, second)

    def test_auxiliary_rewards_stay_binary_and_main_uses_dense(self):
        import tempfile
        torch.set_num_threads(1)
        tk, model, _, grammar = fixture()
        records = [dict(prompt='history', target=''.join(INDEX['0']), task='history_sid', auxiliary=False),
                   dict(prompt='history', target=''.join(INDEX['0'][:3]), task='item_identification', auxiliary=True)]
        with tempfile.TemporaryDirectory() as temp:
            args = InterestRLArguments(output_dir=temp, use_cpu=True, report_to=[],
                per_device_train_batch_size=16, per_device_eval_batch_size=16,
                remove_unused_columns=False)
            trainer = LatentGRPOTrainer(model=model, args=args, processing_class=tk,
                grammar=grammar, mode='4x4', reward_kind='sid_dense',
                train_dataset=Dataset.from_list(records), eval_dataset=Dataset.from_list(records))
            from latent.objective import sid_rewards as real_reward
            with patch('latent.trainer.sid_rewards', wraps=real_reward) as rewards:
                packed = trainer._prepare_inputs([records[0]] * 16 + [records[1]] * 16)
            self.assertEqual([c.kwargs['kind'] for c in rewards.call_args_list], ['sid_dense', 'binary'])
            loss = trainer.compute_loss(trainer.model, packed)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(any(p.grad is not None for p in trainer.model.parameters()))


if __name__ == '__main__':
    unittest.main()
