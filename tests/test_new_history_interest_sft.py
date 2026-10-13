import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from run_new_history_interest_sft import check_recipe, training_command, resume_command, canonical_result, report


class NewHistoryInterestSFTTests(unittest.TestCase):
    def test_recipe_refuses_target_and_unscaled_unit_embeddings(self):
        manifest = dict(k=3, codebook_size=256, collapsed_all_slots=False,
            embedding_recipe=dict(embedding_input='history', contains_target=False,
                pooling='last_valid_token', normalization='l2', dim=2560),
            vq_options=dict(input_scale=math.sqrt(2560)), vq_input_transform=dict(kind='sqrt_dim'))
        check_recipe(manifest)
        for changed in (dict(contains_target=True), dict(embedding_input='history_target')):
            invalid = copy.deepcopy(manifest)
            invalid['embedding_recipe'].update(changed)
            with self.assertRaises(ValueError):check_recipe(invalid)
        invalid = copy.deepcopy(manifest)
        invalid['vq_options']['input_scale'] = 1
        with self.assertRaises(ValueError):check_recipe(invalid)

    def test_serial_command_keeps_global_batch_and_original_causal_training(self):
        from latent.train import parser
        for batch in (16, 32):
            command = training_command(Path('/tmp/experiment'), Path('/tmp/snapshot'), 12345, batch)
            args = parser().parse_args(command[command.index('latent.train') + 1:])
            self.assertEqual(2 * args.batch * args.gradient_accumulation, 1024)
            self.assertEqual(args.sft_eval_fraction, 1)
            self.assertEqual(args.epochs, 15)
            self.assertIn('history_lasttoken_l2_sqrtd', args.labels)
            self.assertIn('--rdzv_backend=static', command)
            self.assertNotIn('latent.parallel_train', command)

    def test_tuning_changes_only_interest_weight_and_rejects_invalid_values(self):
        from latent.train import parser
        root, snapshot = Path('/tmp/experiment'), Path('/tmp/snapshot')
        base = training_command(root, snapshot, 12345)
        tuned = training_command(root, snapshot, 12345, latent_weight=.2)
        self.assertEqual([i for i, (a, b) in enumerate(zip(base, tuned)) if a != b],
                         [base.index('--latent_weight') + 1])
        options = parser().parse_args(tuned[tuned.index('latent.train') + 1:])
        self.assertEqual(options.latent_weight, .2)
        self.assertEqual(options.batch * 2 * options.gradient_accumulation, 1024)
        for invalid in (0, -1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):training_command(root, snapshot, 12345, latent_weight=invalid)

    def test_cross_sid_comparison_preserves_provenance_and_real_item_ranks(self):
        source = [dict(item_id='1')]
        original = dict(metadata=dict(kind='latent', data_sha256='original_data'),
                        rows={m:[dict(row_id=0,output='old1',predict=['old2','old1'])]
                              for m in ('single20','four20_marginal')})
        new = copy.deepcopy(original)
        for rows in new['rows'].values():rows[0].update(output='new1',predict=['new2','new1'])
        a = canonical_result(original, {'1':['old1'],'2':['old2']}, source, [0])
        b = canonical_result(new, {'1':['new1'],'2':['new2']}, source, [0])
        self.assertEqual(a['rows'], b['rows'])
        self.assertEqual(a['metadata']['data_sha256'], 'original_data')
        self.assertEqual(original['rows']['single20'][0]['output'], 'old1')
        with self.assertRaises(ValueError):canonical_result(new, {'1':['new1'],'2':['new2']}, [dict(item_id='2')], [0])
        with self.assertRaises(ValueError):canonical_result(new, {'1':['new1'],'2':['new2']}, source, [1])

    def test_resume_preserves_options_and_rejects_missing_rank_state_or_changed_source(self):
        from latent.train import parser
        from latent.core import sha
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / 'source_snapshot'
            snapshot.mkdir()
            src = snapshot / 'original.txt'
            src.write_text('original')
            (snapshot / 'hashes.json').write_text(json.dumps({'original.txt': sha(src)}))
            cp = root / 'sft/checkpoint-138'
            cp.mkdir(parents=True)
            for name in ('config.json', 'tokenizer_config.json', 'model.safetensors', 'scheduler.pt',
                         'rng_state_0.pth', 'rng_state_1.pth'):
                (cp / name).write_text('state')
            (cp / 'trainer_state.json').write_text(json.dumps(dict(global_step=138, max_steps=4140)))
            ds = cp / 'global_step138'
            ds.mkdir()
            for name in ('bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt',
                         'bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt', 'mp_rank_00_model_states.pt'):
                (ds / name).write_text('state')
            command = training_command(root, snapshot, 12345)
            (root / 'sft.command.json').write_text(json.dumps(command))
            options = vars(parser().parse_args(command[command.index('latent.train') + 1:]))
            (root / 'sft/latent_run.json').write_text(json.dumps(dict(options=options, source_hashes={})))
            resumed = resume_command(root, cp, 23456)
            self.assertEqual(resumed[:-2], [s.replace('master_port=12345', 'master_port=23456') for s in command])
            self.assertEqual(resumed[-2:], ['--resume_from_checkpoint', str(cp)])
            (cp / 'rng_state_1.pth').unlink()
            with self.assertRaises(ValueError):resume_command(root, cp, 23456)
            (cp / 'rng_state_1.pth').write_text('state')
            src.write_text('modified')
            with self.assertRaises(ValueError):resume_command(root, cp, 23456)
            src.write_text('original')
            command[command.index('--batch') + 1] = '16'
            (root / 'sft.command.json').write_text(json.dumps(command))
            with self.assertRaises(ValueError):resume_command(root, cp, 23456)

    def test_report_compares_matching_real_targets_and_all_six_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = dict(rows={m:[dict(row_id=0,output='1',predict=['1','2'])]
                                for m in ('single20','four20_marginal')})
            sid = dict(rows={'direct50':[dict(row_id=0,output='1',predict=['2','1'])]})
            results = dict(new_history_latent=result,old_history_latent=result,
                           new_sid_no_latent=sid,old_sid_no_latent=sid)
            report(Path(tmp), dict(row_ids=[0]), results)
            saved = json.loads((Path(tmp)/'comparison.json').read_text())
            values = saved['results']['new_sid_no_latent']['direct50']
            self.assertEqual(values['Recall@20'], 1)
            self.assertAlmostEqual(values['NDCG@20'], 1/math.log2(3))
            self.assertEqual(values['fraction_with20'], 0)
            results['reference_history_latent'] = copy.deepcopy(result)
            report(Path(tmp), dict(row_ids=[0]), results)
            saved = json.loads((Path(tmp)/'comparison.json').read_text())
            self.assertEqual(set(saved['results']['reference_history_latent']),
                             {'single20', 'four20_marginal'})
            self.assertIn('权重1对照', (Path(tmp)/'comparison.md').read_text())
            results['new_sid_no_latent'] = copy.deepcopy(sid)
            results['new_sid_no_latent']['rows']['direct50'][0]['output'] = 'different'
            with self.assertRaises(ValueError):report(Path(tmp), dict(row_ids=[0]), results)


if __name__ == '__main__':unittest.main()
