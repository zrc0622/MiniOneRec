"""Verify experiment settings and matched, complete two-GPU eval comparisons."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from run_target_interest_sft import training_command, merge_shards, validate_result, write_report


def result(ids, kind='latent'):
    methods=['single20','four20_marginal'] if kind=='latent' else ['direct50']
    rows=[]
    for rid in ids:
        candidates=['target']+[f'item{i}' for i in range(19)]
        if rid%2:candidates[0],candidates[5]=candidates[5],candidates[0]
        rows.append(dict(row_id=rid,output='target',predict=candidates,elapsed_seconds=.1))
    return dict(rows={m:copy.deepcopy(rows) for m in methods},batches={m:[] for m in methods},
                metadata=dict(model='/tmp/model',kind=kind,methods=methods,data_sha256='data',
                              index_sha256='index',source_sha256='code',protocol={},sample_sha256='sample'))


class TargetInterestSFTTests(unittest.TestCase):
    def test_training_command_is_accepted_by_real_trainer_and_keeps_effective_batch(self):
        from latent.train import parser
        for batch in (16,32):
            c=training_command(Path('/tmp/experiment'),Path('/tmp/snapshot'),batch)
            o=parser().parse_args(c[c.index('latent.train')+1:])
            self.assertEqual(2*o.batch*o.gradient_accumulation,1024)
            self.assertEqual(o.stage,'sft')
            self.assertEqual(o.sft_eval_fraction,1)
            self.assertTrue(o.sft_load_best)
            self.assertEqual(o.sft_early_stopping_patience,3)
            self.assertIn('target_vq3',o.labels)

    def test_merge_requires_disjoint_complete_consistent_shards(self):
        samples=[dict(row_ids=[0,2]),dict(row_ids=[1,3])];whole=dict(row_ids=[0,1,2,3])
        parts=[result(s['row_ids']) for s in samples]
        merged=merge_shards(parts,samples,whole)
        self.assertEqual([r['row_id'] for r in merged['rows']['single20']],[0,1,2,3])
        with self.assertRaises(ValueError):merge_shards(parts,samples,dict(row_ids=[0,1,2,3,4]))
        broken=copy.deepcopy(parts);broken[1]['metadata']['index_sha256']='other'
        with self.assertRaises(ValueError):merge_shards(broken,samples,whole)

    def test_result_rejects_duplicate_predictions_and_missing_method(self):
        r=result([0,1]);sample=dict(row_ids=[0,1])
        r['rows']['single20'][0]['predict'].append('target')
        with self.assertRaises(ValueError):validate_result(r,sample)
        r=result([0,1]);del r['rows']['four20_marginal']
        with self.assertRaises(ValueError):validate_result(r,sample)

    def test_comparison_computes_paired_metrics_and_rejects_wrong_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sample=dict(row_ids=[0,1])
            rs={'target_latent_sft':result([0,1]),'history_only_latent_sft':result([0,1]),'no_latent_sft':result([0,1],'sid')}
            write_report(root,sample,rs)
            report=json.loads((root/'comparison.json').read_text())
            m=report['results']['target_latent_sft']['single20']
            self.assertEqual(m['Recall@5'],.5)
            self.assertEqual(m['Recall@20'],1)
            self.assertEqual(m['fraction_with20'],1)
            for rows in rs['history_only_latent_sft']['rows'].values():rows[0]['output']='different_target'
            with self.assertRaises(ValueError):write_report(root,sample,rs)


if __name__=='__main__':unittest.main()
