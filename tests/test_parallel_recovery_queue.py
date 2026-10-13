import json
from pathlib import Path
import tempfile
import unittest

from queue_parallel_recovery import predecessor_complete, idle_pair, recovery_command
from run_parallel_recovery import evaluation_command


class ParallelRecoveryQueueTests(unittest.TestCase):
    def test_waits_for_full_pipeline_not_training_finish_or_dead_coordinator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for phase in ('training_serial_sft', 'ranking_eval'):
                (root/'pipeline_status.json').write_text(json.dumps(dict(phase=phase,coordinator_pid=1)))
                self.assertFalse(predecessor_complete(root, alive=lambda *_: True))
                with self.assertRaises(RuntimeError):predecessor_complete(root, alive=lambda *_: False)
            for phase in ('failed', 'interrupted_server_reboot'):
                (root/'pipeline_status.json').write_text(json.dumps(dict(phase=phase)))
                with self.assertRaises(RuntimeError):predecessor_complete(root)

    def test_complete_requires_export_and_verified_both_protocols_on_both_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'pipeline_status.json').write_text(json.dumps(dict(phase='complete')))
            with self.assertRaises(ValueError):predecessor_complete(root)
            final = root/'sft/final_checkpoint'
            final.mkdir(parents=True)
            for name in ('model.safetensors','config.json','tokenizer_config.json'):(final/name).touch()
            for scope,count in (('sample256',256),('valid20pct',3232)):
                folder=root/'eval'/scope;folder.mkdir(parents=True)
                result=dict(examples=count,verified_same_ids_and_targets=True,
                    results=dict(new_history_latent=dict(single20={},four20_marginal={})))
                (folder/'comparison.json').write_text(json.dumps(result))
            self.assertTrue(predecessor_complete(root))
            result['results']['new_history_latent'].pop('four20_marginal')
            (root/'eval/valid20pct/comparison.json').write_text(json.dumps(result))
            with self.assertRaises(ValueError):predecessor_complete(root)

    def test_completion_does_not_allow_occupied_cards_or_missing_gpu(self):
        ready={g:dict(used_mib=17,utilization=0,pids=[]) for g in ('2','3')}
        self.assertTrue(idle_pair(ready,['2','3']))
        ready['3']['pids']=[123]
        self.assertFalse(idle_pair(ready,['2','3']))
        ready['3']['pids']=[];ready['3']['used_mib']=2000
        self.assertFalse(idle_pair(ready,['2','3']))
        self.assertFalse(idle_pair({'2':ready['2']},['2','3']))

    def test_gpu_pair_propagates_to_frozen_recovery_and_followup_eval(self):
        root=Path('/tmp/parallel');snapshot=Path('/tmp/frozen')
        command=recovery_command(root,snapshot,['2','3'])
        self.assertEqual(command[command.index('--gpus')+1],'2,3')
        self.assertEqual(command[command.index('--resume-source')+1],str(snapshot/'parallel_train.py'))
        runner=command[command.index('--eval-runner')+1]
        followup=evaluation_command(root,runner,['2','3'])
        self.assertEqual(followup[followup.index('--gpus')+1],'2,3')
        self.assertIn('--eval-only',followup)
        self.assertNotIn('2,5',followup)


if __name__ == '__main__':unittest.main()
