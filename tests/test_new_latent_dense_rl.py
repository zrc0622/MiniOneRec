"""Guard new-SID dense RL initialization and two-card candidate budget."""
from pathlib import Path
import unittest
import json
import tempfile
from latent.train import parser,validate_sft_resume
from latent.rl_launch import check_gpu_capacity
from run_new_latent_dense_rl import MODEL,training_command


class NewDenseRLTests(unittest.TestCase):
    def test_gpu_checks_reject_wrong_inventory_busy_cards_and_insufficient_memory(self):
        inventory={'4':dict(total_mib=46080,used_mib=20,pids=[])}
        check_gpu_capacity(inventory,['4'],False,35)
        with self.assertRaises(ValueError):
            check_gpu_capacity(inventory,['5'],False,35)
        inventory['4']['pids']=[123]
        with self.assertRaises(RuntimeError):
            check_gpu_capacity(inventory,['4'],False,35)
        check_gpu_capacity(inventory,['4'],True,35)
        inventory['4']['used_mib']=45000
        with self.assertRaises(RuntimeError):
            check_gpu_capacity(inventory,['4'],True,35)

    def test_fast_entry_changes_only_implementation(self):
        original,_=training_command(Path('/tmp/snapshot'),Path('/tmp/run'),32,port=12345)
        fast,_=training_command(Path('/tmp/snapshot'),Path('/tmp/run'),32,port=12345,fast=True)
        expected=list(original);expected[expected.index('latent.train')]='latent.train_fast'
        self.assertEqual(fast,expected)
        smoke,_=training_command(Path('/tmp/snapshot'),Path('/tmp/run'),32,smoke=True,port=12345,fast=True)
        self.assertIn('latent.train_fast',smoke)
        args=parser().parse_args(fast[fast.index('latent.train_fast')+1:])
        self.assertEqual(args.model,str(MODEL))
        self.assertEqual(args.batch*2*args.gradient_accumulation,1024)
        self.assertEqual(args.mode,'4x4');self.assertEqual(args.reward_kind,'sid_dense')

    def test_weight02_checkpoint_initializes_dense_with_original_rl_weights(self):
        command,_=training_command(Path('/tmp/snapshot'),Path('/tmp/run'),32,port=12345)
        args=parser().parse_args(command[command.index('latent.train')+1:])
        self.assertEqual(args.model,str(MODEL))
        self.assertIn('lw02',args.model)
        self.assertEqual(args.mode,'4x4');self.assertEqual(args.reward_kind,'sid_dense')
        self.assertEqual(args.partial_weight,.2);self.assertEqual(args.latent_weight,1)
        self.assertEqual(args.temperature,1);self.assertEqual(args.beta,.001)
        self.assertEqual(args.epochs,2);self.assertEqual(args.lr,1e-5)
        self.assertEqual(args.rl_eval_fraction,.2);self.assertEqual(args.rl_save_total_limit,3)

    def test_two_card_budget_and_fresh_smoke_initialization(self):
        for batch in (16,32):
            formal,out=training_command(Path('/tmp/snapshot'),Path('/tmp/run'),batch,port=12345)
            smoke,short=training_command(Path('/tmp/snapshot'),Path('/tmp/run'),batch,smoke=True,port=12345)
            args=parser().parse_args(formal[formal.index('latent.train')+1:])
            bounded=parser().parse_args(smoke[smoke.index('latent.train')+1:])
            self.assertEqual(args.batch*2*args.gradient_accumulation,1024)
            self.assertEqual(args.model,bounded.model);self.assertNotEqual(out,short)
            self.assertEqual(bounded.max_steps,2);self.assertEqual(bounded.eval_limit,32)
            self.assertEqual(args.max_steps,-1);self.assertEqual(args.eval_limit,0)

    def test_rl_only_flags_do_not_break_older_sft_checkpoint_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp);checkpoint=output/'checkpoint-1';checkpoint.mkdir()
            for name in ('trainer_state.json','model.safetensors','scheduler.pt'):
                (checkpoint/name).write_text('fixture')
            group=checkpoint/'global_step1';group.mkdir();(group/'zero_optim_states.pt').write_text('fixture')
            args=parser().parse_args(['sft','--model','fixture','--train_file','train','--eval_file','valid',
                '--items','items','--index','index','--output_dir',str(output),
                '--resume_from_checkpoint',str(checkpoint)])
            old=dict(vars(args));old.pop('resume_from_checkpoint');old.pop('rl_eval_fraction');old.pop('rl_save_total_limit')
            (output/'latent_run.json').write_text(json.dumps(dict(options=old,source_hashes={})))
            self.assertEqual(validate_sft_resume(args,output),checkpoint.resolve())
