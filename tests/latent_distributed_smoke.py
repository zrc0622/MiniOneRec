"""CPU: torchrun --nproc_per_node 4 tests/latent_distributed_smoke.py."""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_latent import fixture, INDEX
import torch
from datasets import Dataset
from interest.train import InterestRLArguments
from latent.trainer import LatentGRPOTrainer

torch.set_num_threads(1)
for mode in ('original', '4x4'):
    tk, model, protocol, grammar = fixture()
    tasks = ['history_sid', 'item_identification', 'history_title', 'history_sid', 'history_sid']
    records = [dict(prompt=f'history {i}', target=''.join(INDEX[str(i)][:3] if task == 'item_identification' else INDEX[str(i)]),
                    task=task, auxiliary=task != 'history_sid') for i, task in enumerate(tasks)]
    with tempfile.TemporaryDirectory() as temp:
        args = InterestRLArguments(output_dir=temp, use_cpu=True, report_to=[], max_steps=2,
            learning_rate=1e-3, per_device_train_batch_size=16, per_device_eval_batch_size=16,
            gradient_accumulation_steps=2, remove_unused_columns=False, prediction_loss_only=True,
            eval_strategy='steps', eval_steps=1, save_strategy='no', logging_steps=1, disable_tqdm=True,
            ref_model_sync_steps=1, gradient_checkpointing=True, ddp_find_unused_parameters=False)
        trainer = LatentGRPOTrainer(model=model, args=args, processing_class=tk, grammar=grammar, mode=mode,
            train_dataset=Dataset.from_list(records), eval_dataset=Dataset.from_list(records[:3]))
        trainer.train()
        assert trainer.state.global_step == 2
        assert any('eval_loss' in x for x in trainer.state.log_history)
        values = trainer.accelerator.gather(next(model.parameters()).detach().flatten()[:20])
        for v in values.reshape(trainer.accelerator.num_processes, -1)[1:]:
            torch.testing.assert_close(v, values[:20])
        if trainer.is_world_process_zero():
            print(f'DISTRIBUTED_OK {mode}: 4 ranks, 5 train/3 valid prompt groups, accumulation2, ref_sync1')
if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
