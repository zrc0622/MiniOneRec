"""Optional CPU: torchrun --standalone --nproc_per_node 4 tests/interest_distributed_smoke.py."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_interest import fixture, INDEX
import torch
from datasets import Dataset
from interest.core import initialize_interests
from interest.generation import Grammar
from interest.train import InterestRLArguments
from interest.trainer import InterestGRPOTrainer


torch.set_num_threads(1)
for mode in ("16x1", "4x4"):
    tokenizer, model = fixture()
    initialize_interests(model, tokenizer, INDEX, 2)
    records = [dict(prompt=f"history {i}", target="".join(INDEX[str(i % 4)]),
                    history=[], task="history_sid" if i % 2 else "item_identification", auxiliary=i % 2 == 0)
               for i in range(5)]
    with tempfile.TemporaryDirectory() as path:
        args = InterestRLArguments(output_dir=path, use_cpu=True, report_to=[], max_steps=2,
                                    learning_rate=1e-3, per_device_train_batch_size=16,
                                    per_device_eval_batch_size=16, gradient_accumulation_steps=2,
                                    remove_unused_columns=False, prediction_loss_only=True,
                                    eval_strategy="steps", eval_steps=1, save_strategy="no",
                                    logging_steps=1, disable_tqdm=True, ref_model_sync_steps=1,
                                    gradient_checkpointing=True, ddp_find_unused_parameters=False)
        trainer = InterestGRPOTrainer(model=model, args=args, processing_class=tokenizer,
                                      grammar=Grammar(tokenizer, INDEX, 2), mode=mode,
                                      train_dataset=Dataset.from_list(records), eval_dataset=Dataset.from_list(records[:3]))
        trainer.train()
        assert trainer.state.global_step == 2
        assert any("eval_loss" in x for x in trainer.state.log_history)
        values = trainer.accelerator.gather(next(model.parameters()).detach().flatten()[:20])
        for v in values.reshape(trainer.accelerator.num_processes, -1)[1:]:
            torch.testing.assert_close(v, values[:20])
        if trainer.is_world_process_zero():
            print(f"DISTRIBUTED_OK {mode}: train=5 prompt groups, eval=3, ranks=4, accumulation=2, reference_sync=1")
if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
