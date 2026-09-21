"""CPU regression tests for reference-only cleanup and distributed exit order."""

import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from deepspeed_cleanup import finish_rl_training, patch_reference_bf16_cleanup


class DummyOptim:
    def __init__(self):
        self.param_groups = [{"params": []}]


class BF16Optimizer:
    def __init__(self, real=False):
        self.optimizer = SimpleNamespace() if real else DummyOptim()
        self.using_real_optimizer = real
        self.bf16_groups = []

    def destroy(self):
        # The original failure: inference has a param group but no BF16 groups.
        for i, _ in enumerate(self.optimizer.param_groups):
            for _ in self.bf16_groups[i]:
                pass
        for hook in self._grad_acc_hooks:
            hook.remove()


class Engine:
    def __init__(self, destroy):
        self.destroy = destroy

    def __del__(self):
        self.destroy()


class CleanupTests(unittest.TestCase):
    def setUp(self):
        modules = {name: ModuleType(name) for name in (
            "deepspeed", "deepspeed.runtime", "deepspeed.runtime.bf16_optimizer",
            "deepspeed.runtime.utils",
        )}
        modules["deepspeed"].DeepSpeedEngine = Engine
        modules["deepspeed.runtime.bf16_optimizer"].BF16_Optimizer = BF16Optimizer
        modules["deepspeed.runtime.utils"].DummyOptim = DummyOptim
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_inference_cleanup_handles_missing_groups_and_hooks_repeatedly(self):
        optimizer = BF16Optimizer()
        with self.assertRaises(IndexError):
            optimizer.destroy()
        patch_reference_bf16_cleanup(SimpleNamespace(optimizer=optimizer))
        optimizer.destroy()
        optimizer.destroy()
        self.assertEqual(len(optimizer.optimizer.param_groups), 1)
        self.assertFalse(optimizer.using_real_optimizer)

    def test_cleanup_removes_optional_hooks_once(self):
        optimizer = BF16Optimizer()
        hook = Mock()
        optimizer._grad_acc_hooks = [hook]
        patch_reference_bf16_cleanup(SimpleNamespace(optimizer=optimizer))
        optimizer.destroy()
        optimizer.destroy()
        hook.remove.assert_called_once()

    def test_real_and_other_optimizers_are_not_patched(self):
        for optimizer in (BF16Optimizer(real=True), SimpleNamespace(destroy=Mock()), None):
            before = getattr(optimizer, "destroy", None)
            patch_reference_bf16_cleanup(SimpleNamespace(optimizer=optimizer))
            self.assertEqual(getattr(optimizer, "destroy", None), before)

    def test_engine_cleanup_precedes_process_group_exit_and_runs_once(self):
        calls = []
        active = [True]

        def destroy(name):
            self.assertTrue(active[0], "engine cleanup requires a live group")
            calls.append(name)

        def end():
            calls.append("end")
            active[0] = False

        policy = Engine(lambda: destroy("policy"))
        reference = Engine(lambda: destroy("reference"))
        accelerator = SimpleNamespace(
            wait_for_everyone=lambda: calls.append("barrier"), end_training=end,
        )
        trainer = SimpleNamespace(
            accelerator=accelerator, is_deepspeed_enabled=True,
            ref_model=reference, model_wrapped=policy, deepspeed=policy,
        )
        finish_rl_training(trainer)
        policy.__del__()
        reference.__del__()
        self.assertEqual(calls, ["barrier", "reference", "policy", "end"])

    def test_non_deepspeed_exit(self):
        accelerator = Mock()
        finish_rl_training(SimpleNamespace(accelerator=accelerator, is_deepspeed_enabled=False))
        self.assertEqual([c[0] for c in accelerator.mock_calls], ["wait_for_everyone", "end_training"])

    def test_actual_entrypoint_saves_before_cleanup_and_preserves_save_errors(self):
        # Execute the real final statements without importing the GPU stack or
        # loading models. A failed save must not be reported as a clean finish.
        path = Path(__file__).resolve().parents[1] / "rl.py"
        tree = ast.parse(path.read_text())
        train = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train")
        first = next(i for i, node in enumerate(train.body) if ast.unparse(node) == "trainer.train()")
        code = compile(ast.Module(body=train.body[first:], type_ignores=[]), str(path), "exec")
        import os
        for main_rank in (True, False):
            events = []
            trainer = SimpleNamespace(
                train=lambda: events.append("train"),
                save_model=lambda path: events.append(path),
                is_world_process_zero=lambda: main_rank,
            )
            namespace = dict(
                trainer=trainer, tokenizer=SimpleNamespace(save_pretrained=lambda p: events.append("tokenizer")),
                output_dir="output", os=os, finish_rl_training=lambda t: events.append("cleanup"),
            )
            exec(code, namespace)
            self.assertEqual(events, ["train", "output", "output/final_checkpoint"] +
                             (["tokenizer"] if main_rank else []) + ["cleanup"])
        trainer.save_model = Mock(side_effect=OSError("disk full"))
        namespace["finish_rl_training"] = Mock()
        with self.assertRaisesRegex(OSError, "disk full"):
            exec(code, namespace)
        namespace["finish_rl_training"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
