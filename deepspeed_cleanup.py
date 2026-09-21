"""Local workarounds for DeepSpeed's inference BF16 optimizer teardown."""

from types import MethodType


def _destroy_dummy_bf16_optimizer(optimizer):
    # DeepSpeed 0.18.0 skips BF16 groups/hooks setup for DummyOptim, but its
    # destroy() indexes those absent groups using DummyOptim.param_groups.
    for group in optimizer.bf16_groups:
        for param in group:
            if getattr(param, "_hp_mapping", None) is not None:
                param._hp_mapping = None
    for hook in getattr(optimizer, "_grad_acc_hooks", ()):
        hook.remove()
    optimizer._grad_acc_hooks = []


def patch_reference_bf16_cleanup(engine):
    """Patch only an inference BF16 instance; leave real optimizers unchanged."""
    from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
    from deepspeed.runtime.utils import DummyOptim

    optimizer = engine.optimizer
    if (
        isinstance(optimizer, BF16_Optimizer)
        and isinstance(optimizer.optimizer, DummyOptim)
        and not optimizer.using_real_optimizer
        and not optimizer.bf16_groups
    ):
        optimizer.destroy = MethodType(_destroy_dummy_bf16_optimizer, optimizer)


def _already_destroyed():
    pass


def finish_rl_training(trainer):
    """Call on all ranks after final model/tokenizer saving has succeeded."""
    accelerator = trainer.accelerator
    accelerator.wait_for_everyone()
    if trainer.is_deepspeed_enabled:
        from deepspeed import DeepSpeedEngine

        # The policy's optimizer cleanup uses get_rank(), so engines must be
        # destroyed while the process group is still alive. Trainer may hold
        # several references to the same engine.
        seen = set()
        for engine in (trainer.ref_model, trainer.model_wrapped, trainer.deepspeed):
            if isinstance(engine, DeepSpeedEngine) and id(engine) not in seen:
                engine.destroy()
                # __del__ calls destroy() again; do not repeat cleanup after
                # accelerator.end_training() has released the process group.
                engine.destroy = _already_destroyed
                seen.add(id(engine))
    accelerator.end_training()
