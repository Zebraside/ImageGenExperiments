"""Offline tests for the LightningModule optimizer/scheduler + EMA wiring.

These build a tiny fake ModelBundle so nothing is downloaded.
"""

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR

from imagegen.lit_module import LoRADiffusionModule
from imagegen.models.bundle import ModelBundle


def _bundle(mode="full"):
    return ModelBundle(
        base_model="fake",
        tokenizer=object(),
        text_encoder=nn.Linear(2, 2),
        vae=nn.Linear(2, 2),
        denoiser=nn.Linear(2, 2),
        noise_scheduler=object(),
        prediction_type="epsilon",
        mode=mode,
    )


def _cfg(**train):
    base = {
        "model": {"key": "sd21", "image_size": 512},
        "train": {"mode": "full", "lr": 1e-5, "mixed_precision": "bf16"},
    }
    base["train"].update(train)
    return OmegaConf.create(base)


def test_constant_schedule_returns_plain_optimizer():
    module = LoRADiffusionModule(_bundle(), _cfg())  # lr_schedule defaults to "constant"
    assert isinstance(module.configure_optimizers(), torch.optim.Optimizer)


def test_cosine_schedule_returns_scheduler():
    module = LoRADiffusionModule(_bundle(), _cfg(lr_schedule="cosine", lr_warmup_steps=10, max_steps=100))
    out = module.configure_optimizers()
    assert isinstance(out, dict)
    assert isinstance(out["lr_scheduler"]["scheduler"], LambdaLR)
    assert out["lr_scheduler"]["interval"] == "step"


def test_cond_dropout_off_by_default():
    module = LoRADiffusionModule(_bundle(), _cfg())
    assert module.cond_dropout_prob == 0.0
    caps = ["a", "b", "c"]
    assert module._apply_cond_dropout(caps) is caps  # untouched when disabled


def test_cond_dropout_full_drops_everything():
    module = LoRADiffusionModule(_bundle(), _cfg(cond_dropout_prob=1.0))
    assert module._apply_cond_dropout(["a", "b", "c"]) == ["", "", ""]


def test_ema_off_by_default():
    assert LoRADiffusionModule(_bundle(), _cfg()).use_ema is False


def test_ema_on_only_for_full_mode():
    assert LoRADiffusionModule(_bundle("full"), _cfg(ema=True)).use_ema is True
    # EMA is meaningless for LoRA here -> stays off even if requested.
    assert LoRADiffusionModule(_bundle("lora"), _cfg(ema=True)).use_ema is False


def test_ema_weights_swaps_then_restores():
    module = LoRADiffusionModule(_bundle(), _cfg(ema=True))
    module.on_fit_start()  # builds the EMA shadow == current (raw) weights
    ema_snapshot = module.denoiser.weight.detach().clone()
    # Move the *live* weights away from the EMA so the swap is observable.
    with torch.no_grad():
        module.denoiser.weight.add_(5.0)
    live = module.denoiser.weight.detach().clone()

    with module.ema_weights():  # should expose the EMA (raw) weights
        assert torch.allclose(module.denoiser.weight, ema_snapshot)
        assert not torch.allclose(module.denoiser.weight, live)
    # ...and restore the live weights on exit.
    assert torch.allclose(module.denoiser.weight, live)


def test_ema_checkpoint_round_trips():
    src = LoRADiffusionModule(_bundle(), _cfg(ema=True))
    src.on_fit_start()
    src.ema.step(src.bundle.trainable_parameters())
    ckpt: dict = {}
    src.on_save_checkpoint(ckpt)
    assert "ema" in ckpt

    dst = LoRADiffusionModule(_bundle(), _cfg(ema=True))
    dst.on_load_checkpoint(ckpt)   # stashes state; EMA not built yet
    assert dst.ema is None
    dst.on_fit_start()             # builds EMA and applies the stashed state
    assert dst.ema is not None
    src_shadow = src.ema.state_dict()["shadow_params"]
    dst_shadow = dst.ema.state_dict()["shadow_params"]
    assert all(torch.allclose(a, b) for a, b in zip(src_shadow, dst_shadow))
