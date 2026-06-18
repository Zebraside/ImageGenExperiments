"""Offline tests for the LightningModule optimizer/scheduler + EMA wiring.

These build a tiny fake ModelBundle so nothing is downloaded.
"""

import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR

from imagegen.lit_module import LoRADiffusionModule
from imagegen.models.bundle import ModelBundle


def _bundle(mode="full", noise_scheduler=None, prediction_type="epsilon"):
    return ModelBundle(
        base_model="fake",
        tokenizer=object(),
        text_encoder=nn.Linear(2, 2),
        vae=nn.Linear(2, 2),
        denoiser=nn.Linear(2, 2),
        noise_scheduler=noise_scheduler if noise_scheduler is not None else object(),
        prediction_type=prediction_type,
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


def test_snr_and_val_seed_default_off():
    module = LoRADiffusionModule(_bundle(), _cfg())
    assert module.snr_gamma is None
    assert module.val_seed == 0


def test_loss_weights_none_when_snr_disabled():
    module = LoRADiffusionModule(_bundle(), _cfg())
    assert module._loss_weights(torch.tensor([0, 500, 999])) is None


def test_min_snr_weights_epsilon():
    sched = DDPMScheduler()  # builds alphas_cumprod offline (no download)
    module = LoRADiffusionModule(_bundle(noise_scheduler=sched), _cfg(snr_gamma=5.0))
    timesteps = torch.tensor([0, 250, 500, 999])
    w = module._loss_weights(timesteps)
    assert w is not None
    assert w.shape == timesteps.shape
    assert torch.isfinite(w).all() and (w > 0).all()
    # epsilon: weight = min(SNR, gamma) / SNR <= 1, and == 1 wherever SNR <= gamma.
    assert (w <= 1.0 + 1e-5).all()


def test_min_snr_weights_v_prediction_differs():
    sched = DDPMScheduler()
    eps = LoRADiffusionModule(_bundle(noise_scheduler=sched, prediction_type="epsilon"), _cfg(snr_gamma=5.0))
    vpred = LoRADiffusionModule(
        _bundle(noise_scheduler=sched, prediction_type="v_prediction"), _cfg(snr_gamma=5.0)
    )
    timesteps = torch.tensor([10, 400, 900])
    assert not torch.allclose(eps._loss_weights(timesteps), vpred._loss_weights(timesteps))


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
