"""Offline tests for the loss_type swap + auxiliary-loss plumbing.

Math helpers only (no VAE/LPIPS/facenet downloads); the full pixel-loss path is
exercised by the smoke runs. Mirrors test_lit_module.py's fake-bundle pattern.
"""

import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from omegaconf import OmegaConf

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


def test_loss_type_defaults_to_l2_and_aux_off():
    m = LoRADiffusionModule(_bundle(), _cfg())
    assert m.loss_type == "l2"
    assert m.aux_enabled is False
    assert m.aux_lpips_weight == 0.0 and m.aux_arcface_weight == 0.0


def test_elementwise_l2_matches_mse():
    m = LoRADiffusionModule(_bundle(), _cfg())
    pred, target = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    assert torch.allclose(
        m._elementwise_loss(pred, target), (pred - target) ** 2
    )


def test_elementwise_huber_differs_and_is_gentler_on_outliers():
    m = LoRADiffusionModule(_bundle(), _cfg(loss_type="huber", huber_delta=1.0))
    pred = torch.tensor([5.0])  # large error -> Huber (linear) < L2 (quadratic)
    target = torch.tensor([0.0])
    huber = m._elementwise_loss(pred, target)
    assert torch.isfinite(huber).all()
    assert huber.item() < ((pred - target) ** 2).item()


def test_reduce_loss_applies_snr_weights():
    sched = DDPMScheduler()
    m = LoRADiffusionModule(_bundle(noise_scheduler=sched), _cfg(snr_gamma=5.0))
    pred = torch.randn(3, 4, 8, 8)
    target = torch.randn(3, 4, 8, 8)
    weights = m._loss_weights(torch.tensor([10, 400, 900]))
    weighted = m._reduce_loss(pred, target, weights)
    unweighted = m._reduce_loss(pred, target, None)
    assert torch.isfinite(weighted) and weighted >= 0
    assert not torch.allclose(weighted, unweighted)


def test_predicted_x0_recovers_latents_for_perfect_epsilon():
    # With a perfect epsilon prediction (model_pred == noise), the closed-form
    # x0 estimate must reconstruct the clean latents.
    sched = DDPMScheduler()
    m = LoRADiffusionModule(_bundle(noise_scheduler=sched), _cfg())
    latents = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([100, 700])
    noisy = sched.add_noise(latents, noise, timesteps)
    x0 = m._predicted_x0(noise, noisy, timesteps)  # model_pred == noise
    assert torch.allclose(x0, latents, atol=1e-4)


def test_predicted_x0_recovers_latents_for_perfect_v_prediction():
    sched = DDPMScheduler()
    m = LoRADiffusionModule(
        _bundle(noise_scheduler=sched, prediction_type="v_prediction"), _cfg()
    )
    latents = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([100, 700])
    noisy = sched.add_noise(latents, noise, timesteps)
    v = sched.get_velocity(latents, noise, timesteps)
    x0 = m._predicted_x0(v, noisy, timesteps)  # model_pred == velocity target
    assert torch.allclose(x0, latents, atol=1e-4)


def test_aux_config_parsed():
    m = LoRADiffusionModule(
        _bundle("lora"),
        _cfg(mode="lora", aux_lpips_weight=0.5, aux_every_n_steps=2, aux_max_samples=1),
    )
    assert m.aux_enabled is True
    assert m.aux_lpips_weight == 0.5
    assert m.aux_every_n_steps == 2 and m.aux_max_samples == 1
