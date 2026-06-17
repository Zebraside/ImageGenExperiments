"""Base-model registry + loader.

Each loader returns a :class:`ModelBundle` with the VAE and text encoder frozen.
Depending on ``cfg.train.mode`` the denoiser is either wrapped with LoRA adapters
(only the adapters stay trainable) or left fully trainable.

Models we can train at 512x512 (see the plan for VRAM estimates):
  - "sd21"      Manojb/stable-diffusion-2-1-base        (UNet, epsilon)   <- default
                (mirror of the now-private stabilityai/stable-diffusion-2-1-base)
  - "sd15"      stable-diffusion-v1-5/stable-diffusion-v1-5 (UNet, epsilon)
  - "pixart256" PixArt-alpha/PixArt-XL-2-256x256        (DiT, native 256) <- TODO
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from omegaconf import DictConfig
from peft import LoraConfig
from transformers import CLIPTextModel, CLIPTokenizer

from imagegen.models.bundle import ModelBundle

# Dtype the frozen modules (VAE + text encoder) are cast to, keyed by
# cfg.train.mixed_precision. They run only under no_grad, so storing them in
# half precision saves memory with no effect on the trainable UNet (which stays
# fp32 and is autocast by Lightning). fp16 is intentionally mapped to bf16 for
# the VAE's sake (fp16 VAE can overflow to NaN); fp32 keeps everything as-is.
FROZEN_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.bfloat16, "fp32": torch.float32}


def _load_sd(cfg: DictConfig) -> ModelBundle:
    """Loader for the Stable Diffusion UNet family (1.5 / 2.1 base)."""
    base = cfg.model.base_model

    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")

    # VAE + text encoder are always frozen, and cast to half precision: they
    # only run under no_grad, so this trims weights + activations for free.
    frozen_dtype = FROZEN_DTYPE[cfg.train.mixed_precision]
    vae.requires_grad_(False).to(dtype=frozen_dtype)
    text_encoder.requires_grad_(False).to(dtype=frozen_dtype)

    mode = cfg.train.mode
    if mode == "lora":
        unet.requires_grad_(False)
        lora_config = LoraConfig(
            r=cfg.lora.rank,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=list(cfg.lora.target_modules),
        )
        unet.add_adapter(lora_config)
    elif mode == "full":
        unet.requires_grad_(True)
        if cfg.train.get("grad_checkpointing", False):
            unet.enable_gradient_checkpointing()
    else:
        raise ValueError(f"Unknown train.mode {mode!r} (expected 'lora' or 'full').")

    return ModelBundle(
        base_model=base,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        denoiser=unet,
        noise_scheduler=noise_scheduler,
        prediction_type=noise_scheduler.config.prediction_type,
        mode=mode,
    )


def _load_pixart(cfg: DictConfig) -> ModelBundle:
    # PixArt is a DiT with a T5-XXL text encoder; its embeddings should be
    # precomputed/cached to keep T5 out of the training step. Left as a seam.
    raise NotImplementedError(
        "pixart256 loader not implemented yet. Use 'sd21' or 'sd15' for now."
    )


MODEL_REGISTRY: dict[str, Callable[[DictConfig], ModelBundle]] = {
    "sd21": _load_sd,
    "sd15": _load_sd,
    "pixart256": _load_pixart,
}


def load_model(cfg: DictConfig) -> ModelBundle:
    key = cfg.model.key
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model key {key!r}. Available: {sorted(MODEL_REGISTRY)}."
        )
    return MODEL_REGISTRY[key](cfg)
