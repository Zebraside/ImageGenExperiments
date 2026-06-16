"""A uniform container for the components a diffusion training step needs.

Keeping these behind one dataclass lets the LightningModule stay agnostic to the
specific base model (SD 1.5 / 2.1 today, PixArt later): the factory fills the
bundle and the module just drives ``vae`` / ``text_encoder`` / ``denoiser`` /
``noise_scheduler``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline


@dataclass
class ModelBundle:
    base_model: str
    tokenizer: object
    text_encoder: nn.Module
    vae: nn.Module
    denoiser: nn.Module           # UNet2DConditionModel for the SD family
    noise_scheduler: object       # DDPMScheduler used for the training objective
    prediction_type: str          # "epsilon" or "v_prediction"
    mode: str                     # "lora" | "full"

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.denoiser.parameters() if p.requires_grad]

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def num_total(self) -> int:
        return sum(p.numel() for p in self.denoiser.parameters())

    @torch.no_grad()
    def build_pipeline(self) -> StableDiffusionPipeline:
        """Assemble a sampling pipeline that reuses the in-memory components.

        The scheduler is loaded fresh from the repo (a fast sampler like PNDM),
        not the DDPM scheduler used for training. Components keep their current
        device/dtype, so callers should run generation under the same autocast
        context used for training.
        """
        pipe = StableDiffusionPipeline.from_pretrained(
            self.base_model,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.denoiser,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.set_progress_bar_config(disable=True)
        return pipe
