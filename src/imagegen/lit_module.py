"""LightningModule for diffusion LoRA / full fine-tuning.

Works for the SD UNet family: encode image -> latents, add noise at a random
timestep, predict the noise (or velocity), MSE against the target. Only the
parameters the factory left trainable (LoRA adapters, or the whole UNet in
``full`` mode) receive optimizer state.
"""

from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from omegaconf import DictConfig
from peft.utils import get_peft_model_state_dict

from imagegen.models.bundle import ModelBundle
from imagegen.models.factory import FROZEN_DTYPE


class LoRADiffusionModule(L.LightningModule):
    def __init__(self, bundle: ModelBundle, cfg: DictConfig) -> None:
        super().__init__()
        self.bundle = bundle
        self.cfg = cfg
        self.image_size = cfg.model.image_size

        # Register the sub-modules so Lightning moves them to the right device.
        # Frozen modules are switched to eval() so dropout/other train-only ops stay off.
        self.vae = bundle.vae.eval()
        self.text_encoder = bundle.text_encoder.eval()
        self.denoiser = bundle.denoiser
        self.tokenizer = bundle.tokenizer
        self.noise_scheduler = bundle.noise_scheduler

    # --- training ---------------------------------------------------------

    def _encode_text(self, captions: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return self.text_encoder(tokens)[0]

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        pixel_values = batch["pixel_values"]

        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            encoder_hidden_states = self._encode_text(batch["caption"])

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=self.device,
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        if self.bundle.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:  # epsilon
            target = noise

        model_pred = self.denoiser(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        params = self.bundle.trainable_parameters()
        optimizer = self.cfg.train.get("optimizer", "adamw")
        if optimizer == "adamw_8bit":
            try:
                import bitsandbytes as bnb
            except ImportError as exc:  # pragma: no cover - depends on install
                raise ImportError(
                    "train.optimizer='adamw_8bit' requires bitsandbytes "
                    "(`uv add bitsandbytes`)."
                ) from exc
            return bnb.optim.AdamW8bit(params, lr=self.cfg.train.lr)
        if optimizer == "adamw":
            return torch.optim.AdamW(params, lr=self.cfg.train.lr)
        raise ValueError(
            f"Unknown train.optimizer {optimizer!r} (expected 'adamw' or 'adamw_8bit')."
        )

    # --- sampling / saving ------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        num_images: int,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ) -> list:
        pipe = self.bundle.build_pipeline().to(self.device)
        was_training = self.denoiser.training
        self.denoiser.eval()
        generator = (
            torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        )
        # The sampling callback runs outside the trainer's autocast region, but the
        # frozen VAE/text encoder are in half precision while the UNet stays fp32.
        # Autocast (not pipe.to(dtype=...), which would clobber the fp32 UNet weights)
        # reconciles the mix so the pipeline runs without a dtype mismatch.
        weight_dtype = FROZEN_DTYPE[self.cfg.train.mixed_precision]
        try:
            with torch.autocast(
                self.device.type,
                dtype=weight_dtype,
                enabled=weight_dtype != torch.float32,
            ):
                images = pipe(
                    [prompt] * num_images,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=self.image_size,
                    width=self.image_size,
                    generator=generator,
                ).images
        finally:
            if was_training:
                self.denoiser.train()
        return images

    def save_weights(self, output_dir: str | Path) -> Path:
        """Persist results. LoRA -> portable adapter safetensors; full -> full pipeline."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if self.bundle.mode == "lora":
            lora_state = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(self.denoiser)
            )
            StableDiffusionPipeline.save_lora_weights(
                save_directory=str(out),
                unet_lora_layers=lora_state,
                safe_serialization=True,
            )
        else:
            self.bundle.build_pipeline().save_pretrained(str(out))
        return out
