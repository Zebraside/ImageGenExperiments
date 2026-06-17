"""LightningModule for diffusion LoRA / full fine-tuning.

Works for the SD UNet family: encode image -> latents, add noise at a random
timestep, predict the noise (or velocity), MSE against the target. Only the
parameters the factory left trainable (LoRA adapters, or the whole UNet in
``full`` mode) receive optimizer state.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
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

        # Classifier-free guidance: drop this fraction of captions to the null prompt
        # during training so the UNet learns the unconditional score that inference-time
        # guidance extrapolates from (0.0 disables it).
        self.cond_dropout_prob = float(cfg.train.get("cond_dropout_prob", 0.0))

        # Weight EMA (full fine-tuning only). Built in on_fit_start once the module
        # is on-device; sampled/saved from instead of the raw weights to curb the
        # late-step fine-structure drift full fine-tuning is prone to.
        self.use_ema = bool(cfg.train.get("ema", False)) and bundle.mode == "full"
        self.ema: EMAModel | None = None
        self._ema_state: dict | None = None  # stashed from a checkpoint until EMA exists

    # --- EMA --------------------------------------------------------------

    def on_fit_start(self) -> None:
        if not self.use_ema:
            return
        self.ema = EMAModel(
            self.bundle.trainable_parameters(),
            decay=float(self.cfg.train.get("ema_decay", 0.9999)),
        )
        self.ema.to(self.device)
        if self._ema_state is not None:  # resumed from a checkpoint
            self.ema.load_state_dict(self._ema_state)
            self._ema_state = None

    def on_before_zero_grad(self, optimizer) -> None:
        # Fires once per optimizer step (after step()), so it advances the EMA at
        # the true optimization cadence even under accumulate_grad_batches.
        if self.ema is not None:
            self.ema.step(self.bundle.trainable_parameters())

    @contextmanager
    def ema_weights(self) -> Iterator[None]:
        """Temporarily swap the EMA weights into the denoiser, then restore.

        The live weights are backed up on CPU (not via ``EMAModel.store``, which
        clones them on-device): with EMA resident, the periodic sample pass runs
        with very little VRAM headroom, so the swap must not add a full GPU copy.
        """
        if self.ema is None:
            yield
            return
        params = self.bundle.trainable_parameters()
        backup = [p.detach().cpu().clone() for p in params]
        self.ema.copy_to(params)
        try:
            yield
        finally:
            for p, b in zip(params, backup):
                p.data.copy_(b.to(p.device))

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.ema is not None:
            checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # EMA isn't a registered module, so restore it manually. on_fit_start may
        # run after this, so stash the state and apply it once the EMA is built.
        self._ema_state = checkpoint.get("ema")

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

    def _apply_cond_dropout(self, captions: list[str]) -> list[str]:
        """Replace a random per-sample fraction of captions with the empty (null) prompt.

        The empty string tokenizes to SD's unconditional sequence -- the same input the
        pipeline feeds the negative branch of classifier-free guidance.
        """
        if self.cond_dropout_prob <= 0.0:
            return captions
        keep = torch.rand(len(captions)) >= self.cond_dropout_prob
        return [c if keep[i] else "" for i, c in enumerate(captions)]

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        pixel_values = batch["pixel_values"]

        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            encoder_hidden_states = self._encode_text(self._apply_cond_dropout(batch["caption"]))

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

    def _build_optimizer(self) -> torch.optim.Optimizer:
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

    def configure_optimizers(self):
        optimizer = self._build_optimizer()

        schedule = self.cfg.train.get("lr_schedule", "constant")
        if schedule == "constant":
            return optimizer
        if schedule != "cosine":
            raise ValueError(
                f"Unknown train.lr_schedule {schedule!r} (expected 'constant' or 'cosine')."
            )

        max_steps = self.cfg.train.get("max_steps")
        if not max_steps or max_steps < 0:
            raise ValueError("train.lr_schedule='cosine' requires a positive train.max_steps.")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.cfg.train.get("lr_warmup_steps", 0)),
            num_training_steps=int(max_steps),
        )
        # interval="step": the scheduler advances once per optimizer step, matching
        # num_training_steps == max_steps (which Lightning also counts in opt steps).
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

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
        # Release cached-but-unused blocks so the sampling pass has room -- with EMA
        # resident, training leaves little free VRAM (measured ~0.6 GB at batch 16).
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
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
            with self.ema_weights(), torch.autocast(
                self.device.type,
                dtype=weight_dtype,
                enabled=weight_dtype != torch.float32,
            ):
                # One image per pipeline call (not [prompt] * num_images in a single
                # batched call): at 512 the batched sample pass is the run's VRAM peak
                # and OOMs a 16 GB card. Reusing the one seeded generator across calls
                # keeps each image a distinct, deterministic draw.
                images = []
                for _ in range(num_images):
                    images.extend(
                        pipe(
                            prompt,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            height=self.image_size,
                            width=self.image_size,
                            generator=generator,
                        ).images
                    )
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
            # Persist the EMA weights (what we sample from), not the raw ones.
            with self.ema_weights():
                self.bundle.build_pipeline().save_pretrained(str(out))
        return out
