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
from diffusers.training_utils import EMAModel, compute_snr
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

        # min-SNR-gamma loss weighting (Hang et al. 2023): down-weights low-noise
        # timesteps the model trivially fits, balancing the per-timestep loss scales.
        # None keeps plain (unweighted) MSE.
        snr_gamma = cfg.train.get("snr_gamma")
        self.snr_gamma = float(snr_gamma) if snr_gamma is not None else None

        # Seed offset for the deterministic validation pass (see validation_step).
        self.val_seed = int(cfg.train.get("val_seed", 0))

        # Base reconstruction loss on the noise/velocity target: L2 (MSE, default)
        # or Huber/smooth-L1, which is gentler on the high-error outliers that
        # dominate MSE. min-SNR weighting applies identically to either.
        self.loss_type = str(cfg.train.get("loss_type", "l2"))
        self.huber_delta = float(cfg.train.get("huber_delta", 1.0))

        # Auxiliary pixel-space losses (LoRA experiments). Both reconstruct the
        # predicted x0, VAE-decode it, and compare to the real image: LPIPS for
        # perceptual similarity, ArcFace (facenet identity embedding) for identity
        # preservation. Weight 0.0 disables a term (the default for exps 1-4, so
        # they pay nothing). The decode keeps a graph, so it is restricted to the
        # first aux_max_samples of the batch every aux_every_n_steps to bound VRAM.
        self.aux_lpips_weight = float(cfg.train.get("aux_lpips_weight", 0.0))
        self.aux_arcface_weight = float(cfg.train.get("aux_arcface_weight", 0.0))
        self.aux_every_n_steps = int(cfg.train.get("aux_every_n_steps", 1))
        self.aux_max_samples = int(cfg.train.get("aux_max_samples", 1))
        # LPIPS backbone: 'alex' is much lighter (memory/compute) than 'vgg' for the
        # in-graph perceptual loss on a 16 GB card; both are valid LPIPS variants.
        self.aux_lpips_net = str(cfg.train.get("aux_lpips_net", "alex"))
        self.aux_enabled = self.aux_lpips_weight > 0 or self.aux_arcface_weight > 0
        # Frozen perceptual/face nets, built lazily on-device. Held in a plain dict
        # so they are NOT registered as submodules (kept out of the state_dict /
        # EMA, which only tracks the trainable UNet).
        self._aux_nets: dict = {}

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

    def _loss_weights(self, timesteps: torch.Tensor) -> torch.Tensor | None:
        """Per-sample min-SNR-gamma weights for ``timesteps``; ``None`` when disabled.

        epsilon prediction: ``min(SNR, gamma) / SNR``; v-prediction adds the +1 term
        (``min(SNR, gamma) / (SNR + 1)``) since the velocity target already folds in
        the signal component.
        """
        if self.snr_gamma is None:
            return None
        snr = compute_snr(self.noise_scheduler, timesteps)
        clamped = snr.clamp(max=self.snr_gamma)
        if self.bundle.prediction_type == "v_prediction":
            return clamped / (snr + 1)
        return clamped / snr

    def _elementwise_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Unreduced reconstruction error, L2 (MSE) or Huber per ``loss_type``."""
        if self.loss_type == "huber":
            return F.huber_loss(pred, target, reduction="none", delta=self.huber_delta)
        if self.loss_type == "l2":
            return F.mse_loss(pred, target, reduction="none")
        raise ValueError(f"Unknown train.loss_type {self.loss_type!r} (expected 'l2' or 'huber').")

    def _reduce_loss(
        self, model_pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None
    ) -> torch.Tensor:
        err = self._elementwise_loss(model_pred.float(), target.float())
        if weights is None:
            return err.mean()
        # Per-sample error, weighted by min-SNR, then averaged over the batch.
        per_sample = err.mean(dim=list(range(1, model_pred.ndim)))
        return (per_sample * weights.to(per_sample.device)).mean()

    def _diffusion_loss(
        self,
        batch: dict,
        *,
        generator: torch.Generator | None = None,
        apply_dropout: bool = True,
        return_parts: bool = False,
    ):
        """Shared training/validation objective.

        Passing a ``generator`` makes the noise + timestep draw (and the VAE sample)
        deterministic, so the validation pass scores the same triples every time.
        ``apply_dropout`` gates classifier-free-guidance caption dropout (training only;
        validation measures the conditional loss). With ``return_parts`` the loss is
        returned alongside the tensors the auxiliary pixel losses need (the differentiable
        ``model_pred`` and the inputs to reconstruct the predicted x0).
        """
        pixel_values = batch["pixel_values"]
        captions = self._apply_cond_dropout(batch["caption"]) if apply_dropout else batch["caption"]

        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample(generator=generator)
            latents = latents * self.vae.config.scaling_factor
            encoder_hidden_states = self._encode_text(captions)

        noise = torch.randn(
            latents.shape, generator=generator, device=self.device, dtype=latents.dtype
        )
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            generator=generator,
            device=self.device,
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        if self.bundle.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:  # epsilon
            target = noise

        model_pred = self.denoiser(noisy_latents, timesteps, encoder_hidden_states).sample

        weights = self._loss_weights(timesteps)
        loss = self._reduce_loss(model_pred, target, weights)
        if not return_parts:
            return loss
        parts = {
            "model_pred": model_pred,
            "noisy_latents": noisy_latents,
            "timesteps": timesteps,
            "pixel_values": pixel_values,
        }
        return loss, parts

    # --- auxiliary pixel-space losses (LPIPS / ArcFace) -------------------

    def _lpips_net(self):
        if "lpips" not in self._aux_nets:
            import lpips

            net = lpips.LPIPS(net=self.aux_lpips_net, verbose=False).to(self.device).eval()
            net.requires_grad_(False)
            self._aux_nets["lpips"] = net
        return self._aux_nets["lpips"]

    def _facenet(self):
        if "facenet" not in self._aux_nets:
            from facenet_pytorch import InceptionResnetV1

            net = InceptionResnetV1(pretrained="vggface2").to(self.device).eval()
            net.requires_grad_(False)
            self._aux_nets["facenet"] = net
        return self._aux_nets["facenet"]

    def _predicted_x0(
        self, model_pred: torch.Tensor, noisy_latents: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Closed-form one-step x0 estimate from the model output (eps or v)."""
        acp = self.noise_scheduler.alphas_cumprod.to(self.device)[timesteps]
        acp = acp.view(-1, *([1] * (noisy_latents.ndim - 1)))
        sqrt_acp = acp.sqrt()
        sqrt_one_minus = (1 - acp).sqrt()
        if self.bundle.prediction_type == "v_prediction":
            return sqrt_acp * noisy_latents - sqrt_one_minus * model_pred
        return (noisy_latents - sqrt_one_minus * model_pred) / sqrt_acp

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """VAE-decode latents to an image in [-1, 1] (graph kept for backprop)."""
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents.to(self.vae.dtype)).sample
        return image.clamp(-1, 1)

    def _aux_pixel_loss(self, parts: dict) -> tuple[torch.Tensor, dict]:
        """Weighted LPIPS + ArcFace loss on the decoded predicted x0 vs the real image."""
        n = min(self.aux_max_samples, parts["model_pred"].shape[0])
        pred_x0 = self._predicted_x0(
            parts["model_pred"][:n], parts["noisy_latents"][:n], parts["timesteps"][:n]
        )
        decoded = self._decode_latents(pred_x0).float()  # [-1, 1]
        real = parts["pixel_values"][:n].float()  # already [-1, 1]

        total = decoded.new_zeros(())
        logs: dict = {}
        if self.aux_lpips_weight > 0:
            # lpips expects NCHW in [-1, 1] and returns a per-image distance. Compare
            # at 256 (downsampled) to keep the perceptual-net activations within VRAM.
            dec_lp = F.interpolate(decoded, size=256, mode="bilinear", align_corners=False)
            real_lp = F.interpolate(real, size=256, mode="bilinear", align_corners=False)
            lpips_val = self._lpips_net()(dec_lp, real_lp).mean()
            total = total + self.aux_lpips_weight * lpips_val
            logs["train/loss_lpips"] = lpips_val.detach()
        if self.aux_arcface_weight > 0:
            pred_r = F.interpolate(decoded, size=160, mode="bilinear", align_corners=False)
            real_r = F.interpolate(real, size=160, mode="bilinear", align_corners=False)
            net = self._facenet()
            emb_pred = net(pred_r)
            emb_real = net(real_r)  # frozen net, but keep simple (no_grad not required)
            arc_val = (1 - F.cosine_similarity(emb_pred, emb_real, dim=1)).mean()
            total = total + self.aux_arcface_weight * arc_val
            logs["train/loss_arcface"] = arc_val.detach()
        return total, logs

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        run_aux = self.aux_enabled and (self.global_step % self.aux_every_n_steps == 0)
        if run_aux:
            base, parts = self._diffusion_loss(batch, return_parts=True)
            aux, logs = self._aux_pixel_loss(parts)
            loss = base + aux
            self.log("train/loss_diffusion", base, on_step=True, on_epoch=False)
            for key, value in logs.items():
                self.log(key, value, on_step=True, on_epoch=False)
        else:
            loss = self._diffusion_loss(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # Deterministic per-batch generator: the same (image, noise, timestep) triples
        # are scored at every validation, so val/loss is comparable across steps and
        # meaningful for checkpoint selection (unlike the noisy per-step train loss).
        # Validation runs on the live weights; with a moderate ema_decay the EMA tracks
        # them closely, so this is a faithful proxy for the saved (EMA) weights.
        generator = torch.Generator(device=self.device).manual_seed(self.val_seed + batch_idx)
        loss = self._diffusion_loss(batch, generator=generator, apply_dropout=False)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # Image-quality evaluation (see imagegen.evaluate): this step drives the test
        # loop so FidCallback's on_test_* hooks fire (real-face capture -> FID/KID). It
        # also logs a held-out reconstruction loss, scored with the same deterministic
        # per-batch generator as validation so test/loss is comparable across models.
        generator = torch.Generator(device=self.device).manual_seed(self.val_seed + batch_idx)
        loss = self._diffusion_loss(batch, generator=generator, apply_dropout=False)
        self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
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
