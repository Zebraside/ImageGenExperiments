"""Training callbacks."""

from __future__ import annotations

from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import WandbLogger


class SampleImageCallback(L.Callback):
    """Sample images during training; save to disk and log to W&B.

    Two sources are visualized. On a global-step (or epoch) interval the
    **trigger prompt** is sampled from the training loop. When a validation split
    exists, the **held-out validation captions** are also sampled at the end of
    every validation epoch (including the pre-training sanity check), so
    generation quality on unseen prompts is tracked over time. Images are always
    written to ``output_dir/samples`` and, when a WandbLogger is active, also
    logged there.
    """

    def __init__(
        self,
        prompt: str,
        output_dir: str,
        num_samples: int = 4,
        every_n_steps: int = 250,
        every_n_epochs: int = 0,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.prompt = prompt
        self.sample_dir = Path(output_dir) / "samples"
        self.num_samples = num_samples
        self.every_n_steps = every_n_steps
        self.every_n_epochs = every_n_epochs
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self._last_sampled_step = -1
        # Captions captured from the first validation batch (shuffle=False -> the
        # same held-out prompts every run). Empty when there is no validation split.
        self._val_prompts: list[str] = []

    def _save_and_log(self, trainer, tag: str, images, captions, key: str) -> None:
        """Write ``images`` to disk under ``tag`` and log them to W&B if active.

        ``tag`` prefixes the saved filenames (e.g. ``step000010`` or ``val_step000500``);
        ``key`` is the W&B image panel (e.g. ``samples`` or ``val_samples``).
        """
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            img.save(self.sample_dir / f"{tag}_{i}.png")

        if isinstance(trainer.logger, WandbLogger):
            # Use the logger's own log_image (logs at W&B's current step) rather than
            # experiment.log(step=...): an explicit past step is rejected as
            # non-monotonic once training metrics have advanced the step pointer.
            trainer.logger.log_image(
                key=key,
                images=list(images),
                caption=list(captions),
            )

    def _sample_and_log(self, trainer, pl_module, tag: str) -> None:
        """Generate from the trigger prompt, save to disk, and log to W&B if active."""
        images = pl_module.generate(
            prompt=self.prompt,
            num_images=self.num_samples,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            seed=self.seed,
        )
        self._save_and_log(
            trainer, tag, images, [self.prompt] * len(images), key="samples"
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = trainer.global_step
        if self.every_n_steps <= 0 or step == 0 or step % self.every_n_steps != 0:
            return
        # on_train_batch_end fires once per micro-batch, but global_step is constant
        # across an accumulation window -- guard against sampling the same step twice
        # (or N times) under accumulate_grad_batches > 1.
        if step == self._last_sampled_step:
            return
        self._last_sampled_step = step
        self._sample_and_log(trainer, pl_module, tag=f"step{step:06d}")

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.every_n_epochs <= 0:
            return
        # current_epoch is the just-completed 0-indexed epoch, so +1 fires after
        # epochs 10, 20, ... when every_n_epochs == 10.
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs != 0:
            return
        self._sample_and_log(trainer, pl_module, tag=f"epoch{epoch:04d}")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        # Capture the held-out prompts to visualize from the first batch. The val
        # loader is shuffle=False, so this is the same deterministic set every run.
        if batch_idx == 0:
            self._val_prompts = list(batch["caption"][: self.num_samples])

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self._val_prompts:
            return  # no validation split -> nothing to visualize
        # One image per distinct caption (generate batches a single prompt, so loop).
        images, captions = [], []
        for prompt in self._val_prompts:
            generated = pl_module.generate(
                prompt=prompt,
                num_images=1,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                seed=self.seed,
            )
            images.extend(generated)
            captions.extend([prompt] * len(generated))
        self._save_and_log(
            trainer,
            f"val_step{trainer.global_step:06d}",
            images,
            captions,
            key="val_samples",
        )


class PeriodicWeightSave(L.Callback):
    """Periodically persist portable weights during a long run (crash-safety).

    Every ``every_n_steps`` optimizer steps it calls ``pl_module.save_weights`` into
    ``output_dir/checkpoints/step{global_step:06d}`` -- the same portable format as
    the final save (LoRA adapter safetensors / full pipeline), so each checkpoint is
    directly loadable by ``imagegen.evaluate`` / ``scripts.generate_report``. Final
    weights are still written by ``train.py`` after ``fit``.
    """

    def __init__(self, output_dir: str, every_n_steps: int = 2000) -> None:
        super().__init__()
        self.ckpt_dir = Path(output_dir) / "checkpoints"
        self.every_n_steps = int(every_n_steps)
        self._last_saved_step = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = trainer.global_step
        if self.every_n_steps <= 0 or step == 0 or step % self.every_n_steps != 0:
            return
        # on_train_batch_end fires once per micro-batch; global_step is constant across
        # an accumulation window -- guard against saving the same step more than once.
        if step == self._last_saved_step:
            return
        self._last_saved_step = step
        pl_module.save_weights(self.ckpt_dir / f"step{step:06d}")
