"""Training callbacks."""

from __future__ import annotations

from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import WandbLogger


class SampleImageCallback(L.Callback):
    """Periodically sample from the trigger prompt; save to disk and log to W&B.

    Hooked to the training loop (we have no validation set) on a global-step
    interval. Images are always written to ``output_dir/samples`` and, when a
    WandbLogger is active, also logged there.
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

    def _sample_and_log(self, trainer, pl_module, tag: str, log_step: int) -> None:
        """Generate from the trigger prompt, save to disk, and log to W&B if active.

        ``tag`` prefixes the saved filenames (e.g. ``step000010`` or ``epoch0010``);
        ``log_step`` is the (monotonic) step passed to W&B.
        """
        images = pl_module.generate(
            prompt=self.prompt,
            num_images=self.num_samples,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            seed=self.seed,
        )

        self.sample_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            img.save(self.sample_dir / f"{tag}_{i}.png")

        if isinstance(trainer.logger, WandbLogger):
            import wandb

            trainer.logger.experiment.log(
                {"samples": [wandb.Image(img, caption=self.prompt) for img in images]},
                step=log_step,
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = trainer.global_step
        if self.every_n_steps <= 0 or step == 0 or step % self.every_n_steps != 0:
            return
        self._sample_and_log(trainer, pl_module, tag=f"step{step:06d}", log_step=step)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.every_n_epochs <= 0:
            return
        # current_epoch is the just-completed 0-indexed epoch, so +1 fires after
        # epochs 10, 20, ... when every_n_epochs == 10.
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs != 0:
            return
        self._sample_and_log(
            trainer, pl_module, tag=f"epoch{epoch:04d}", log_step=trainer.global_step
        )
