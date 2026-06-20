"""Training entrypoint.

    uv run python -m imagegen.train --config configs/lora_sd21.yaml [dotlist overrides...]

Example overrides (OmegaConf dotlist):
    train.max_steps=2  data.limit=8  logging.wandb_project=null
"""

from __future__ import annotations

import argparse
import warnings

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from imagegen.callbacks import PeriodicWeightSave, SampleImageCallback
from imagegen.data import ImageFolderDataModule
from imagegen.fid_callback import FidCallback
from imagegen.lit_module import LoRADiffusionModule
from imagegen.models import load_model

_PRECISION = {"bf16": "bf16-mixed", "fp16": "16-mixed", "fp32": "32-true"}


def parse_config() -> DictConfig:
    parser = argparse.ArgumentParser(description="Train a diffusion LoRA / full fine-tune.")
    parser.add_argument("--config", required=True, help="Path to an OmegaConf YAML config.")
    parser.add_argument("overrides", nargs="*", help="Dotlist overrides, e.g. train.lr=5e-5")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    return cfg


def build_logger(cfg: DictConfig):
    if cfg.logging.wandb_project:
        # name= gives each experiment a meaningful W&B run name (what is being
        # tested) instead of W&B's random auto-name; None keeps the auto-name.
        return WandbLogger(
            project=cfg.logging.wandb_project,
            name=cfg.logging.get("run_name"),
            save_dir=cfg.train.output_dir,
        )
    return CSVLogger(save_dir=cfg.train.output_dir, name="logs")


def main() -> None:
    # VAE and text encoder are intentionally frozen in eval(); suppress Lightning's noise about it.
    warnings.filterwarnings("ignore", message="Found .* module\\(s\\) in eval mode")

    torch.set_float32_matmul_precision("high")

    cfg = parse_config()
    L.seed_everything(cfg.seed, workers=True)

    val_size = cfg.data.get("val_size")
    val_enabled = bool(val_size)

    datamodule = ImageFolderDataModule(
        root=cfg.data.root,
        caption=cfg.train.trigger_prompt,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.model.image_size,
        limit=cfg.data.get("limit"),
        val_size=val_size,
        seed=cfg.seed,
    )

    bundle = load_model(cfg)
    print(
        f"[{cfg.model.key}] mode={cfg.train.mode} "
        f"trainable={bundle.num_trainable():,} / total={bundle.num_total():,} "
        f"({100 * bundle.num_trainable() / max(bundle.num_total(), 1):.2f}%)"
    )
    module = LoRADiffusionModule(bundle, cfg)

    # Mid-training .ckpt snapshots are disabled; only the final portable weights
    # (save_weights below) are persisted.
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        SampleImageCallback(
            prompt=cfg.train.trigger_prompt,
            output_dir=cfg.train.output_dir,
            num_samples=cfg.train.num_samples,
            every_n_steps=cfg.train.sample_every_n_steps,
            every_n_epochs=cfg.train.get("sample_every_n_epochs", 0),
            num_inference_steps=cfg.train.sample_steps,
            guidance_scale=cfg.train.guidance_scale,
        ),
    ]

    # Periodic portable-weight checkpoints for long runs (crash-safety). Off unless
    # train.save_every_n_steps is set, so existing configs are unaffected.
    if cfg.train.get("save_every_n_steps"):
        callbacks.append(
            PeriodicWeightSave(
                output_dir=cfg.train.output_dir,
                every_n_steps=cfg.train.save_every_n_steps,
            )
        )

    # FID-as-eval-metric experiment: only when explicitly requested and a held-out
    # split exists to draw real reference images from.
    if cfg.train.get("compute_fid", False) and val_enabled:
        callbacks.append(
            FidCallback(
                trigger_prompt=cfg.train.trigger_prompt,
                num_samples=cfg.train.get("fid_num_samples", 64),
                real_images=cfg.train.get("fid_real_images", 256),
                num_inference_steps=cfg.train.get("fid_steps", 25),
                guidance_scale=cfg.train.guidance_scale,
                seed=cfg.train.get("val_seed", 0),
            )
        )

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        precision=_PRECISION[cfg.train.mixed_precision],
        max_steps=cfg.train.get("max_steps") or -1,
        max_epochs=cfg.train.get("max_epochs"),
        # Wall-clock budget ("DD:HH:MM:SS"); training stops at whichever of
        # max_steps / max_time is reached first. None disables it.
        max_time=cfg.train.get("max_time"),
        overfit_batches=cfg.train.get("overfit_batches", 0),
        accumulate_grad_batches=cfg.train.get("accumulate_grad_batches", 1),
        gradient_clip_val=cfg.train.gradient_clip_val,
        enable_checkpointing=False,
        log_every_n_steps=cfg.logging.log_every,
        # Run validation on a step cadence (one "epoch" is the whole dataset here),
        # else disable it entirely so the val_dataloader is never requested.
        val_check_interval=cfg.train.get("val_check_interval") if val_enabled else None,
        limit_val_batches=1.0 if val_enabled else 0,
        # Run a validation pass before training (Lightning's sanity check) so the
        # callback logs baseline validation images at step 0. self.log metrics are
        # discarded during sanity, but the image hook still fires.
        num_sanity_val_steps=cfg.train.get("num_sanity_val_steps", 2),
        callbacks=callbacks,
        logger=build_logger(cfg),
    )

    trainer.fit(module, datamodule)

    out = module.save_weights(cfg.train.output_dir)
    print(f"Saved {cfg.train.mode} weights to {out}")


if __name__ == "__main__":
    main()
