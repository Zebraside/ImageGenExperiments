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
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from imagegen.callbacks import SampleImageCallback
from imagegen.data import ImageFolderDataModule
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
        return WandbLogger(project=cfg.logging.wandb_project, save_dir=cfg.train.output_dir)
    return CSVLogger(save_dir=cfg.train.output_dir, name="logs")


def main() -> None:
    # VAE and text encoder are intentionally frozen in eval(); suppress Lightning's noise about it.
    warnings.filterwarnings("ignore", message="Found .* module\\(s\\) in eval mode")

    torch.set_float32_matmul_precision("high")

    cfg = parse_config()
    L.seed_everything(cfg.seed, workers=True)

    datamodule = ImageFolderDataModule(
        root=cfg.data.root,
        caption=cfg.train.trigger_prompt,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.model.image_size,
        limit=cfg.data.get("limit"),
    )

    bundle = load_model(cfg)
    print(
        f"[{cfg.model.key}] mode={cfg.train.mode} "
        f"trainable={bundle.num_trainable():,} / total={bundle.num_total():,} "
        f"({100 * bundle.num_trainable() / max(bundle.num_total(), 1):.2f}%)"
    )
    module = LoRADiffusionModule(bundle, cfg)

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.train.output_dir,
            every_n_train_steps=cfg.train.ckpt_every_n_steps,
            save_top_k=-1,
        ),
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

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        precision=_PRECISION[cfg.train.mixed_precision],
        max_steps=cfg.train.get("max_steps") or -1,
        max_epochs=cfg.train.get("max_epochs"),
        overfit_batches=cfg.train.get("overfit_batches", 0),
        gradient_clip_val=cfg.train.gradient_clip_val,
        log_every_n_steps=cfg.logging.log_every,
        callbacks=callbacks,
        logger=build_logger(cfg),
    )

    trainer.fit(module, datamodule)

    out = module.save_weights(cfg.train.output_dir)
    print(f"Saved {cfg.train.mode} weights to {out}")


if __name__ == "__main__":
    main()
