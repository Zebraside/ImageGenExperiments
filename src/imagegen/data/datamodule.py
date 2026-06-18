"""LightningDataModule wrapping :class:`ImageFolderDataset`.

By default only ``train_dataloader`` is provided. When ``val_size`` is set, a
deterministic held-out validation split is carved off (see ``setup``) and a
``val_dataloader`` is exposed so a comparable ``val/loss`` can drive checkpoint
selection -- see ``imagegen.lit_module.LoRADiffusionModule.validation_step``.
"""

from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from imagegen.data.dataset import ImageFolderDataset


class ImageFolderDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str | Path,
        caption: str,
        batch_size: int,
        num_workers: int,
        image_size: int,
        limit: int | None = None,
        val_size: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.root = root
        self.caption = caption
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.limit = limit
        self.val_size = val_size
        self.seed = seed
        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_ds is not None:
            return
        full = ImageFolderDataset(
            self.root,
            caption=self.caption,
            image_size=self.image_size,
            limit=self.limit,
        )
        # Carve off a deterministic validation split when requested. Capped at 1/5 of
        # the data so training always keeps the lion's share; if that rounds to 0
        # (tiny smoke datasets) validation stays disabled.
        n_val = min(self.val_size, len(full) // 5) if self.val_size else 0
        if n_val > 0:
            gen = torch.Generator().manual_seed(self.seed)
            self.train_ds, self.val_ds = random_split(full, [len(full) - n_val, n_val], generator=gen)
        else:
            self.train_ds, self.val_ds = full, None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            # Keep the final partial batch so small runs (e.g. data.limit < batch_size)
            # still yield a batch instead of an empty loader.
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader | None:
        # No validation split -> no loader (the trainer is also told via
        # limit_val_batches=0, so this is never called in that case).
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )
