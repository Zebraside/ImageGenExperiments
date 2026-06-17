"""LightningDataModule wrapping :class:`ImageFolderDataset`.

There is no validation set (the training loop is driven by the sample callback,
see ``imagegen.callbacks``), so only ``train_dataloader`` is provided.
"""

from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

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
    ) -> None:
        super().__init__()
        self.root = root
        self.caption = caption
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.limit = limit
        self.dataset: ImageFolderDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.dataset is None:
            self.dataset = ImageFolderDataset(
                self.root,
                caption=self.caption,
                image_size=self.image_size,
                limit=self.limit,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            # Keep the final partial batch so small runs (e.g. data.limit < batch_size)
            # still yield a batch instead of an empty loader.
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )
