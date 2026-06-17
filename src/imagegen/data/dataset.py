"""Recursive image-folder dataset for diffusion training.

Walks ``root`` for images, normalizes each to a ``[-1, 1]`` CHW tensor (the VAE
convention the training step relies on), and resolves a per-image caption from a
sidecar ``<image>.txt`` (written by ``scripts/caption_dataset.py`` /
``scripts/normalize_captions.py``), falling back to a shared caption when an
image has none. ``IMAGE_EXTS`` is the single source of truth for which files
count as images -- the caption scripts import it so all consumers agree.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Lowercase suffixes (with the leading dot) treated as images. Imported by the
# caption scripts, which test ``p.suffix.lower() in IMAGE_EXTS``.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class ImageFolderDataset(Dataset):
    """Images found recursively under ``root``, each with a resolved caption.

    Args:
        root: directory searched recursively for images.
        caption: fallback caption used when an image has no (or an empty) sidecar.
        image_size: images are resized to ``(image_size, image_size)``.
        limit: if set, keep only the first ``limit`` images (smoke tests / overfit).
    """

    def __init__(
        self,
        root: str | Path,
        caption: str,
        image_size: int = 512,
        limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.caption = caption
        self.image_size = image_size

        paths = sorted(
            p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not paths:
            raise RuntimeError(
                f"No images ({sorted(IMAGE_EXTS)}) found under {self.root}."
            )
        if limit is not None:
            paths = paths[:limit]
        self.paths = paths

        # ToTensor -> [0, 1] CHW float32; Normalize(0.5, 0.5) -> [-1, 1].
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def _resolve_caption(self, path: Path) -> str:
        """Sidecar ``<image>.txt`` caption if present and non-empty, else the fallback."""
        sidecar = path.with_suffix(".txt")
        if sidecar.exists():
            text = sidecar.read_text(encoding="utf-8").strip()
            if text:
                return text
        return self.caption

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        return {
            "pixel_values": self.transform(image),
            "caption": self._resolve_caption(path),
        }
