import pytest
import torch
from PIL import Image

from imagegen.data.dataset import ImageFolderDataset


def _make_images(root, n=5, size=256):
    for i in range(n):
        Image.new("RGB", (size, size), color=(i * 10, 0, 0)).save(root / f"img_{i}.png")
    # also exercise nested dirs + a non-square source
    sub = root / "nested"
    sub.mkdir()
    Image.new("RGB", (128, 200), color=(0, 255, 0)).save(sub / "odd.jpg")


def test_returns_normalized_tensor_and_caption(tmp_path):
    _make_images(tmp_path, n=3)
    ds = ImageFolderDataset(tmp_path, caption="a face", image_size=256)

    assert len(ds) == 4  # 3 png + 1 nested jpg
    sample = ds[0]
    assert sample["caption"] == "a face"

    px = sample["pixel_values"]
    assert px.shape == (3, 256, 256)
    assert px.dtype == torch.float32
    assert px.min() >= -1.0 and px.max() <= 1.0


def test_sidecar_caption_overrides_fallback(tmp_path):
    _make_images(tmp_path, n=2)
    # Sidecar for the first image; the second has none -> fallback caption.
    (tmp_path / "img_0.txt").write_text("  a smiling person  \n", encoding="utf-8")
    ds = ImageFolderDataset(tmp_path, caption="fallback")

    by_name = {p.stem: i for i, p in enumerate(ds.paths)}
    assert ds[by_name["img_0"]]["caption"] == "a smiling person"  # stripped, from file
    assert ds[by_name["img_1"]]["caption"] == "fallback"          # no sidecar


def test_empty_sidecar_falls_back(tmp_path):
    _make_images(tmp_path, n=1)
    (tmp_path / "img_0.txt").write_text("   \n", encoding="utf-8")
    ds = ImageFolderDataset(tmp_path, caption="fallback")
    assert ds[0]["caption"] == "fallback"


def test_limit_caps_dataset(tmp_path):
    _make_images(tmp_path, n=5)
    ds = ImageFolderDataset(tmp_path, caption="x", limit=2)
    assert len(ds) == 2


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ImageFolderDataset(tmp_path / "nope", caption="x")


def test_empty_root_raises(tmp_path):
    with pytest.raises(RuntimeError):
        ImageFolderDataset(tmp_path, caption="x")
