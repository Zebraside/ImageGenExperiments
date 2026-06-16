import os

import pytest
from omegaconf import OmegaConf

from imagegen.models.factory import MODEL_REGISTRY, load_model


def _cfg(**train):
    base = {
        "model": {"key": "sd21", "base_model": "Manojb/stable-diffusion-2-1-base", "image_size": 256},
        "lora": {"rank": 8, "alpha": 8, "dropout": 0.0, "target_modules": ["to_q", "to_k", "to_v", "to_out.0"]},
        "train": {"mode": "lora", "grad_checkpointing": False, "lr": 1e-4},
    }
    base["train"].update(train)
    return OmegaConf.create(base)


# --- offline tests ---------------------------------------------------------

def test_unknown_key_raises():
    cfg = _cfg()
    cfg.model.key = "does-not-exist"
    with pytest.raises(KeyError):
        load_model(cfg)


def test_pixart_not_implemented():
    cfg = _cfg()
    cfg.model.key = "pixart256"
    with pytest.raises(NotImplementedError):
        load_model(cfg)


def test_registry_keys():
    assert {"sd21", "sd15", "pixart256"} <= set(MODEL_REGISTRY)


# --- networked tests (download the base model; opt in) ---------------------

requires_download = pytest.mark.skipif(
    not os.environ.get("RUN_DOWNLOAD_TESTS"),
    reason="set RUN_DOWNLOAD_TESTS=1 to download the base model and run.",
)


@requires_download
def test_lora_freezes_base():
    bundle = load_model(_cfg(mode="lora"))
    # Only LoRA adapters trainable -> a tiny fraction of total params.
    assert 0 < bundle.num_trainable() < bundle.num_total() * 0.05


@requires_download
def test_full_mode_trains_everything():
    bundle = load_model(_cfg(mode="full"))
    assert bundle.num_trainable() == bundle.num_total()
