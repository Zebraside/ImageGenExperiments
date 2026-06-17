# imagegen

Image generation deep learning model training pipeline.

Python dependencies are managed with [uv](https://docs.astral.sh/uv/). Torch is installed
from the CUDA 12.4 wheel index (`pytorch-cu124`) for GPU training.

## Setup

```bash
uv sync
```

This creates a `.venv` and installs all dependencies pinned in `uv.lock`.

## Verify the install

```bash
uv run scripts/check_torch.py
```

Expected output: the torch version, `cuda available: True`, the GPU name, and a `PASS`
line confirming a GPU matmul succeeded.

## Prepare the dataset

```bash
uv run scripts/prepare_dataset.py    # download FFHQ-512 + extract PNGs & caption sidecars -> /workspace/data/ffhq512/
uv run scripts/caption_dataset.py    # (optional) re-caption every image with BLIP -> sidecar <image>.txt
uv run scripts/normalize_captions.py # (optional) rewrite captions into one clean form (in place)
```

`prepare_dataset.py` pulls native 512x512 FFHQ from `Ryan-sjtu/ffhq512-caption` and writes
each row's caption to a sidecar `.txt`, so the dataset arrives captioned. The two caption
scripts are only needed if you'd rather (re-)generate captions yourself: `caption_dataset.py`
runs BLIP, and `normalize_captions.py` uses a small local LLM (`Qwen/Qwen2.5-1.5B-Instruct`)
to standardize them. Training reads each image's caption from its sidecar `.txt`; images
without one fall back to `train.trigger_prompt` in the config. All scripts are idempotent —
re-runs skip work that's already done.

## Project layout

```
src/imagegen/        # package code
configs/             # training configs (OmegaConf YAML)
scripts/             # standalone scripts (check_torch, prepare_dataset, caption_dataset, normalize_captions)
```

## Common commands

```bash
uv run scripts/check_torch.py   # verify torch + GPU
uv run ruff check .             # lint
uv run pytest                   # tests
uv add <package>                # add a dependency
```
