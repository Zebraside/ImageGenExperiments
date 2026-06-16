# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git / Commit conventions

- Do **not** add a `Co-Authored-By:` trailer or otherwise credit Claude/the assistant as a co-author.
- Write each commit message as a short bullet list summarizing what was done.
- When starting new work, create a new branch and work in it (don't commit directly to `main`).
- When merging a branch into `main`, squash its commits before merging.

## Commands

Dependencies are managed with **uv**; everything runs through `uv run`.

```bash
uv sync                              # create .venv, install pinned deps (torch from CUDA 12.4 index)
uv run scripts/check_torch.py        # verify torch + GPU (prints PASS on a successful GPU matmul)
uv run ruff check .                  # lint
uv run pytest                        # tests
uv run pytest tests/test_dataset.py::test_limit_caps_dataset   # single test
uv add <package>                     # add a dependency
```

Training:

```bash
uv run python -m imagegen.train --config configs/lora_sd21.yaml      # LoRA fine-tune SD 2.1 @ 256
uv run python -m imagegen.train --config configs/full_sd21.yaml      # full UNet fine-tune
uv run imagegen-overfit                                              # single-batch sanity check (configs/overfit.yaml)
uv run python -m imagegen.train --config configs/lora_sd21.yaml train.max_steps=2 data.limit=8 logging.wandb_project=null  # quick smoke run
```

Any config leaf is overridable via OmegaConf dotlist args appended to the command (`train.lr=5e-5`, `data.batch_size=4`).

### Tests that need the network
`tests/test_factory.py` has tests that download the base model; they are skipped unless `RUN_DOWNLOAD_TESTS=1` is set. Everything else (captions, dataset, registry wiring) runs fully offline against synthetic data.

## Dataset preparation

Three idempotent scripts build `/workspace/data/ffhq256/` (re-runs skip completed work):

```bash
uv run scripts/prepare_dataset.py     # download FFHQ-256 parquet, extract PNGs -> /workspace/data/ffhq256/
uv run scripts/caption_dataset.py     # BLIP-caption each image -> sidecar <image>.txt
uv run scripts/normalize_captions.py  # rewrite captions in place via Qwen2.5-1.5B-Instruct
```

Training reads each image's caption from its sidecar `.txt`; images without one fall back to `train.trigger_prompt`.

## Architecture

The training stack is **PyTorch Lightning + diffusers + PEFT**, driven entirely by an OmegaConf YAML config. The flow in `imagegen.train.main`:

1. **`models/factory.load_model(cfg)`** — a registry keyed by `cfg.model.key` (`sd21`, `sd15` share `_load_sd`; `pixart256` is a stubbed seam). Loads the SD components, **always freezes the VAE and text encoder**, and depending on `cfg.train.mode`:
   - `lora` — freezes the UNet and attaches LoRA adapters (`cfg.lora.*`); only adapters stay trainable.
   - `full` — whole UNet trainable, optional gradient checkpointing.
   Returns a **`ModelBundle`** (`models/bundle.py`) — a dataclass that hides the base-model specifics behind `vae` / `text_encoder` / `denoiser` / `noise_scheduler` and exposes `trainable_parameters()`, param counts, and `build_pipeline()` (a fresh fast sampler for generation, reusing the in-memory weights). This boundary is what lets the LightningModule stay model-agnostic for future backbones like PixArt.
2. **`lit_module.LoRADiffusionModule`** — the training objective: VAE-encode image → latents, add noise at a random timestep, predict noise (`epsilon`) or velocity (`v_prediction`, switched on `bundle.prediction_type`), MSE loss. Only `bundle.trainable_parameters()` get optimizer state. `save_weights()` writes portable LoRA adapter safetensors in `lora` mode, or a full pipeline in `full` mode.
3. **`data/datamodule.ImageFolderDataModule`** wraps **`data/dataset.ImageFolderDataset`** — recursively loads images under `data.root`, normalizes to `[-1, 1]` (VAE convention), and resolves the per-image sidecar caption with fallback to the trigger prompt.
4. **`callbacks.SampleImageCallback`** — periodically samples from the trigger prompt (step- or epoch-based), saves to `output_dir/samples`, and logs to W&B when active. There is no validation set, so it hooks the training loop.

Logging: a `WandbLogger` when `logging.wandb_project` is set, otherwise a `CSVLogger`.

`captions.py` holds the **deterministic, model-free** caption logic (prompt construction + `clean_caption` cleanup) split out from `scripts/normalize_captions.py` so it is unit-testable without downloading the LLM.

## Notes

- `configs/default.yaml` is a stale example with an older schema (`model.name: unet`, `base_channels`) and is **not** used by the training pipeline. The live configs are `lora_sd21.yaml`, `full_sd21.yaml`, and `overfit.yaml`.
- The base model is `Manojb/stable-diffusion-2-1-base`, a public mirror of the now-private `stabilityai/stable-diffusion-2-1-base`.
- `data/` and `outputs/` are gitignored.
