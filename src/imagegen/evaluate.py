"""Image-quality evaluation entrypoint.

Runs every trained solution (plus the un-fine-tuned base model) through PyTorch
Lightning's built-in test loop (``Trainer.test``) and reports FID + KID against the
held-out real FFHQ faces. The metric machinery lives in the test loop -- a
``test_step`` on :class:`~imagegen.lit_module.LoRADiffusionModule` and the
``on_test_*`` hooks of :class:`~imagegen.fid_callback.FidCallback` -- so this module
is only a thin orchestrator: for each solution it loads the trained weights into a
``ModelBundle``, calls ``trainer.test``, and tabulates the logged metrics.

Every model is scored on an identical protocol (same held-out prompts/seeds, same
steps/guidance/resolution, same real reference set) so the numbers are comparable.

    uv run python -m imagegen.evaluate                      # all 8 models, 512/512
    uv run python -m imagegen.evaluate --only base,exp2_lora --num-gen 16 --num-real 16
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from imagegen.data import ImageFolderDataModule
from imagegen.fid_callback import FidCallback
from imagegen.lit_module import LoRADiffusionModule
from imagegen.models import load_model

_PRECISION = {"bf16": "bf16-mixed", "fp16": "16-mixed", "fp32": "32-true"}

CONFIGS_DIR = Path("configs")
DEFAULT_OUT = Path("/workspace/outputs/report")

# (key, label, mode, config). mode drives weight loading:
#   base -> raw base model (no fine-tune reference); full -> load the saved
#   pipeline dir; lora -> base + the saved adapter. `config` is the YAML whose
#   model/lora/data settings the run was trained with (must match to reload LoRA).
RUNS = [
    {"key": "base", "label": "Base SD 2.1 (no fine-tune)", "mode": "base", "config": "exp2_lora"},
    {"key": "exp3_full_fid", "label": "Full FT (L2)", "mode": "full", "config": "exp3_full_fid"},
    {"key": "exp4_full_huber", "label": "Full FT (Huber)", "mode": "full", "config": "exp4_full_huber"},
    {"key": "exp2_lora", "label": "LoRA (L2)", "mode": "lora", "config": "exp2_lora"},
    {"key": "exp5_lora_lpips", "label": "LoRA + LPIPS (Alex)", "mode": "lora", "config": "exp5_lora_lpips"},
    {"key": "exp6_lora_arcface", "label": "LoRA + ArcFace", "mode": "lora", "config": "exp6_lora_arcface"},
    {"key": "exp7_lora_lpips_vgg", "label": "LoRA + LPIPS (VGG)", "mode": "lora", "config": "exp7_lora_lpips_vgg"},
    {"key": "exp8_lora_lpips_low", "label": "LoRA + LPIPS (w=0.05)", "mode": "lora", "config": "exp8_lora_lpips_low"},
    {"key": "full_lora_l2", "label": "LoRA-L2 (full run)", "mode": "lora", "config": "full_lora_l2"},
]


def resolve_runs(only: str | None) -> list[dict]:
    """Select runs by comma-separated key, preserving the declared order."""
    if not only:
        return list(RUNS)
    keep = {k.strip() for k in only.split(",") if k.strip()}
    return [r for r in RUNS if r["key"] in keep]


# --- weight loading ------------------------------------------------------


def build_eval_cfg(run: dict, args: argparse.Namespace) -> DictConfig:
    """The run's training config, re-pointed for an evaluation (test) pass."""
    cfg = OmegaConf.load(CONFIGS_DIR / f"{run['config']}.yaml")
    cfg.data.val_size = args.num_real  # held-out split must hold >= the real budget
    cfg.data.batch_size = args.real_batch
    cfg.data.limit = None
    if run["mode"] == "full":
        # The saved output_dir is a full diffusers pipeline (EMA weights merged in);
        # loading every component from it reconstructs the fine-tuned model.
        cfg.model.base_model = cfg.train.output_dir
        cfg.train.mode = "full"
        cfg.train.grad_checkpointing = False
    elif run["mode"] == "lora":
        cfg.train.mode = "lora"  # fresh adapter; trained weights loaded in load_trained_module
    elif run["mode"] == "base":
        cfg.train.mode = "full"  # raw base UNet, no adapter
        cfg.train.grad_checkpointing = False
    else:
        raise ValueError(f"Unknown run mode {run['mode']!r}")
    return cfg


def _load_lora_weights(denoiser, path: Path) -> None:
    """Load saved adapter safetensors into the PEFT-adapted denoiser.

    The inverse of ``LoRADiffusionModule.save_weights`` (which writes diffusers-format,
    ``unet.``-prefixed keys via ``convert_state_dict_to_diffusers``).
    """
    from diffusers.utils import convert_unet_state_dict_to_peft
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    raw = load_file(str(path))
    unet_only = {k[len("unet.") :]: v for k, v in raw.items() if k.startswith("unet.")}
    peft_state = convert_unet_state_dict_to_peft(unet_only)
    result = set_peft_model_state_dict(denoiser, peft_state)
    missing = getattr(result, "unexpected_keys", None)
    if missing:
        print(f"  [warn] {len(missing)} unexpected LoRA keys ignored")


def load_trained_module(cfg: DictConfig, run: dict) -> LoRADiffusionModule:
    bundle = load_model(cfg)
    if run["mode"] == "lora":
        _load_lora_weights(
            bundle.denoiser, Path(cfg.train.output_dir) / "pytorch_lora_weights.safetensors"
        )
    return LoRADiffusionModule(bundle, cfg)


# --- per-run evaluation --------------------------------------------------


def evaluate_run(run: dict, args: argparse.Namespace) -> dict:
    cfg = build_eval_cfg(run, args)
    L.seed_everything(cfg.seed, workers=True)

    datamodule = ImageFolderDataModule(
        root=cfg.data.root,
        caption=cfg.train.trigger_prompt,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.model.image_size,
        limit=None,
        val_size=args.num_real,
        seed=cfg.seed,
    )
    module = load_trained_module(cfg, run)
    fid_cb = FidCallback(
        trigger_prompt=cfg.train.trigger_prompt,
        num_samples=args.num_gen,
        real_images=args.num_real,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
    )
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        precision=_PRECISION[cfg.train.mixed_precision],
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        callbacks=[fid_cb],
    )
    trainer.test(module, datamodule)

    cm = trainer.callback_metrics

    def _get(name):
        v = cm.get(name)
        return float(v) if v is not None else None

    metrics = {
        "label": run["label"],
        "fid": _get("test/fid"),
        "kid": _get("test/kid"),
        "kid_std": _get("test/kid_std"),
        "loss": _get("test/loss"),
    }

    # Free the model before loading the next one (full models are ~5 GB each).
    del trainer, module, datamodule, fid_cb
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


# --- reporting (pure) ----------------------------------------------------


def render_markdown(results: dict, base_key: str = "base") -> str:
    """Ranked FID/KID table (best FID first), with ΔFID vs the base model."""
    base_fid = results.get(base_key, {}).get("fid")
    ranked = sorted(
        results.items(),
        key=lambda kv: (kv[1].get("fid") is None, kv[1].get("fid") or 0.0),
    )
    lines = [
        "# Image-quality evaluation (FID / KID)\n",
        "Each model generates a fixed budget of images from held-out FFHQ captions "
        "(identical prompts/seeds/steps/guidance) and is scored against the held-out "
        "real faces via `torchmetrics` FID + KID. **Lower is better** for both. ΔFID "
        "is vs the un-fine-tuned base model (negative = improvement).\n",
        "| rank | model | FID ↓ | KID ↓ | ΔFID vs base | test/loss |",
        "|------|-------|-------|-------|--------------|-----------|",
    ]
    for i, (key, m) in enumerate(ranked, start=1):
        fid = m.get("fid")
        kid = m.get("kid")
        kid_std = m.get("kid_std")
        loss = m.get("loss")
        fid_s = f"{fid:.2f}" if fid is not None else "—"
        if kid is not None:
            kid_s = f"{kid:.4f}" + (f" ± {kid_std:.4f}" if kid_std is not None else "")
        else:
            kid_s = "—"
        if fid is not None and base_fid is not None:
            d = fid - base_fid
            delta_s = "0.00 (base)" if key == base_key else f"{d:+.2f}"
        else:
            delta_s = "—"
        loss_s = f"{loss:.4f}" if loss is not None else "—"
        lines.append(f"| {i} | {m.get('label', key)} (`{key}`) | {fid_s} | {kid_s} | {delta_s} | {loss_s} |")
    lines.append("")
    return "\n".join(lines)


def plot_metrics(results: dict, path: Path, base_key: str = "base") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = [(k, v) for k, v in results.items() if v.get("fid") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in ((axes[0], "fid", "FID ↓"), (axes[1], "kid", "KID ↓")):
        ranked = sorted(items, key=lambda kv: kv[1].get(metric) or 0.0)
        labels = [v.get("label", k) for k, v in ranked]
        vals = [v.get(metric) or 0.0 for _, v in ranked]
        colors = ["#d62728" if k == base_key else "#1f77b4" for k, _ in ranked]
        ax.barh(range(len(vals)), vals, color=colors)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title)
        for y, val in enumerate(vals):
            ax.text(val, y, f" {val:.3g}", va="center", fontsize=7)
    fig.suptitle("Image quality vs held-out real FFHQ faces (base highlighted)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def write_outputs(out: Path, results: dict, args: argparse.Namespace) -> None:
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "num_generated": args.num_gen,
            "num_real": args.num_real,
            "steps": args.steps,
            "guidance": args.guidance,
            "seed": args.seed,
            "metric": "torchmetrics FID + KID (InceptionV3-2048, normalize=True)",
        },
        "results": results,
    }
    (out / "quality_metrics.json").write_text(json.dumps(payload, indent=2))
    (out / "quality_metrics.md").write_text(render_markdown(results))
    try:
        plot_metrics(results, out / "quality_metrics.png")
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"[plot] skipped: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description="FID/KID image-quality eval via Trainer.test")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--only", default=None, help="comma-separated run keys")
    ap.add_argument("--num-gen", type=int, default=512, help="generated images per model")
    ap.add_argument("--num-real", type=int, default=512, help="held-out real reference images")
    ap.add_argument("--real-batch", type=int, default=8, help="batch size for the real loader")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    out = Path(args.out)

    results: dict = {}
    for run in resolve_runs(args.only):
        print(f"\n=== evaluate {run['key']} ({run['mode']}) ===")
        try:
            results[run["key"]] = evaluate_run(run, args)
        except Exception as exc:
            print(f"[fail] {run['key']}: {exc}")
            results[run["key"]] = {"label": run["label"], "fid": None, "error": str(exc)}
        write_outputs(out, results, args)  # incremental: keep partial results on disk

    print(f"\n[done] wrote quality_metrics.{{json,md,png}} to {out}")
    print(render_markdown(results))


if __name__ == "__main__":
    main()
