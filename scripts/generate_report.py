"""Visual report for the experiment sweep.

Produces, under ``--out`` (default /workspace/outputs/report):
  - initial_capabilities.png  -- base SD 2.1 on a fixed prompt/seed grid (the
    model's starting point, before any fine-tuning).
  - <run>_progression.png     -- per-run strip of the periodic training samples
    (fixed seed/prompt), left = early step, right = final: the quality
    progression across checkpoints.
  - comparison_final.png      -- base vs every run's final weights on the same
    fixed prompts/seeds.
  - report.md                 -- embeds the above + a best-effort metrics table.

All generation uses one fixed prompt set + seeds so every panel is comparable.

    uv run python scripts/generate_report.py
    uv run python scripts/generate_report.py --only base,exp1_full_baseline
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless box: no display
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from diffusers import StableDiffusionPipeline
from PIL import Image

BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
OUTPUTS_ROOT = Path("/workspace/outputs")

# (key, label, output_dir, mode). "base" is the untrained reference.
# exp1 (plain full-FT baseline) is skipped: exp3 is the full-FT L2 reference (+FID).
RUNS = [
    ("base", "Base SD 2.1 (no fine-tune)", None, "base"),
    ("exp3_full_fid", "Full FT (L2)", "exp3_full_fid", "full"),
    ("exp4_full_huber", "Full FT (Huber)", "exp4_full_huber", "full"),
    ("exp2_lora", "LoRA (L2)", "exp2_lora", "lora"),
    ("exp5_lora_lpips", "LoRA + LPIPS (Alex)", "exp5_lora_lpips", "lora"),
    ("exp6_lora_arcface", "LoRA + ArcFace", "exp6_lora_arcface", "lora"),
    ("exp7_lora_lpips_vgg", "LoRA + LPIPS (VGG)", "exp7_lora_lpips_vgg", "lora"),
    ("exp8_lora_lpips_low", "LoRA + LPIPS (w=0.05)", "exp8_lora_lpips_low", "lora"),
]

# Fixed evaluation grid (prompts x seeds) shared by every panel.
PROMPTS = [
    "a high-quality photo of a person's face",
    "a portrait photo of a smiling young woman",
    "a portrait photo of an elderly man with a beard",
    "a studio headshot of a person with curly hair",
]
SEEDS = [0, 1, 2, 3]
STEPS = 30
GUIDANCE = 5.0
DTYPE = torch.bfloat16  # SD 2.1 VAE is unstable in fp16; bf16 is safe + fast


def load_pipeline(mode: str, output_dir: str | None) -> StableDiffusionPipeline:
    if mode == "base":
        pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL, safety_checker=None, requires_safety_checker=False, torch_dtype=DTYPE
        )
    elif mode == "full":
        pipe = StableDiffusionPipeline.from_pretrained(
            str(OUTPUTS_ROOT / output_dir),
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=DTYPE,
        )
    elif mode == "lora":
        pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL, safety_checker=None, requires_safety_checker=False, torch_dtype=DTYPE
        )
        pipe.load_lora_weights(str(OUTPUTS_ROOT / output_dir))
    else:
        raise ValueError(mode)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda" if torch.cuda.is_available() else "cpu")


def generate_grid(pipe: StableDiffusionPipeline) -> list[Image.Image]:
    device = pipe.device
    images = []
    for prompt, seed in zip(PROMPTS, SEEDS):
        gen = torch.Generator(device=device).manual_seed(seed)
        out = pipe(
            prompt,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            height=512,
            width=512,
            generator=gen,
        )
        images.append(out.images[0])
    return images


def free(pipe) -> None:
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_row(images, labels, title, path) -> None:
    """One row of images with per-column labels."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img)
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_matrix(rows, row_labels, col_labels, title, path) -> None:
    """Grid: one row per model, one column per prompt/seed."""
    nrows, ncols = len(rows), len(col_labels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows + 0.5))
    axes = axes.reshape(nrows, ncols)
    for r, (images, rlabel) in enumerate(zip(rows, row_labels)):
        for c in range(ncols):
            ax = axes[r][c]
            ax.imshow(images[c])
            ax.axis("off")
            if r == 0:
                ax.set_title(col_labels[c], fontsize=8)
            if c == 0:
                ax.text(
                    -0.05, 0.5, rlabel, fontsize=9, ha="right", va="center",
                    rotation=90, transform=ax.transAxes,
                )
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def progression_strip(run_key: str, output_dir: str, out: Path) -> Path | None:
    """Tile the periodic trigger-prompt samples (stepNNNNNN_0.png) into one strip."""
    sample_dir = OUTPUTS_ROOT / output_dir / "samples"
    pngs = sorted(sample_dir.glob("step*_0.png"))
    if not pngs:
        return None
    imgs = [Image.open(p) for p in pngs]
    labels = [p.stem.replace("_0", "").replace("step", "step ") for p in pngs]
    path = out / f"{run_key}_progression.png"
    save_row(imgs, labels, f"{run_key}: training progression (fixed seed/prompt)", path)
    return path


def fetch_metrics(run_names: list[str]) -> dict:
    """Best-effort final val/loss + val/fid per W&B run name (skipped on any error)."""
    try:
        import wandb

        api = wandb.Api()
        entity = api.default_entity
        runs = api.runs(f"{entity}/imagegen")
        out = {}
        for r in runs:
            if r.name in run_names:
                out[r.name] = {
                    k: r.summary.get(k)
                    for k in ("val/loss", "val/fid", "train/loss")
                    if r.summary.get(k) is not None
                }
        return out
    except Exception as exc:  # pragma: no cover - network/auth dependent
        print(f"[metrics] skipped W&B fetch: {exc}")
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTPUTS_ROOT / "report"))
    ap.add_argument("--only", default=None, help="comma-separated run keys to include")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    selected = RUNS
    if args.only:
        keep = set(args.only.split(","))
        selected = [r for r in RUNS if r[0] in keep]

    col_labels = [f"seed {s}\n{p[:28]}…" for p, s in zip(PROMPTS, SEEDS)]
    final_rows, row_labels, present = [], [], []

    for key, label, output_dir, mode in selected:
        if mode != "base" and not (OUTPUTS_ROOT / output_dir).exists():
            print(f"[skip] {key}: {OUTPUTS_ROOT / output_dir} missing (run not finished?)")
            continue
        print(f"[generate] {key} ({mode})")
        pipe = load_pipeline(mode, output_dir)
        images = generate_grid(pipe)
        free(pipe)

        if key == "base":
            save_row(images, col_labels, "Initial capabilities: base SD 2.1",
                     out / "initial_capabilities.png")
        else:
            progression_strip(key, output_dir, out)

        final_rows.append(images)
        row_labels.append(label)
        present.append((key, label, output_dir, mode))

    if final_rows:
        save_matrix(final_rows, row_labels, col_labels,
                    "Final weights: base vs all experiments", out / "comparison_final.png")

    metrics = fetch_metrics(
        ["lora-r32-l2-2k", "full-ft-fid-2k", "full-ft-huber-2k",
         "lora-lpips-aux-2k", "lora-arcface-aux-2k",
         "lora-lpips-vgg-2k", "lora-lpips-alex-w05-2k"]
    )
    write_markdown(out, present, metrics)
    print(f"[done] report written to {out}")


def write_markdown(out: Path, present, metrics: dict) -> None:
    lines = [
        "# Experiment visual report\n",
        "Base Stable Diffusion 2.1 vs the experiment sweep. All runs use "
        "**effective batch 8**, **2000 optimizer steps**, and the same "
        "prompts/seeds. The plain full-FT baseline (exp1) is omitted because exp3 "
        "trains identically (full-FT, L2) and adds the FID metric.\n",
        "## Initial capabilities (base model)\n",
        "![initial](initial_capabilities.png)\n",
        "## Final comparison (all runs)\n",
        "![comparison](comparison_final.png)\n",
        "## Training progression per run\n",
    ]
    for key, label, output_dir, mode in present:
        if mode == "base":
            continue
        strip = out / f"{key}_progression.png"
        if strip.exists():
            lines.append(f"### {label} (`{key}`)\n")
            lines.append(f"![{key}]({key}_progression.png)\n")
    if metrics:
        lines.append("## Final metrics (from W&B)\n")
        lines.append("| run | val/loss | val/fid |")
        lines.append("|-----|----------|---------|")
        for name, vals in metrics.items():
            vl = vals.get("val/loss")
            vf = vals.get("val/fid")
            vl = f"{vl:.4f}" if isinstance(vl, (int, float)) else "—"
            vf = f"{vf:.2f}" if isinstance(vf, (int, float)) else "—"
            lines.append(f"| {name} | {vl} | {vf} |")
        lines.append("")
        lines.append(
            "_Note: `full-ft-huber` reports the Huber objective, so its `val/loss` "
            "magnitude is not directly comparable to the L2 runs. Lower FID is better._\n"
        )
    (out / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
