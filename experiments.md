# Experiments

Runs on FFHQ-512, **effective batch 8** across all (full-FT uses
accumulate_grad_batches=2; LoRA uses batch 8 / accumulate 1; aux-LoRA uses
batch 4 / accumulate 2), same trigger prompt / seed / sampling params. Tracked
in W&B project `imagegen` under the listed run name. Every run trains for a fixed
**2000 optimizer steps** (equal-steps comparison across methods). Run names carry a
`-2k` suffix to distinguish them from the earlier 800-step history.

| # | Experiment | Config | W&B run name | Status |
|---|------------|--------|--------------|--------|
| 1 | Baseline. Full finetune (L2) | `configs/exp1_full_baseline.yaml` | — | [~] skipped (redundant with #3) |
| 2 | Switch to LoRA (L2) | `configs/exp2_lora.yaml` | `lora-r32-l2-2k` | [x] FID 54.45 (best) |
| 3 | Full FT + FID metric (full-FT L2 reference) | `configs/exp3_full_fid.yaml` | `full-ft-fid-2k` | [x] FID 58.06 |
| 4 | Huber instead of L2 (full FT) | `configs/exp4_full_huber.yaml` | `full-ft-huber-2k` | [x] FID 56.65 |
| 5 | LPIPS auxiliary loss, Alex w=0.1 (LoRA) | `configs/exp5_lora_lpips.yaml` | `lora-lpips-aux-2k` | [x] FID 91.72 (worst trained) |
| 6 | ArcFace identity loss, w=0.2 (LoRA) | `configs/exp6_lora_arcface.yaml` | `lora-arcface-aux-2k` | [x] FID 63.19 |
| 7 | LPIPS aux, VGG backbone w=0.1 (LoRA) | `configs/exp7_lora_lpips_vgg.yaml` | `lora-lpips-vgg-2k` | [x] FID 84.86 |
| 8 | LPIPS aux, Alex w=0.05 (LoRA) | `configs/exp8_lora_lpips_low.yaml` | `lora-lpips-alex-w05-2k` | [x] FID 82.65 |

Seven runs at 2000 steps. Carried over from the 800-step sweep: full-FT (L2 &
Huber) and LoRA-L2 give clean adaptation; FID improves then plateaus (~162);
**LPIPS-Alex aux over-textures even at w=0.1**, while ArcFace at 0.2 stays clean.
This round extends every run to 2000 steps and adds two LPIPS probes — #7 swaps the
backbone to VGG and #8 halves the weight to 0.05 — to test whether the texture
artifacts are an AlexNet/weight effect. Report: `/workspace/outputs/report/`.

## Image-quality evaluation (FID / KID)

**Metric.** Realism vs the held-out real FFHQ faces, measured by **FID** (primary)
and **KID** (companion, unbiased at small N), both from InceptionV3-2048 features.
Run through Lightning's built-in test loop (`Trainer.test` → `test_step` /
`test_dataloader` / `FidCallback.on_test_*`); the orchestrator is `imagegen.evaluate`.
Every model is scored on an **identical protocol**: 512 generated images from
held-out captions vs 512 held-out real faces, fixed seeds, 30 steps, guidance 5.0,
512×512. Lower is better. Run: `uv run python -m imagegen.evaluate`.

| rank | model | FID ↓ | KID ↓ | ΔFID vs base |
|------|-------|-------|-------|--------------|
| 1 | LoRA (L2) `exp2_lora` | **54.45** | 0.0036 | −47.04 |
| 2 | Full FT (Huber) `exp4_full_huber` | 56.65 | 0.0026 | −44.84 |
| 3 | Full FT (L2) `exp3_full_fid` | 58.06 | 0.0032 | −43.42 |
| 4 | LoRA + ArcFace `exp6_lora_arcface` | 63.19 | 0.0063 | −38.29 |
| 5 | LoRA + LPIPS w=0.05 `exp8_lora_lpips_low` | 82.65 | 0.0311 | −18.84 |
| 6 | LoRA + LPIPS VGG `exp7_lora_lpips_vgg` | 84.86 | 0.0310 | −16.62 |
| 7 | LoRA + LPIPS Alex `exp5_lora_lpips` | 91.72 | 0.0407 | −9.77 |
| 8 | Base SD 2.1 (no fine-tune) `base` | 101.48 | 0.0328 | 0.00 |

**Findings.** (1) Every fine-tune beats base (101 → 54–92). (2) **Plain LoRA-L2 wins
(FID 54.45)** — the simplest adapter gives the best realism, edging out full fine-tuning
(56–58) at a fraction of the trainable params. (3) **Pixel-space LPIPS aux loss is
clearly harmful**: it pushes FID to 83–92 and roughly 10× the KID of the clean runs —
the over-texturing seen visually is now quantified. The two "rescue" probes only soften
it: VGG (84.86) ≈ half-weight Alex (82.65) < Alex w=0.1 (91.72), but none recover the
54–58 of the LPIPS-free runs. (4) **ArcFace identity loss is benign** (63.19) — a mild
FID cost vs plain LoRA, no texture blow-up. (5) `test/loss` (held-out diffusion MSE) does
**not** track FID (e.g. full-Huber has the lowest loss 0.045 but not the best FID),
confirming FID/KID measure something the training loss does not. Charts:
`/workspace/outputs/report/quality_metrics.png`; raw numbers: `quality_metrics.json`.

Notes:
- #1 is skipped: #3 trains identically (full-FT, L2) and additionally logs FID,
  so it doubles as the full-FT baseline. #4 (Huber) compares against #3.
- FID (#3) is logged as `val/fid` (generated vs held-out real faces) — a
  distribution metric, not a differentiable loss term.
- Aux-loss runs (#5–#8) reconstruct the predicted x0, VAE-decode it, and compare
  to the real image (LPIPS at 256px / facenet identity at 160px); they run on LoRA
  at batch 4 so the in-graph decode + perceptual nets fit 16 GB. #7 uses the VGG
  LPIPS backbone (vs Alex), #8 uses Alex at half weight (0.05).
- Run all: `bash scripts/run_experiments.sh`
- Visual report: `uv run python scripts/generate_report.py`
