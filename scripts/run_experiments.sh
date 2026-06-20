#!/usr/bin/env bash
# Run the experiments sequentially (single 16 GB GPU; full-FT runs peak
# ~15.6 GB so they cannot share the card). Each run logs to W&B under the
# run_name baked into its config, and tees console output to its output dir.
# Output dirs are wiped first to clear any smoke-test pollution.
set -u
cd /root/ImageGenExperiments
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # curbs fragmentation for the aux-loss runs

# exp1_full_baseline is intentionally skipped: exp3 trains identically (full-FT, L2)
# and additionally logs FID, so it serves as the full-FT reference.
CONFIGS=(
  exp2_lora
  exp3_full_fid
  exp4_full_huber
  exp5_lora_lpips
  exp6_lora_arcface
  exp7_lora_lpips_vgg
  exp8_lora_lpips_low
)

for cfg in "${CONFIGS[@]}"; do
  out="/workspace/outputs/$cfg"
  echo "============================================================"
  echo "RUN: $cfg  ($(date -u +%H:%M:%S) UTC)"
  echo "============================================================"
  rm -rf "$out"
  mkdir -p "$out"
  uv run python -m imagegen.train --config "configs/$cfg.yaml" 2>&1 | tee "$out/train.log"
  if [ "${PIPESTATUS[0]}" -eq 0 ]; then
    echo "RUN_OK: $cfg ($(date -u +%H:%M:%S) UTC)"
  else
    echo "RUN_FAIL: $cfg ($(date -u +%H:%M:%S) UTC)"
  fi
done
echo "ALL_RUNS_DONE"
