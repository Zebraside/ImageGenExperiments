#!/usr/bin/env bash
# Fast smoke test of every experiment config: 2 steps, tiny data, no W&B.
# Validates the train/val/save path (incl. FID + aux-loss) and that VRAM holds.
set -u
cd /root/ImageGenExperiments

COMMON="train.max_steps=2 data.limit=16 data.val_size=8 \
  train.accumulate_grad_batches=1 train.val_check_interval=2 \
  train.num_sanity_val_steps=0 train.num_samples=1 train.sample_steps=5 \
  train.sample_every_n_steps=99999 logging.wandb_project=null"

declare -A EXTRA=(
  [exp3_full_fid]="train.fid_num_samples=2 train.fid_real_images=2 train.fid_steps=5"
)

for cfg in exp1_full_baseline exp2_lora exp3_full_fid exp4_full_huber exp5_lora_lpips exp6_lora_arcface; do
  echo "============================================================"
  echo "SMOKE: $cfg"
  echo "============================================================"
  # shellcheck disable=SC2086
  uv run python -m imagegen.train --config configs/$cfg.yaml $COMMON ${EXTRA[$cfg]:-} \
    && echo "SMOKE_OK: $cfg" || echo "SMOKE_FAIL: $cfg (exit $?)"
done
echo "ALL_SMOKE_DONE"
