"""FID / KID image-quality callback.

FID and KID are distribution-level metrics, not differentiable per-batch losses,
so they live in a callback rather than the training objective. At the end of a
validation/test epoch we generate a fixed budget of images from the held-out
prompts and score them against a cached set of real held-out faces with
``torchmetrics``' Frechet / Kernel Inception Distance.

Two stages share the same machinery:

* **validation** (during ``Trainer.fit``) logs ``val/fid`` only -- a cheap,
  directional quality signal that does not slow training.
* **test** (``Trainer.test``, see ``imagegen.evaluate``) logs ``test/fid`` **and**
  ``test/kid`` -- the comparable image-quality evaluation run on every solution.
  KID is (near-)unbiased at the few-hundred-sample budget a 16 GB card allows, so
  it backstops FID's small-sample bias.

The sample budgets are deliberately modest, so the validation metric stays cheap
during short runs; the test stage is given a larger, uniform budget by
``imagegen.evaluate`` so the numbers are comparable across models.
"""

from __future__ import annotations

import numpy as np
import torch
from lightning.pytorch import Callback


class FidCallback(Callback):
    def __init__(
        self,
        trigger_prompt: str,
        num_samples: int = 64,
        real_images: int = 256,
        num_inference_steps: int = 25,
        guidance_scale: float = 4.0,
        seed: int = 0,
        kid_subset_size: int = 50,
    ) -> None:
        super().__init__()
        self.trigger_prompt = trigger_prompt
        self.num_samples = int(num_samples)
        self.real_images = int(real_images)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.seed = int(seed)
        # KID samples `subset_size` features per subset; it must not exceed the
        # smaller of the real / generated pools (smoke runs use only a handful).
        self.kid_subset_size = min(int(kid_subset_size), self.num_samples, self.real_images)

        self._fid = None  # lazily built torchmetrics FID, kept on the module device
        self._kid = None  # lazily built torchmetrics KID (test stage only)
        self._real = []  # cached real images in [0,1], NCHW float, capped at real_images
        self._real_ready = False
        self._prompts: list[str] = []  # held-out captions used as generation prompts

    # --- real-image + prompt capture -------------------------------------

    def _capture(self, batch, batch_idx) -> None:
        # Cache a fixed pool of real reference images (denormalize [-1,1] -> [0,1])
        # and accumulate the held-out captions that drive generation -- up to the
        # generation budget, so a large test run samples diverse prompts rather than
        # cycling one batch. The val/test loader is shuffle=False, so this is the
        # same deterministic set every run.
        imgs = (batch["pixel_values"].detach().float().cpu() + 1.0) / 2.0
        caps = list(batch["caption"])
        for img, cap in zip(imgs, caps):
            if not self._real_ready and len(self._real) < self.real_images:
                self._real.append(img.clamp(0, 1))
            if len(self._prompts) < self.num_samples:
                self._prompts.append(cap)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        self._capture(batch, batch_idx)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        self._capture(batch, batch_idx)

    # --- metric compute ---------------------------------------------------

    def _ensure_metrics(self, device, with_kid: bool) -> None:
        if self._fid is None:
            from torchmetrics.image.fid import FrechetInceptionDistance

            self._fid = FrechetInceptionDistance(feature=2048, normalize=True)
        self._fid = self._fid.to(device)
        if with_kid and self._kid is None:
            from torchmetrics.image.kid import KernelInceptionDistance

            self._kid = KernelInceptionDistance(
                feature=2048, subset_size=self.kid_subset_size, normalize=True
            )
        if with_kid:
            self._kid = self._kid.to(device)

    def _prompt_for(self, i: int) -> str:
        if self._prompts:
            return self._prompts[i % len(self._prompts)]
        return self.trigger_prompt

    @torch.no_grad()
    def _run_stage(self, pl_module, stage: str, with_kid: bool) -> None:
        if not self._real:
            return  # no held-out split -> nothing to reference against
        device = pl_module.device
        self._ensure_metrics(device, with_kid)
        self._fid.reset()
        if with_kid:
            self._kid.reset()

        # Real reference distribution (cached once; mark ready so we stop growing it).
        self._real_ready = True
        for img in self._real:
            x = img.unsqueeze(0).to(device)
            self._fid.update(x, real=True)
            if with_kid:
                self._kid.update(x, real=True)

        # Generated distribution: one image per call (generate batches a single
        # prompt), cycling the held-out prompts, a distinct seed per image so the
        # set is diverse rather than the same face repeated.
        for i in range(self.num_samples):
            images = pl_module.generate(
                prompt=self._prompt_for(i),
                num_images=1,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                seed=self.seed + i,
            )
            arr = np.asarray(images[0].convert("RGB"), dtype=np.float32) / 255.0
            fake = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            self._fid.update(fake, real=False)
            if with_kid:
                self._kid.update(fake, real=False)

        fid = self._fid.compute().float()
        pl_module.log(f"{stage}/fid", fid, prog_bar=True, on_epoch=True, sync_dist=False)
        if with_kid:
            kid_mean, kid_std = self._kid.compute()
            pl_module.log(f"{stage}/kid", kid_mean.float(), prog_bar=True, on_epoch=True, sync_dist=False)
            pl_module.log(f"{stage}/kid_std", kid_std.float(), on_epoch=True, sync_dist=False)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        # Cheap directional signal during training: FID only.
        self._run_stage(pl_module, stage="val", with_kid=False)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        # Full image-quality evaluation: FID + KID.
        self._run_stage(pl_module, stage="test", with_kid=True)
