"""Offline tests for FidCallback orchestration.

The torchmetrics FID metric is stubbed (no Inception download): we verify the
callback caches real images + prompts from validation batches, generates the
right number of images, and logs a finite val/fid.
"""

import torch
from PIL import Image

from imagegen.fid_callback import FidCallback


class _FakeFid:
    def __init__(self):
        self.real, self.fake, self.reset_calls = 0, 0, 0

    def to(self, _device):
        return self

    def reset(self):
        self.reset_calls += 1
        self.real = self.fake = 0

    def update(self, _img, real):
        if real:
            self.real += 1
        else:
            self.fake += 1

    def compute(self):
        return torch.tensor(12.5)


class _FakeKid:
    def __init__(self):
        self.real, self.fake = 0, 0

    def to(self, _device):
        return self

    def reset(self):
        self.real = self.fake = 0

    def update(self, _img, real):
        if real:
            self.real += 1
        else:
            self.fake += 1

    def compute(self):
        return torch.tensor(0.03), torch.tensor(0.001)


class _FakeModule:
    device = torch.device("cpu")

    def __init__(self):
        self.calls, self.logged = 0, {}

    def generate(self, **kwargs):
        self.calls += 1
        return [Image.new("RGB", (64, 64))]

    def log(self, name, value, **_kwargs):
        self.logged[name] = float(value)


def _batch(n, caps):
    return {"pixel_values": torch.rand(n, 3, 64, 64) * 2 - 1, "caption": caps}


def test_caches_reals_and_prompts_then_logs_fid():
    cb = FidCallback("trigger", num_samples=4, real_images=6)
    cb._fid = _FakeFid()  # inject stub -> no Inception download
    module = _FakeModule()

    cb.on_validation_batch_end(None, module, None, _batch(4, ["a", "b", "c", "d"]), 0)
    cb.on_validation_batch_end(None, module, None, _batch(4, ["e", "f", "g", "h"]), 1)
    assert len(cb._real) == 6  # capped at real_images across batches
    assert cb._prompts == ["a", "b", "c", "d"]  # only the first batch's captions

    cb.on_validation_epoch_end(None, module)
    assert module.calls == 4  # num_samples generated images
    assert cb._fid.real == 6 and cb._fid.fake == 4
    assert module.logged["val/fid"] == 12.5


def test_prompts_cycle_when_fewer_than_num_samples():
    cb = FidCallback("trigger", num_samples=5, real_images=4)
    cb._fid = _FakeFid()
    module = _FakeModule()
    cb.on_validation_batch_end(None, module, None, _batch(2, ["x", "y"]), 0)
    cb.on_validation_epoch_end(None, module)
    assert module.calls == 5  # cycles ["x","y"] to fill the budget


def test_noop_without_real_images():
    cb = FidCallback("trigger")
    module = _FakeModule()
    cb.on_validation_epoch_end(None, module)  # nothing cached
    assert module.calls == 0
    assert "val/fid" not in module.logged


def test_test_stage_logs_fid_and_kid():
    cb = FidCallback("trigger", num_samples=4, real_images=6)
    cb._fid = _FakeFid()  # inject stubs -> no Inception download
    cb._kid = _FakeKid()
    module = _FakeModule()

    cb.on_test_batch_end(None, module, None, _batch(4, ["a", "b", "c", "d"]), 0)
    cb.on_test_batch_end(None, module, None, _batch(4, ["e", "f", "g", "h"]), 1)
    assert len(cb._real) == 6

    cb.on_test_epoch_end(None, module)
    assert module.calls == 4
    assert cb._fid.real == 6 and cb._fid.fake == 4
    assert cb._kid.real == 6 and cb._kid.fake == 4  # KID gets the same updates
    assert module.logged["test/fid"] == 12.5
    assert abs(module.logged["test/kid"] - 0.03) < 1e-5
    assert abs(module.logged["test/kid_std"] - 0.001) < 1e-5


def test_test_prompts_accumulate_across_batches():
    cb = FidCallback("trigger", num_samples=6, real_images=8)
    cb._fid, cb._kid = _FakeFid(), _FakeKid()
    module = _FakeModule()
    cb.on_test_batch_end(None, module, None, _batch(4, ["a", "b", "c", "d"]), 0)
    cb.on_test_batch_end(None, module, None, _batch(4, ["e", "f", "g", "h"]), 1)
    # captions accumulate up to num_samples (diverse prompts for a large eval)
    assert cb._prompts == ["a", "b", "c", "d", "e", "f"]
