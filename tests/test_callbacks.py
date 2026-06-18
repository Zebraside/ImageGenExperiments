"""Tests for SampleImageCallback's step gating (offline; generate() is stubbed)."""

from imagegen.callbacks import SampleImageCallback


class _FakeTrainer:
    def __init__(self, step):
        self.global_step = step
        self.logger = object()  # not a WandbLogger -> the W&B branch is skipped


class _FakeModule:
    def __init__(self):
        self.calls = 0
        self.prompts = []

    def generate(self, **kwargs):
        self.calls += 1
        self.prompts.append(kwargs.get("prompt"))
        return []  # no images -> nothing written to disk


def _callback(tmp_path, every_n_steps=500, num_samples=1):
    return SampleImageCallback(
        prompt="p",
        output_dir=str(tmp_path),
        num_samples=num_samples,
        every_n_steps=every_n_steps,
    )


def test_samples_once_per_step_under_accumulation(tmp_path):
    cb = _callback(tmp_path)
    module = _FakeModule()
    trainer = _FakeTrainer(step=500)
    # Several micro-batches share the same global_step (accumulate_grad_batches > 1).
    for _ in range(4):
        cb.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=0)
    assert module.calls == 1  # deduped to a single sample pass


def test_advancing_step_allows_another_sample(tmp_path):
    cb = _callback(tmp_path)
    module = _FakeModule()
    cb.on_train_batch_end(_FakeTrainer(500), module, None, None, 0)
    cb.on_train_batch_end(_FakeTrainer(1000), module, None, None, 0)
    assert module.calls == 2


def test_non_multiple_and_zero_steps_skip(tmp_path):
    cb = _callback(tmp_path)
    module = _FakeModule()
    cb.on_train_batch_end(_FakeTrainer(0), module, None, None, 0)    # step 0 never samples
    cb.on_train_batch_end(_FakeTrainer(501), module, None, None, 0)  # not a multiple
    assert module.calls == 0


def test_validation_samples_one_image_per_captured_caption(tmp_path):
    cb = _callback(tmp_path, num_samples=2)
    module = _FakeModule()
    trainer = _FakeTrainer(step=500)
    batch = {"caption": ["a", "b", "c"]}  # more than num_samples
    cb.on_validation_batch_end(trainer, module, None, batch, batch_idx=0)
    cb.on_validation_epoch_end(trainer, module)
    # Capped at num_samples (2), one generate() call per distinct caption.
    assert module.calls == 2
    assert module.prompts == ["a", "b"]


def test_validation_only_captures_first_batch(tmp_path):
    cb = _callback(tmp_path, num_samples=2)
    module = _FakeModule()
    trainer = _FakeTrainer(step=500)
    cb.on_validation_batch_end(trainer, module, None, {"caption": ["a", "b"]}, batch_idx=0)
    cb.on_validation_batch_end(trainer, module, None, {"caption": ["x", "y"]}, batch_idx=1)
    cb.on_validation_epoch_end(trainer, module)
    assert module.prompts == ["a", "b"]  # later batches don't overwrite the captured set


def test_validation_epoch_end_without_split_is_noop(tmp_path):
    cb = _callback(tmp_path)
    module = _FakeModule()
    cb.on_validation_epoch_end(_FakeTrainer(500), module)  # no captions captured
    assert module.calls == 0
