"""Verify that the torch install works, including GPU compute.

Run with:  uv run scripts/check_torch.py
Exits non-zero if CUDA is unavailable or the GPU computation fails.
"""

import sys

import torch


def main() -> int:
    print(f"torch version : {torch.__version__}")
    print(f"compiled CUDA : {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("FAIL: CUDA is not available to torch.")
        return 1

    device = torch.device("cuda:0")
    print(f"device name   : {torch.cuda.get_device_name(device)}")

    # Small matmul on the GPU, then move the result back to the CPU.
    a = torch.randn(512, 512, device=device)
    b = torch.randn(512, 512, device=device)
    c = (a @ b).cpu()

    if not torch.isfinite(c).all():
        print("FAIL: GPU matmul produced non-finite values.")
        return 1

    print(f"matmul result : shape={tuple(c.shape)}, mean={c.mean().item():.4f}")
    print("PASS: torch is installed and the GPU is usable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
