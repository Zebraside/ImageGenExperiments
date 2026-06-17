"""Generate per-image captions for an image folder with a local BLIP model.

Run with:  uv run scripts/caption_dataset.py

FFHQ ships no captions, so we caption every face with BLIP and write the text to a
sidecar file next to each image (``<image>.png`` -> ``<image>.txt``). That's the layout
``imagegen.data.dataset.ImageFolderDataset`` reads: it uses the sidecar caption when
present and falls back to the fixed trigger prompt otherwise.

Images are discovered recursively under ``--root`` using the same extension set as the
dataset. The model runs on GPU in fp16 when CUDA is available. The run is idempotent and
resumable: an image whose ``.txt`` already exists is skipped (use ``--overwrite`` to
re-caption), so re-runs and the partial 29k->70k extraction both work.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from imagegen.data.dataset import IMAGE_EXTS

DEFAULT_ROOT = Path("/workspace/data/ffhq512")
DEFAULT_MODEL = "Salesforce/blip-image-captioning-large"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Caption an image folder with BLIP -> sidecar .txt.")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Image folder (recursive).")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF BLIP captioning model id.")
    p.add_argument("--batch-size", type=int, default=32, help="Images per BLIP forward pass.")
    p.add_argument("--max-new-tokens", type=int, default=30, help="Max caption length (tokens).")
    p.add_argument("--limit", type=int, default=None, help="Cap #images (smoke tests).")
    p.add_argument("--overwrite", action="store_true", help="Re-caption images that already have a .txt.")
    return p.parse_args()


def find_images(root: Path, limit: int | None) -> list[Path]:
    if not root.exists():
        print(f"Root {root!r} does not exist. Run scripts/prepare_dataset.py first.", file=sys.stderr)
        sys.exit(1)
    paths = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if limit is not None:
        paths = paths[:limit]
    return paths


def main() -> int:
    args = parse_args()

    paths = find_images(args.root, args.limit)
    if not paths:
        print(f"No images ({sorted(IMAGE_EXTS)}) found under {args.root!r}.", file=sys.stderr)
        return 1

    todo = paths if args.overwrite else [p for p in paths if not p.with_suffix(".txt").exists()]
    pre_skipped = len(paths) - len(todo)
    print(f"Found {len(paths)} images under {args.root}; {len(todo)} to caption "
          f"({pre_skipped} already have a .txt).")
    if not todo:
        print("Nothing to do.")
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {args.model} on {device} ({dtype})...")
    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    written = 0
    for start in range(0, len(todo), args.batch_size):
        batch_paths = todo[start : start + args.batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device, dtype)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        captions = processor.batch_decode(out, skip_special_tokens=True)

        for path, caption in zip(batch_paths, captions):
            path.with_suffix(".txt").write_text(caption.strip() + "\n", encoding="utf-8")
            written += 1
        print(f"  captioned {written}/{len(todo)} (last: {captions[-1].strip()!r})")

    print(f"\nSummary: written={written} skipped={pre_skipped} total_images={len(paths)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
