"""Normalize sidecar captions into one clean form with a small local LLM.

Run with:  uv run scripts/normalize_captions.py

Reads each ``<image>.txt`` produced by ``scripts/caption_dataset.py`` and rewrites it into a
single clean natural phrase (filler/artifacts removed, lowercase, no trailing period) using
``Qwen/Qwen2.5-1.5B-Instruct``. Captions are overwritten in place.

The run is resumable without keeping a backup: relative caption paths that have been
normalized are appended to a state file (``--state``); a re-run skips them. Use ``--overwrite``
to re-normalize everything regardless of state.

Images are discovered recursively under ``--root`` using the dataset's extension set; only
images that already have a ``.txt`` are processed (so this can run while/after captioning).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from imagegen.captions import build_messages, clean_caption
from imagegen.data.dataset import IMAGE_EXTS

DEFAULT_ROOT = Path("data/ffhq256")
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_STATE = Path("outputs/normalize_state.txt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rewrite sidecar captions into one clean form.")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Image folder (recursive).")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF instruct model id.")
    p.add_argument("--batch-size", type=int, default=64, help="Captions per generate() call.")
    p.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens per rewrite.")
    p.add_argument("--limit", type=int, default=None, help="Cap #captions (smoke tests).")
    p.add_argument("--overwrite", action="store_true", help="Ignore state; re-normalize all.")
    p.add_argument("--state", type=Path, default=DEFAULT_STATE, help="Resumable progress file.")
    return p.parse_args()


def find_caption_paths(root: Path, limit: int | None) -> list[Path]:
    """Sidecar ``.txt`` paths that have a matching image, found in a single walk.

    One ``rglob`` pass collects image and ``.txt`` stems and intersects them in memory --
    no per-file ``is_file``/``exists`` stat, which matters when ``root`` is on a slow
    (e.g. FUSE) filesystem with 100k+ entries.
    """
    if not root.exists():
        print(f"Root {root!r} does not exist.", file=sys.stderr)
        sys.exit(1)
    image_stems: set[Path] = set()
    txt_by_stem: dict[Path, Path] = {}
    for p in root.rglob("*"):
        suffix = p.suffix.lower()
        if suffix in IMAGE_EXTS:
            image_stems.add(p.with_suffix(""))
        elif suffix == ".txt":
            txt_by_stem[p.with_suffix("")] = p
    captions = sorted(txt_by_stem[s] for s in image_stems if s in txt_by_stem)
    if limit is not None:
        captions = captions[:limit]
    return captions


def load_done(state: Path) -> set[str]:
    if not state.exists():
        return set()
    return {line.strip() for line in state.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> int:
    args = parse_args()

    captions = find_caption_paths(args.root, args.limit)
    if not captions:
        print(f"No sidecar .txt captions found under {args.root!r}.", file=sys.stderr)
        return 1

    done = set() if args.overwrite else load_done(args.state)
    todo = [c for c in captions if str(c.relative_to(args.root)) not in done]
    skipped = len(captions) - len(todo)
    print(f"Found {len(captions)} captions under {args.root}; {len(todo)} to normalize "
          f"({skipped} already done per {args.state}).")
    if not todo:
        print("Nothing to do.")
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {args.model} on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    args.state.parent.mkdir(parents=True, exist_ok=True)
    normalized = 0
    with args.state.open("a", encoding="utf-8") as state_f:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start : start + args.batch_size]
            raws = [c.read_text(encoding="utf-8").strip() for c in batch]
            prompts = [
                tokenizer.apply_chat_template(
                    build_messages(r), tokenize=False, add_generation_prompt=True
                )
                for r in raws
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            # Decode only the newly generated continuation (drop the prompt tokens).
            gen = out[:, inputs["input_ids"].shape[1] :]
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)

            for path, text in zip(batch, decoded):
                caption = clean_caption(text)
                path.write_text(caption + "\n", encoding="utf-8")
                state_f.write(f"{path.relative_to(args.root)}\n")
                normalized += 1
            state_f.flush()
            print(f"  normalized {normalized}/{len(todo)} "
                  f"(last: {raws[-1]!r} -> {clean_caption(decoded[-1])!r})")

    print(f"\nSummary: normalized={normalized} skipped={skipped} total_captions={len(captions)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
