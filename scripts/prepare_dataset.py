"""Download FFHQ (merkol/ffhq-256) and extract its 256x256 face images to disk.

Run with:  uv run scripts/prepare_dataset.py

merkol/ffhq-256 ships the full 70,000 FFHQ faces, already at 256x256, packed into
~15 Parquet shards with embedded PNG bytes (~7.4 GB). Packed shards download as a
handful of large files, so there's none of the per-file API-rate-limit pain of a
loose-file repo.

Stage 1 downloads the Parquet shards into data/ffhq256_parquet/. Stage 2 decodes
every row and writes it as data/ffhq256/<shard>/<row>.png (the layout the folder
DataModule reads). Because the stored bytes are already 256x256 PNG, Stage 2 just
writes them out verbatim -- no resize/re-encode.

Both stages are idempotent: re-runs skip already-downloaded shards and any output
PNG that already exists. Exits non-zero if a row is missing image bytes.
"""

# Enable hf-transfer before importing huggingface_hub so it takes effect.
import os

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

REPO_ID = "merkol/ffhq-256"
PARQUET_ROOT = Path("/workspace/data/ffhq256_parquet")
DST_ROOT = Path("/workspace/data/ffhq256")


def download() -> None:
    print(f"Downloading {REPO_ID} parquet shards -> {PARQUET_ROOT} (resumable)...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(PARQUET_ROOT),
        allow_patterns=["*.parquet"],
    )
    print("Download complete.")


def extract() -> int:
    shards = sorted(PARQUET_ROOT.rglob("*.parquet"))
    if not shards:
        print(f"No .parquet shards found under {PARQUET_ROOT!r}.", file=sys.stderr)
        return 1
    print(f"Found {len(shards)} parquet shards. Extracting PNGs -> {DST_ROOT}")

    written = skipped = 0
    errors: list[str] = []
    for shard_idx, shard in enumerate(shards):
        out_dir = DST_ROOT / f"{shard_idx:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        pf = pq.ParquetFile(shard)
        row = 0
        for batch in pf.iter_batches(batch_size=512, columns=["image"]):
            for cell in batch.column("image").to_pylist():
                dst = out_dir / f"{row:06d}.png"
                row += 1
                if dst.exists():
                    skipped += 1
                    continue
                data = cell.get("bytes") if isinstance(cell, dict) else cell
                if not data:
                    errors.append(f"{shard.name} row {row - 1}: no image bytes")
                    continue
                dst.write_bytes(data)  # already 256x256 PNG; no re-encode
                written += 1
        print(f"  shard {shard_idx:03d} ({shard.name}): {row} rows "
              f"(written={written} skipped={skipped})")

    print(f"\nSummary: written={written} skipped={skipped} errors={len(errors)}")
    if errors:
        print("Failures:")
        for e in errors[:50]:
            print(f"  {e}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more")
        return 1
    return 0


def main() -> int:
    download()
    return extract()


if __name__ == "__main__":
    sys.exit(main())
