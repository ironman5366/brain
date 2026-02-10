"""
Combine multiple preprocessed EEG datasets into a single safetensors + parquet.

Each source dataset is expected to have:
  <prefix>-{split}.safetensors        (key: "samples", shape [N, 32, 125])
  <prefix>-{split}-encodec.safetensors (key: "audio_embeds", shape [N, 128, 75])
  <prefix>-{split}-metadata.parquet

To add a new dataset, append its prefix to SOURCES and re-run.

Usage:
    uv run python -m data.combine --name combined-4src
"""

import argparse
from pathlib import Path

import polars as pl
import torch
from safetensors.torch import load_file, save_file

DATA_DIR = Path("/kreka/research/willy/side/brain_datasets/nmed-processed")

SOURCES = [
    "nmed-32ch",
    "songfam",
    "musicemo",
    "musin-g",
]


def combine_split(name: str, split: str):
    all_samples = []
    all_audio_embeds = []
    all_metadata = []

    for source in SOURCES:
        samples_path = DATA_DIR / f"{source}-{split}.safetensors"
        encodec_path = DATA_DIR / f"{source}-{split}-encodec.safetensors"
        metadata_path = DATA_DIR / f"{source}-{split}-metadata.parquet"

        if not samples_path.exists():
            print(f"  SKIP {source} ({split}): {samples_path} not found")
            continue
        if not encodec_path.exists():
            print(f"  SKIP {source} ({split}): {encodec_path} not found")
            continue

        samples = load_file(samples_path)["samples"]
        audio_embeds = load_file(encodec_path)["audio_embeds"]

        assert len(samples) == len(audio_embeds), (
            f"{source}: samples ({len(samples)}) != audio_embeds ({len(audio_embeds)})"
        )

        all_samples.append(samples)
        all_audio_embeds.append(audio_embeds)

        if metadata_path.exists():
            meta = pl.read_parquet(metadata_path).with_columns(
                pl.lit(source).alias("source")
            )
            all_metadata.append(meta)

        print(f"  {source}: {len(samples):,} samples")

    if not all_samples:
        print(f"  No sources found for {split}, skipping")
        return

    combined_samples = torch.cat(all_samples, dim=0)
    combined_audio_embeds = torch.cat(all_audio_embeds, dim=0)

    print(f"  Total: {len(combined_samples):,} samples, shape {list(combined_samples.shape)}")

    save_file(
        {"samples": combined_samples},
        DATA_DIR / f"{name}-{split}.safetensors",
    )
    save_file(
        {"audio_embeds": combined_audio_embeds},
        DATA_DIR / f"{name}-{split}-encodec.safetensors",
    )

    if all_metadata:
        combined_meta = pl.concat(all_metadata, how="diagonal")
        combined_meta.write_parquet(DATA_DIR / f"{name}-{split}-metadata.parquet")

    print(f"  Saved to {DATA_DIR / name}-{split}.*")


def main():
    parser = argparse.ArgumentParser(description="Combine preprocessed EEG datasets")
    parser.add_argument("--name", required=True, help="Output file prefix")
    args = parser.parse_args()

    print(f"=== Combining {len(SOURCES)} sources: {', '.join(SOURCES)} ===")
    print(f"Output prefix: {args.name}\n")

    for split in ["train", "val"]:
        print(f"[{split}]")
        combine_split(args.name, split)
        print()

    print("Done!")


if __name__ == "__main__":
    main()
