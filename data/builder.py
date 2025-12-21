# Builtin imports
from typing import Literal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import random

# Internal imports
from data.alljoined import (
    fetch_alljoined,
    load_alljoined_file,
    DEFAULT_TMIN,
    DEFAULT_TMAX,
)
from constants import (
    DEFAULT_SFREQ,
    DEFAULT_NORMALIZATION,
)

# External imports
import polars as pl
from pydantic import BaseModel
from tqdm import tqdm
import mne
import torch
from safetensors.torch import save_file

mne.set_log_level(verbose="WARNING")

DATA_DIR = Path("/kreka/research/willy/side/brain_datasets/")


class DatasetConfig(BaseModel):
    name: str

    # Epoch timing
    tmin: float = DEFAULT_TMIN
    tmax: float = DEFAULT_TMAX
    target_sfreq: float = DEFAULT_SFREQ
    normalization: Literal["recording"] | Literal["epoch"] | Literal["none"] = (
        DEFAULT_NORMALIZATION
    )

    train_split: float = 0.9
    val_split: float = 0.1


def save_split(
    samples: list[dict],
    mask_indices: torch.Tensor,
    safetensors_path: Path,
    metadata_path: Path,
):
    """
    Stack sample tensors and save to safetensors, save metadata to parquet.
    """
    # Stack all sample tensors
    tensors = torch.stack([s["sample"] for s in samples])

    # Save tensors to safetensors
    save_file(
        {"sparse_samples": tensors, "mask_indices": mask_indices}, safetensors_path
    )

    # Build metadata (exclude tensor fields)
    metadata_df = pl.from_dicts(samples)

    metadata_df = metadata_df.drop("sample", "mask")

    metadata_df.write_parquet(metadata_path)

    print(f"Saved {tensors.shape} samples to {safetensors_path}")
    print(f"Saved metadata to {metadata_path}")


def build():
    config = DatasetConfig(name="alljoined-epochs-2025-12-21")

    all_samples = []
    mask_indices = None

    # Fetch and load samples from each source
    alljoined_files = list(fetch_alljoined())
    with ThreadPoolExecutor(32) as executor:
        futs = []
        for file_row in alljoined_files:
            futs.append(
                executor.submit(
                    load_alljoined_file,
                    file_row,
                    target_sfreq=config.target_sfreq,
                    tmin=config.tmin,
                    tmax=config.tmax,
                    normalization=config.normalization,
                )
            )

        for fut in tqdm(futs, desc="Loading alljoined data..."):
            file_samples = fut.result()

            # Verify mask consistency (all files should have same active channels)
            if file_samples:
                file_mask = file_samples[0]["mask"]
                if mask_indices is None:
                    mask_indices = file_mask
                else:
                    assert torch.equal(file_mask, mask_indices), (
                        "Masks differ between files, storage format needs to be changed"
                    )

            all_samples.extend(file_samples)

    print(f"Loaded: {len(all_samples):,} samples")

    # Subject-level splitting
    unique_subjects = list(set(s["subject"] for s in all_samples))
    random.shuffle(unique_subjects)

    n_train = int(len(unique_subjects) * config.train_split)
    train_subjects = set(unique_subjects[:n_train])
    val_subjects = set(unique_subjects[n_train:])

    train_samples = [s for s in all_samples if s["subject"] in train_subjects]
    val_samples = [s for s in all_samples if s["subject"] in val_subjects]

    print(f"Train: {len(train_samples):,} samples from {len(train_subjects)} subjects")
    print(f"Val: {len(val_samples):,} samples from {len(val_subjects)} subjects")

    # Save splits
    out_dir = DATA_DIR / config.name
    out_dir.mkdir(exist_ok=True, parents=True)

    save_split(
        train_samples,
        mask_indices,
        out_dir / f"{config.name}-train.safetensors",
        out_dir / f"{config.name}-train-metadata.parquet",
    )

    save_split(
        val_samples,
        mask_indices,
        out_dir / f"{config.name}-val.safetensors",
        out_dir / f"{config.name}-val-metadata.parquet",
    )


if __name__ == "__main__":
    build()
