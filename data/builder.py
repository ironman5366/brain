# Builtin imports
from typing import Literal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Internal imports
from data.alljoined import fetch_alljoined
from constants import (
    DEFAULT_WINDOW_SECONDS,
    DEFAULT_SFREQ,
    DEFAULT_NORMALIZATION,
    DEFAULT_OVERLAP,
)
from utils import mne_to_tensor

# External imports
import polars as pl
from pydantic import BaseModel
from tqdm import tqdm
import mne
import vortex as vx
import pyarrow.parquet as pq

mne.set_log_level(verbose="WARNING")

DATA_DIR = Path("/kreka/research/willy/side/brain_datasets/")


class DatasetConfig(BaseModel):
    name: str

    overlap: float = DEFAULT_OVERLAP
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    target_sfreq: float = DEFAULT_SFREQ
    normalization: Literal["recording"] | Literal["window"] | Literal["none"] = (
        DEFAULT_NORMALIZATION
    )

    train_split: float = 0.9
    val_split: float = 0.1


def load_file(config: DatasetConfig, data_row) -> list[dict]:
    raw_data = mne.io.read_raw_edf(data_row["file"])
    tens, mask = mne_to_tensor(
        raw_data,
        overlap=config.overlap,
        window_seconds=config.window_seconds,
        target_sfreq=config.target_sfreq,
        normalization=config.normalization,
        verbose=False,
    )

    sparse_tensor = tens[:, mask]
    mask_vals = mask.nonzero().squeeze()

    rows = []

    for i, sample in enumerate(sparse_tensor):
        rows.append(
            {
                "sample": sample.numpy(),
                "mask_indices": mask_vals.numpy(),
                "idx": i,
                **data_row,
            }
        )

    return rows


def build():
    config = DatasetConfig(name="alljoined-initial")

    samples = []

    # Fetch samples from each source
    alljoined_files = list(fetch_alljoined())
    with ThreadPoolExecutor(32) as executor:
        futs = []

        for file_row in alljoined_files:
            futs.append(executor.submit(load_file, config, file_row))

        for fut in tqdm(futs, desc="Loading alljoined data..."):
            samples.extend(fut.result())

    # Split
    df = pl.from_dicts(samples)
    print(f"Loaded: {len(df):,} samples")

    # Split on subject level to avoid data leakage
    unique_subjects = df.select("subject").unique().sample(fraction=1.0, shuffle=True)
    n_train = int(len(unique_subjects) * config.train_split)
    print("har")

    train_subjects = unique_subjects[:n_train]["subject"]
    val_subjects = unique_subjects[n_train:]["subject"]

    train_df = df.filter(pl.col("subject").is_in(train_subjects))
    val_df = df.filter(pl.col("subject").is_in(val_subjects))

    print(f"Train: {len(train_df):,} samples from {len(train_subjects)} subjects")
    print(f"Val: {len(val_df):,} samples from {len(val_subjects)} subjects")

    # Save as vortex files
    out_dir = DATA_DIR / config.name
    out_dir.mkdir(exist_ok=True, parents=True)

    train_file = out_dir / f"{config.name}-train.parquet"
    pq.write_table(train_df.to_arrow(), train_file)
    print(f"Wrote train split to {str(train_file)}")

    val_file = out_dir / f"{config.name}-val.parquet"
    pq.write_table(val_df.to_arrow(), val_file)
    print(f"Wrote val split to {str(val_file)}")


if __name__ == "__main__":
    build()
