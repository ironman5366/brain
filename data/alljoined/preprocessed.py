"""
Sanity check my preprocessing by using the alljoined exported npy files
"""

# Builtin imports
from pathlib import Path

# External imports
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import polars as pl
import numpy as np
from safetensors.torch import save_file, load_file


def build_preprocessed(preprocessed_dir: Path, out_path: Path):
    for split, split_train_file, split_partition in [
        ("test", "preprocessed_eeg_test_flat.npy", "stim_test"),
        ("train", "preprocessed_eeg_training_flat.npy", "stim_train"),
    ]:
        print(f"Processing split {split}, {split_train_file}, {split_partition}")

        all_metadata = None
        all_samples = None

        for ch in tqdm(list(sorted(preprocessed_dir.iterdir()))):
            if ch.is_dir() and ch.name.startswith("sub-"):
                metadata_file = ch / "experiment_metadata_categories.parquet"
                metadata_df = pl.read_parquet(metadata_file)
                metadata_df = metadata_df.filter(
                    pl.col("partition") == "stim_train", ~pl.col("dropped")
                )

                # If we have an existing one, select the subset of columns that the existing one has
                if all_metadata is not None:
                    metadata_df = metadata_df.select(all_metadata.columns)

                print("schema", metadata_df.schema)

                if all_metadata is None:
                    all_metadata = metadata_df
                else:
                    all_metadata = all_metadata.vstack(metadata_df)

                train_file = ch / "preprocessed_eeg_training_flat.npy"
                samples = np.load(train_file, allow_pickle=True)[
                    "preprocessed_eeg_data"
                ]

                assert len(metadata_df) == len(samples), "Data len != metadata len"
                samples_torch = torch.from_numpy(samples)
                print(f"Loaded {samples_torch.shape} samples from subject {ch.name}")

                if all_samples is None:
                    all_samples = samples_torch
                else:
                    all_samples = torch.cat((all_samples, samples_torch))

        # Another sanity check
        assert len(all_metadata) == len(all_samples), "Badness"
        print(
            f"Loaded {all_samples.shape} samples from {all_metadata.unique('subject').count()} subjects"
        )

        metadata_path = out_path / f"{split}-metadata.parquet"
        all_metadata.write_parquet(metadata_path)
        print(f"wrote {split} metadata to {metadata_path}")

        tensors_path = out_path / f"{split}.safetensors"
        save_file({"samples": all_samples}, tensors_path)
        print(f"wrote {split} data to {tensors_path}")


class AJPreprocessedDataset(Dataset):
    def __init__(self, built_dir: Path, split: str):
        # Crawl the preprocessed dir, find all the subject metadata files
        samples_file = built_dir / f"{split}.safetensors"
        assert samples_file.exists()

        metadata_file = built_dir / f"{split}-metadata.parquet"
        assert metadata_file.exists()

        self.samples = load_file(samples_file)["samples"].to(dtype=torch.float32)
        print(f"Loaded {self.samples.shape} samples from {samples_file}")

        self.metadata = pl.read_parquet(metadata_file)
        print(f"loaded {len(self.metadata)} metadata rows from {metadata_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {"sample": self.samples[idx], "row": self.metadata[idx]}


class AJPreprocessedClassificationDataset(AJPreprocessedDataset):
    def __init__(self, built_dir: Path, split: str, class_col: str):
        super().__init__(built_dir, split)
        self.class_col = class_col

        self.distinct = (
            self.metadata.select(class_col).unique().sort(by=class_col).with_row_index()
        )
        self.vals_to_ids = {}
        for i, row in enumerate(
            self.distinct.select(self.class_col).iter_rows(named=True)
        ):
            self.vals_to_ids[row[self.class_col]] = i

        self.class_dim = len(self.distinct)

        print(f"Classifying on {class_col}, {self.class_dim} classes, {self.distinct}")

    def __getitem__(self, idx):
        it = super().__getitem__(idx)
        row = it["row"]
        tensor = it["sample"]

        row_class = row.select(self.class_col)[0].item()
        row_it = self.vals_to_ids[row_class]

        return tensor, torch.tensor(row_it, dtype=torch.long)
