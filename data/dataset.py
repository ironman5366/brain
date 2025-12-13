# Builtin imports
from pathlib import Path

# External imports
from torch.utils.data import Dataset
import polars as pl
import torch
import numpy as np
from tqdm import tqdm


class EEGDataset(Dataset):
    def __init__(self, path: Path):
        print(f"Loading {path}...")
        self.data = pl.read_parquet(path)

        print("Loading samples...")
        samples = []
        for row in tqdm(self.data.iter_rows(named=True)):
            samples.append(np.array(row["sample"]))
        ns = np.array(samples)
        print(f"Started ns = {ns.shape}")

        self.samples = torch.Tensor(ns)
        print(f"Samples = {self.samples}")

        # Don't store double the mem
        self.data = self.data.drop("sample")
        print(f"Samples shape {self.samples.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.samples[index]
