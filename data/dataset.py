# Builtin imports
from pathlib import Path

# Internal imports
from constants import NUM_CHANNELS

# External imports
from torch.utils.data import Dataset
import torch
from safetensors.torch import load_file


class EEGDataset(Dataset):
    def __init__(self, samples_path: Path):
        print(f"Loading samples from {samples_path}...")
        samples_data = load_file(samples_path)

        # De-sparsify the data
        mask_indices = samples_data["mask_indices"]
        self.mask = torch.zeros(NUM_CHANNELS, dtype=torch.bool)
        self.mask[mask_indices] = True

        sparse_samples = samples_data["sparse_samples"]
        self.dense_samples = torch.zeros(NUM_CHANNELS, sparse_samples.shape[-1])
        self.dense_samples[self.mask] = sparse_samples

        print(
            f"Sparse samples shape {sparse_samples.shape}, dense samples shape {self.dense_samples.shape}"
        )

    def __len__(self):
        return len(self.dense_samples)

    def __getitem__(self, idx):
        return self.dense_samples[idx]
