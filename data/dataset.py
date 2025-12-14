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

        mask_indices = samples_data["mask_indices"]
        self.mask = torch.zeros(NUM_CHANNELS, dtype=torch.bool)
        self.mask[mask_indices] = True
        self.sparse_samples = samples_data["sparse_samples"]

    def __len__(self):
        return len(self.sparse_samples)

    def __getitem__(self, idx):
        # De-sparsify the data
        # [ACTIVE_CHANNELS, SAMPLE_LEN]
        sparse_sample = self.sparse_samples[idx]
        dense_sample = torch.zeros(NUM_CHANNELS, sparse_sample.shape[-1])
        dense_sample[self.mask] = sparse_sample
        return dense_sample


class SparseDataset(EEGDataset):
    def __getitem__(self, idx):
        return self.sparse_samples[idx]
