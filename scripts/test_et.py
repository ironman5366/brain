# Builtin imports
from pathlib import Path

# Internal imports
from data.dataset import EEGDataset
from models.et import EEGMAE, EEGMAEConfig

# External imports
import torch
from safetensors.torch import load_file


def main():
    data_path = Path(
        "/kreka/research/willy/side/brain_datasets/alljoined-2025-12-13/alljoined-2025-12-13-val.safetensors"
    )
    ds = EEGDataset(data_path)

    first_samples = torch.stack([ds[i] for i in range(0, 4)])
    print("sample shape", first_samples.shape)

    config = EEGMAEConfig(dim=1024, depth=6, heads=8, mlp_dim=2048, dim_head=64)
    m = EEGMAE.from_config(config)

    m(first_samples)


if __name__ == "__main__":
    main()
