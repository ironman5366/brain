# Builtin imports
from pathlib import Path

# Internal imports
from data.dataset import MaskedEEGDataset, SparseDataset
from models.et import EEGMAE, EEGMAEConfig

# External imports
import torch
from safetensors.torch import load_file


def main():
    data_path = Path(
        "/kreka/research/willy/side/brain_datasets/alljoined-2025-12-13/alljoined-2025-12-13-val.safetensors"
    )
    ds = SparseDataset(data_path)

    first_samples = torch.stack([ds[i] for i in range(0, 4)])
    print("sample shape", first_samples.shape)

    config = EEGMAEConfig(
        encoder_dim=1024,
        encoder_mlp_dim=4096,
        decoder_dim=512,
        decoder_mlp_dim=2048,
        masking_ratio=0.5,
        depth=6,
        heads=16,
        dim_head=64,
        mask_on="samples",
        max_tokens=256,
        sequence_len=32,
    )
    m = EEGMAE.from_config(config)

    m(first_samples)


if __name__ == "__main__":
    main()
