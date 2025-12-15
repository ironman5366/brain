# Builtin imports
from pathlib import Path

# Internal imports
from models.vit import EEGViT
from models.mae import EEGViTMAE
from data.dataset import EEGDataset

# External imports
import torch


def main():
    data_path = Path(
        "/kreka/research/willy/side/brain_datasets/alljoined-2025-12-13/alljoined-2025-12-13-val.safetensors"
    )
    ds = EEGDataset(data_path)
    first_sample = ds[0].unsqueeze(0)
    print(f"First sample shape {first_sample.shape}")

    print("Loading vit...")
    vit = EEGViT(
        sample_len=first_sample.shape[-1],
        patch_len=16,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
    )

    print("Loading MAE...")
    mae = EEGViTMAE(
        encoder=vit,
        decoder_dim=512,
    )

    out = vit(first_sample)
    print("Vit out shape", out.shape)

    mae_out = mae(first_sample)
    print("Mae out", mae_out)

    rand = torch.randn(first_sample.shape)
    mae_rand_out = mae(rand)
    print("Mae rand out", mae_rand_out)


if __name__ == "__main__":
    main()
