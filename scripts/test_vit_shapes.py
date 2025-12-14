# Builtin imports
from pathlib import Path

# Internal imports
from models.vit import EEGViT
from data.dataset import EEGDataset


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

    print("Loaded...")
    out = vit(first_sample)
    print("Out shape", out.shape)


if __name__ == "__main__":
    main()
