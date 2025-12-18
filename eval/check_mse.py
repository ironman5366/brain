# Builtin imports
from pathlib import Path

# Internal imports
from data.dataset import EEGDataset, SparseDataset
from models.et import EEGMAE

# External imports
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

BATCH_SIZE = 256

ALLJOINED_BASE_PATH = Path(
    "/kreka/research/willy/side/brain_datasets/alljoined-window-2025-12-14/"
)
TRAIN_PATH = ALLJOINED_BASE_PATH / "alljoined-window-2025-12-14-train.safetensors"
VAL_PATH = ALLJOINED_BASE_PATH / "alljoined-window-2025-12-14-val.safetensors"

DEVICE = "cuda"


def main():
    # checkpoint_path = "checkpoints/alljoined_et_v1_2025_12_15/final"
    checkpoint_path = "checkpoints/alljoined_channel_v1_2025_12_17/final"

    with torch.inference_mode():
        print(f"Loading checkpoint from {checkpoint_path}...")
        m = EEGMAE.from_pretrained(checkpoint_path).to(DEVICE)

        print("Starting dataset...")
        ds = SparseDataset(
            samples_path=VAL_PATH,
        )
        dl = DataLoader(ds, num_workers=8, shuffle=True, batch_size=BATCH_SIZE)

        losses = []
        for batch in tqdm(dl):
            batch = batch.to(DEVICE)
            loss = m(batch)["loss"]
            print(f"Batch {batch.shape} - {loss:.3f}")
            losses.append(loss)

        avg = sum(losses) / len(losses)
        print(f"Avg loss over {len(ds)} - {avg}")


if __name__ == "__main__":
    main()
