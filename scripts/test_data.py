# Internal imports
from data.dataset import EEGDataset

# External imports
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():
    print("init...")
    ds = EEGDataset(
        path="/kreka/research/willy/side/brain_datasets/alljoined-initial/alljoined-initial-train.parquet"
    )
    print("dl...")
    dl = DataLoader(ds, batch_size=64, num_workers=8)
    n = next(dl)
    print(f"N is {n.shape}")


if __name__ == "__main__":
    main()
