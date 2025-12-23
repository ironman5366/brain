# Builtin imports
from pathlib import Path
import pickle as pkl
import random
from typing import Generator

# External imports
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import polars as pl
from safetensors.torch import save_file, load_file

# Constants
DEAP_RAW_DIR = Path(
    "/kreka/research/willy/side/brain_datasets/deap/deap-dataset/data_preprocessed_python"
)
DEAP_OUT_DIR = Path("/kreka/research/willy/side/brain_datasets/deap_processed")
DEAP_N_CHANNELS = 32  # Keep only first 32 EEG channels
DEAP_BASELINE_SAMPLES = 384  # First 384 samples are baseline
DEAP_WINDOW_SIZE = 128  # Chunk into windows of 128 samples
LABEL_THRESHOLD = 5.0  # Threshold for binary classification


def fetch_deap(data_dir: Path = DEAP_RAW_DIR) -> Generator[Path, None, None]:
    """
    Yield file paths for all DEAP subject pkl files.
    """
    for f in sorted(data_dir.glob("s*.dat")):
        yield f


def load_deap_file(file_path: Path) -> list[dict]:
    """
    Load a single DEAP pkl file and return list of windowed sample dicts.

    Each pkl file contains:
    - 'data': (40, 40, 8064) = 40 trials x 40 channels x 8064 samples
    - 'labels': (40, 4) = 40 trials x 4 labels (valence, arousal, dominance, liking)

    Processing:
    - Keep only first 32 EEG channels
    - Remove first 384 baseline samples
    - Chunk remaining samples into windows of 128
    """
    # Load pkl with proper encoding
    with open(file_path, "rb") as f:
        sample = pkl.load(f, encoding="iso-8859-1")

    data = sample["data"]  # (40, 40, 8064)
    labels = sample["labels"]  # (40, 4)

    # Extract subject ID from filename (e.g., "s01" from "s01.dat")
    subject_id = file_path.stem

    # Process data:
    # 1. Keep only first 32 EEG channels
    # 2. Remove first 384 baseline samples
    data = data[:, :DEAP_N_CHANNELS, DEAP_BASELINE_SAMPLES:]  # (40, 32, 7680)

    n_trials, n_channels, n_samples = data.shape
    n_windows = n_samples // DEAP_WINDOW_SIZE  # 7680 / 128 = 60

    samples = []

    for trial_id in range(n_trials):
        trial_data = data[trial_id]  # (32, 7680)
        trial_labels = labels[trial_id]  # (4,)

        valence, arousal, dominance, liking = trial_labels

        for window_id in range(n_windows):
            start = window_id * DEAP_WINDOW_SIZE
            end = start + DEAP_WINDOW_SIZE

            window_data = trial_data[:, start:end]  # (32, 128)
            window_tensor = torch.from_numpy(window_data).to(torch.float32)

            samples.append(
                {
                    "sample": window_tensor,
                    "subject_id": subject_id,
                    "trial_id": trial_id,
                    "window_id": window_id,
                    # Continuous labels
                    "valence": float(valence),
                    "arousal": float(arousal),
                    "dominance": float(dominance),
                    "liking": float(liking),
                    # Binary labels
                    "valence_high": valence >= LABEL_THRESHOLD,
                    "arousal_high": arousal >= LABEL_THRESHOLD,
                    "dominance_high": dominance >= LABEL_THRESHOLD,
                    "liking_high": liking >= LABEL_THRESHOLD,
                }
            )

    return samples


def save_split(
    samples: list[dict],
    safetensors_path: Path,
    metadata_path: Path,
):
    """
    Stack sample tensors and save to safetensors, save metadata to parquet.
    """
    # Stack all sample tensors
    tensors = torch.stack([s["sample"] for s in samples])

    # Save tensors to safetensors
    save_file({"samples": tensors}, safetensors_path)

    # Build metadata (exclude tensor field)
    metadata_records = [{k: v for k, v in s.items() if k != "sample"} for s in samples]
    metadata_df = pl.from_dicts(metadata_records)
    metadata_df.write_parquet(metadata_path)

    print(f"Saved {tensors.shape} samples to {safetensors_path}")
    print(f"Saved {len(metadata_df)} metadata rows to {metadata_path}")


def build_deap(
    data_dir: Path = DEAP_RAW_DIR,
    out_dir: Path = DEAP_OUT_DIR,
    train_split: float = 0.9,
):
    """
    Build DEAP dataset: load all subjects, split by subject, save to safetensors + parquet.
    """
    all_samples = []

    # Load all samples from all subjects
    for file_path in tqdm(list(fetch_deap(data_dir)), desc="Loading DEAP subjects"):
        file_samples = load_deap_file(file_path)
        all_samples.extend(file_samples)

    print(f"Loaded {len(all_samples):,} total windows")

    # Subject-level splitting
    unique_subjects = list(set(s["subject_id"] for s in all_samples))
    random.shuffle(unique_subjects)

    n_train = int(len(unique_subjects) * train_split)
    train_subjects = set(unique_subjects[:n_train])
    val_subjects = set(unique_subjects[n_train:])

    train_samples = [s for s in all_samples if s["subject_id"] in train_subjects]
    val_samples = [s for s in all_samples if s["subject_id"] in val_subjects]

    print(f"Train: {len(train_samples):,} windows from {len(train_subjects)} subjects")
    print(f"Val: {len(val_samples):,} windows from {len(val_subjects)} subjects")

    # Create output directory
    out_dir.mkdir(exist_ok=True, parents=True)

    # Save splits
    save_split(
        train_samples,
        out_dir / "train.safetensors",
        out_dir / "train-metadata.parquet",
    )

    save_split(
        val_samples,
        out_dir / "val.safetensors",
        out_dir / "val-metadata.parquet",
    )

    print(f"Done! Output saved to {out_dir}")


class DEAPDataset(Dataset):
    """
    PyTorch Dataset for DEAP data stored in safetensors + parquet format.
    """

    def __init__(self, built_dir: Path, split: str):
        """
        Args:
            built_dir: Directory containing the built dataset
            split: Either "train" or "val"
        """
        samples_file = built_dir / f"{split}.safetensors"
        metadata_file = built_dir / f"{split}-metadata.parquet"

        assert samples_file.exists(), f"Samples file not found: {samples_file}"
        assert metadata_file.exists(), f"Metadata file not found: {metadata_file}"

        self.samples = load_file(samples_file)["samples"].to(dtype=torch.float32)
        self.metadata = pl.read_parquet(metadata_file)

        print(f"Loaded {len(self.samples)} samples from {samples_file}")
        print(f"Loaded {len(self.metadata)} metadata rows from {metadata_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {"sample": self.samples[idx], "row": self.metadata[idx]}


class DEAPRegressionDataset(DEAPDataset):
    """
    DEAP Dataset for regression tasks with continuous labels.
    Returns (tensor, label) tuples where label is a float tensor.
    """

    def __init__(self, built_dir: Path, split: str, label_col: str = "valence"):
        """
        Args:
            built_dir: Directory containing the built dataset
            split: Either "train" or "val"
            label_col: Column to use for regression (e.g., "valence", "arousal", "dominance", "liking")
        """
        super().__init__(built_dir, split)
        self.label_col = label_col

        # Pre-extract labels as tensor for efficiency
        self.labels = torch.tensor(
            self.metadata[label_col].to_list(), dtype=torch.float32
        )

        print(f"Regression on {label_col}, range: [{self.labels.min():.2f}, {self.labels.max():.2f}]")

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


class DEAPMultiRegressionDataset(DEAPDataset):
    """
    DEAP Dataset for multi-output regression (all 4 emotion dimensions).
    Returns (tensor, labels) where labels is a float tensor of shape (4,).
    """

    LABEL_COLS = ["valence", "arousal", "dominance", "liking"]

    def __init__(self, built_dir: Path, split: str):
        """
        Args:
            built_dir: Directory containing the built dataset
            split: Either "train" or "val"
        """
        super().__init__(built_dir, split)

        # Pre-extract all labels as tensor [N, 4]
        self.labels = torch.tensor(
            [self.metadata[col].to_list() for col in self.LABEL_COLS],
            dtype=torch.float32,
        ).T  # Transpose to [N, 4]

        print(f"Multi-regression on {self.LABEL_COLS}")
        for i, col in enumerate(self.LABEL_COLS):
            print(f"  {col}: [{self.labels[:, i].min():.2f}, {self.labels[:, i].max():.2f}]")

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


class DEAPClassificationDataset(DEAPDataset):
    """
    DEAP Dataset for classification tasks (binary labels).
    Returns (tensor, class_id) tuples.
    """

    def __init__(self, built_dir: Path, split: str, class_col: str = "valence_high"):
        """
        Args:
            built_dir: Directory containing the built dataset
            split: Either "train" or "val"
            class_col: Column to use for classification (e.g., "valence_high", "arousal_high")
        """
        super().__init__(built_dir, split)
        self.class_col = class_col

        # Build class mapping
        self.distinct = (
            self.metadata.select(class_col).unique().sort(by=class_col).with_row_index()
        )
        self.vals_to_ids = {}
        for row in self.distinct.select(self.class_col).iter_rows(named=True):
            self.vals_to_ids[row[self.class_col]] = len(self.vals_to_ids)

        self.class_dim = len(self.distinct)

        print(f"Classifying on {class_col}, {self.class_dim} classes: {self.distinct}")

    def __getitem__(self, idx):
        tensor = self.samples[idx]
        row = self.metadata[idx]

        row_class = row.select(self.class_col)[0].item()
        class_id = self.vals_to_ids[row_class]

        return tensor, torch.tensor(class_id, dtype=torch.long)


if __name__ == "__main__":
    build_deap()
