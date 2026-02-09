# Builtin imports
from pathlib import Path

# Internal imports
from constants import NUM_CHANNELS

# External imports
from torch.utils.data import Dataset
import torch
from safetensors.torch import load_file
import polars as pl


class MaskedEEGDataset(Dataset):
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


class SparseDataset(MaskedEEGDataset):
    def __getitem__(self, idx):
        return self.sparse_samples[idx]


def get_metadata(p: Path):
    metadata_file = p.parent / f"{p.stem}-metadata.parquet"
    print(metadata_file)
    return pl.read_parquet(metadata_file)


class SparseMetadataDataset(SparseDataset):
    def __init__(self, samples_path: Path):
        super().__init__(samples_path)
        self.metadata = get_metadata(samples_path)


class SparseClassificationDataset(SparseMetadataDataset):
    def __init__(self, samples_path: Path, class_col: str):
        super().__init__(samples_path)
        self.class_col = class_col

        self.distinct = (
            self.metadata.select(class_col).unique().sort(by=class_col).with_row_index()
        )
        self.vals_to_ids = {}
        for i, row in enumerate(
            self.distinct.select(self.class_col).iter_rows(named=True)
        ):
            self.vals_to_ids[row[self.class_col]] = i

        self.class_dim = len(self.distinct)

        print(f"Classifying on {class_col}, {self.class_dim} classes, {self.distinct}")

    def __getitem__(self, idx):
        tensor = super().__getitem__(idx)
        row = self.metadata[idx]
        row_class = row.select(self.class_col)[0].item()
        row_it = self.vals_to_ids[row_class]
        return tensor, torch.tensor(row_it, dtype=torch.long)


class SparseAudioEmbedDataset(SparseDataset):
    """EEG samples paired with pre-computed audio embeddings."""

    def __init__(self, samples_path: Path, audio_embeds_path: str):
        super().__init__(samples_path)
        print(f"Loading audio embeddings from {audio_embeds_path}...")
        self.audio_embeds = load_file(audio_embeds_path)["audio_embeds"]
        assert len(self.audio_embeds) == len(self.sparse_samples), (
            f"Audio embed count {len(self.audio_embeds)} != EEG count {len(self.sparse_samples)}"
        )

    def __getitem__(self, idx):
        return self.sparse_samples[idx], self.audio_embeds[idx]


class DenseAudioEmbedDataset(Dataset):
    """EEG samples (dense, no mask) paired with audio embeddings."""

    def __init__(self, samples_path: Path, audio_embeds_path: str):
        print(f"Loading samples from {samples_path}...")
        self.samples = load_file(samples_path)["samples"]
        print(f"Loading audio embeddings from {audio_embeds_path}...")
        self.audio_embeds = load_file(audio_embeds_path)["audio_embeds"]
        assert len(self.audio_embeds) == len(self.samples), (
            f"Audio embed count {len(self.audio_embeds)} != EEG count {len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.audio_embeds[idx]


class CombinedAudioEmbedDataset(Dataset):
    """Concatenates multiple audio-embed datasets with identical channel layout."""

    def __init__(self, *datasets):
        self.datasets = datasets
        self.cumulative_sizes = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        for ds_idx, cumsize in enumerate(self.cumulative_sizes):
            if idx < cumsize:
                offset = self.cumulative_sizes[ds_idx - 1] if ds_idx > 0 else 0
                return self.datasets[ds_idx][idx - offset]
        raise IndexError(f"Index {idx} out of range for {len(self)} samples")


class HBNAudioEmbedDataset(Dataset):
    """EEG samples paired with movie audio embeddings via index lookup.

    Efficient for HBN where all subjects watch the same 4 movies —
    stores only ~654 unique audio windows instead of duplicating them
    across ~2,639 subjects (~250 KB vs ~60 GB).
    """

    def __init__(self, samples_path: Path, audio_embeds_path: str):
        print(f"Loading samples from {samples_path}...")
        self.samples = load_file(samples_path)["samples"]

        print(f"Loading movie audio embeddings from {audio_embeds_path}...")
        embeds_data = load_file(audio_embeds_path)
        self.movie_embeds = dict(embeds_data)

        # Load metadata for (task_name, window_idx) lookup
        metadata_path = samples_path.parent / f"{samples_path.stem}-metadata.parquet"
        print(f"Loading metadata from {metadata_path}...")
        meta = pl.read_parquet(metadata_path)
        self.task_names = meta["task_name"].to_list()
        self.window_indices = meta["window_idx"].to_list()

        assert len(self.samples) == len(self.task_names), (
            f"Sample count {len(self.samples)} != metadata rows {len(self.task_names)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        task_name = self.task_names[idx]
        window_idx = self.window_indices[idx]
        audio_embed = self.movie_embeds[task_name][window_idx]
        return self.samples[idx], audio_embed


class ThingsEEGClassificationDataset(Dataset):
    def __init__(self, samples_path: Path, class_col: str):
        self.class_col = class_col

        print(f"Loading samples from {samples_path}...")
        self.samples = load_file(samples_path)["samples"]
        self.metadata = get_metadata(samples_path)
        self.distinct = (
            self.metadata.select(class_col).unique().sort(by=class_col).with_row_index()
        )
        self.vals_to_ids = {}
        for i, row in enumerate(
            self.distinct.select(self.class_col).iter_rows(named=True)
        ):
            self.vals_to_ids[row[self.class_col]] = i

        self.class_dim = len(self.distinct)

        print(f"Classifying on {class_col}, {self.class_dim} classes, {self.distinct}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tensor = self.samples[idx]
        row = self.metadata[idx]

        row_class = row.select(self.class_col)[0].item()
        row_it = self.vals_to_ids[row_class]

        # TODO: probably better to do this with autocast
        return tensor.to(torch.float32), torch.tensor(row_it, dtype=torch.long)
