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


class ContinuousAudioEmbedDataset(Dataset):
    """Multi-second EEG sequences paired with audio embeddings.

    EEG: (N, W, C, T) — W seconds of C channels with T samples each.
    Audio: (N, W, 128, 75) — W seconds of EnCodec embeddings.
    Optionally includes sequence lengths for variable-length (full-song) data.
    """

    def __init__(self, samples_path: Path, audio_embeds_path: str, lengths_path: str | None = None):
        super().__init__()
        print(f"Loading continuous samples from {samples_path}...")
        samples_data = load_file(samples_path)
        self.sparse_samples = samples_data["sparse_samples"]  # (N, W, C, T)

        print(f"Loading continuous audio embeddings from {audio_embeds_path}...")
        self.audio_embeds = load_file(audio_embeds_path)["audio_embeds"]  # (N, W, 128, 75)

        assert len(self.audio_embeds) == len(self.sparse_samples), (
            f"Audio embed count {len(self.audio_embeds)} != EEG count {len(self.sparse_samples)}"
        )

        # Variable-length support
        self.lengths = None
        if lengths_path:
            lengths_p = Path(lengths_path)
            if lengths_p.exists():
                print(f"Loading sequence lengths from {lengths_path}...")
                self.lengths = load_file(lengths_path)["lengths"]  # (N,)

        print(f"  EEG shape: {self.sparse_samples.shape}")
        print(f"  Audio shape: {self.audio_embeds.shape}")
        if self.lengths is not None:
            print(f"  Lengths: min={self.lengths.min().item()}, max={self.lengths.max().item()}")

    def __len__(self):
        return len(self.sparse_samples)

    def __getitem__(self, idx):
        eeg = self.sparse_samples[idx]      # (W, C, T)
        audio = self.audio_embeds[idx]       # (W, 128, 75)
        if self.lengths is not None:
            return eeg, audio, self.lengths[idx]
        return eeg, audio


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
