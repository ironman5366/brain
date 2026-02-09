"""
Build the DS005876 Song Familiarity dataset into safetensors + parquet format.

Processes 29 subjects of 32-channel EEG recorded during song familiarity detection.
Extracts music-listening segments, downsamples to 125 Hz, windows into 1-second
epochs, and normalizes.

Usage:
    uv run python -m data.songfam.build
"""

from pathlib import Path
import random

import mne
import numpy as np
import polars as pl
import torch
from safetensors.torch import save_file
from tqdm import tqdm

from data.songfam.songs import (
    CHANNEL_NAMES,
    SONGFAM_DATA_DIR,
    SONGFAM_OUTPUT_DIR,
    SONGFAM_SFREQ,
    SUBJECT_IDS,
    TARGET_SFREQ,
)

WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = int(TARGET_SFREQ * WINDOW_SECONDS)  # 125
NORMALIZATION = "epoch"
TRAIN_SPLIT = 0.9
EPS = 1e-8


def load_events(subject_id: str) -> pl.DataFrame:
    """Load BIDS events.tsv for a subject."""
    path = (
        SONGFAM_DATA_DIR / subject_id / "eeg"
        / f"{subject_id}_task-songfamiliarity_events.tsv"
    )
    return pl.read_csv(path, separator="\t")


def load_beh(subject_id: str) -> pl.DataFrame:
    """Load behavioral data for a subject."""
    path = (
        SONGFAM_DATA_DIR / subject_id / "beh"
        / f"{subject_id}_task-songfamiliarity_beh.tsv"
    )
    return pl.read_csv(path, separator="\t")


def load_eeg(subject_id: str) -> mne.io.BaseRaw:
    """Load EEGLAB .set file for a subject."""
    path = (
        SONGFAM_DATA_DIR / subject_id / "eeg"
        / f"{subject_id}_task-songfamiliarity_eeg.set"
    )
    raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose=False)
    return raw


def process_subject(subject_id: str) -> list[dict]:
    """Process a single subject's EEG data into windowed samples."""
    raw = load_eeg(subject_id)
    events_df = load_events(subject_id)
    beh_df = load_beh(subject_id)

    # Downsample to target rate
    if raw.info["sfreq"] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, verbose=False)

    # Pick only the 32 EEG channels (in the target order)
    raw.pick(CHANNEL_NAMES)

    eeg_data = torch.from_numpy(raw.get_data()).float()  # (32, T)

    # Find trial onsets (value == "1" marks trial start with stim_file)
    trial_events = events_df.filter(pl.col("value") == "1")

    samples = []
    for trial_idx in range(len(trial_events)):
        trial_row = trial_events.row(trial_idx, named=True)
        onset_sec = trial_row["onset"]
        stim_file = trial_row["stim_file"]

        if stim_file == "n/a" or stim_file is None:
            continue

        # Find matching behavioral data by trial order
        # beh trials are 1-indexed and sequential
        if trial_idx >= len(beh_df):
            continue
        beh_row = beh_df.row(trial_idx, named=True)

        song_dur = beh_row["songDur"]
        responded = beh_row.get("responded", 0)
        rt = beh_row.get("rt", None)

        # Extract the music-listening segment
        start_sample = int(onset_sec * TARGET_SFREQ)
        end_sample = start_sample + int(song_dur * TARGET_SFREQ)

        if end_sample > eeg_data.shape[1]:
            continue

        segment = eeg_data[:, start_sample:end_sample]  # (32, dur_samples)

        # Window into 1-second epochs
        n_windows = segment.shape[1] // WINDOW_SAMPLES
        if n_windows == 0:
            continue

        usable = n_windows * WINDOW_SAMPLES
        windowed = segment[:, :usable].reshape(32, n_windows, WINDOW_SAMPLES)
        windowed = windowed.permute(1, 0, 2)  # (W, 32, 125)

        # Per-epoch normalization
        if NORMALIZATION == "epoch":
            mean = windowed.mean(dim=-1, keepdim=True)
            std = windowed.std(dim=-1, keepdim=True)
            windowed = (windowed - mean) / (std + EPS)

        for w_idx in range(n_windows):
            samples.append({
                "sample": windowed[w_idx],
                "subject_id": subject_id,
                "song_filename": stim_file,
                "song_dur": float(song_dur),
                "window_idx": w_idx,
                "window_start_sec": w_idx * WINDOW_SECONDS,
                "responded": int(responded) if responded is not None else 0,
                "rt": float(rt) if rt is not None and rt != "n/a" else -1.0,
            })

    return samples


def build():
    print("=== DS005876 Song Familiarity Dataset Builder ===")

    # Process all subjects
    print(f"\n[1/2] Processing {len(SUBJECT_IDS)} subjects...")
    all_samples = []
    for subject_id in tqdm(SUBJECT_IDS, desc="Subjects"):
        try:
            subject_samples = process_subject(subject_id)
            all_samples.extend(subject_samples)
        except Exception as e:
            print(f"  WARNING: Failed to process {subject_id}: {e}")

    print(f"  Total samples: {len(all_samples):,}")

    # Split and save
    print("\n[2/2] Splitting and saving...")
    unique_subjects = list(set(s["subject_id"] for s in all_samples))
    random.seed(42)
    random.shuffle(unique_subjects)

    n_train = int(len(unique_subjects) * TRAIN_SPLIT)
    train_subjects = set(unique_subjects[:n_train])
    val_subjects = set(unique_subjects[n_train:])

    train_samples = [s for s in all_samples if s["subject_id"] in train_subjects]
    val_samples = [s for s in all_samples if s["subject_id"] in val_subjects]

    print(f"  Train: {len(train_samples):,} from {len(train_subjects)} subjects")
    print(f"  Val: {len(val_samples):,} from {len(val_subjects)} subjects")

    SONGFAM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, split_samples in [("train", train_samples), ("val", val_samples)]:
        if not split_samples:
            continue

        tensors = torch.stack([s["sample"] for s in split_samples])

        save_file(
            {"samples": tensors},
            SONGFAM_OUTPUT_DIR / f"songfam-{split_name}.safetensors",
        )

        metadata = pl.from_dicts([
            {k: v for k, v in s.items() if k != "sample"}
            for s in split_samples
        ])
        metadata.write_parquet(
            SONGFAM_OUTPUT_DIR / f"songfam-{split_name}-metadata.parquet",
        )

        print(f"  Saved {split_name}: {tensors.shape}")

    print("\nDone!")


if __name__ == "__main__":
    build()
