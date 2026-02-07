"""
Build the NMED-T dataset into safetensors + parquet format.

Usage:
    uv run python -m data.nmed.build
"""

from pathlib import Path
import random
import shutil
import zipfile

import h5py
import numpy as np
import polars as pl
import scipy.io
import torch
from safetensors.torch import save_file
from tqdm import tqdm

from data.nmed.songs import (
    EGI_CHANNEL_NAMES,
    NMED_AUDIO_DIR,
    NMED_DATA_DIR,
    NMED_MUSIC_DIR,
    NMED_OUTPUT_DIR,
    NMED_SFREQ,
    SONGS,
    SUBJECT_IDS,
)
from utils import standardize_epochs

WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = int(NMED_SFREQ * WINDOW_SECONDS)  # 125
NORMALIZATION = "epoch"
TRAIN_SPLIT = 0.9


def load_behavioral_ratings() -> np.ndarray:
    """Load behavioral ratings (familiarity, enjoyment).

    Returns: (2, 10, 20) array — [familiarity/enjoyment, songs, subjects]
    """
    f = h5py.File(NMED_DATA_DIR / "behavioralRatings.mat", "r")
    ratings = np.array(f["behavioralRatings"])  # (2, 10, 20)
    f.close()
    return ratings


def load_participant_info() -> list[dict]:
    """Load participant demographics."""
    mat = scipy.io.loadmat(NMED_DATA_DIR / "participantInfo.mat")
    info = mat["participantInfo"][0]
    participants = []
    for p in info:
        participants.append({
            "id": str(p["id"][0]),
            "age": int(p["age"][0, 0]),
            "years_training": float(p["nYearsTraining"][0, 0]),
            "weekly_listening": float(p["weeklyListening"][0, 0]),
        })
    return participants


def prepare_audio():
    """Extract and normalize audio files into data/nmed/audio/."""
    NMED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    for song in SONGS:
        out_path = NMED_AUDIO_DIR / f"song_{song.id:02d}.flac"
        if out_path.exists():
            continue

        source = song.audio_source
        if ":" in source:
            # File inside a zip
            zip_name, inner_name = source.split(":", 1)
            zip_path = NMED_MUSIC_DIR / zip_name
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Find the matching entry (handle encoding differences)
                matching = [n for n in zf.namelist() if n.endswith(".flac") and "01 " in n]
                if not matching:
                    print(f"WARNING: No matching FLAC in {zip_path}")
                    continue
                with zf.open(matching[0]) as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        else:
            src_path = NMED_MUSIC_DIR / source
            if src_path.exists():
                shutil.copy2(src_path, out_path)
            else:
                print(f"WARNING: Audio not found: {src_path}")

        print(f"  Audio: {song.title} -> {out_path.name}")


def process_songs(
    ratings: np.ndarray,
    participant_info: list[dict],
) -> list[dict]:
    """Process all imputed song files into windowed samples with metadata."""
    all_samples = []
    mask = None

    participant_map = {p["id"]: p for p in participant_info}

    for song in tqdm(SONGS, desc="Processing songs"):
        mat_path = NMED_DATA_DIR / f"song{song.file_number}_Imputed.mat"
        mat = scipy.io.loadmat(mat_path)

        data_key = f"data{song.file_number}"
        subs_key = f"subs{song.file_number}"

        eeg_data = mat[data_key]  # (125, T, 20)
        subs = mat[subs_key]  # (1, 20) object array of subject IDs

        n_channels, n_times, n_subjects = eeg_data.shape
        n_windows = n_times // WINDOW_SAMPLES

        # Get subject IDs for this song
        sub_ids = [str(subs[0, i][0]) for i in range(n_subjects)]

        # Song-level ratings: familiarity and enjoyment
        song_idx = song.id - 1  # 0-indexed
        familiarity = ratings[0, song_idx]  # (20,)
        enjoyment = ratings[1, song_idx]  # (20,)

        for subj_idx in range(n_subjects):
            sub_id = sub_ids[subj_idx]

            # Extract this subject's continuous EEG: (125, T)
            subj_eeg = eeg_data[:, :, subj_idx]

            # Window into (n_windows, 125, WINDOW_SAMPLES)
            usable_samples = n_windows * WINDOW_SAMPLES
            subj_eeg_trimmed = subj_eeg[:, :usable_samples]
            windows = subj_eeg_trimmed.reshape(
                n_channels, n_windows, WINDOW_SAMPLES
            )  # (125, W, 125)
            windows = windows.transpose(1, 0, 2)  # (W, 125, 125)
            windows = torch.from_numpy(windows.copy()).to(torch.float32)

            # Map EGI channels to standard grid
            standardized, window_mask = standardize_epochs(
                windows,
                EGI_CHANNEL_NAMES,
                normalization=NORMALIZATION,
            )

            if mask is None:
                mask = window_mask
            else:
                assert torch.equal(mask, window_mask), "Mask mismatch between subjects"

            # Extract only active channels (sparse representation)
            mask_indices = torch.where(mask)[0]
            sparse_windows = standardized[:, mask, :]  # (W, C_active, 125)

            for w_idx in range(sparse_windows.shape[0]):
                all_samples.append({
                    "sample": sparse_windows[w_idx],
                    "subject_id": sub_id,
                    "song_id": song.id,
                    "song_name": song.title,
                    "artist": song.artist,
                    "tempo_bpm": song.tempo_bpm,
                    "window_idx": w_idx,
                    "window_start_sec": w_idx * WINDOW_SECONDS,
                    "familiarity": int(familiarity[subj_idx]),
                    "enjoyment": int(enjoyment[subj_idx]),
                })

    return all_samples, mask


def save_split(
    samples: list[dict],
    mask_indices: torch.Tensor,
    safetensors_path: Path,
    metadata_path: Path,
):
    """Stack sample tensors and save to safetensors + parquet."""
    tensors = torch.stack([s["sample"] for s in samples])

    save_file(
        {"sparse_samples": tensors, "mask_indices": mask_indices},
        safetensors_path,
    )

    metadata_df = pl.from_dicts([{k: v for k, v in s.items() if k != "sample"} for s in samples])
    metadata_df.write_parquet(metadata_path)

    print(f"Saved {tensors.shape} samples to {safetensors_path}")
    print(f"Saved metadata to {metadata_path}")


def build():
    print("=== NMED-T Dataset Builder ===")

    # Step 1: Prepare audio files
    print("\n[1/4] Preparing audio files...")
    prepare_audio()

    # Step 2: Load metadata
    print("\n[2/4] Loading metadata...")
    ratings = load_behavioral_ratings()
    participant_info = load_participant_info()
    print(f"  Ratings shape: {ratings.shape}")
    print(f"  Participants: {len(participant_info)}")

    # Step 3: Process songs
    print("\n[3/4] Processing songs...")
    all_samples, mask = process_songs(ratings, participant_info)
    mask_indices = torch.where(mask)[0]
    print(f"  Total samples: {len(all_samples):,}")
    print(f"  Active channels: {mask_indices.shape[0]}")

    # Step 4: Split and save
    print("\n[4/4] Splitting and saving...")
    unique_subjects = list(set(s["subject_id"] for s in all_samples))
    random.seed(42)
    random.shuffle(unique_subjects)

    n_train = int(len(unique_subjects) * TRAIN_SPLIT)
    train_subjects = set(unique_subjects[:n_train])
    val_subjects = set(unique_subjects[n_train:])

    train_samples = [s for s in all_samples if s["subject_id"] in train_subjects]
    val_samples = [s for s in all_samples if s["subject_id"] in val_subjects]

    print(f"  Train: {len(train_samples):,} samples from {len(train_subjects)} subjects")
    print(f"  Val: {len(val_samples):,} samples from {len(val_subjects)} subjects")

    NMED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_split(
        train_samples,
        mask_indices,
        NMED_OUTPUT_DIR / "nmed-train.safetensors",
        NMED_OUTPUT_DIR / "nmed-train-metadata.parquet",
    )

    save_split(
        val_samples,
        mask_indices,
        NMED_OUTPUT_DIR / "nmed-val.safetensors",
        NMED_OUTPUT_DIR / "nmed-val-metadata.parquet",
    )

    print("\nDone!")


if __name__ == "__main__":
    build()
