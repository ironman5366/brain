"""
Build the EAV dataset into safetensors + parquet format.

Processes 42 subjects of 30-channel EEG recorded during conversational emotion
tasks. Loads pre-segmented trials from .mat files, downsamples from 500 Hz to
125 Hz, windows into 1-second epochs, maps channels to the standard 428-channel
sparse grid, and normalizes.

Also extracts audio for every trial into a flat directory:
  - Speaking trials: copies .wav from Audio/ dir
  - Listening trials: extracts audio from .mp4 via ffmpeg

Usage:
    uv run python -m data.eav.build               # sparse 428-grid
    uv run python -m data.eav.build --channels 32  # dense 32ch (for combiner)
"""

import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import scipy.io
import scipy.signal
import torch
from safetensors.torch import save_file
from tqdm import tqdm

from data.eav.songs import (
    CHANNEL_NAMES,
    CONDITION_NAMES,
    EAV_AUDIO_DIR,
    EAV_DATA_DIR,
    EAV_OUTPUT_DIR,
    EAV_SFREQ,
    EMOTION_NAMES,
    SUBJECT_IDS,
    TARGET_SFREQ,
)
from data.eav.channel_mapping import EAV_TO_32CH_MAP
from data.musicemo.songs import TARGET_CHANNELS
from utils import standardize_epochs

WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = int(TARGET_SFREQ * WINDOW_SECONDS)  # 125
NORMALIZATION = "epoch"
TRAIN_SPLIT = 0.9
DOWNSAMPLE_FACTOR = EAV_SFREQ // TARGET_SFREQ  # 4


def audio_filename(subject_id: str, trial_idx: int) -> str:
    """Canonical audio filename for a trial."""
    return f"{subject_id}_trial{trial_idx:03d}.wav"


def prepare_audio():
    """Extract/copy audio for all trials into EAV_AUDIO_DIR.

    Speaking trials: copy .wav from subject's Audio/ dir.
    Listening trials: extract audio from .mp4 via ffmpeg.
    """
    EAV_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    n_prepared = 0
    n_skipped = 0
    n_failed = 0

    for subject_id in tqdm(SUBJECT_IDS, desc="Preparing audio"):
        subject_dir = EAV_DATA_DIR / subject_id
        audio_dir = subject_dir / "Audio"
        video_dir = subject_dir / "Video"

        for trial_idx in range(200):
            out_path = EAV_AUDIO_DIR / audio_filename(subject_id, trial_idx)
            if out_path.exists():
                n_skipped += 1
                continue

            prefix = f"{trial_idx + 1:03d}_"

            # Try .wav first (speaking trials)
            found = False
            if audio_dir.exists():
                for f in audio_dir.iterdir():
                    if f.name.startswith(prefix) and f.suffix == ".wav":
                        shutil.copy2(f, out_path)
                        found = True
                        break

            # Fall back to extracting from .mp4 (listening trials)
            if not found and video_dir.exists():
                for f in video_dir.iterdir():
                    if f.name.startswith(prefix) and f.suffix == ".mp4":
                        result = subprocess.run(
                            [
                                "ffmpeg", "-i", str(f),
                                "-vn", "-acodec", "pcm_s16le",
                                "-ar", "44100", "-ac", "1",
                                "-y", str(out_path),
                            ],
                            capture_output=True,
                        )
                        if result.returncode == 0:
                            found = True
                        else:
                            n_failed += 1
                        break

            if found:
                n_prepared += 1
            elif not out_path.exists():
                n_failed += 1

    print(f"  Audio: {n_prepared} prepared, {n_skipped} already existed, {n_failed} failed")


def load_subject(subject_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load EEG data and labels for a subject.

    Returns:
        eeg: (n_trials, n_channels, n_times) array
        labels: (n_trials,) array of label indices 0-9
    """
    eeg_dir = EAV_DATA_DIR / subject_id / "EEG"
    eeg_path = eeg_dir / f"{subject_id}_eeg.mat"
    label_path = eeg_dir / f"{subject_id}_eeg_label.mat"

    mat = scipy.io.loadmat(str(eeg_path))
    # Try 'seg1' first, fall back to 'seg' (matches reference code)
    eeg = mat.get("seg1")
    if eeg is None or np.ndim(eeg) != 3:
        eeg = mat["seg"]
    # eeg shape: (n_times, n_channels, n_trials) = (10000, 30, 200)

    mat_y = scipy.io.loadmat(str(label_path))
    label_onehot = mat_y["label"]  # (10, n_trials)
    labels = np.argmax(label_onehot, axis=0)  # (n_trials,)

    # Transpose to (n_trials, n_channels, n_times)
    eeg = np.transpose(eeg, (2, 1, 0))

    return eeg, labels


def process_subject(subject_id: str) -> list[dict]:
    """Process a single subject's EEG into windowed, standardized samples."""
    eeg, labels = load_subject(subject_id)
    n_trials, n_channels, n_times = eeg.shape

    # Downsample from 500 Hz to 125 Hz
    eeg_ds = scipy.signal.resample_poly(eeg, up=1, down=DOWNSAMPLE_FACTOR, axis=2)
    # eeg_ds shape: (n_trials, 30, 2500)

    n_times_ds = eeg_ds.shape[2]
    n_windows = n_times_ds // WINDOW_SAMPLES  # 2500 / 125 = 20

    samples = []

    for trial_idx in range(n_trials):
        label_idx = int(labels[trial_idx])
        trial_eeg = eeg_ds[trial_idx]  # (30, 2500)

        # Window into 1-second epochs: (n_windows, 30, 125)
        usable = n_windows * WINDOW_SAMPLES
        windowed = trial_eeg[:, :usable].reshape(
            n_channels, n_windows, WINDOW_SAMPLES
        )
        windowed = np.transpose(windowed, (1, 0, 2))  # (n_windows, 30, 125)
        windowed = torch.from_numpy(windowed.copy()).to(torch.float32)

        # Map to standard 428-channel grid + normalize
        standardized, mask = standardize_epochs(
            windowed,
            CHANNEL_NAMES,
            normalization=NORMALIZATION,
        )

        # Extract sparse representation
        mask_indices = torch.where(mask)[0]
        sparse_windows = standardized[:, mask, :]  # (n_windows, C_active, 125)

        for w_idx in range(n_windows):
            samples.append({
                "sample": sparse_windows[w_idx],
                "subject_id": subject_id,
                "trial_idx": trial_idx,
                "window_idx": w_idx,
                "window_start_sec": w_idx * WINDOW_SECONDS,
                "emotion": EMOTION_NAMES[label_idx],
                "condition": CONDITION_NAMES[label_idx],
                "label_idx": label_idx,
                "audio_filename": audio_filename(subject_id, trial_idx),
            })

    return samples, mask


def apply_channel_mapping(eeg_30ch: torch.Tensor) -> torch.Tensor:
    """Apply precomputed 30->32 channel IDW mapping.

    Args:
        eeg_30ch: (30, T) tensor of 30-channel EEG data

    Returns:
        (32, T) tensor of mapped 32-channel EEG data
    """
    n_samples = eeg_30ch.shape[1]
    eeg_32ch = torch.zeros(len(TARGET_CHANNELS), n_samples)

    for t_idx, t_name in enumerate(TARGET_CHANNELS):
        contributors = EAV_TO_32CH_MAP[t_name]
        for src_idx, weight in contributors:
            eeg_32ch[t_idx] += weight * eeg_30ch[src_idx]

    return eeg_32ch


def process_subject_32ch(subject_id: str) -> list[dict]:
    """Process a single subject's EEG into 32-channel windowed samples."""
    eeg, labels = load_subject(subject_id)
    n_trials, n_channels, n_times = eeg.shape

    eeg_ds = scipy.signal.resample_poly(eeg, up=1, down=DOWNSAMPLE_FACTOR, axis=2)
    n_times_ds = eeg_ds.shape[2]
    n_windows = n_times_ds // WINDOW_SAMPLES

    samples = []
    EPS = 1e-8

    for trial_idx in range(n_trials):
        label_idx = int(labels[trial_idx])
        trial_eeg = torch.from_numpy(eeg_ds[trial_idx].copy()).to(torch.float32)

        # Map 30ch -> 32ch
        eeg_32ch = apply_channel_mapping(trial_eeg)  # (32, 2500)

        # Window into 1-second epochs
        usable = n_windows * WINDOW_SAMPLES
        windowed = eeg_32ch[:, :usable].reshape(32, n_windows, WINDOW_SAMPLES)
        windowed = windowed.permute(1, 0, 2)  # (n_windows, 32, 125)

        # Per-epoch normalization
        if NORMALIZATION == "epoch":
            mean = windowed.mean(dim=-1, keepdim=True)
            std = windowed.std(dim=-1, keepdim=True)
            windowed = (windowed - mean) / (std + EPS)

        for w_idx in range(n_windows):
            samples.append({
                "sample": windowed[w_idx],
                "subject_id": subject_id,
                "trial_idx": trial_idx,
                "window_idx": w_idx,
                "window_start_sec": w_idx * WINDOW_SECONDS,
                "emotion": EMOTION_NAMES[label_idx],
                "condition": CONDITION_NAMES[label_idx],
                "label_idx": label_idx,
                "audio_filename": audio_filename(subject_id, trial_idx),
            })

    return samples


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

    metadata_df = pl.from_dicts([
        {k: v for k, v in s.items() if k != "sample"} for s in samples
    ])
    metadata_df.write_parquet(metadata_path)

    print(f"Saved {tensors.shape} samples to {safetensors_path}")
    print(f"Saved metadata to {metadata_path}")


def save_split_dense(
    samples: list[dict],
    safetensors_path: Path,
    metadata_path: Path,
):
    """Save dense (no mask) samples to safetensors + parquet."""
    tensors = torch.stack([s["sample"] for s in samples])

    save_file({"samples": tensors}, safetensors_path)

    metadata_df = pl.from_dicts([
        {k: v for k, v in s.items() if k != "sample"} for s in samples
    ])
    metadata_df.write_parquet(metadata_path)

    print(f"Saved {tensors.shape} samples to {safetensors_path}")
    print(f"Saved metadata to {metadata_path}")


def split_and_save(all_samples, save_fn, prefix):
    """Subject-level train/val split and save."""
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

    EAV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_fn(
        train_samples,
        EAV_OUTPUT_DIR / f"{prefix}-train.safetensors",
        EAV_OUTPUT_DIR / f"{prefix}-train-metadata.parquet",
    )

    save_fn(
        val_samples,
        EAV_OUTPUT_DIR / f"{prefix}-val.safetensors",
        EAV_OUTPUT_DIR / f"{prefix}-val-metadata.parquet",
    )


def build(channels: int = 30):
    print(f"=== EAV Dataset Builder ({channels}ch) ===")
    print(f"  Source: {EAV_DATA_DIR}")
    print(f"  Output: {EAV_OUTPUT_DIR}")
    print(f"  Resample: {EAV_SFREQ} Hz -> {TARGET_SFREQ} Hz")

    # Prepare audio files
    print("\n[1/3] Preparing audio files...")
    prepare_audio()

    # Process all subjects
    print(f"\n[2/3] Processing {len(SUBJECT_IDS)} subjects...")
    all_samples = []

    if channels == 32:
        for subject_id in tqdm(SUBJECT_IDS, desc="Subjects (32ch)"):
            try:
                subject_samples = process_subject_32ch(subject_id)
                all_samples.extend(subject_samples)
            except Exception as e:
                print(f"  WARNING: Failed to process {subject_id}: {e}")

        print(f"  Total samples: {len(all_samples):,}")
        print(f"  Channels: 32 (IDW-mapped)")

        # Emotion/condition distribution
        emotions = {}
        for s in all_samples:
            key = f"{s['emotion']}_{s['condition']}"
            emotions[key] = emotions.get(key, 0) + 1
        for k, v in sorted(emotions.items()):
            print(f"    {k}: {v:,}")

        print("\n[3/3] Splitting and saving...")
        split_and_save(all_samples, save_split_dense, "eav-32ch")
    else:
        mask = None
        for subject_id in tqdm(SUBJECT_IDS, desc="Subjects"):
            try:
                subject_samples, subject_mask = process_subject(subject_id)
                all_samples.extend(subject_samples)

                if mask is None:
                    mask = subject_mask
                else:
                    assert torch.equal(mask, subject_mask), (
                        f"Mask mismatch for {subject_id}"
                    )
            except Exception as e:
                print(f"  WARNING: Failed to process {subject_id}: {e}")

        if not all_samples:
            print("  ERROR: No samples produced. Exiting.")
            return

        mask_indices = torch.where(mask)[0]
        print(f"  Total samples: {len(all_samples):,}")
        print(f"  Active channels: {mask_indices.shape[0]}")

        # Emotion/condition distribution
        emotions = {}
        for s in all_samples:
            key = f"{s['emotion']}_{s['condition']}"
            emotions[key] = emotions.get(key, 0) + 1
        for k, v in sorted(emotions.items()):
            print(f"    {k}: {v:,}")

        print("\n[3/3] Splitting and saving...")

        def save_fn(samples, st_path, meta_path):
            save_split(samples, mask_indices, st_path, meta_path)

        split_and_save(all_samples, save_fn, "eav")

    print("\nDone!")


if __name__ == "__main__":
    import sys
    ch = 32 if "--channels" in sys.argv and "32" in sys.argv else 30
    build(channels=ch)
