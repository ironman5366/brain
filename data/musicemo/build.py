"""
Build the DS002721 Music Emotion dataset into safetensors + parquet format.

Processes 31 subjects of 19-channel EEG recorded during music-induced emotion
listening. Extracts music-listening segments from runs 2-5, maps 19 channels
to 32-channel target via IDW spatial interpolation, downsamples to 125 Hz,
windows into 1-second epochs, and normalizes.

Usage:
    uv run python -m data.musicemo.build
"""

from pathlib import Path
import random

import mne
import numpy as np
import polars as pl
import torch
from safetensors.torch import save_file
from tqdm import tqdm

from data.musicemo.channel_mapping import (
    SOURCE_CHANNELS_MODERN,
    TENTWENTY_TO_32CH_MAP,
)
from data.musicemo.songs import (
    CHANNEL_NAMES,
    EVENT_MUSIC_START,
    MUSICEMO_DATA_DIR,
    MUSICEMO_OUTPUT_DIR,
    MUSICEMO_SFREQ,
    MUSIC_RUNS,
    OLD_TO_MODERN_NAMES,
    SUBJECT_IDS,
    TARGET_CHANNELS,
    TARGET_SFREQ,
)

WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = int(TARGET_SFREQ * WINDOW_SECONDS)  # 125
NORMALIZATION = "epoch"
TRAIN_SPLIT = 0.9
EPS = 1e-8

# Music clips are 12 seconds each; the EEG recording window is ~20s but we
# only want the 12-16s of actual music playback.
MUSIC_DURATION_SEC = 12.0


def apply_channel_mapping(eeg_19ch: torch.Tensor) -> torch.Tensor:
    """Apply precomputed 19→32 channel IDW mapping.

    Args:
        eeg_19ch: (19, T) tensor of 19-channel EEG data

    Returns:
        (32, T) tensor of mapped 32-channel EEG data
    """
    n_samples = eeg_19ch.shape[1]
    eeg_32ch = torch.zeros(len(TARGET_CHANNELS), n_samples)

    for t_idx, t_name in enumerate(TARGET_CHANNELS):
        contributors = TENTWENTY_TO_32CH_MAP[t_name]
        for src_idx, weight in contributors:
            eeg_32ch[t_idx] += weight * eeg_19ch[src_idx]

    return eeg_32ch


def load_events(subject_id: str, run: int) -> pl.DataFrame:
    """Load BIDS events.tsv for a subject/run."""
    path = (
        MUSICEMO_DATA_DIR / subject_id / "eeg"
        / f"{subject_id}_task-run{run}_events.tsv"
    )
    return pl.read_csv(path, separator="\t")


def load_eeg(subject_id: str, run: int) -> mne.io.BaseRaw:
    """Load EDF file for a subject/run."""
    path = (
        MUSICEMO_DATA_DIR / subject_id / "eeg"
        / f"{subject_id}_task-run{run}_eeg.edf"
    )
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
    return raw


def find_music_trials(events_df: pl.DataFrame) -> list[dict]:
    """Extract music trial info from events.

    Stimulus codes span 302-657 (not just 301-360 as documented). The hundreds
    digit varies by trial; the clip number is always code % 100.
    We identify stimulus codes as any event code 300-699 that isn't a known
    standard event type.

    Returns list of dicts with keys: onset_sec, stim_code, mp3_filename.
    """
    trials = []

    # Known non-stimulus event codes
    known_codes = (
        {257, 259, 263, 786, 788, 1092, 32768, 33568, 33569, 33570,
         33571, 33572, 33573, 33574, 33575}
        | set(range(800, 808))   # question codes
        | set(range(833, 842))   # response codes
        | set(range(901, 910))   # answer codes
    )

    # Find stimulus ID events: codes 300-699 that aren't known standard codes
    stim_events = events_df.filter(
        (pl.col("trial_type") >= 300)
        & (pl.col("trial_type") <= 699)
        & ~pl.col("trial_type").is_in(list(known_codes))
    )

    for row in stim_events.iter_rows(named=True):
        stim_code = int(row["trial_type"])
        onset_sec = float(row["onset"])
        mp3_num = stim_code % 100
        mp3_filename = f"{mp3_num:03d}.mp3"

        # Look for corresponding music playback start (788) near this onset
        music_starts = events_df.filter(
            (pl.col("trial_type") == EVENT_MUSIC_START)
            & (pl.col("onset") >= onset_sec - 1.0)
            & (pl.col("onset") <= onset_sec + 5.0)
        )

        if len(music_starts) > 0:
            # Use the music start onset as the actual playback start
            play_onset = float(music_starts.row(0, named=True)["onset"])
        else:
            # Fall back to the stimulus code onset
            play_onset = onset_sec

        trials.append({
            "onset_sec": play_onset,
            "stim_code": stim_code,
            "mp3_filename": mp3_filename,
        })

    return trials


def process_subject(subject_id: str) -> list[dict]:
    """Process a single subject's EEG data into windowed samples."""
    samples = []

    for run in MUSIC_RUNS:
        eeg_path = (
            MUSICEMO_DATA_DIR / subject_id / "eeg"
            / f"{subject_id}_task-run{run}_eeg.edf"
        )
        if not eeg_path.exists():
            continue

        raw = load_eeg(subject_id, run)
        events_df = load_events(subject_id, run)

        # Rename channels to modern names for consistent ordering
        rename_map = {}
        for ch_name in raw.ch_names:
            if ch_name in OLD_TO_MODERN_NAMES:
                rename_map[ch_name] = OLD_TO_MODERN_NAMES[ch_name]
        if rename_map:
            raw.rename_channels(rename_map)

        # Pick the 19 EEG channels (using modern names)
        raw.pick(SOURCE_CHANNELS_MODERN)

        # Downsample to target rate
        if raw.info["sfreq"] != TARGET_SFREQ:
            raw.resample(TARGET_SFREQ, verbose=False)

        eeg_data = torch.from_numpy(raw.get_data()).float()  # (19, T)

        # Apply 19→32 channel mapping
        eeg_32ch = apply_channel_mapping(eeg_data)  # (32, T)

        # Find music trials in this run
        trials = find_music_trials(events_df)

        for trial in trials:
            onset_sec = trial["onset_sec"]
            start_sample = int(onset_sec * TARGET_SFREQ)
            end_sample = start_sample + int(MUSIC_DURATION_SEC * TARGET_SFREQ)

            if end_sample > eeg_32ch.shape[1]:
                end_sample = eeg_32ch.shape[1]

            if start_sample >= eeg_32ch.shape[1]:
                continue

            segment = eeg_32ch[:, start_sample:end_sample]  # (32, dur_samples)

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
                    "run": run,
                    "mp3_filename": trial["mp3_filename"],
                    "stim_code": trial["stim_code"],
                    "window_idx": w_idx,
                    "window_start_sec": w_idx * WINDOW_SECONDS,
                })

    return samples


def build():
    print("=== DS002721 Music Emotion Dataset Builder ===")

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

    if not all_samples:
        print("  ERROR: No samples produced. Exiting.")
        return

    # Count unique clips
    unique_clips = set(s["mp3_filename"] for s in all_samples)
    print(f"  Unique clips: {len(unique_clips)}")

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

    MUSICEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, split_samples in [("train", train_samples), ("val", val_samples)]:
        if not split_samples:
            continue

        tensors = torch.stack([s["sample"] for s in split_samples])

        save_file(
            {"samples": tensors},
            MUSICEMO_OUTPUT_DIR / f"musicemo-{split_name}.safetensors",
        )

        metadata = pl.from_dicts([
            {k: v for k, v in s.items() if k != "sample"}
            for s in split_samples
        ])
        metadata.write_parquet(
            MUSICEMO_OUTPUT_DIR / f"musicemo-{split_name}-metadata.parquet",
        )

        print(f"  Saved {split_name}: {tensors.shape}")

    print("\nDone!")


if __name__ == "__main__":
    build()
