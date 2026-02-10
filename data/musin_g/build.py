"""
Build the MUSIN-G (ds003774) dataset into safetensors + parquet format.

Processes 20 subjects × 12 sessions (one song per session) of 128-channel EGI
EEG recorded during passive music listening. Maps to 32 channels via IDW spatial
interpolation (same EGI system as NMED-T), downsamples to 125 Hz, windows into
1-second epochs, and normalizes.

Usage:
    uv run python -m data.musin_g.build
"""

import random
from pathlib import Path

import mne
import numpy as np
import polars as pl
import torch
from safetensors.torch import save_file
from tqdm import tqdm

from data.musin_g.songs import (
    EGI_CHANNEL_NAMES,
    MUSIN_G_BEH_PATH,
    MUSIN_G_DATA_DIR,
    MUSIN_G_OUTPUT_DIR,
    MUSIN_G_SFREQ,
    SONGS,
    SONGS_BY_ID,
    SUBJECT_IDS,
    TARGET_SFREQ,
)
from data.songfam.channel_mapping import EGI_TO_32CH_MAP, TARGET_CHANNELS
from utils import map_egi_to_32ch

WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = int(TARGET_SFREQ * WINDOW_SECONDS)  # 125
NORMALIZATION = "epoch"
TRAIN_SPLIT = 0.9
EPS = 1e-8


def load_behavioral_data() -> dict[tuple[int, int], dict]:
    """Load behavioral ratings from stimuli/Behavioural_data.

    Returns:
        Dict mapping (subject_num, song_id) to {enjoyment, familiarity}.
    """
    rows = {}
    with open(MUSIN_G_BEH_PATH) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            subj = int(parts[0])
            song_id = int(parts[1])
            enjoyment = int(parts[2])
            familiarity = int(parts[3])
            rows[(subj, song_id)] = {
                "enjoyment": enjoyment,
                "familiarity": familiarity,
            }
    return rows


def find_set_file(subject_id: str, session_num: int) -> Path:
    """Find the .set file for a given subject and session."""
    ses_dir = MUSIN_G_DATA_DIR / subject_id / f"ses-{session_num:02d}" / "eeg"
    # Run number matches session number
    pattern = f"{subject_id}_ses-{session_num:02d}_task-MusicListening_run-*_eeg.set"
    matches = list(ses_dir.glob(pattern))
    if len(matches) == 1:
        return matches[0]
    # Fallback: try any .set file in the directory
    matches = list(ses_dir.glob("*.set"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No .set file found in {ses_dir}")


def find_events_file(subject_id: str, session_num: int) -> Path:
    """Find the events.tsv file for a given subject and session."""
    ses_dir = MUSIN_G_DATA_DIR / subject_id / f"ses-{session_num:02d}" / "eeg"
    matches = list(ses_dir.glob("*_events.tsv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No events.tsv found in {ses_dir}")


def parse_music_segment(
    events_path: Path, n_total_samples: int, song_duration_sec: int,
) -> tuple[float, float]:
    """Parse events.tsv to find the music-listening segment.

    The events files contain the full experiment log. We filter to events
    within this recording's sample range and find the music onset (stim)
    and offset (opyp) markers. If no offset marker is found, falls back
    to onset + known song duration.

    Returns:
        (onset_sec, offset_sec) of the music segment.
    """
    events = pl.read_csv(events_path, separator="\t")

    # Filter to events within this recording's sample range
    events = events.filter(pl.col("sample") < n_total_samples)

    # Music onset: 'stim' marker (first trial uses 'stim', later trials use 'stm+')
    onset_rows = events.filter(
        pl.col("value").is_in(["stim", "stm+"])
    )
    if len(onset_rows) == 0:
        raise ValueError(f"No music onset event (stim/stm+) found in {events_path}")

    onset_sec = onset_rows["onset"][0]

    # Music offset: 'opyp' or 'fxnd' marker
    offset_rows = events.filter(
        pl.col("value").is_in(["opyp", "fxnd"])
    )
    if len(offset_rows) > 0:
        offset_sec = offset_rows["onset"][0]
    else:
        # Fallback: use onset + known song duration
        offset_sec = onset_sec + song_duration_sec

    return float(onset_sec), float(offset_sec)


def process_session(
    subject_id: str,
    session_num: int,
    song: "Song",
    beh_data: dict,
) -> list[dict]:
    """Process one session (one song) for one subject."""
    set_path = find_set_file(subject_id, session_num)
    events_path = find_events_file(subject_id, session_num)

    # Load EEG
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)

    n_total_samples = raw.n_times

    # Parse events for music segment
    onset_sec, offset_sec = parse_music_segment(
        events_path, n_total_samples, song.duration_sec,
    )

    # Downsample to target rate
    if raw.info["sfreq"] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, verbose=False)

    # Pick EGI channels: E1-E124, rename E129->Cz, drop E125-E128
    raw_ch_names = raw.ch_names
    keep_channels = []
    rename_map = {}
    for ch in raw_ch_names:
        if ch == "E129":
            keep_channels.append(ch)
            rename_map[ch] = "Cz"
        elif ch.startswith("E"):
            num = int(ch[1:])
            if 1 <= num <= 124:
                keep_channels.append(ch)

    raw.pick(keep_channels)
    if rename_map:
        raw.rename_channels(rename_map)

    # Adjust onset/offset for downsampled rate
    onset_sample = int(onset_sec * TARGET_SFREQ)
    offset_sample = int(offset_sec * TARGET_SFREQ)

    # Clamp to recording bounds
    offset_sample = min(offset_sample, raw.n_times)
    if onset_sample >= offset_sample:
        return []

    # Extract music segment: (125, T_music)
    eeg_data = torch.from_numpy(
        raw.get_data()[:, onset_sample:offset_sample]
    ).float()

    n_channels = eeg_data.shape[0]

    # Window into 1-second epochs
    n_windows = eeg_data.shape[1] // WINDOW_SAMPLES
    if n_windows == 0:
        return []

    usable = n_windows * WINDOW_SAMPLES
    windowed = eeg_data[:, :usable].reshape(n_channels, n_windows, WINDOW_SAMPLES)
    windowed = windowed.permute(1, 0, 2)  # (W, 125, 125)

    # Map EGI → 32ch via spatial IDW
    mapped = map_egi_to_32ch(
        windowed, EGI_TO_32CH_MAP, TARGET_CHANNELS,
        normalization=NORMALIZATION,
    )  # (W, 32, 125)

    # Look up behavioral data
    subj_num = int(subject_id.split("-")[1])
    beh = beh_data.get((subj_num, song.id), {})

    samples = []
    for w_idx in range(mapped.shape[0]):
        samples.append({
            "sample": mapped[w_idx],
            "subject_id": subject_id,
            "song_id": song.id,
            "song_name": song.title,
            "artist": song.artist,
            "genre": song.genre,
            "tempo_bpm": song.tempo_bpm,
            "window_idx": w_idx,
            "window_start_sec": w_idx * WINDOW_SECONDS,
            "familiarity": beh.get("familiarity", -1),
            "enjoyment": beh.get("enjoyment", -1),
        })

    return samples


def build():
    print("=== MUSIN-G Dataset Builder (32ch) ===")

    # Load behavioral data
    print("\n[1/3] Loading behavioral data...")
    beh_data = load_behavioral_data()
    print(f"  Loaded ratings for {len(beh_data)} subject-song pairs")

    # Process all subjects and sessions
    print(f"\n[2/3] Processing {len(SUBJECT_IDS)} subjects × {len(SONGS)} songs...")
    all_samples = []
    for subject_id in tqdm(SUBJECT_IDS, desc="Subjects"):
        for session_num, song in enumerate(SONGS, start=1):
            try:
                session_samples = process_session(
                    subject_id, session_num, song, beh_data,
                )
                all_samples.extend(session_samples)
            except Exception as e:
                print(f"  WARNING: Failed {subject_id}/ses-{session_num:02d}: {e}")

    print(f"  Total samples: {len(all_samples):,}")

    # Subject-level train/val split
    print("\n[3/3] Splitting and saving...")
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

    MUSIN_G_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, split_samples in [("train", train_samples), ("val", val_samples)]:
        if not split_samples:
            continue

        tensors = torch.stack([s["sample"] for s in split_samples])

        save_file(
            {"samples": tensors},
            MUSIN_G_OUTPUT_DIR / f"musin-g-{split_name}.safetensors",
        )

        metadata = pl.from_dicts([
            {k: v for k, v in s.items() if k != "sample"}
            for s in split_samples
        ])
        metadata.write_parquet(
            MUSIN_G_OUTPUT_DIR / f"musin-g-{split_name}-metadata.parquet",
        )

        print(f"  Saved {split_name}: {tensors.shape}")

    print("\nDone!")


if __name__ == "__main__":
    build()
