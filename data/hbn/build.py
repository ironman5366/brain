"""
Build the HBN movie-watching EEG dataset into safetensors + parquet format.

Processes up to 2,639 subjects of 129-channel EEG recorded during movie watching.
Extracts video-watching segments, downsamples to 125 Hz, maps to 32-channel layout,
windows into 1-second epochs, and normalizes.

Usage:
    uv run python -m data.hbn.build [--max-subjects N] [--release R1]
"""

import argparse
import random
from pathlib import Path

import mne
import polars as pl
import torch
from safetensors.torch import save_file
from tqdm import tqdm

from data.hbn.movies import (
    EGI129_TO_32CH_MAP,
    HBN_BIDS_DIR,
    HBN_OUTPUT_DIR,
    MOVIE_BY_TASK,
    MOVIE_TASK_NAMES,
    RELEASES,
    TARGET_SFREQ,
)
from data.songfam.channel_mapping import TARGET_CHANNELS
from utils import map_egi_to_32ch

WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = int(TARGET_SFREQ * WINDOW_SECONDS)  # 125
NORMALIZATION = "epoch"
TRAIN_SPLIT = 0.9
EPS = 1e-8

# Expected channel count for HBN EGI data
EXPECTED_N_CHANNELS = 129


def scan_subjects(
    max_subjects: int | None = None,
    releases: list[str] | None = None,
) -> list[dict]:
    """Scan BIDS releases and return a manifest of valid subjects.

    Returns list of dicts with keys: subject_id, release, tasks (list of valid task names).
    """
    if releases is None:
        releases = RELEASES

    subjects = {}

    for release in releases:
        release_dir = HBN_BIDS_DIR / release
        participants_path = release_dir / "participants.tsv"
        if not participants_path.exists():
            print(f"  Skipping {release}: no participants.tsv")
            continue

        df = pl.read_csv(participants_path, separator="\t")

        for row in df.iter_rows(named=True):
            sub_id = row["participant_id"]

            available_tasks = []
            for task_name in MOVIE_TASK_NAMES:
                if row.get(task_name) != "available":
                    continue

                # Verify .set file exists
                set_path = (
                    release_dir
                    / sub_id
                    / "eeg"
                    / f"{sub_id}_task-{task_name}_eeg.set"
                )
                if set_path.exists():
                    available_tasks.append(task_name)

            if available_tasks:
                subjects[sub_id] = {
                    "subject_id": sub_id,
                    "release": release,
                    "tasks": available_tasks,
                }

    subject_list = list(subjects.values())

    if max_subjects is not None:
        subject_list = subject_list[:max_subjects]

    return subject_list


def process_subject_task(
    sub_id: str,
    release: str,
    task_name: str,
) -> list[dict]:
    """Process one (subject, task) pair into windowed 32-channel samples."""
    base = HBN_BIDS_DIR / release / sub_id / "eeg"
    set_path = base / f"{sub_id}_task-{task_name}_eeg.set"
    events_path = base / f"{sub_id}_task-{task_name}_events.tsv"

    # Load events, find video_start and video_stop
    events_df = pl.read_csv(events_path, separator="\t")
    start_rows = events_df.filter(pl.col("value") == "video_start")
    stop_rows = events_df.filter(pl.col("value") == "video_stop")

    if len(start_rows) == 0 or len(stop_rows) == 0:
        return []

    video_start_sec = float(start_rows["onset"][0])
    video_stop_sec = float(stop_rows["onset"][0])

    # Load EEG
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)

    # Validate channel count
    n_ch = len(raw.ch_names)
    if n_ch < EXPECTED_N_CHANNELS:
        return []

    # Resample to 125 Hz
    if raw.info["sfreq"] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, verbose=False)

    eeg_data = torch.from_numpy(raw.get_data()).float()  # (n_ch, T)

    # Extract video segment
    start_sample = int(video_start_sec * TARGET_SFREQ)
    stop_sample = int(video_stop_sec * TARGET_SFREQ)

    if stop_sample > eeg_data.shape[1]:
        stop_sample = eeg_data.shape[1]

    segment = eeg_data[:EXPECTED_N_CHANNELS, start_sample:stop_sample]  # (129, dur)

    # Window into 1-second epochs
    n_windows = segment.shape[1] // WINDOW_SAMPLES
    if n_windows == 0:
        return []

    usable = n_windows * WINDOW_SAMPLES
    windowed = segment[:, :usable].reshape(EXPECTED_N_CHANNELS, n_windows, WINDOW_SAMPLES)
    windowed = windowed.permute(1, 0, 2)  # (W, 129, 125)

    # Map 129ch -> 32ch via IDW
    mapped = map_egi_to_32ch(
        windowed,
        EGI129_TO_32CH_MAP,
        TARGET_CHANNELS,
        normalization=NORMALIZATION,
    )  # (W, 32, 125)

    movie = MOVIE_BY_TASK[task_name]
    samples = []
    for w_idx in range(n_windows):
        samples.append({
            "sample": mapped[w_idx],
            "subject_id": sub_id,
            "task_name": task_name,
            "movie_title": movie.title,
            "window_idx": w_idx,
            "window_start_sec": w_idx * WINDOW_SECONDS,
        })

    return samples


def process_subject(subject_info: dict) -> list[dict]:
    """Process all available tasks for one subject."""
    samples = []
    sub_id = subject_info["subject_id"]
    release = subject_info["release"]

    for task_name in subject_info["tasks"]:
        try:
            task_samples = process_subject_task(sub_id, release, task_name)
            samples.extend(task_samples)
        except Exception as e:
            print(f"  WARNING: {sub_id}/{task_name}: {e}")

    return samples


def build(max_subjects: int | None = None, releases: list[str] | None = None):
    print("=== HBN Movie Watching EEG Dataset Builder ===")

    # Step 1: Scan for valid subjects
    print("\n[1/3] Scanning subjects...")
    subject_manifest = scan_subjects(max_subjects=max_subjects, releases=releases)
    total_tasks = sum(len(s["tasks"]) for s in subject_manifest)
    print(f"  Found {len(subject_manifest)} subjects, {total_tasks} (subject, task) pairs")

    # Step 2: Process all subjects
    print(f"\n[2/3] Processing subjects...")
    all_samples = []
    for subject_info in tqdm(subject_manifest, desc="Subjects"):
        subject_samples = process_subject(subject_info)
        all_samples.extend(subject_samples)

    print(f"  Total samples: {len(all_samples):,}")

    if not all_samples:
        print("  No samples produced. Exiting.")
        return

    # Step 3: Split and save
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

    HBN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, split_samples in [("train", train_samples), ("val", val_samples)]:
        if not split_samples:
            continue

        tensors = torch.stack([s["sample"] for s in split_samples])

        save_file(
            {"samples": tensors},
            HBN_OUTPUT_DIR / f"hbn-{split_name}.safetensors",
        )

        metadata = pl.from_dicts([
            {k: v for k, v in s.items() if k != "sample"}
            for s in split_samples
        ])
        metadata.write_parquet(
            HBN_OUTPUT_DIR / f"hbn-{split_name}-metadata.parquet",
        )

        print(f"  Saved {split_name}: {tensors.shape}")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build HBN movie watching EEG dataset")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Limit number of subjects (for testing)")
    parser.add_argument("--release", type=str, default=None,
                        help="Process only this release (e.g. cmi_bids_R1)")
    args = parser.parse_args()

    releases = [args.release] if args.release else None
    build(max_subjects=args.max_subjects, releases=releases)
