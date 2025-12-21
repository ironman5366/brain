# Builtin imports
import json
from pathlib import Path

# External imports
import mne
import polars as pl
import torch

# Internal imports
from utils import standardize_epochs

# Location you've downloaded https://huggingface.co/datasets/Alljoined/Alljoined-1.6M
ALLJOINED_BASE_DIR = Path("/kreka/research/willy/side/alljoined")

# Default epoch timing (to match Alljoined preprocessing)
DEFAULT_TMIN = -0.2  # 200ms before stimulus
DEFAULT_TMAX = 1.0  # 1s after stimulus


def parse_markers(json_path: Path | str) -> list[dict]:
    """
    Parse JSON metadata file to extract stimulus and behavioral markers.

    Returns list of dicts with keys:
        - label: "stim_test", "stim_train", "behav", or "oddball"
        - image_id: int (for stim markers) or response code (for behav)
        - index: sequence number
        - start_datetime: ISO timestamp string
    """
    json_path = Path(json_path)
    if not json_path.exists():
        return []

    data = json.loads(json_path.read_text())
    markers = []

    for m in data.get("Markers", []):
        markers.append(
            {
                "label": m["label"],
                "image_id": m["value"],
                "index": m["index"],
                "start_datetime": m["startDatetime"],
            }
        )

    return markers


def load_experiment_metadata(subject_num: int) -> dict[int, dict] | None:
    """
    Load experiment_metadata_categories.parquet for a subject.

    Returns a dict mapping image_id -> metadata dict, or None if not available.
    The metadata dict contains: category_name, super_category, category_num,
    super_category_id, dropped, etc.
    """
    parquet_path = (
        ALLJOINED_BASE_DIR
        / "preprocessed_eeg"
        / f"sub-{subject_num:02d}"
        / "experiment_metadata_categories.parquet"
    )

    if not parquet_path.exists():
        return None

    df = pl.read_parquet(parquet_path)

    # Extract image_id from image_path (last 5 digits before extension)
    df = df.with_columns(
        pl.col("image_path").str.slice(-9, 5).cast(pl.Int64).alias("image_id")
    )

    # Build lookup dict
    # Note: Same image_id can appear multiple times (different sessions/blocks)
    # We'll just take the category info which should be the same
    lookup = {}
    for row in df.iter_rows(named=True):
        image_id = row["image_id"]
        if image_id not in lookup:
            lookup[image_id] = {
                "category_name": row.get("category_name"),
                "super_category": row.get("super_category"),
                "category_num": row.get("category_num"),
                "super_category_id": row.get("super_category_id"),
            }

    return lookup


def extract_subject_num(subject_id: str) -> int:
    """Extract numeric subject ID from string like 'alljoined::sub-01'."""
    # Format: "alljoined::sub-XX"
    return int(subject_id.split("sub-")[-1])


# Cache for experiment metadata per subject
_metadata_cache: dict[int, dict[int, dict] | None] = {}


def load_alljoined_file(
    file_row: dict,
    target_sfreq: float = 256,
    tmin: float = DEFAULT_TMIN,
    tmax: float = DEFAULT_TMAX,
    normalization: str = "epoch",
) -> list[dict]:
    """
    Load a single Alljoined EDF file and return a list of dicts, one per epoch.

    This is the main entry point for loading Alljoined data. It:
    1. Extracts stimulus-locked epochs
    2. Adds category metadata from the experiment parquet
    3. Returns samples ready for the builder

    Args:
        file_row: Dict from fetch_alljoined() with file, subject, session, block, metadata
        target_sfreq: Target sampling frequency
        tmin: Epoch start relative to stimulus
        tmax: Epoch end relative to stimulus
        normalization: "epoch", "recording", or "none"

    Returns:
        List of sample dicts with keys: sample, mask, epoch_idx, image_id,
        partition, onset, seq_num, category_name, super_category, etc.
    """
    # Extract epochs
    epochs_tensor, mask, event_metadata = extract_alljoined_epochs(
        file_row["file"],
        target_sfreq=target_sfreq,
        tmin=tmin,
        tmax=tmax,
        normalization=normalization,
        verbose=False,
    )

    if len(event_metadata) == 0:
        return []

    # Get mask indices for sparse storage
    mask_indices = mask.nonzero().squeeze()

    # Load category metadata for this subject (cached)
    subject_num = extract_subject_num(file_row["subject"])
    if subject_num not in _metadata_cache:
        _metadata_cache[subject_num] = load_experiment_metadata(subject_num)
    category_lookup = _metadata_cache[subject_num]

    # Build one dict per epoch
    samples = []
    for epoch_idx, event in enumerate(event_metadata):
        # Extract sparse representation for this epoch
        sparse_sample = epochs_tensor[epoch_idx, mask]  # (n_active_channels, n_samples)

        sample = {
            "sample": sparse_sample,
            "mask": mask_indices,
            "epoch_idx": epoch_idx,
            "image_id": event["image_id"],
            "partition": event["partition"],
            "onset": event["onset"],
            "seq_num": event["seq_num"],
            **file_row,
        }

        # Add category info if available
        if category_lookup is not None:
            cat_info = category_lookup.get(event["image_id"])
            if cat_info:
                sample.update(cat_info)

        samples.append(sample)

    return samples


def extract_alljoined_epochs(
    edf_path: str | Path,
    target_sfreq: float = 256,
    tmin: float = DEFAULT_TMIN,
    tmax: float = DEFAULT_TMAX,
    baseline: tuple = (None, 0),
    normalization: str = "epoch",
    event_regexp: str = "stim.*",
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """
    Extract stimulus-locked epochs from an Alljoined EDF file.

    This is alljoined-specific because it parses the alljoined annotation format:
    "stim_test,16641,-1,1" or prefixed "session_X,block_Y,stim_test,16641,-1,1"

    Channel filtering is handled by standardize_epochs() which uses the standard
    10-20 channel whitelist.

    Args:
        edf_path: Path to EDF file
        target_sfreq: Target sampling frequency (Hz)
        tmin: Epoch start time relative to stimulus (seconds)
        tmax: Epoch end time relative to stimulus (seconds)
        baseline: Baseline correction tuple (start, end) in seconds
        normalization: "epoch", "recording", or "none"
        event_regexp: Regex to filter annotation labels
        verbose: Whether to print progress

    Returns:
        data: (n_epochs, NUM_CHANNELS, n_samples) tensor
        mask: (NUM_CHANNELS,) bool tensor indicating active channels
        events: list of dicts with {image_id, partition, onset, seq_num}
    """
    from constants import NUM_CHANNELS

    # Load raw data
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Fix EMOTIV capitalization issue (Afz should be AFz for standard montage)
    if "Afz" in raw.ch_names:
        raw.rename_channels({"Afz": "AFz"})

    # Resample if needed
    if raw.info["sfreq"] != target_sfreq:
        if verbose:
            print(f"Resampling from {raw.info['sfreq']} -> {target_sfreq}")
        raw.resample(target_sfreq)

    # Get events from annotations
    events, event_id = mne.events_from_annotations(raw, regexp=event_regexp, verbose=False)

    if len(events) == 0:
        if verbose:
            print("No events found matching pattern")
        return torch.zeros((0, NUM_CHANNELS, 0)), torch.zeros(NUM_CHANNELS, dtype=torch.bool), []

    # Create epochs
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        verbose=False,
    )

    # Get epoch data as tensor
    epoch_data = torch.from_numpy(epochs.get_data()).to(torch.float32)

    if verbose:
        print(f"Created {len(epochs)} epochs, shape: {epoch_data.shape}")

    # Standardize channels (whitelist-based) and normalize
    standardized, mask = standardize_epochs(
        epoch_data,
        epochs.ch_names,
        normalization=normalization,
    )

    # Parse event descriptions to extract alljoined-specific metadata
    # Format: "stim_test,16641,-1,1" or "session_X,block_Y,stim_test,16641,-1,1"
    code_to_desc = {v: k for k, v in event_id.items()}
    event_metadata = []

    for i, event in enumerate(epochs.events):
        desc = code_to_desc[event[2]]
        parts = desc.split(",")

        # Handle both formats:
        # Simple: "stim_test,16641,-1,1"
        # Prefixed: "session_1,block_1,stim_test,16641,-1,1"
        if parts[0].startswith("stim_"):
            partition = parts[0]
            image_id = int(parts[1])
            seq_num = int(parts[3])
        else:
            # Skip prefix parts until we find stim_
            partition = "unknown"
            image_id = -1
            seq_num = -1
            for j, p in enumerate(parts):
                if p.startswith("stim_"):
                    partition = p
                    image_id = int(parts[j + 1])
                    seq_num = int(parts[j + 3])
                    break

        onset = epochs.events[i, 0] / epochs.info["sfreq"]

        event_metadata.append({
            "image_id": image_id,
            "partition": partition,
            "onset": onset,
            "seq_num": seq_num,
        })

    return standardized, mask, event_metadata


def fetch_alljoined():
    raw_dir = ALLJOINED_BASE_DIR / "raw_eeg"

    for subj_dir in raw_dir.iterdir():
        # Like "alljoined::sub-01"
        subj_id = f"alljoined::{subj_dir.name}"

        for session_dir in subj_dir.iterdir():
            session_id = f"{subj_id}::{session_dir.name}"

            for block_dir in session_dir.iterdir():
                if not block_dir.is_dir():
                    continue

                if not block_dir.name.startswith("block_"):
                    print(f"Malformed block directory {block_dir}, skipping...")
                    continue

                block_id = f"{session_id}::{block_dir.name}"

                metadata_files = list(block_dir.glob("*.json"))
                edf_files = list(block_dir.glob("*.edf"))

                # Make sure we're not missing something
                assert len(edf_files) == 1, f"Not exactly 1 edf file in {block_dir}"

                metadata_file = None
                if metadata_files:
                    assert len(metadata_files) == 1, (
                        f"More than 1 metadata file in {block_dir}"
                    )
                    metadata_file = metadata_files[0]

                yield {
                    "file": str(edf_files[0]),
                    "subject": subj_id,
                    "session": session_id,
                    "block": block_id,
                    "metadata": str(metadata_file),
                }
