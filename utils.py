# External imports
import mne
import torch

# Internal imports
from constants import (
    DEFAULT_SFREQ,
    DEFAULT_WINDOW_SECONDS,
    DEFAULT_NORMALIZATION,
    CHANNEL_TO_IDX,
    NUM_CHANNELS,
)


def normalize_channel_name(name: str) -> str:
    """
    Normalize channel name by removing common prefixes/suffixes.
    """
    name = name.upper().strip()

    # Remove common prefixes
    prefixes = ["EEG ", "EEG-", "EEG_"]
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix) :]

    # Remove common suffixes (references)
    suffixes = ["-REF", "-LE", "-AVG", "-AR", "-M1", "-M2", "-A1", "-A2", "_REF"]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    return name.strip()


def mne_to_tensor(
    raw: mne.io.BaseRaw,
    target_sfreq: float = DEFAULT_SFREQ,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    overlap: float = 0.0,  # 0.0 = no overlap, 0.5 = 50% overlap
    normalization: str = DEFAULT_NORMALIZATION,  # "window", "recording", or "none"
    eps: float = 1e-8,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert MNE Raw object to standardized windowed tensors.

    Args:
        raw: MNE Raw object (from EDF, BDF, etc.)
        target_sfreq: Target sampling frequency in Hz
        window_seconds: Window length in seconds
        overlap: Fraction of overlap between windows (0.0 to <1.0)
        normalization: Normalization strategy:
            - "window": z-score each window/channel independently
            - "recording": z-score per channel across entire recording
            - "none": no normalization

    Returns:
        data: torch.Tensor of shape (n_windows, NUM_CHANNELS, window_samples)
        mask: torch.Tensor of shape (NUM_CHANNELS,), True where channel has real data
    """
    # Resample if needed
    if raw.info["sfreq"] != target_sfreq:
        print(f"Resamping from {raw.info['sfreq']} -> {target_sfreq}")
        raw = raw.copy().resample(target_sfreq)

    # Get the raw data
    raw_data = torch.from_numpy(raw.get_data())  # (n_channels_in_file, n_times)

    if verbose:
        print("raw data shape", raw_data.shape)

    n_times = raw_data.shape[1]
    window_samples = int(target_sfreq * window_seconds)

    # Map to standard channels
    standardized = torch.zeros((NUM_CHANNELS, n_times), dtype=torch.float32)
    mask = torch.zeros(NUM_CHANNELS, dtype=torch.bool)

    for file_idx, ch_name in enumerate(raw.ch_names):
        normalized = normalize_channel_name(ch_name)
        if normalized in CHANNEL_TO_IDX:
            std_idx = CHANNEL_TO_IDX[normalized]
            standardized[std_idx] = raw_data[file_idx]
            mask[std_idx] = True
        else:
            if verbose:
                print(f"Skipping channel: {ch_name}")

    # Recording-level normalization (per channel, across all time)
    if normalization == "recording":
        for ch_idx in range(NUM_CHANNELS):
            if mask[ch_idx]:
                mean = standardized[ch_idx].mean()
                std = standardized[ch_idx].std()
                standardized[ch_idx] = (standardized[ch_idx] - mean) / (std + eps)

    # Chunk into windows
    step = int(window_samples * (1 - overlap))
    n_windows = (n_times - window_samples) // step + 1

    if n_windows <= 0:
        raise ValueError(
            f"Recording too short: {n_times} samples < {window_samples} window size"
        )

    data = torch.zeros((n_windows, NUM_CHANNELS, window_samples), dtype=torch.float32)

    for i in range(n_windows):
        start = i * step
        end = start + window_samples
        data[i] = standardized[:, start:end]

    # Window-level normalization (per window, per channel)
    if normalization == "window":
        for i in range(n_windows):
            for ch_idx in range(NUM_CHANNELS):
                if mask[ch_idx]:
                    mean = data[i, ch_idx].mean()
                    std = data[i, ch_idx].std()
                    data[i, ch_idx] = (data[i, ch_idx] - mean) / (std + eps)

    return data, mask
