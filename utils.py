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
    overlap: float = 0.0,
    normalization: str = DEFAULT_NORMALIZATION,  # "window", "recording", or "none"
    eps: float = 1e-8,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Resample if needed
    if raw.info["sfreq"] != target_sfreq:
        if verbose:
            print(f"Resamping from {raw.info['sfreq']} -> {target_sfreq}")
        raw = raw.copy().resample(target_sfreq)

    raw_data = torch.from_numpy(raw.get_data()).to(
        torch.float32
    )  # (n_ch_in_file, n_times)

    if verbose:
        print("raw data shape", raw_data.shape)

    n_times = raw_data.shape[1]
    window_samples = int(target_sfreq * window_seconds)

    standardized = torch.zeros((NUM_CHANNELS, n_times), dtype=torch.float32)
    mask = torch.zeros(NUM_CHANNELS, dtype=torch.bool)

    for file_idx, ch_name in enumerate(raw.ch_names):
        normalized_name = normalize_channel_name(ch_name)
        std_idx = CHANNEL_TO_IDX.get(normalized_name)
        if std_idx is not None:
            standardized[std_idx] = raw_data[file_idx]
            mask[std_idx] = True
        else:
            if verbose:
                print(f"Skipping channel: {ch_name}")

    # Recording-level normalization (match original: unbiased=True default)
    if normalization == "recording":
        x = standardized[mask]  # (n_real_channels, n_times)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)  # unbiased=True by default (matches your code)
        standardized[mask] = (x - mean) / (std + eps)

    # Windowing (vectorized)
    step = int(window_samples * (1 - overlap))
    if step <= 0:
        raise ValueError(f"Invalid overlap={overlap}: step would be {step}")

    if n_times < window_samples:
        raise ValueError(
            f"Recording too short: {n_times} samples < {window_samples} window size"
        )

    win = standardized.unfold(dimension=1, size=window_samples, step=step)  # (C, W, S)
    n_windows = win.shape[1]
    if n_windows <= 0:
        raise ValueError(
            f"Recording too short for step={step}: got n_windows={n_windows}"
        )

    data = win.permute(1, 0, 2).contiguous()  # (W, C, S)

    # Window-level normalization (match original: only masked channels, unbiased=True default)
    if normalization == "window":
        x = data[:, mask, :]  # (W, C_real, S)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)  # unbiased=True default
        data[:, mask, :] = (x - mean) / (std + eps)

    elif normalization != "none":
        raise ValueError(f"Unknown normalization strategy: {normalization}")

    return data, mask


def standardize_epochs(
    epoch_data: torch.Tensor,
    ch_names: list[str],
    normalization: str = "epoch",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standardize epoch data to fixed channel positions and apply normalization.

    This is a source-agnostic function that maps arbitrary channel names to
    the standard 10-20 positions and applies normalization.

    Args:
        epoch_data: (n_epochs, n_channels, n_times) raw epoch tensor
        ch_names: list of channel names matching epoch_data's channel dimension
        normalization: "epoch", "recording", or "none"
        eps: Small constant for numerical stability

    Returns:
        standardized: (n_epochs, NUM_CHANNELS, n_times) tensor with channels
                      mapped to standard positions
        mask: (NUM_CHANNELS,) bool tensor indicating active channels
    """
    n_epochs, _, n_times = epoch_data.shape

    standardized = torch.zeros((n_epochs, NUM_CHANNELS, n_times), dtype=torch.float32)
    mask = torch.zeros(NUM_CHANNELS, dtype=torch.bool)

    for file_idx, ch_name in enumerate(ch_names):
        normalized_name = normalize_channel_name(ch_name)
        std_idx = CHANNEL_TO_IDX.get(normalized_name)
        if std_idx is not None:
            standardized[:, std_idx, :] = epoch_data[:, file_idx, :]
            mask[std_idx] = True

    # Apply normalization
    if normalization == "recording":
        # Normalize across all epochs using full recording stats
        x = standardized[:, mask, :]  # (n_epochs, n_real_channels, n_times)
        mean = x.mean(dim=(0, 2), keepdim=True)  # Mean across epochs and time
        std = x.std(dim=(0, 2), keepdim=True)
        standardized[:, mask, :] = (x - mean) / (std + eps)
    elif normalization == "epoch":
        # Normalize each epoch independently
        x = standardized[:, mask, :]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        standardized[:, mask, :] = (x - mean) / (std + eps)
    elif normalization != "none":
        raise ValueError(f"Unknown normalization strategy: {normalization}")

    return standardized, mask
