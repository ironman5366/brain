"""EEG band power computation using Welch's PSD method via BrainFlow."""

import numpy as np
from brainflow.data_filter import DataFilter

# Standard EEG frequency bands: (name, low_hz, high_hz, description)
BANDS = [
    ("delta", 0.5, 4.0, "Deep sleep"),
    ("theta", 4.0, 8.0, "Drowsiness, meditation, memory"),
    ("alpha", 8.0, 13.0, "Relaxed, eyes closed"),
    ("beta", 13.0, 30.0, "Active thinking, focus"),
    ("gamma", 30.0, 100.0, "Higher cognitive processing"),
]

BAND_RANGES = [(low, high) for _, low, high, _ in BANDS]


def compute_band_powers(
    data: np.ndarray,
    eeg_channels: list[int],
    sampling_rate: int,
) -> dict:
    """
    Compute EEG band powers from a data array.

    Args:
        data: Full board data array (num_rows, num_samples) from EEGStream.get_recent_data()
        eeg_channels: Row indices for EEG channels
        sampling_rate: Board sampling rate in Hz

    Returns dict with:
      - bands: list of {name, low, high, power, relative, stddev, description}
      - total_power: float
      - window_samples: int
    """
    actual_samples = data.shape[1]
    if actual_samples < sampling_rate:
        return {"error": "Insufficient data", "window_samples": actual_samples}

    # get_custom_band_powers returns (avg_powers, std_powers) across channels
    avg_powers, std_powers = DataFilter.get_custom_band_powers(
        data, BAND_RANGES, eeg_channels, sampling_rate, True
    )

    total_power = float(np.sum(avg_powers))

    bands = []
    for i, (name, low, high, desc) in enumerate(BANDS):
        bands.append({
            "name": name,
            "low": low,
            "high": high,
            "power": float(avg_powers[i]),
            "relative": float(avg_powers[i] / total_power) if total_power > 0 else 0,
            "stddev": float(std_powers[i]),
            "description": desc,
        })

    return {
        "bands": bands,
        "total_power": total_power,
        "window_samples": actual_samples,
        "sampling_rate": sampling_rate,
        "num_channels": len(eeg_channels),
    }
