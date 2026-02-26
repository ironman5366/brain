"""EEG frequency spectrum (PSD) computation using Welch's method via BrainFlow."""

import numpy as np
from brainflow.data_filter import DataFilter, WindowOperations


def compute_spectrum(
    data: np.ndarray,
    eeg_channels: list[int],
    sampling_rate: int,
    nfft: int = 512,
) -> dict:
    """
    Compute averaged PSD across EEG channels using Welch's method.

    Args:
        data: Full board data array (num_rows, num_samples) from EEGStream.get_recent_data()
        eeg_channels: Row indices for EEG channels
        sampling_rate: Board sampling rate in Hz
        nfft: FFT window size

    Returns dict with:
      - frequencies: list of frequency bin centers (Hz)
      - amplitudes_db: list of power values in dB (10*log10)
      - nfft: int
      - sampling_rate: int
      - num_channels: int
    """
    actual_samples = data.shape[1]
    if actual_samples < nfft:
        return {"error": f"Insufficient data ({actual_samples} samples, need {nfft})", "window_samples": actual_samples}

    overlap = nfft // 2
    all_psds = []

    for ch in eeg_channels:
        channel_data = data[ch].copy()
        # Welch PSD: returns (amplitudes, frequencies)
        psd = DataFilter.get_psd_welch(
            channel_data, nfft, overlap, sampling_rate,
            WindowOperations.HANNING.value
        )
        all_psds.append(psd[0])  # amplitudes
        frequencies = psd[1]

    # Average across channels
    avg_psd = np.mean(all_psds, axis=0)

    # Convert to dB, clamp minimum to avoid log(0)
    avg_psd = np.maximum(avg_psd, 1e-12)
    amplitudes_db = 10.0 * np.log10(avg_psd)

    return {
        "frequencies": frequencies.tolist(),
        "amplitudes_db": amplitudes_db.tolist(),
        "nfft": nfft,
        "sampling_rate": sampling_rate,
        "num_channels": len(eeg_channels),
        "window_samples": actual_samples,
    }
